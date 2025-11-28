//! Fusion graph representation.

use crate::{DType, HashMap, LazyOp, OpTensor, RVec, Shape, TensorId};
use slotmap::{SlotMap, new_key_type};

use super::{FusionGroup, OpPattern};

new_key_type! {
    /// Unique identifier for a node in the fusion graph.
    pub struct FusionNodeId;
}

/// A node in the fusion graph representing a single operation.
#[derive(Debug, Clone)]
pub struct FusionNode {
    /// Unique identifier
    pub id: FusionNodeId,

    /// The fusion group this node belongs to (None if not yet grouped)
    pub group: Option<FusionGroupId>,

    /// ID of the tensor this node produces
    pub tensor_id: TensorId,

    /// Operation pattern for fusion analysis
    pub pattern: OpPattern,

    /// Input node IDs (dependencies)
    pub inputs: RVec<FusionNodeId>,

    /// Nodes that consume this output
    pub consumers: RVec<FusionNodeId>,

    /// The underlying lazy operation
    pub op: LazyOp,

    /// Output shape
    pub shape: Shape,

    /// Output data type
    pub dtype: DType,
}

new_key_type! {
    /// Unique identifier for a fusion group.
    pub struct FusionGroupId;
}

/// The complete fusion graph for a computation.
///
/// This is a DAG where nodes represent operations and edges represent
/// data dependencies. Nodes are grouped into [`FusionGroup`]s that will
/// be compiled into single kernels.
#[derive(Debug)]
pub struct FusionGraph {
    /// All nodes in the graph
    pub nodes: SlotMap<FusionNodeId, FusionNode>,

    /// All fusion groups
    pub groups: SlotMap<FusionGroupId, FusionGroup>,

    /// Mapping from TensorId to FusionNodeId
    tensor_to_node: HashMap<TensorId, FusionNodeId>,

    /// Nodes that haven't been assigned to a group yet
    pub ungrouped: Vec<FusionNodeId>,

    /// Final output nodes of the computation
    pub outputs: Vec<FusionNodeId>,
}

impl FusionGraph {
    /// Create an empty fusion graph.
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            groups: SlotMap::with_key(),
            tensor_to_node: HashMap::default(),
            ungrouped: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Build a fusion graph from a post-order traversal of tensors.
    pub fn from_post_order(tensors: &[&OpTensor]) -> Self {
        let mut graph = Self::new();

        // Phase 1: Create nodes for each tensor
        for tensor in tensors {
            let id = graph.add_node(tensor);
            graph.ungrouped.push(id);
        }

        // Phase 2: Build edges based on operation sources
        for tensor in tensors {
            let node_id = graph.tensor_to_node[&tensor.id()];
            let mut inputs = RVec::new();

            for src in tensor.op().srcs() {
                if let Some(&src_node_id) = graph.tensor_to_node.get(&src.id()) {
                    inputs.push(src_node_id);
                    // Add this node as a consumer of the source
                    graph.nodes[src_node_id].consumers.push(node_id);
                }
            }

            graph.nodes[node_id].inputs = inputs;
        }

        // Phase 3: Identify output nodes (nodes with no consumers)
        graph.outputs = graph
            .nodes
            .iter()
            .filter(|(_, node)| node.consumers.is_empty())
            .map(|(id, _)| id)
            .collect();

        // Phase 4: Run fusion analysis
        graph.fuse();

        graph
    }

    /// Add a node for a tensor.
    fn add_node(&mut self, tensor: &OpTensor) -> FusionNodeId {
        let op = tensor.op().clone();
        let pattern = op.pattern();

        let node = FusionNode {
            id: FusionNodeId::default(), // Will be set by slotmap
            group: None,
            tensor_id: tensor.id(),
            pattern,
            inputs: RVec::new(),
            consumers: RVec::new(),
            op,
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
        };

        let id = self.nodes.insert(node);
        self.nodes[id].id = id;
        self.tensor_to_node.insert(tensor.id(), id);
        id
    }

    /// Look up the node ID for a tensor.
    pub fn node_for_tensor(&self, tensor_id: TensorId) -> Option<FusionNodeId> {
        self.tensor_to_node.get(&tensor_id).copied()
    }

    /// Get a node by ID.
    pub fn node(&self, id: FusionNodeId) -> Option<&FusionNode> {
        self.nodes.get(id)
    }

    /// Get a group by ID.
    pub fn group(&self, id: FusionGroupId) -> Option<&FusionGroup> {
        self.groups.get(id)
    }

    /// Iterate over all fusion groups.
    pub fn iter_groups(&self) -> impl Iterator<Item = (FusionGroupId, &FusionGroup)> {
        self.groups.iter()
    }

    /// Returns nodes in topological order (dependencies before dependents).
    pub fn topological_order(&self) -> Vec<FusionNodeId> {
        let mut result = Vec::with_capacity(self.nodes.len());
        let mut visited = HashMap::default();

        for (id, _) in &self.nodes {
            self.topo_visit(id, &mut visited, &mut result);
        }

        result
    }

    fn topo_visit(
        &self,
        node_id: FusionNodeId,
        visited: &mut HashMap<FusionNodeId, bool>,
        result: &mut Vec<FusionNodeId>,
    ) {
        if let Some(&done) = visited.get(&node_id) {
            if !done {
                panic!("Cycle detected in fusion graph");
            }
            return;
        }

        visited.insert(node_id, false); // Mark as in-progress

        let node = &self.nodes[node_id];
        for &input_id in &node.inputs {
            self.topo_visit(input_id, visited, result);
        }

        visited.insert(node_id, true); // Mark as done
        result.push(node_id);
    }

    /// Check if two nodes can be fused according to TVM-style rules.
    pub fn can_fuse(&self, producer: FusionNodeId, consumer: FusionNodeId) -> bool {
        let prod = &self.nodes[producer];
        let cons = &self.nodes[consumer];

        // Rule 5: Opaque blocks all fusion
        if prod.pattern == OpPattern::Opaque || cons.pattern == OpPattern::Opaque {
            return false;
        }

        // Producer must have single consumer for simple fusion
        // (multi-consumer fusion is more complex)
        if prod.consumers.len() > 1 {
            return false;
        }

        // Already in same group
        if prod.group.is_some() && prod.group == cons.group {
            return false;
        }

        match (prod.pattern, cons.pattern) {
            // Rule 1 & 2: Elementwise/Injective chains
            (OpPattern::Elemwise, OpPattern::Elemwise) => true,
            (OpPattern::Elemwise, OpPattern::Injective) => true,
            (OpPattern::Injective, OpPattern::Elemwise) => true,
            (OpPattern::Injective, OpPattern::Injective) => true,

            // Rule 3: Reduce can absorb elemwise/injective prologue
            (OpPattern::Elemwise, OpPattern::Reduce) => true,
            (OpPattern::Injective, OpPattern::Reduce) => true,

            // Rule 3 (negative): Reduce output cannot fuse forward
            // The reduction produces a value that must be materialized
            (OpPattern::Reduce, _) => false,

            // Rule 4: ComputeIntensive can absorb elemwise epilogue
            (OpPattern::ComputeIntensive, OpPattern::Elemwise) => {
                // Additional check: consumer must be a valid epilogue op
                cons.op.is_gemm_epilogue_candidate()
            }

            // Rule 4 (negative): Cannot fuse into compute-intensive
            // Would change the tiling structure
            (_, OpPattern::ComputeIntensive) => false,

            _ => false,
        }
    }
}

impl Default for FusionGraph {
    fn default() -> Self {
        Self::new()
    }
}

