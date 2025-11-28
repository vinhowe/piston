//! Fusion group representation.

use super::{FusionNodeId, OpPattern};

/// A group of operations that will be fused into a single kernel.
///
/// The group has:
/// - An "anchor" node that determines the dispatch strategy
/// - A list of nodes in topological order
/// - External inputs (nodes outside the group that feed into it)
/// - A single output node
#[derive(Debug, Clone)]
pub struct FusionGroup {
    /// The "anchor" or "master" operation that determines dispatch.
    ///
    /// - For elementwise chains: the last operation
    /// - For compute-intensive: the matmul/conv
    /// - For reductions: the reduction itself
    pub anchor: FusionNodeId,

    /// All nodes in this group, in topological order.
    pub nodes: Vec<FusionNodeId>,

    /// The dominant pattern (highest pattern value in group).
    /// Determines the codegen strategy.
    pub dominant_pattern: OpPattern,

    /// External inputs: nodes outside this group that feed into it.
    pub external_inputs: Vec<FusionNodeId>,

    /// The output node of this group.
    pub output: FusionNodeId,
}

impl FusionGroup {
    /// Create a new fusion group with a single anchor node.
    pub fn new(anchor: FusionNodeId, pattern: OpPattern) -> Self {
        Self {
            anchor,
            nodes: vec![anchor],
            dominant_pattern: pattern,
            external_inputs: Vec::new(),
            output: anchor,
        }
    }

    /// Add a node to this group (as a prologue).
    ///
    /// The node is added at the beginning of the nodes list
    /// to maintain topological order (inputs before outputs).
    pub fn add_prologue(&mut self, node: FusionNodeId, pattern: OpPattern) {
        self.nodes.insert(0, node);
        self.dominant_pattern = self.dominant_pattern.dominant(pattern);
    }

    /// Add a node to this group (as an epilogue).
    ///
    /// The node is added at the end and becomes the new output.
    pub fn add_epilogue(&mut self, node: FusionNodeId, pattern: OpPattern) {
        self.nodes.push(node);
        self.dominant_pattern = self.dominant_pattern.dominant(pattern);
        self.output = node;
    }

    /// Returns true if this group contains compute-intensive ops.
    pub fn is_compute_intensive(&self) -> bool {
        self.dominant_pattern == OpPattern::ComputeIntensive
    }

    /// Returns true if this group contains reductions.
    pub fn is_reduce(&self) -> bool {
        self.dominant_pattern == OpPattern::Reduce
    }

    /// Returns true if this is a pure elementwise/injective group.
    pub fn is_elemwise_only(&self) -> bool {
        self.dominant_pattern <= OpPattern::Injective
    }

    /// Returns the number of nodes in this group.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if this group has only one node.
    pub fn is_singleton(&self) -> bool {
        self.nodes.len() == 1
    }
}

