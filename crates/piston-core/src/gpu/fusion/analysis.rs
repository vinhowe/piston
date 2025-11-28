//! Fusion analysis pass.

use super::{FusionGraph, FusionGroup, FusionGroupId, FusionNodeId};

impl FusionGraph {
    /// Run the fusion analysis pass to group nodes.
    ///
    /// Uses a greedy algorithm similar to TVM's partition_graph:
    /// 1. Process nodes in reverse topological order (outputs first)
    /// 2. For each node, try to join existing groups or create new ones
    pub fn fuse(&mut self) {
        // Process in reverse topological order so we process consumers before producers
        let topo_order: Vec<_> = self.topological_order().into_iter().rev().collect();

        for node_id in topo_order {
            if self.nodes[node_id].group.is_some() {
                continue; // Already grouped
            }

            self.try_fuse_node(node_id);
        }

        // Update external inputs for each group
        self.compute_external_inputs();
    }

    /// Try to fuse a node into an existing group or create a new one.
    fn try_fuse_node(&mut self, node_id: FusionNodeId) {
        let node = &self.nodes[node_id];
        let pattern = node.pattern;

        // Strategy 1: If this node has exactly one consumer that's already grouped,
        // try to join that group as a prologue
        if node.consumers.len() == 1 {
            let consumer_id = node.consumers[0];
            if let Some(group_id) = self.nodes[consumer_id].group {
                if self.can_fuse(node_id, consumer_id) {
                    self.add_to_group_as_prologue(node_id, group_id);
                    return;
                }
            }
        }

        // Strategy 2: Try to fuse with inputs (as epilogue to an input's group)
        let inputs = self.nodes[node_id].inputs.clone();
        for input_id in inputs {
            let input_node = &self.nodes[input_id];

            // Input must be ungrouped or be the output of its group
            let can_extend = input_node.group.is_none()
                || self
                    .groups
                    .get(input_node.group.unwrap())
                    .map(|g| g.output == input_id)
                    .unwrap_or(false);

            if can_extend && self.can_fuse(input_id, node_id) {
                // If input has no group, create one
                let group_id = input_node.group.unwrap_or_else(|| {
                    let group = FusionGroup::new(input_id, self.nodes[input_id].pattern);
                    let id = self.groups.insert(group);
                    self.nodes[input_id].group = Some(id);
                    id
                });

                self.add_to_group_as_epilogue(node_id, group_id);
                return;
            }
        }

        // Strategy 3: Create a singleton group
        let group = FusionGroup::new(node_id, pattern);
        let group_id = self.groups.insert(group);
        self.nodes[node_id].group = Some(group_id);
    }

    /// Add a node to a group as a prologue (input-side).
    fn add_to_group_as_prologue(&mut self, node_id: FusionNodeId, group_id: FusionGroupId) {
        let pattern = self.nodes[node_id].pattern;
        self.nodes[node_id].group = Some(group_id);
        self.groups[group_id].add_prologue(node_id, pattern);
    }

    /// Add a node to a group as an epilogue (output-side).
    fn add_to_group_as_epilogue(&mut self, node_id: FusionNodeId, group_id: FusionGroupId) {
        let pattern = self.nodes[node_id].pattern;
        self.nodes[node_id].group = Some(group_id);
        self.groups[group_id].add_epilogue(node_id, pattern);
    }

    /// Compute external inputs for each group.
    fn compute_external_inputs(&mut self) {
        let group_ids: Vec<_> = self.groups.keys().collect();

        for group_id in group_ids {
            let group = &self.groups[group_id];
            let group_nodes: std::collections::HashSet<_> = group.nodes.iter().cloned().collect();

            let mut external_inputs = Vec::new();

            for &node_id in &group.nodes {
                let node = &self.nodes[node_id];
                for &input_id in &node.inputs {
                    // If input is not in this group, it's external
                    if !group_nodes.contains(&input_id) {
                        if !external_inputs.contains(&input_id) {
                            external_inputs.push(input_id);
                        }
                    }
                }
            }

            self.groups[group_id].external_inputs = external_inputs;
        }
    }

    /// Get statistics about the fusion results.
    pub fn stats(&self) -> FusionStats {
        let total_nodes = self.nodes.len();
        let total_groups = self.groups.len();

        let singleton_groups = self.groups.values().filter(|g| g.is_singleton()).count();

        let fused_nodes = self
            .groups
            .values()
            .filter(|g| !g.is_singleton())
            .map(|g| g.len())
            .sum::<usize>();

        let max_group_size = self.groups.values().map(|g| g.len()).max().unwrap_or(0);

        let elemwise_groups = self.groups.values().filter(|g| g.is_elemwise_only()).count();

        let reduce_groups = self.groups.values().filter(|g| g.is_reduce()).count();

        let compute_groups = self
            .groups
            .values()
            .filter(|g| g.is_compute_intensive())
            .count();

        FusionStats {
            total_nodes,
            total_groups,
            singleton_groups,
            fused_nodes,
            max_group_size,
            elemwise_groups,
            reduce_groups,
            compute_groups,
        }
    }
}

/// Statistics about fusion results.
#[derive(Debug, Clone)]
pub struct FusionStats {
    /// Total number of operations
    pub total_nodes: usize,
    /// Total number of fusion groups (= number of kernels)
    pub total_groups: usize,
    /// Groups with only one node (no fusion)
    pub singleton_groups: usize,
    /// Nodes that are in multi-node groups
    pub fused_nodes: usize,
    /// Maximum number of nodes in a single group
    pub max_group_size: usize,
    /// Groups that are purely elementwise/injective
    pub elemwise_groups: usize,
    /// Groups containing reductions
    pub reduce_groups: usize,
    /// Groups containing compute-intensive ops
    pub compute_groups: usize,
}

impl std::fmt::Display for FusionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fusion Statistics:")?;
        writeln!(f, "  Total nodes:      {}", self.total_nodes)?;
        writeln!(f, "  Total groups:     {}", self.total_groups)?;
        writeln!(f, "  Singleton groups: {}", self.singleton_groups)?;
        writeln!(f, "  Fused nodes:      {}", self.fused_nodes)?;
        writeln!(f, "  Max group size:   {}", self.max_group_size)?;
        writeln!(f, "  Elemwise groups:  {}", self.elemwise_groups)?;
        writeln!(f, "  Reduce groups:    {}", self.reduce_groups)?;
        writeln!(f, "  Compute groups:   {}", self.compute_groups)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Add integration tests with actual tensors
}

