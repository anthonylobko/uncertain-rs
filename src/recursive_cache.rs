// Implementation for recursive cached sampling with intermediate caching
use crate::cache::dist_cache;
use crate::{Uncertain, computation::ComputationNode};
use std::collections::HashMap;

impl Uncertain<f64> {
    /// Take samples with recursive caching - ensures all nodes (leaves and intermediates)
    /// use cached samples and are evaluated consistently
    #[must_use]
    pub fn take_samples_cached_recursive(&self, count: usize) -> Vec<f64> {
        // First check if we already have this cached at the top level
        // Try to get from cache without inserting empty vec
        let cache = dist_cache();
        if let Some(existing) = cache.get_samples(self.id, count) {
            return existing;
        }
        
        // Build a map of all nodes in the computation graph
        let mut node_map = HashMap::new();
        collect_all_nodes(&self.node, &mut node_map);
        
        // Recursively cache all nodes bottom-up
        let result = cache_node_recursive(&self.node, &node_map, count);
        
        // Cache the final result
        cache.get_or_compute_samples(self.id, count, || result.clone());
        
        result
    }
}

/// Collect all nodes in the computation graph and create Uncertain wrappers
fn collect_all_nodes(
    node: &ComputationNode<f64>,
    nodes: &mut HashMap<uuid::Uuid, ComputationNode<f64>>,
) {
    match node {
        ComputationNode::Leaf { id, .. } => {
            nodes.insert(*id, node.clone());
        }
        ComputationNode::BinaryOp { left, right, .. } => {
            // First process children
            collect_all_nodes(left, nodes);
            collect_all_nodes(right, nodes);
            // Note: BinaryOp nodes don't have their own UUID in the current structure
            // They're identified by the Uncertain wrapper's UUID
        }
        ComputationNode::UnaryOp { operand, .. } => {
            collect_all_nodes(operand, nodes);
        }
        ComputationNode::Conditional {
            condition,
            if_true,
            if_false,
        } => {
            collect_all_nodes_bool(condition, nodes);
            collect_all_nodes(if_true, nodes);
            collect_all_nodes(if_false, nodes);
        }
    }
}

fn collect_all_nodes_bool(
    _node: &ComputationNode<bool>,
    _nodes: &mut HashMap<uuid::Uuid, ComputationNode<f64>>,
) {
    // Skip bool nodes for now
}

/// Recursively cache a node and all its dependencies
fn cache_node_recursive(
    node: &ComputationNode<f64>,
    node_map: &HashMap<uuid::Uuid, ComputationNode<f64>>,
    count: usize,
) -> Vec<f64> {
    match node {
        ComputationNode::Leaf { id, sample } => {
            // For leaves, use the standard caching mechanism
            let leaf_uncertain = Uncertain {
                id: *id,
                sample_fn: sample.clone(),
                node: node.clone(),
            };
            leaf_uncertain.take_samples_cached(count)
        }
        
        ComputationNode::BinaryOp {
            left,
            right,
            operation,
        } => {
            // First ensure children are cached
            let left_samples = cache_node_recursive(left, node_map, count);
            let right_samples = cache_node_recursive(right, node_map, count);
            
            // Now compute this node's samples using the cached children
            let mut results = Vec::with_capacity(count);
            for i in 0..count {
                let result = operation.apply(left_samples[i], right_samples[i]);
                results.push(result);
            }
            
            // Note: We can't cache this directly since BinaryOp doesn't have a UUID
            // The caching happens at the Uncertain wrapper level
            results
        }
        
        ComputationNode::UnaryOp { operand, operation } => {
            // First ensure operand is cached
            let operand_samples = cache_node_recursive(operand, node_map, count);
            
            // Now compute this node's samples using the cached operand
            let mut results = Vec::with_capacity(count);
            for i in 0..count {
                let result = match operation {
                    crate::computation::UnaryOperation::Map(func) => func(operand_samples[i]),
                    crate::computation::UnaryOperation::Filter(_) => operand_samples[i],
                };
                results.push(result);
            }
            
            results
        }
        
        ComputationNode::Conditional { .. } => {
            panic!("Conditional nodes not supported for f64 recursive caching")
        }
    }
}