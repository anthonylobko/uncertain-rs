// Minimal implementation for recursive cached sampling
use crate::cache::dist_cache;
use crate::{Uncertain, computation::ComputationNode};
use std::collections::HashMap;

impl Uncertain<f64> {
    /// Take samples with recursive caching - ensures all leaf distributions
    /// use cached samples and are evaluated with the same sample indices
    #[must_use]
    pub fn take_samples_cached_recursive(&self, count: usize) -> Vec<f64> {
        // First check if we already have this cached at the top level
        // We need to check the cache but NOT insert an empty vec if not found
        // So we can't use get_or_compute here
        // For now, just skip this optimization since we can't directly check the cache

        // Collect all leaf node IDs and pre-generate their cached samples
        let leaf_map = collect_leaves(&self.node);

        // Ensure all leaves have cached samples
        for leaf_uncertain in leaf_map.values() {
            // This will cache the samples if not already cached
            let _ = leaf_uncertain.take_samples_cached(count);
        }

        // Now generate samples by evaluating the computation graph
        // but using cached leaf samples by index
        let mut results = Vec::with_capacity(count);

        for sample_idx in 0..count {
            let value = evaluate_with_cached_leaves(&self.node, &leaf_map, sample_idx, count);
            results.push(value);
        }

        // Cache the final result using get_or_compute pattern
        // The result is already computed, so we just ensure it's cached
        let _ = dist_cache().get_or_compute_samples(self.id, count, || results.clone());

        results
    }
}

/// Collect all leaf nodes and create Uncertain wrappers for them
fn collect_leaves(node: &ComputationNode<f64>) -> HashMap<uuid::Uuid, Uncertain<f64>> {
    let mut leaves = HashMap::new();
    collect_leaves_recursive(node, &mut leaves);
    leaves
}

fn collect_leaves_recursive(
    node: &ComputationNode<f64>,
    leaves: &mut HashMap<uuid::Uuid, Uncertain<f64>>,
) {
    match node {
        ComputationNode::Leaf { id, sample } => {
            if !leaves.contains_key(id) {
                // Create an Uncertain wrapper for this leaf
                // This preserves the original UUID
                let leaf_uncertain = Uncertain {
                    id: *id,
                    sample_fn: sample.clone(),
                    node: node.clone(),
                };
                leaves.insert(*id, leaf_uncertain);
            }
        }
        ComputationNode::BinaryOp { left, right, .. } => {
            collect_leaves_recursive(left, leaves);
            collect_leaves_recursive(right, leaves);
        }
        ComputationNode::UnaryOp { operand, .. } => {
            collect_leaves_recursive(operand, leaves);
        }
        ComputationNode::Conditional {
            condition,
            if_true,
            if_false,
        } => {
            // For f64, we don't expect conditionals, but handle anyway
            collect_leaves_recursive_bool(condition, leaves);
            collect_leaves_recursive(if_true, leaves);
            collect_leaves_recursive(if_false, leaves);
        }
    }
}

fn collect_leaves_recursive_bool(
    _node: &ComputationNode<bool>,
    _leaves: &mut HashMap<uuid::Uuid, Uncertain<f64>>,
) {
    // Skip bool nodes for now - they're not f64
}

/// Evaluate the computation graph using cached samples at the given index
fn evaluate_with_cached_leaves(
    node: &ComputationNode<f64>,
    leaf_map: &HashMap<uuid::Uuid, Uncertain<f64>>,
    sample_idx: usize,
    sample_count: usize,
) -> f64 {
    match node {
        ComputationNode::Leaf { id, sample } => {
            // Try to get cached sample for this leaf
            if let Some(leaf_uncertain) = leaf_map.get(id) {
                // Get the cached samples for this leaf via the cache
                let cached_samples =
                    dist_cache().get_or_compute_samples(leaf_uncertain.id, sample_count, || {
                        // This should not be called since we pre-cached
                        leaf_uncertain.samples().take(sample_count).collect()
                    });
                if let Some(value) = cached_samples.get(sample_idx) {
                    return *value;
                }
            }
            // Fallback to direct sampling
            sample()
        }

        ComputationNode::BinaryOp {
            left,
            right,
            operation,
        } => {
            let left_val = evaluate_with_cached_leaves(left, leaf_map, sample_idx, sample_count);
            let right_val = evaluate_with_cached_leaves(right, leaf_map, sample_idx, sample_count);
            operation.apply(left_val, right_val)
        }

        ComputationNode::UnaryOp { operand, operation } => {
            let operand_val =
                evaluate_with_cached_leaves(operand, leaf_map, sample_idx, sample_count);
            match operation {
                crate::computation::UnaryOperation::Map(func) => func(operand_val),
                crate::computation::UnaryOperation::Filter(_) => operand_val,
            }
        }

        ComputationNode::Conditional { .. } => {
            panic!("Conditional nodes not supported for f64 recursive caching")
        }
    }
}
