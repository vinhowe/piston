//! Fused kernel code generation.
//!
//! This module provides code generators for different fusion patterns:
//! - [`FusedElemwiseCodegen`] - For elementwise/injective chains
//! - [`FusedReduceCodegen`] - For reductions with prologues
//! - [`FusedComputeCodegen`] - For compute-intensive ops with epilogues

use super::{FusionGraph, FusionGroup, OpPattern};
use crate::{
    BinaryOp, KernelSource, LazyOp, OperationError, UnaryOp, WgslFragment, WorkgroupSize,
};

/// Configuration for fused kernel generation.
#[derive(Debug, Clone)]
pub struct FusionCodegenConfig {
    /// Maximum number of operations to fuse into a single kernel
    pub max_fused_ops: usize,
    /// Enable GEMM epilogue fusion
    pub fuse_gemm_epilogue: bool,
    /// Enable reduction prologue fusion
    pub fuse_reduce_prologue: bool,
}

impl Default for FusionCodegenConfig {
    fn default() -> Self {
        Self {
            max_fused_ops: 16,
            fuse_gemm_epilogue: true,
            fuse_reduce_prologue: true,
        }
    }
}

/// Code generator for fused elementwise/injective operations.
///
/// Generates a single kernel that:
/// 1. Loads inputs from global memory
/// 2. Applies all fused operations in sequence
/// 3. Writes final output to global memory
pub struct FusedElemwiseCodegen<'a> {
    group: &'a FusionGroup,
    graph: &'a FusionGraph,
    config: FusionCodegenConfig,
}

impl<'a> FusedElemwiseCodegen<'a> {
    pub fn new(group: &'a FusionGroup, graph: &'a FusionGraph) -> Self {
        Self {
            group,
            graph,
            config: FusionCodegenConfig::default(),
        }
    }

    pub fn with_config(mut self, config: FusionCodegenConfig) -> Self {
        self.config = config;
        self
    }

    /// Generate the fused kernel source.
    pub fn generate(&self, _workgroup_size: &WorkgroupSize) -> Result<KernelSource, OperationError> {
        let mut fragment = WgslFragment::new(4096);

        // Generate bindings for external inputs
        for (i, &input_id) in self.group.external_inputs.iter().enumerate() {
            let _node = &self.graph.nodes[input_id];
            fragment.write(format!(
                "@group(0) @binding({}) var<storage, read> input_{}: array<vec4<f32>>;\n",
                i, i
            ));
        }

        // Output binding
        let output_binding = self.group.external_inputs.len();
        fragment.write(format!(
            "@group(0) @binding({}) var<storage, read_write> output: array<vec4<f32>>;\n",
            output_binding
        ));

        // Uniform binding
        fragment.write(format!(
            "@group(1) @binding(0) var<uniform> metadata: Meta;\n\n"
        ));

        // Metadata struct (just numel for now)
        fragment.write("struct Meta { numel: u32 }\n\n");

        // Helper functions for unary ops
        self.generate_helper_functions(&mut fragment);

        // Main function
        fragment.write("@compute @workgroup_size(64, 1, 1)\n");
        fragment.write("fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n");
        fragment.write("    let index = gid.x;\n");
        fragment.write("    if (index >= metadata.numel / 4u) { return; }\n\n");

        // Generate fused computation
        for (i, &node_id) in self.group.nodes.iter().enumerate() {
            let node = &self.graph.nodes[node_id];
            let var_name = format!("v{}", i);
            let expr = self.generate_node_expr(node, i)?;
            fragment.write(format!("    let {} = {};\n", var_name, expr));
        }

        // Write output
        let final_var = format!("v{}", self.group.nodes.len() - 1);
        fragment.write(format!("    output[index] = {};\n", final_var));
        fragment.write("}\n");

        Ok(KernelSource(fragment.0.into()))
    }

    fn generate_helper_functions(&self, fragment: &mut WgslFragment) {
        // Check which functions we need
        let needs_gelu = self.group.nodes.iter().any(|&id| {
            matches!(
                &self.graph.nodes[id].op,
                LazyOp::Unary(u) if matches!(u.op, UnaryOp::Gelu)
            )
        });

        let needs_silu = self.group.nodes.iter().any(|&id| {
            matches!(
                &self.graph.nodes[id].op,
                LazyOp::Unary(u) if matches!(u.op, UnaryOp::Silu | UnaryOp::Sigmoid)
            )
        });

        let needs_tanh = needs_gelu
            || self.group.nodes.iter().any(|&id| {
                matches!(
                    &self.graph.nodes[id].op,
                    LazyOp::Unary(u) if matches!(u.op, UnaryOp::Tanh)
                )
            });

        if needs_tanh {
            fragment.write(
                r#"
fn safe_tanh(x: vec4<f32>) -> vec4<f32> {
    return select(tanh(x), sign(x), abs(x) >= vec4<f32>(10.));
}
"#,
            );
        }

        if needs_gelu {
            fragment.write(
                r#"
fn gelu(val: vec4<f32>) -> vec4<f32> {
    let cdf = vec4<f32>(0.5) + vec4<f32>(0.5) * safe_tanh(val * (vec4<f32>(0.035677407)
            * (val * val) + vec4<f32>(0.7978846)));
    return val * cdf;
}
"#,
            );
        }

        if needs_silu {
            fragment.write(
                r#"
fn sigmoid(val: vec4<f32>) -> vec4<f32> {
    return select(1.0 / (1.0 + exp(-val)), exp(val) / (1.0 + exp(val)), val >= vec4<f32>(0.));
}
fn silu(val: vec4<f32>) -> vec4<f32> {
    return val * sigmoid(val);
}
"#,
            );
        }

        fragment.write("\n");
    }

    fn generate_node_expr(
        &self,
        node: &super::FusionNode,
        _node_idx: usize,
    ) -> Result<String, OperationError> {
        let input_expr = |idx: usize| -> String { self.input_var_name(node, idx) };

        match &node.op {
            LazyOp::Unary(unary) => {
                let input = input_expr(0);
                let expr = match unary.op {
                    UnaryOp::Relu => format!("max({}, vec4<f32>(0.0))", input),
                    UnaryOp::Gelu => format!("gelu({})", input),
                    UnaryOp::Silu => format!("silu({})", input),
                    UnaryOp::Sigmoid => format!("sigmoid({})", input),
                    UnaryOp::Tanh => format!("safe_tanh({})", input),
                    UnaryOp::Exp => format!("exp({})", input),
                    UnaryOp::Log => format!("log({})", input),
                    UnaryOp::Sqrt => format!("sqrt({})", input),
                    UnaryOp::Neg => format!("-({})", input),
                    UnaryOp::Abs => format!("abs({})", input),
                    UnaryOp::Square => format!("({}) * ({})", input, input),
                    UnaryOp::Reciprocal => format!("1.0 / ({})", input),
                    _ => {
                        return Err(OperationError::CompileError(format!(
                            "Unsupported unary op in fusion: {:?}",
                            unary.op
                        )))
                    }
                };
                Ok(expr)
            }

            LazyOp::Binary(binary) => {
                let lhs = input_expr(0);
                let rhs = match &binary.rhs {
                    crate::TensorTypeOrScalarEnum::Tensor(_) => input_expr(1),
                    crate::TensorTypeOrScalarEnum::Scalar(s) => format!("vec4<f32>({})", s),
                };

                let expr = match binary.op {
                    BinaryOp::Add => format!("({}) + ({})", lhs, rhs),
                    BinaryOp::Sub => format!("({}) - ({})", lhs, rhs),
                    BinaryOp::Mul => format!("({}) * ({})", lhs, rhs),
                    BinaryOp::Div => format!("({}) / ({})", lhs, rhs),
                    BinaryOp::Maximum => format!("max({}, {})", lhs, rhs),
                    BinaryOp::Minimum => format!("min({}, {})", lhs, rhs),
                    BinaryOp::Pow => format!("pow(abs({}), {}) * sign({})", lhs, rhs, lhs),
                };
                Ok(expr)
            }

            LazyOp::Affine(affine) => {
                let input = input_expr(0);
                let mul = affine.mul;
                let add = affine.add;
                Ok(format!(
                    "fma({}, vec4<f32>({}), vec4<f32>({}))",
                    input, mul, add
                ))
            }

            // For const/generators at start of chain
            LazyOp::Const => {
                // This should be an external input
                Ok(self.input_var_name(node, 0))
            }

            _ => Err(OperationError::CompileError(format!(
                "Unsupported op in elemwise fusion: {:?}",
                node.op.name()
            ))),
        }
    }

    fn input_var_name(&self, node: &super::FusionNode, input_idx: usize) -> String {
        if input_idx >= node.inputs.len() {
            // This might be an external input directly (for first node)
            if let Some(pos) = self
                .group
                .external_inputs
                .iter()
                .position(|&id| id == node.id)
            {
                return format!("input_{}[index]", pos);
            }
            panic!("Input index {} out of bounds for node", input_idx);
        }

        let input_id = node.inputs[input_idx];

        // Check if input is in this group (use local var) or external (use buffer)
        if let Some(pos) = self.group.nodes.iter().position(|&id| id == input_id) {
            format!("v{}", pos)
        } else {
            // External input - find its binding index
            let ext_idx = self
                .group
                .external_inputs
                .iter()
                .position(|&id| id == input_id)
                .unwrap_or_else(|| panic!("Input {:?} not found in external inputs", input_id));
            format!("input_{}[index]", ext_idx)
        }
    }
}

/// Placeholder for compute-intensive (GEMM) + epilogue fusion.
pub struct FusedComputeCodegen<'a> {
    #[allow(dead_code)]
    group: &'a FusionGroup,
    #[allow(dead_code)]
    graph: &'a FusionGraph,
}

impl<'a> FusedComputeCodegen<'a> {
    pub fn new(group: &'a FusionGroup, graph: &'a FusionGraph) -> Self {
        Self { group, graph }
    }

    /// Generate GEMM kernel with fused epilogue.
    pub fn generate(&self, _workgroup_size: &WorkgroupSize) -> Result<KernelSource, OperationError> {
        // TODO: Implement GEMM with fused bias + activation
        // This requires modifying the existing GEMM kernel to accept epilogue code
        Err(OperationError::CompileError(
            "GEMM epilogue fusion not yet implemented".to_string(),
        ))
    }
}

/// Placeholder for reduction + prologue fusion.
pub struct FusedReduceCodegen<'a> {
    #[allow(dead_code)]
    group: &'a FusionGroup,
    #[allow(dead_code)]
    graph: &'a FusionGraph,
}

impl<'a> FusedReduceCodegen<'a> {
    pub fn new(group: &'a FusionGroup, graph: &'a FusionGraph) -> Self {
        Self { group, graph }
    }

    /// Generate reduction kernel with fused prologue operations.
    pub fn generate(&self, _workgroup_size: &WorkgroupSize) -> Result<KernelSource, OperationError> {
        // TODO: Implement reduction with fused prologue
        // This requires modifying reduction kernels to apply prologue inline
        Err(OperationError::CompileError(
            "Reduction prologue fusion not yet implemented".to_string(),
        ))
    }
}

/// Select the appropriate codegen for a fusion group.
pub fn select_codegen<'a>(
    group: &'a FusionGroup,
    graph: &'a FusionGraph,
) -> Box<dyn FusionCodegen + 'a> {
    match group.dominant_pattern {
        OpPattern::Elemwise | OpPattern::Injective => {
            Box::new(FusedElemwiseCodegen::new(group, graph))
        }
        OpPattern::Reduce => Box::new(FusedReduceCodegen::new(group, graph)),
        OpPattern::ComputeIntensive => Box::new(FusedComputeCodegen::new(group, graph)),
        OpPattern::Opaque => {
            // Shouldn't happen - opaque ops should be singleton groups
            panic!("Cannot generate fused kernel for opaque pattern")
        }
    }
}

/// Trait for fused kernel code generators.
pub trait FusionCodegen {
    fn generate(&self, workgroup_size: &WorkgroupSize) -> Result<KernelSource, OperationError>;
}

impl FusionCodegen for FusedElemwiseCodegen<'_> {
    fn generate(&self, workgroup_size: &WorkgroupSize) -> Result<KernelSource, OperationError> {
        self.generate(workgroup_size)
    }
}

impl FusionCodegen for FusedReduceCodegen<'_> {
    fn generate(&self, workgroup_size: &WorkgroupSize) -> Result<KernelSource, OperationError> {
        self.generate(workgroup_size)
    }
}

impl FusionCodegen for FusedComputeCodegen<'_> {
    fn generate(&self, workgroup_size: &WorkgroupSize) -> Result<KernelSource, OperationError> {
        self.generate(workgroup_size)
    }
}

