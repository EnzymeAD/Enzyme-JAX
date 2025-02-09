use crate::model::*;
use crate::optimize::*;
use crate::rewrites::*;
use cxx::CxxVector;
use egg::*;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs::*;
use std::process::{Command, Stdio};
use std::time::*;
use std::{borrow::Borrow, collections::HashMap};

#[cxx::bridge(namespace = "tensat")]
pub mod ffi {
    #[derive(Debug)]
    enum Type {
        i1,
        i32,
        i64,
        bf16,
        f32,
        f64,
    }

    #[derive(Debug)]
    enum Ops {
        Var,
        Num,
        Vec,
        Input,
        Index,
        CompareOp,
        BroadcastInDimOp,
        ConvertOp,
        ReduceOp,
        ReshapeOp,
        GatherOp,
        SelectOp,
        ConcatenateOp,
        DotGeneralOp,
        ConvolutionOp,
        PadOp,
        SliceOp,
        TransposeOp,
        MulOp,
        AddOp,
        DivOp,
        SubtractOp,
        MinOp,
        MaxOp,
        NegOp,
        TanhOp,
        ExpOp,
        IotaOp,
        // ConstantOp,
        DynamicUpdateSliceOp,
        DynamicSliceOp,
        ScatterOp,
        BlackBox,
        ReturnOp,
        SSplit0,
        SSplit1,
        MatchRank,
        InferReshape,
    }

    #[derive(Debug, Clone)]
    struct Node {
        op: Ops,
        label: String,
        operands: Vec<i32>,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Tensor {
        pub shape: Vec<i64>,
        pub element_type: Type,
    }

    // CXX won't let me construct a Vec<Vec<i32>>, so we use Vec<ffi::Shape> instead
    // TODO: We should replace all the &[i32]s we see in Rust ffi function arguments
    // to Vec<Shape> or similar. rust::Slice in CXX is quite error prone, because
    // a common pattern is to create a std::vector then create a slice out of it,
    // but the data is easily corrupted by the vector going out of scope.

    // Note that this should only used in the above case, there are purpose-built
    // structures for Tensor and Node. In all other cases, use the builtin rust types.
    #[derive(Debug)]
    struct Vector {
        pub vec: Vec<i64>,
    }

    // Similarly, we're creating a Matrix type for vecs of vecs (padding)
    #[derive(Debug)]
    struct Matrix {
        pub mat: Vec<Vector>,
    }

    // take floats from c++ and wrap them into f32s below
    extern "Rust" {
        type Mdl;
        type CppGraphConverter;
        type TensorData;
        type TensorInfo;
        fn new_converter() -> Box<CppGraphConverter>;
        // Exposing the constructor functions with Box<TensorInfo>
        fn new_input(
            self: &mut CppGraphConverter,
            block_arg_number: i64,
            tensor: Tensor,
        ) -> Box<TensorInfo>;
        fn new_index(
            self: &mut CppGraphConverter,
            index: i64,
            inpt: &TensorInfo,
        ) -> Box<TensorInfo>;
        fn new_compare_op(
            self: &mut CppGraphConverter,
            inpt_1: &TensorInfo,
            inpt_2: &TensorInfo,
            comparison_direction: i64,
            comparison_type: i64,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_broadcast_in_dim(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            dimensions: Vec<i64>,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_convert_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            output_type: i64,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_reduce_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            dimensions: Vec<i64>,
            outputs: &Vec<Tensor>,
        ) -> Box<TensorInfo>;
        fn new_reshape_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        pub fn new_gather_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            start_indices: &TensorInfo,
            output: Tensor,
            offset_dims: Vec<i64>,
            collapsed_slice_dims: Vec<i64>,
            operand_batching_dims: Vec<i64>,
            start_indices_batching_dims: Vec<i64>,
            start_index_map: Vec<i64>,
            index_vector_dim: i64,
            slice_sizes: Vec<i64>,
            indices_are_sorted: i64,
        ) -> Box<TensorInfo>;
        fn new_select_op(
            self: &mut CppGraphConverter,
            pred: &TensorInfo,
            on_true: &TensorInfo,
            on_false: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_concatenate_op(
            self: &mut CppGraphConverter,
            inputs: &[*mut TensorInfo],
            dimension: i64,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_convolution_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            windowStrides: Vec<i64>,
            padding: Vec<Vector>,
            lhsDilation: Vec<i64>,
            rhsDilation: Vec<i64>,
            windowReversal: Vec<bool>,
            inputBatchDimension: i64,
            inputFeatureDimension: i64,
            inputSpatialDimension: Vec<i64>,
            kernelInputFeatureDimension: i64,
            kernelOutputFeatureDimension: i64,
            kernelSpatialDimension: Vec<i64>,
            outputBatchDimension: i64,
            outputFeatureDimension: i64,
            outputSpatialDimension: Vec<i64>,
            featureGroupCount: i64,
            batchGroupCount: i64,
            precision_config: Vec<i64>,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_dot_general_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            lhs_batching_dimensions: Vec<i64>,
            rhs_batching_dimensions: Vec<i64>,
            lhs_contracting_dimensions: Vec<i64>,
            rhs_contracting_dimensions: Vec<i64>,
            precision_config: Vec<i64>,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_pad_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            padding_value: &TensorInfo,
            edge_padding_low: Vec<i64>,
            edge_padding_high: Vec<i64>,
            interior_padding: Vec<i64>,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_slice_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            start_indices: Vec<i64>,
            limit_indices: Vec<i64>,
            strides: Vec<i64>,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_transpose_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            permutation: Vec<i64>,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_mul_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_add_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_div_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_subtract_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_min_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_max_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_neg_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_tanh_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_exp_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_iota_op(
            self: &mut CppGraphConverter,
            iota_dimension: i64,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_dynamic_update_slice_op(
            self: &mut CppGraphConverter,
            operand: &TensorInfo,
            update: &TensorInfo,
            start_indices: &TensorInfo,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_dynamic_slice_op(
            self: &mut CppGraphConverter,
            operand: &TensorInfo,
            start_indices: &TensorInfo,
            slice_sizes: i64,
            output: Tensor,
        ) -> Box<TensorInfo>;
        fn new_scatter_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            scatter_indices: &TensorInfo,
            updates: &TensorInfo,
            dimension_numbers: i64,
            outputs: &Vec<Tensor>,
        ) -> Box<TensorInfo>;
        fn new_blackbox_op(
            self: &mut CppGraphConverter,
            inpts: &[*mut TensorInfo],
            captured: &[*mut TensorInfo], // values that appear in a block that was declared outside
            cpp_num: i64,
            outputs: &Vec<Tensor>,
        ) -> Box<TensorInfo>;
        fn new_return_op(
            self: &mut CppGraphConverter,
            inpts: &[*mut TensorInfo],
        ) -> Box<TensorInfo>;
        fn optimize(self: &CppGraphConverter) -> Vec<Node>;
        fn print_rec_expr(self: &CppGraphConverter);
        fn pretty_print_rec_expr(self: &CppGraphConverter, width: i32);
    }

    unsafe extern "C++" {
        include!("EqualitySaturation.h");

        fn get_cost(
            op: Ops,
            operands: Vec<Tensor>,
            other_vector_args: Vec<Vector>,
            int_args: Vec<i64>,
            matrix_args: Vec<Matrix>,
        ) -> Vec<u64>;

        fn get_graph_cost(graph: Vec<Node>) -> u64;

        fn get_shape(
            op: Ops,
            operands: Vec<Tensor>,
            other_vector_args: Vec<Vector>,
            int_args: Vec<i64>,
            matrix_args: Vec<Matrix>,
        ) -> Vec<Tensor>;

        fn apply_mlir_rewrite(nodes: Vec<Node>, roots: Vec<Tensor>) -> Box<CppGraphConverter>;
    }
}

/// Struct for converting a model specified using our Rust interface to RecExpr
///
/// The RecExpr is growed on the fly when member functions are called. Uses a
/// Hashmap to store the map of scalar nodes to their indices into the RecExpr to
/// avoid replication.
#[derive(Default)]
pub struct CppGraphConverter {
    rec_expr: RecExpr<Mdl>,
    scalar_map: HashMap<i64, Id>,
    name_gen: NameGen,
    blackbox_cpp_num_to_tensorinfo: HashMap<i64, TensorInfo>,
}

pub fn new_converter() -> Box<CppGraphConverter> {
    Box::new(CppGraphConverter::default())
}

impl ffi::Type {
    pub fn from_str(s: &str) -> Option<ffi::Type> {
        match s {
            "i1" => Some(ffi::Type::i1),
            "i32" => Some(ffi::Type::i32),
            "i64" => Some(ffi::Type::i64),
            "bf16" => Some(ffi::Type::bf16),
            "f32" => Some(ffi::Type::f32),
            "f64" => Some(ffi::Type::f64),
            _ => None,
        }
    }
}

impl ffi::Ops {
    pub fn from_mdl(m: &Mdl) -> ffi::Ops {
        use ffi::Ops;
        match m {
            Mdl::Var(_) => Ops::Var,
            Mdl::Num(_) => Ops::Num,
            Mdl::Vec(_) => Ops::Vec,
            Mdl::Input(_) => Ops::Input,
            Mdl::Index(_) => Ops::Index,
            Mdl::CompareOp(_) => Ops::CompareOp,
            Mdl::BroadcastInDimOp(_) => Ops::BroadcastInDimOp,
            Mdl::ConvertOp(_) => Ops::ConvertOp,
            Mdl::ReduceOp(_) => Ops::ReduceOp,
            Mdl::ReshapeOp(_) => Ops::ReshapeOp,
            Mdl::GatherOp(_) => Ops::GatherOp,
            Mdl::SelectOp(_) => Ops::SelectOp,
            Mdl::ConcatenateOp(_) => Ops::ConcatenateOp,
            Mdl::DotGeneralOp(_) => Ops::DotGeneralOp,
            Mdl::PadOp(_) => Ops::PadOp,
            Mdl::SliceOp(_) => Ops::SliceOp,
            Mdl::TransposeOp(_) => Ops::TransposeOp,
            Mdl::ConvolutionOp(_) => Ops::ConvolutionOp,
            Mdl::MulOp(_) => Ops::MulOp,
            Mdl::AddOp(_) => Ops::AddOp,
            Mdl::DivOp(_) => Ops::DivOp,
            Mdl::SubtractOp(_) => Ops::SubtractOp,
            Mdl::MinOp(_) => Ops::MinOp,
            Mdl::MaxOp(_) => Ops::MaxOp,
            Mdl::NegOp(_) => Ops::NegOp,
            Mdl::TanhOp(_) => Ops::TanhOp,
            Mdl::ExpOp(_) => Ops::ExpOp,
            Mdl::IotaOp(_) => Ops::IotaOp,
            Mdl::DynamicUpdateSliceOp(_) => Ops::DynamicUpdateSliceOp,
            Mdl::DynamicSliceOp(_) => Ops::DynamicSliceOp,
            Mdl::ScatterOp(_) => Ops::ScatterOp,
            Mdl::BlackBox(_) => Ops::BlackBox,
            Mdl::ReturnOp(_) => Ops::ReturnOp,
            Mdl::SSplit0(_) => Ops::SSplit0,
            Mdl::SSplit1(_) => Ops::SSplit1,
            Mdl::MatchRank(_) => Ops::MatchRank,
            Mdl::InferReshape(_) => Ops::InferReshape,
        }
    }
}

/// The APIs of GraphConverter are (intended to) match TASO's so that we can easily
/// construct TASO graphs using this class
impl CppGraphConverter {
    pub fn rec_expr(self) -> RecExpr<Mdl> {
        self.rec_expr
    }

    fn vec_node(&mut self, vec: Vec<i64>) -> Id {
        let vec: Vec<Id> = vec.iter().map(|n| self.add_or_get_val(*n)).collect();
        let node = Mdl::Vec(vec);
        let id = self.rec_expr.add(node);
        id
    }

    fn add_or_get_val(&mut self, val: i64) -> Id {
        match self.scalar_map.get(&val) {
            Some(id) => *id,
            None => {
                let node = Mdl::Num(val);
                let id = self.rec_expr.add(node);
                self.scalar_map.insert(val, id);
                id
            }
        }
    }

    fn tensor_data(tensors: Vec<ffi::Tensor>) -> TensorData {
        TensorData {
            tensors,
            name: None,
            need_infer_shape: false,
        }
    }

    // Wrapper functions for C++ side
    pub fn new_input(&mut self, block_arg_number: i64, tensor: ffi::Tensor) -> Box<TensorInfo> {
        let name = format!("input_{}", block_arg_number)
            + "@"
            + &tensor.shape.iter().join("_")
            + "@"
            + (format!("{:?}", tensor.element_type).as_str());
        let node = Mdl::Var(Symbol::from(name));
        let name_id = self.rec_expr.add(node);
        let block_arg_node_id = self.add_or_get_val(block_arg_number as i64);
        let new_node = Mdl::Input([name_id, block_arg_node_id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![tensor]),
        };
        Box::new(res)
    }

    pub fn new_index(&mut self, index: i64, inpt: &TensorInfo) -> Box<TensorInfo> {
        let index_num_node = self.add_or_get_val(index);
        let new_node = Mdl::Index([index_num_node, inpt.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![inpt.tensor_data.tensors
                [index as usize]
                .clone()]),
        };
        Box::new(res)
    }

    pub fn new_compare_op(
        &mut self,
        inpt_1: &TensorInfo,
        inpt_2: &TensorInfo,
        comparison_direction: i64,
        comparison_type: i64,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let comparison_direction_node = self.add_or_get_val(comparison_direction);
        let comparison_type_node = self.add_or_get_val(comparison_type);
        let new_node = Mdl::CompareOp([
            inpt_1.id,
            inpt_2.id,
            comparison_direction_node,
            comparison_type_node,
        ]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_broadcast_in_dim(
        &mut self,
        inpt: &TensorInfo,
        dimensions: Vec<i64>,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let dimensions_id = self.vec_node(dimensions);
        let shape_id = self.vec_node(output.shape.clone());
        let new_node = Mdl::BroadcastInDimOp([inpt.id, dimensions_id, shape_id]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_convert_op(
        &mut self,
        inpt: &TensorInfo,
        output_type: i64,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let output_type_node = self.add_or_get_val(output_type);
        let new_node = Mdl::ConvertOp([inpt.id, output_type_node]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_reduce_op(
        &mut self,
        inpt: &TensorInfo,
        dimensions: Vec<i64>,
        outputs: &Vec<ffi::Tensor>,
    ) -> Box<TensorInfo> {
        let dimensions_id = self.vec_node(dimensions);
        let new_node = Mdl::ReduceOp([inpt.id, dimensions_id]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(outputs.clone()),
        };
        Box::new(res)
    }

    pub fn new_reshape_op(&mut self, inpt: &TensorInfo, output: ffi::Tensor) -> Box<TensorInfo> {
        let shape_id = self.vec_node(output.shape.clone());
        let new_node = Mdl::ReshapeOp([inpt.id, shape_id]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_gather_op(
        &mut self,
        inpt: &TensorInfo,
        start_indices: &TensorInfo,
        output: ffi::Tensor,
        offset_dims: Vec<i64>,
        collapsed_slice_dims: Vec<i64>,
        operand_batching_dims: Vec<i64>,
        start_indices_batching_dims: Vec<i64>,
        start_index_map: Vec<i64>,
        index_vector_dim: i64,
        slice_sizes: Vec<i64>,
        indices_are_sorted: i64,
    ) -> Box<TensorInfo> {
        let offset_dims_id = self.vec_node(offset_dims);
        let collapsed_slice_dims_id = self.vec_node(collapsed_slice_dims);
        let operand_batching_dims_id = self.vec_node(operand_batching_dims);
        let start_indices_batching_dims_id = self.vec_node(start_indices_batching_dims);
        let start_index_map_id = self.vec_node(start_index_map);
        let slice_sizes_id = self.vec_node(slice_sizes);
        let index_vector_dim_id = self.add_or_get_val(index_vector_dim);
        let indices_are_sorted_id = self.add_or_get_val(indices_are_sorted);

        let new_node = Mdl::GatherOp([
            inpt.id,
            start_indices.id,
            offset_dims_id,
            collapsed_slice_dims_id,
            operand_batching_dims_id,
            start_indices_batching_dims_id,
            start_index_map_id,
            index_vector_dim_id,
            slice_sizes_id,
            indices_are_sorted_id,
        ]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_select_op(
        &mut self,
        pred: &TensorInfo,
        on_true: &TensorInfo,
        on_false: &TensorInfo,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let new_node = Mdl::SelectOp([pred.id, on_true.id, on_false.id]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    fn new_tensorinfo_vec(&mut self, inputs: &[*mut TensorInfo]) -> Id {
        let tensor_infos: Vec<&TensorInfo> = inputs.iter().map(|&ptr| unsafe { &*ptr }).collect();
        let inputs_node = Mdl::Vec(tensor_infos.iter().map(|i| i.id).collect());
        self.rec_expr.add(inputs_node)
    }

    pub fn new_concatenate_op(
        &mut self,
        inputs: &[*mut TensorInfo],
        dimension: i64,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let inputs_id = self.new_tensorinfo_vec(inputs);
        let dimension_id = self.add_or_get_val(dimension);
        let new_node = Mdl::ConcatenateOp([inputs_id, dimension_id]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_convolution_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        window_strides: Vec<i64>,
        padding: Vec<ffi::Vector>,
        lhs_dilation: Vec<i64>,
        rhs_dilation: Vec<i64>,
        window_reversal: Vec<bool>,
        input_batch_dimension: i64,
        input_feature_dimension: i64,
        input_spatial_dimensions: Vec<i64>,
        kernel_input_feature_dimension: i64,
        kernel_output_feature_dimension: i64,
        kernel_spatial_dimensions: Vec<i64>,
        output_batch_dimension: i64,
        output_feature_dimension: i64,
        output_spatial_dimensions: Vec<i64>,
        feature_group_count: i64,
        batch_group_count: i64,
        precision_config: Vec<i64>,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let window_strides_node_id = self.vec_node(window_strides);
        let lhs_dilation_node_id = self.vec_node(lhs_dilation);
        let rhs_dilation_node_id = self.vec_node(rhs_dilation);

        // We could add a bool element type vec?
        let window_reversal_node_id =
            self.vec_node(window_reversal.iter().map(|x| *x as i64).collect());
        let input_spatial_dimensions_node_id = self.vec_node(input_spatial_dimensions);
        let kernel_spatial_dimensions_node_id = self.vec_node(kernel_spatial_dimensions);
        let output_spatial_dimensions_node_id = self.vec_node(output_spatial_dimensions);
        let precision_config_node_id = self.vec_node(precision_config);

        let padding_node_ids: Vec<Id> = padding
            .into_iter()
            .map(|pad| self.vec_node(pad.vec))
            .collect::<Vec<Id>>();
        let padding_node_id = self.rec_expr.add(Mdl::Vec(padding_node_ids));

        let new_node = Mdl::ConvolutionOp([
            lhs.id,
            rhs.id,
            window_strides_node_id,
            padding_node_id,
            lhs_dilation_node_id,
            rhs_dilation_node_id,
            window_reversal_node_id,
            self.add_or_get_val(input_batch_dimension),
            self.add_or_get_val(input_feature_dimension),
            input_spatial_dimensions_node_id,
            self.add_or_get_val(kernel_input_feature_dimension),
            self.add_or_get_val(kernel_output_feature_dimension),
            kernel_spatial_dimensions_node_id,
            self.add_or_get_val(output_batch_dimension),
            self.add_or_get_val(output_feature_dimension),
            output_spatial_dimensions_node_id,
            self.add_or_get_val(feature_group_count),
            self.add_or_get_val(batch_group_count),
            precision_config_node_id,
        ]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_dot_general_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        lhs_batching_dimensions: Vec<i64>,
        rhs_batching_dimensions: Vec<i64>,
        lhs_contracting_dimensions: Vec<i64>,
        rhs_contracting_dimensions: Vec<i64>,
        precision_config: Vec<i64>,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let lhs_batch_dim_name_id = self.vec_node(lhs_batching_dimensions);
        let rhs_batch_dim_name_id = self.vec_node(rhs_batching_dimensions);
        let lhs_contract_dim_name_id = self.vec_node(lhs_contracting_dimensions);
        let rhs_contract_dim_name_id = self.vec_node(rhs_contracting_dimensions);
        let precision_config_id = self.vec_node(precision_config);

        let new_node = Mdl::DotGeneralOp([
            lhs.id,
            rhs.id,
            lhs_batch_dim_name_id,
            rhs_batch_dim_name_id,
            lhs_contract_dim_name_id,
            rhs_contract_dim_name_id,
            precision_config_id,
        ]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_pad_op(
        &mut self,
        inpt: &TensorInfo,
        padding_value: &TensorInfo,
        edge_padding_low: Vec<i64>,
        edge_padding_high: Vec<i64>,
        interior_padding: Vec<i64>,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let edge_padding_low_id = self.vec_node(edge_padding_low);
        let edge_padding_high_id = self.vec_node(edge_padding_high);
        let interior_padding_id = self.vec_node(interior_padding);

        let new_node = Mdl::PadOp([
            inpt.id,
            padding_value.id,
            edge_padding_low_id,
            edge_padding_high_id,
            interior_padding_id,
        ]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_slice_op(
        &mut self,
        inpt: &TensorInfo,
        start_indices: Vec<i64>,
        limit_indices: Vec<i64>,
        strides: Vec<i64>,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let start_indices_id = self.vec_node(start_indices);
        let limit_indices_id = self.vec_node(limit_indices);
        let strides_id = self.vec_node(strides);
        let new_node = Mdl::SliceOp([inpt.id, start_indices_id, limit_indices_id, strides_id]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_transpose_op(
        &mut self,
        inpt: &TensorInfo,
        permutation: Vec<i64>,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let permutation_id = self.vec_node(permutation);
        let new_node = Mdl::TransposeOp([inpt.id, permutation_id]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_mul_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let new_node = Mdl::MulOp([lhs.id, rhs.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_add_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let new_node = Mdl::AddOp([lhs.id, rhs.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_div_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let new_node = Mdl::DivOp([lhs.id, rhs.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_subtract_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let new_node = Mdl::SubtractOp([lhs.id, rhs.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_min_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let new_node = Mdl::MinOp([lhs.id, rhs.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_max_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let new_node = Mdl::MaxOp([lhs.id, rhs.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_neg_op(&mut self, inpt: &TensorInfo, output: ffi::Tensor) -> Box<TensorInfo> {
        let new_node = Mdl::NegOp([inpt.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_tanh_op(&mut self, inpt: &TensorInfo, output: ffi::Tensor) -> Box<TensorInfo> {
        let new_node = Mdl::TanhOp([inpt.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_exp_op(&mut self, inpt: &TensorInfo, output: ffi::Tensor) -> Box<TensorInfo> {
        let new_node = Mdl::ExpOp([inpt.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_iota_op(&mut self, iota_dimension: i64, output: ffi::Tensor) -> Box<TensorInfo> {
        let iota_dim_id = self.add_or_get_val(iota_dimension);
        let shape_id = self.vec_node(output.shape.clone());
        let new_node = Mdl::IotaOp([iota_dim_id, shape_id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_dynamic_update_slice_op(
        &mut self,
        operand: &TensorInfo,
        update: &TensorInfo,
        start_indices: &TensorInfo,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let new_node = Mdl::DynamicUpdateSliceOp([operand.id, update.id, start_indices.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_dynamic_slice_op(
        &mut self,
        operand: &TensorInfo,
        start_indices: &TensorInfo,
        slice_sizes: i64,
        output: ffi::Tensor,
    ) -> Box<TensorInfo> {
        let slice_sizes_id = self.add_or_get_val(slice_sizes);
        let new_node = Mdl::DynamicSliceOp([operand.id, start_indices.id, slice_sizes_id]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![output]),
        };
        Box::new(res)
    }

    pub fn new_scatter_op(
        &mut self,
        inpt: &TensorInfo,
        scatter_indices: &TensorInfo,
        updates: &TensorInfo,
        dimension_numbers: i64,
        outputs: &Vec<ffi::Tensor>,
    ) -> Box<TensorInfo> {
        let dimension_numbers_id = self.add_or_get_val(dimension_numbers);
        let new_node = Mdl::ScatterOp([
            inpt.id,
            scatter_indices.id,
            updates.id,
            dimension_numbers_id,
        ]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(outputs.clone()),
        };
        Box::new(res)
    }

    pub fn new_blackbox_op(
        &mut self,
        inpts: &[*mut TensorInfo],
        captured: &[*mut TensorInfo],
        cpp_num: i64,
        outputs: &Vec<ffi::Tensor>,
    ) -> Box<TensorInfo> {
        let cpp_num_node = self.add_or_get_val(cpp_num);
        let inputs_id = self.new_tensorinfo_vec(inpts);
        let captured_id = self.new_tensorinfo_vec(captured);
        let new_node = Mdl::BlackBox([cpp_num_node, inputs_id, captured_id]);

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(outputs.clone()),
        };
        self.blackbox_cpp_num_to_tensorinfo
            .insert(cpp_num, res.clone());
        Box::new(res)
    }

    pub fn new_return_op(&mut self, inpts: &[*mut TensorInfo]) -> Box<TensorInfo> {
        let inputs_id = self.new_tensorinfo_vec(inpts);
        let new_node = Mdl::ReturnOp([inputs_id]);
        // Returns do not produce values!
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: CppGraphConverter::tensor_data(vec![]),
        };
        Box::new(res)
    }

    pub fn print_rec_expr(&self) {
        println!("{:?}", self.rec_expr)
    }

    pub fn pretty_print_rec_expr(&self, width: i32) {
        println!("{}", self.rec_expr.pretty(width as usize))
    }

    fn convert_to_node(
        egraph: &EGraph<Mdl, TensorAnalysis>,
        to_egraph: &HashMap<Id, Id>,
        rec_expr: &RecExpr<Mdl>,
    ) -> Vec<ffi::Node> {
        let mut res: Vec<ffi::Node> = Vec::new();

        let index = |id: Id| (usize::from(id) as i32); // TODO: this is probably wrong
        let convert = |operands: &[Id]| {
            operands
                .iter()
                .map(|id: &Id| index(*id))
                .collect::<Vec<i32>>()
        };

        let rec_expr_ref = rec_expr.as_ref();

        for (i, mdl) in rec_expr_ref.iter().enumerate() {
            let op = ffi::Ops::from_mdl(mdl);

            let new_node = |operands: &[Id]| ffi::Node {
                op,
                label: "".to_string(),
                operands: convert(operands),
            };

            let node = match mdl {
                Mdl::Var(label) => ffi::Node {
                    op,
                    label: label.to_string(),
                    operands: vec![],
                },
                Mdl::Num(num) => ffi::Node {
                    op,
                    label: "".to_string(),
                    operands: vec![*num as i32],
                },
                Mdl::InferReshape([input]) => {
                    let input_index = index(*input);
                    let id = to_egraph[&Id::from(i)];
                    let mut operands: Vec<i32> = (&egraph[id]).data.tensors[0]
                        .shape
                        .iter()
                        .map(|x| *x as i32)
                        .collect();
                    operands.insert(0, input_index);
                    ffi::Node {
                        op,
                        label: "".to_string(),
                        operands,
                    }
                }
                // TODO: More clever pattern matching
                Mdl::Vec(ops) => new_node(ops),
                Mdl::Input(ops) => new_node(ops),
                Mdl::Index(ops) => new_node(ops),
                Mdl::ReshapeOp(ops) => new_node(ops),
                Mdl::ConcatenateOp(ops) => new_node(ops),
                Mdl::DotGeneralOp(ops) => new_node(ops),
                Mdl::SliceOp(ops) => new_node(ops),
                Mdl::TransposeOp(ops) => new_node(ops),
                Mdl::BroadcastInDimOp(ops) => new_node(ops),
                Mdl::ConvolutionOp(ops) => new_node(ops),
                Mdl::MulOp(ops) => new_node(ops),
                Mdl::AddOp(ops) => new_node(ops),
                Mdl::DivOp(ops) => new_node(ops),
                Mdl::SubtractOp(ops) => new_node(ops),
                Mdl::MinOp(ops) => new_node(ops),
                Mdl::MaxOp(ops) => new_node(ops),
                Mdl::NegOp(ops) => new_node(ops),
                Mdl::TanhOp(ops) => new_node(ops),
                Mdl::ExpOp(ops) => new_node(ops),
                Mdl::IotaOp(ops) => new_node(ops),
                Mdl::PadOp(ops) => new_node(ops),
                Mdl::ReturnOp(ops) => new_node(ops),
                Mdl::BlackBox(ops) => new_node(ops),
                Mdl::SSplit0(ops) => new_node(ops),
                Mdl::SSplit1(ops) => new_node(ops),
                Mdl::MatchRank(ops) => new_node(ops),
                _ => unimplemented!(),
            };

            res.push(node);
        }

        res
    }

    pub fn get_end_to_end_cost(
        egraph: &EGraph<Mdl, TensorAnalysis>,
        to_egraph: &HashMap<Id, Id>,
        rec_expr: &RecExpr<Mdl>,
    ) -> u64 {
        let nodes = CppGraphConverter::convert_to_node(egraph, to_egraph, rec_expr);
        ffi::get_graph_cost(nodes)
    }

    pub fn optimize<'a>(&'a self) -> Vec<ffi::Node> {
        let start = &self.rec_expr;

        // Configuration
        let n_sec: u64 = env::var("SATURATION_TIME_LIMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3600);
        let no_cycle = env::var("NO_CYCLE").unwrap_or("true".to_string()) == "true"; // disallow cycle in egraph?
        let filter_after = env::var("FILTER_AFTER").unwrap_or("true".to_string()) == "true"; // vanilla filtering or efficient filtering
        let iter_limit = 10000;
        let node_limit = 5000000; // max nodes in e-graph

        let path = std::env::current_dir().unwrap();
        println!("The current directory is {}", path.display());
        let rule_file = "src/enzyme_ad/jax/deps/tensat/converted.txt";
        let multi_file = "src/enzyme_ad/jax/deps/tensat/converted_multi.txt";

        let learned_rules =
            read_to_string(rule_file).expect("Something went wrong reading the rule file");
        let time_limit_sec = Duration::new(n_sec, 0);
        let split_rules: Vec<&str> = learned_rules
            .split("\n")
            .filter(|x| !x.is_empty())
            .collect();
        let do_filter_after = no_cycle && filter_after;
        let analysis = TensorAnalysis::new(&self.blackbox_cpp_num_to_tensorinfo);
        let mut rules = rules_from_str(split_rules, do_filter_after);

        // let mut custom_rules: Vec<Rewrite<Mdl, TensorAnalysis>> = vec![
        //     rewrite!("transpose-of-transpose";
        //              "(TransposeOp (TransposeOp ?x ?p) ?p)" => "?x" if decreasing_perm("?p")),
        //     rewrite!("flatten-concat";
        //              "(ConcatenateOp ?v ?d)" => { FlattenConcat {
        //              vec: "?v".parse().unwrap(),
        //              dim: "?d".parse().unwrap(),
        //     }}),
        //     rewrite!("merge-slices";
        //              "(ConcatenateOp (Vec (SliceOp ?x ?s1 ?l1 ?s) (SliceOp ?x ?s2 ?l2 ?s)) ?d)" => { MergeSlices {
        //              x: "?x".parse().unwrap(),
        //              s1: "?s1".parse().unwrap(),
        //              s2: "?s2".parse().unwrap(),
        //              l1: "?l1".parse().unwrap(),
        //              l2: "?l2".parse().unwrap(),
        //              strides: "?s".parse().unwrap(),
        //             dim: "?d".parse().unwrap()
        //     }}),
        //     rewrite!("concat-dot";
        //              "(DotGeneralOp (ConcatenateOp (Vec ?a ?b) ?d1) (ConcatenateOp (Vec ?c ?d) ?d2) ?lb ?rb ?lc ?rc ?p)"
        //              => "(AddOp (DotGeneralOp ?a ?c ?lb ?rb ?lc ?rc ?p) (DotGeneralOp ?b ?d ?lb ?rb ?lc ?rc ?p))"
        //              if concat_dot_compatible("?lc", "?d1", "?rc", "?d2")),
        // ];

        // rules.append(&mut custom_rules);

        if env::var("EQSAT_RULES").unwrap_or("true".to_string()) == "false" {
            rules.clear();
        }

        let mut mlir_rules: Vec<Rewrite<Mdl, TensorAnalysis>> = MlirRewrites::all()
            .iter()
            .map(|r| {
                rewrite!(r.to_string();
                          (r.to_ast().to_string().parse::<Pattern<Mdl>>().unwrap())
                          => { MlirRewriteApplier { rewrite: r.clone(), no_cycle, filter_after, }})
            })
            .collect();

        if env::var("ENZYME_RULES").unwrap_or("true".to_string()) != "false" {
            rules.append(&mut mlir_rules);
        }

        let iter_multi = 2;
        let node_multi = 30000;
        let learned_rules =
            read_to_string(multi_file).expect("Something went wrong reading the multi rule file");
        let multi_rules: Vec<(&str, bool)> = learned_rules
            .split("\n")
            .filter(|x| !x.is_empty())
            .map(|x| (x, /*symmetric=*/ false))
            .collect();

        let use_multi = env::var("MULTI_RULES").unwrap_or("true".to_string()) != "false";

        let mut multi_patterns = MultiPatterns::with_rules(
            multi_rules,
            no_cycle,
            iter_multi,
            filter_after,
            node_multi,
            n_sec,
        );

        let runner = if use_multi {
            Runner::<Mdl, TensorAnalysis, ()>::new(analysis)
                .with_node_limit(node_limit)
                .with_time_limit(time_limit_sec)
                .with_iter_limit(iter_limit)
                .with_expr(&start)
                .with_hook(move |runner| multi_patterns.run_one(runner))
        } else {
            Runner::<Mdl, TensorAnalysis, ()>::new(analysis)
                .with_node_limit(node_limit)
                .with_time_limit(time_limit_sec)
                .with_iter_limit(iter_limit)
                .with_expr(&start)
        };

        let start_time = Instant::now();
        let mut runner = runner.run(&rules[..]);
        println!("RUNNER RUN done");
        if do_filter_after {
            // Do cycle removal after the final iteration
            remove_cycle_by_order(&mut runner);
        }
        let sat_duration = start_time.elapsed();
        let num_iter_sat = runner.iterations.len() - 1;

        println!("Runner complete!");
        println!("  Nodes: {}", runner.egraph.total_size());
        println!("  Classes: {}", runner.egraph.number_of_classes());
        println!("  Stopped: {:?}", runner.stop_reason.unwrap());
        println!("  Time taken: {:?}", sat_duration);
        println!("  Number of iterations: {:?}", num_iter_sat);

        let (num_enodes, num_classes, avg_nodes_per_class, num_edges, num_programs) =
            get_stats(&runner.egraph);
        println!("  Average nodes per class: {}", avg_nodes_per_class);
        println!("  Number of edges: {}", num_edges);
        println!("  Number of programs: {}", num_programs);

        let (egraph, root) = (runner.egraph, runner.roots[0]);
        let cost_model: CostModel = CostModel::new();
        // let (best, ext_secs) = extract_by_greedy(&egraph, root, &cost_model);

        // println!("{}", best);
        let global_extractor = GlobalExtractor::new(
            &egraph,
            &cost_model,
            root,
        );

        let candidate = extract_by_optimization(global_extractor, OptimizationMethod::SimulatedAnnealing);
        println!("optimized candidate: {:?}", candidate);

        // println!(
        //     "end-to-end cost: {}",
        //     CppGraphConverter::get_end_to_end_cost(&egraph, &to_egraph, &best),
        // );

        let (best, ext_secs, to_egraph) = extract_by_ilp(&egraph, root, &cost_model);
        CppGraphConverter::convert_to_node(&egraph, &to_egraph, &best)
    }
}

fn extract_by_greedy(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    cost_model: &CostModel,
) -> (RecExpr<Mdl>, f32) {
    let tnsr_cost = TensorCost { egraph, cost_model };
    let start_time = Instant::now();
    let mut extractor = Extractor::new(egraph, tnsr_cost);
    let (best_cost, best) = extractor.find_best(root);
    let duration = start_time.elapsed();

    println!("Extractor complete!");
    println!("  Time taken: {:?}", duration);
    println!("  Best cost: {:?}", best_cost);
    let ext_secs = duration.as_secs_f32();

    (best, ext_secs)
}

fn extract_by_ilp(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    cost_model: &CostModel,
) -> (RecExpr<Mdl>, f32, HashMap<Id, Id>) {
    // Prepare data for ILP formulation, save to json
    let (m_id_map, e_m, h_i, cost_i, fus_cost_i, fus_i, g_i, root_m, i_to_nodes, blacklist_i) =
        prep_ilp_data(egraph, root, cost_model);

    println!("prepped ilp data");
    let data = json!({
        "e_m": e_m,
        "h_i": h_i,
        "cost_i": cost_i,
        "fus_cost_i": fus_cost_i,
        "fus_i": fus_i,
        "g_i": g_i,
        "root_m": root_m,
        "blacklist_i": blacklist_i,
    });
    let data_str = serde_json::to_string(&data).expect("Fail to convert json to string");
    create_dir_all("./tmp");
    write("./tmp/ilp_data.json", data_str).expect("Unable to write file");

    // Call python script to run ILP
    let order_var_int = false;
    let class_constraint = true;
    let no_order = false;
    let initialise_with_greedy = false;
    let fusion_costs: bool = std::env::var("FUSION_COSTS")
        .unwrap_or(String::from("false"))
        .parse()
        .unwrap();
    let mut arg_vec = vec!["src/enzyme_ad/jax/deps/tensat/extractor/extract.py"];
    if order_var_int {
        arg_vec.push("--order_var_int");
    }
    if class_constraint {
        arg_vec.push("--eclass_constraint");
    }
    if no_order {
        arg_vec.push("--no_order");
    }
    if fusion_costs {
        println!("running with fusion costs");
        arg_vec.push("--fusion-costs");
    }
    if initialise_with_greedy {
        // Get node_to_i map
        let node_to_i: HashMap<Mdl, usize> = (&i_to_nodes)
            .iter()
            .enumerate()
            .map(|(i, node)| (node.clone(), i))
            .collect();

        let tnsr_cost = TensorCost {
            egraph: egraph,
            cost_model: cost_model,
        };
        let mut extractor = Extractor::new(egraph, tnsr_cost);
        let (i_list, m_list) = get_init_solution(egraph, root, &extractor.costs, &g_i, &node_to_i);

        // Store initial solution
        let solution_data = json!({
            "i_list": i_list,
            "m_list": m_list,
        });
        let sol_data_str =
            serde_json::to_string(&solution_data).expect("Fail to convert json to string");
        write("./tmp/init_sol.json", sol_data_str).expect("Unable to write file");

        arg_vec.push("--initialize");
    }
    let time_lim = "1000";
    let num_thread = "8";
    arg_vec.push("--time_lim_sec");
    arg_vec.push(time_lim);
    arg_vec.push("--num_thread");
    arg_vec.push(num_thread);

    let child = Command::new("python3")
        .args(&arg_vec)
        .spawn()
        .expect("failed to execute child");
    let output = child.wait_with_output().expect("failed to get output");

    if output.status.success() {
        // Read back solved results, construct optimized graph
        let solved_str = read_to_string("./tmp/solved.json")
            .expect("Something went wrong reading the solved file");
        let solved_data: SolvedResults =
            serde_json::from_str(&solved_str).expect("JSON was not well-formatted");

        let mut node_picked: HashMap<Id, Mdl> = HashMap::new();
        for (i, x_i) in solved_data.solved_x.iter().enumerate() {
            if *x_i == 1 {
                let eclass_id = m_id_map[g_i[i]];
                if node_picked.contains_key(&eclass_id) {
                    println!("Duplicate node in eclass");
                    println!("{}", node_picked.get(&eclass_id).unwrap().display_op());
                    println!("{}", i_to_nodes[i].display_op());
                    continue;
                }
                //assert!(!node_picked.contains_key(&eclass_id));
                node_picked.insert(eclass_id, i_to_nodes[i].clone());
            }
        }

        let mut expr = RecExpr::default();
        let mut added_memo: HashMap<Id, Id> = Default::default();
        let mut to_egraph: HashMap<Id, Id> = Default::default();
        let _ = construct_best_rec(
            &node_picked,
            root,
            &mut added_memo,
            &mut to_egraph,
            egraph,
            &mut expr,
        );
        (expr, solved_data.time, to_egraph)
    } else {
        panic!("Python script failed");
    }
}

// this is copied from main.rs
fn get_stats(egraph: &EGraph<Mdl, TensorAnalysis>) -> (usize, usize, f32, usize, f32) {
    let num_enodes = egraph.total_size();
    let num_classes = egraph.number_of_classes();
    let avg_nodes_per_class = num_enodes as f32 / (num_classes as f32);
    let num_edges = egraph
        .classes()
        .fold(0, |acc, c| c.iter().fold(0, |sum, n| n.len() + sum) + acc);
    let num_programs = egraph
        .classes()
        .fold(0.0, |acc, c| acc + (c.len() as f32).log2());
    (
        num_enodes,
        num_classes,
        avg_nodes_per_class,
        num_edges,
        num_programs,
    )
}

/// Struct for generating new names for weight tensors in the model
///
/// Generates names like w1, w2...
#[derive(Default)]
pub struct NameGen {
    count_input: i32,
    count_weight: i32,
}

impl NameGen {
    pub fn new_weight_name(&mut self) -> String {
        let name = format!("w_{}", self.count_weight);
        self.count_weight += 1;
        name
    }

    pub fn new_input_name(&mut self) -> String {
        let name = format!("input_{}", self.count_input);
        self.count_input += 1;
        name
    }
}
