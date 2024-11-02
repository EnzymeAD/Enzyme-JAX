#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

//use rand::prelude::*;
use rand;
use std::convert::TryInto;
use std::time::{Duration, Instant};
use std::{collections::HashMap, collections::HashSet};
use {crate::ffi_utils::*, crate::input::ffi, crate::rewrites::*};

use egg::*;

define_language! {
  pub enum Mdl {
      "input"              = Input([Id; 2]),  // takes Var: name@dim1_dim2, block_arg_number
      "CompareOp"          = CompareOp([Id; 4]), // input1, input2, comparison_direction,
                                                           // comparsion_type
      "BroadcastInDimOp"   = BroadcastInDimOp([Id; 2]), // input, broadcast_dimensions
      // TODO: we might need the input type as well.
      "ConvertOp"          = ConvertOp([Id; 2]), // input, output_tyoe.
      // TODO: we probably won't have any rewrites for reduces. Maybe function pointers for the
      // body
      "ReduceOp"           = ReduceOp([Id; 2]), // input, init_values, dimensions, body
      "ReshapeOp"          = ReshapeOp([Id; 2]), // input, shape
      "GatherOp"           = GatherOp([Id; 10]),
      "SelectOp"           = SelectOp([Id; 3]), // pred, on_true, on_false
      "ConcatenateOp"      = ConcatenateOp([Id; 2]), // inputs, dimension
      "ConvolutionOp"      = ConvolutionOp([Id; 19]), // LOTS of inputs
      "DotGeneralOp"       = DotGeneralOp([Id; 7]), // lhs, rhs, ..., shape
      "PadOp"              = PadOp([Id; 5]), // input, padding_value, edge_padding_low,
                                                       // edge_padding_high, interior_padding
      "SliceOp"            = SliceOp([Id; 4]), // input, start_indices, limit_indices, strides
      "TransposeOp"        = TransposeOp([Id; 2]), // input, permutation
      // BINARY OPS
      "MulOp"              = MulOp([Id; 2]),
      "AddOp"              = AddOp([Id; 2]),
      "DivOp"              = DivOp([Id; 2]),
      "SubtractOp"         = SubtractOp([Id; 2]),
      "MinOp"              = MinOp([Id; 2]),
      "MaxOp"              = MaxOp([Id; 2]),
      // UNARY OPS
      "NegOp"              = NegOp([Id; 1]), // input
      "TanhOp"             = TanhOp([Id; 1]), // input
      "ExpOp"              = ExpOp([Id; 1]), // input
      // MISC OPS
      "IotaOp"             = IotaOp([Id; 2]), // iota_dimension, output_shape
      // "ConstantOp"         = ConstantOp([Id; 0]),
      "DynamicUpdateSliceOp" = DynamicUpdateSliceOp([Id; 3]), // operand, update, start_indices
      "DynamicSliceOp"     = DynamicSliceOp([Id; 3]), // operand, start_indices, slice_sizes
      // Complete pain, has arity 12
      "ScatterOp"          = ScatterOp([Id; 4]), // input, scatter_indices, updates, dimension_numbers
      "ReturnOp"           = ReturnOp([Id; 1]),
      "BlackBox"           = BlackBox([Id; 3]),  // id, args, captured values (last two should be vecs)
      "Vec"                = Vec(Vec<Id>),
      "Index"              = Index([Id; 2]),   // index, input. for indexing into ops with multiple result Values.
      // SHORTHANDS (not 1:1 with stablehlo)
      //
      // Let axis' = max(0, len(shape(input)) - len(shape(orig_1))) + axis.
      //
      // TODO: We might need a symmetric "left leaning" version for a different rewrite
      //
      // (SSplit0 input axis orig_0) means:
      // split input on axis' dimension, taking the left component.
      // The split point is such that the result has the same shape as orig_0 on axis'.
      //
      // (SSplit1 input axis orig_1) means:
      // split input on axis' dimension, taking the right component.
      // The split point is such that the result has the same shape as axis'.

      // This translates to a StableHLO SliceOp, with all the slices being [0..shape(input)[d]) in every
      // dimension d except axis', and the axis slice being [0..shape(orig_0)[axis']) for SSplit0,
      // and [shape(input)[axis'] - shape(orig_1)[axis']..shape(input)[axis']) for SSplit1.
      //
      // This allows embedding "splits" in syntactic rewrites (similarly to TASO) without keeping track of the
      // split tree, nor having a custom Applier to get the shape of inputs.
      //
      // These will only be constructed by being on the RHS of rewrites, rather than from the input StableHLO
      // module.
      "SSplit0"             = SSplit0([Id; 3]),  // input, axis, orig_0
      "SSplit1"             = SSplit1([Id; 3]),  // input, axis, orig_1
      // (MatchRank input ref) means:
      // If len(shape(input)) < len(shape(ref)), reshape input such that len(shape(input')) = len(shape(ref)),
      // and shape(input') = shape(input) + [1, 1, ...].
      //
      // Otherwise, reshape input such that shape(input') = shape(input)[0..len(shape(ref))).
      // It is an error to have any index i in [len(shape(ref))..len(shape(input))) such that shape(input)[i] != 1.
      //
      // This allows certain rewrites with ConcatenateOps with axis larger than the rank of the operands, as
      // StableHLO doesn't have implicit casting.
      "MatchRank"         = MatchRank([Id; 2]),  // input, ref
      // (InferReshape input) means:
      // When this node is being merged into an eclass, use the shape of an existing eclass to create a ReshapeOp.
      // Hence, this should only appear as the root node of RHS of a rewrite.
      "InferReshape"      = InferReshape([Id; 1]),  // input
      // MISC
      Num(i64),
      Var(Symbol),
  }
}

impl Mdl {
    pub fn clone_with_mapping(
        &self,
        mapping: &HashMap<Id, Id>
    ) -> Mdl {
        let f = |x: Id| {
            mapping.get(&x).unwrap().clone()
        };

        // TODO: Find a better way of doing this?
        match self {
            Mdl::Num(_) | Mdl::Var(_) => self.clone(),
            Mdl::Vec(x) => Mdl::Vec(x.iter().map(|x| f(*x)).collect()),
            Mdl::Input(x) => Mdl::Input(x.map(f)),
            Mdl::CompareOp(x) => Mdl::CompareOp(x.map(f)),
            Mdl::BroadcastInDimOp(x) => Mdl::BroadcastInDimOp(x.map(f)),
            Mdl::ConvertOp(x) => Mdl::ConvertOp(x.map(f)),
            Mdl::ReduceOp(x) => Mdl::ReduceOp(x.map(f)),
            Mdl::ReshapeOp(x) => Mdl::ReshapeOp(x.map(f)),
            Mdl::GatherOp(x) => Mdl::GatherOp(x.map(f)),
            Mdl::SelectOp(x) => Mdl::SelectOp(x.map(f)),
            Mdl::ConcatenateOp(x) => Mdl::ConcatenateOp(x.map(f)),
            Mdl::ConvolutionOp(x) => Mdl::ConvolutionOp(x.map(f)),
            Mdl::DotGeneralOp(x) => Mdl::DotGeneralOp(x.map(f)),
            Mdl::PadOp(x) => Mdl::PadOp(x.map(f)),
            Mdl::SliceOp(x) => Mdl::SliceOp(x.map(f)),
            Mdl::TransposeOp(x) => Mdl::TransposeOp(x.map(f)),
            Mdl::MulOp(x) => Mdl::MulOp(x.map(f)),
            Mdl::AddOp(x) => Mdl::AddOp(x.map(f)),
            Mdl::DivOp(x) => Mdl::DivOp(x.map(f)),
            Mdl::SubtractOp(x) => Mdl::SubtractOp(x.map(f)),
            Mdl::MinOp(x) => Mdl::MinOp(x.map(f)),
            Mdl::MaxOp(x) => Mdl::MaxOp(x.map(f)),
            Mdl::NegOp(x) => Mdl::NegOp(x.map(f)),
            Mdl::TanhOp(x) => Mdl::TanhOp(x.map(f)),
            Mdl::ExpOp(x) => Mdl::ExpOp(x.map(f)),
            Mdl::IotaOp(x) => Mdl::IotaOp(x.map(f)),
            Mdl::DynamicUpdateSliceOp(x) => Mdl::DynamicUpdateSliceOp(x.map(f)),
            Mdl::DynamicSliceOp(x) => Mdl::DynamicSliceOp(x.map(f)),
            Mdl::ScatterOp(x) => Mdl::ScatterOp(x.map(f)),
            Mdl::ReturnOp(x) => Mdl::ReturnOp(x.map(f)),
            Mdl::BlackBox(x) => Mdl::BlackBox(x.map(f)),
            Mdl::Index(x) => Mdl::Index(x.map(f)),
            Mdl::SSplit0(x) => Mdl::SSplit0(x.map(f)),
            Mdl::SSplit1(x) => Mdl::SSplit1(x.map(f)),
            Mdl::MatchRank(x) => Mdl::MatchRank(x.map(f)),
            Mdl::InferReshape(x) => Mdl::InferReshape(x.map(f)),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DataKind {
    Name,
    Scalar,
    Tnsr,
    TnsrTuple,
}

impl Default for DataKind {
    fn default() -> Self {
        DataKind::Name
    }
}

// Struct for storing shape and value-related metadata for tensors. This
// is the base metadata struct that is used by Analysis as well.
#[derive(Clone, Debug)]
pub struct TensorData {
    // In StableHLO, each operation can have multiple results (for
    // example, see ScatterOp).
    //
    // To handle this, we assume that all operations have multiple
    // results, hence the Vecs below. We can access the i-th element of
    // an operation x by using an Index node: (Index (Num i) x).
    // For an Index node, we simply store singleton Vecs for the two
    // fields below. Then, we always take the 0-th result of any
    // operation that appears as an operand. This allows us to omit
    // (Index 0) for the common case of using the only element in
    // the operation.
    /// The list of results of this tensor
    pub tensors: Vec<ffi::Tensor>,
    /// Is the root node InferReshape?
    pub need_infer_shape: bool,
    /// The name string of this eclass if it is a Name type
    pub name: Option<&'static str>,

}

// Struct for storing information of a tensor. This is passed between functions
// during graph creation.
#[derive(Clone)]
pub struct TensorInfo {
    /// Id into the RecExpr constructed
    pub id: Id,
    pub tensor_data: TensorData,
}
/// Struct for metadata analysis
///
/// In this analysis, it calls functions on the TASO side (e.g. graph.matmul())
/// to create (or get) new ops/nodes and stores pointers to the output tensors.
/// TASO will measure and store the runtime cost when creating a new op/node.
pub struct TensorAnalysis {
    /// Record blacklisted nodes for filtering cycles
    pub blacklist_nodes: HashSet<Mdl>,
    /// Newly added nodes by order
    pub newly_added: Vec<Mdl>,
    pub blackbox_cpp_num_to_shape: HashMap<i64, TensorInfo>,
}

impl<'a> TensorAnalysis {
    pub fn new(blackbox_cpp_num_to_shape: &HashMap<i64, TensorInfo>) -> Self {
        TensorAnalysis {
            blacklist_nodes: HashSet::<Mdl>::new(),
            newly_added: Vec::<Mdl>::new(),
            blackbox_cpp_num_to_shape: blackbox_cpp_num_to_shape.clone(),
        }
    }
}

impl Analysis<Mdl> for TensorAnalysis {
    type Data = TensorData;

    fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
        assert!(!to.need_infer_shape || !from.need_infer_shape);
        if !to.need_infer_shape && !from.need_infer_shape {
            assert!(to.tensors == from.tensors, "to: {:?}, from: {:?}", to, from);
            false
        } else {
            assert!(to.tensors.len() == from.tensors.len());
            for i in 0..to.tensors.len() {
                assert!(to.tensors[i].element_type == from.tensors[i].element_type);
            }
            if to.need_infer_shape {
                *to = from;
                true
            } else {
                false
            }
        }
    }

    fn make(egraph: &EGraph<Mdl, Self>, enode: &Mdl) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;

        // Helper function to create ffi::Tensor from Vec<i32>.
        fn map_to_tensor(vec: Vec<i32>) -> ffi::Tensor {
            ffi::Tensor {
                shape: vec.into_iter().map(|x| x as i64).collect(),
                element_type: ffi::Type::i32, 
            }
        }

        let get_num = |id| {
            for node in egraph[id].iter() {
                if let Mdl::Num(x) = node {
                    return x;
                }
            }
            panic!("no num found");
        };

        match enode {
            Mdl::Num(_) | Mdl::Vec(_) => TensorData {
                tensors: vec![ffi::Tensor {
                    shape: vec![],
                    element_type: ffi::Type::i32,
                }],
                name: Some("num"),
                need_infer_shape: false,
            },
            Mdl::Var(name) => {
                let shape: Vec<i64> = name
                    .as_str()
                    .split("@")
                    .nth(1)
                    .expect(&("Invalid Var name: check shape, name: ".to_owned() + name.as_str()))
                    .split('_')
                    .filter(|&x| !x.is_empty()) // if we have a 0-rank shape, this turns out to be [""]
                    .map(|x| x.parse().unwrap())
                    .collect();

                let element_type: ffi::Type = ffi::Type::from_str(
                    name.as_str()
                        .split("@")
                        .nth(2)
                        .expect("Invalid Var name: check type"),
                )
                .expect("Invalid Var name: check type");

                TensorData {
                    tensors: vec![ffi::Tensor {
                        shape,
                        element_type,
                    }],
                    name: Some(name.as_str()),
                    need_infer_shape: false,
                }
            }
            Mdl::Input([node, _block_arg_number]) => x(node).clone(),
            Mdl::Index([index, input]) => {
                let index = *get_num(*index) as usize;
                let input = x(input);
                TensorData {
                    tensors: vec![input.tensors[index].clone()],
                    name: None,
                    need_infer_shape: false,
                }
            }
            Mdl::BlackBox(inputs) => {
                let cpp_num = get_num(inputs[0]);
                let shape_vec = egraph.analysis.blackbox_cpp_num_to_shape[cpp_num]
                    .tensor_data
                    .tensors
                    .iter()
                    .map(|t| t.clone())
                    .collect();
                TensorData {
                    tensors: shape_vec,
                    name: None,
                    need_infer_shape: false,
                }
            }
            Mdl::ReturnOp(_) => TensorData {
                tensors: vec![],
                name: None,
                need_infer_shape: false,
            },
            Mdl::InferReshape([input]) => {
                let input = x(input);
                TensorData {
                    tensors: vec![ffi::Tensor {
                        shape: vec![],
                        element_type: input.tensors[0].element_type,
                    }],
                    name: None,
                    need_infer_shape: true,
                }
            },
            x => TensorData {
                tensors: create_stablehlo_op(egraph, x, ffi::get_shape),
                name: None,
                need_infer_shape: false,
            },
        }
    }

    fn modify(_egraph: &mut EGraph<Mdl, Self>, _id: Id) {}
}
