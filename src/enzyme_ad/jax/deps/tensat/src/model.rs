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
      "Index"              = Index([Id; 2]),
      Var(Symbol),
      Num(i64),
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
        assert!(to.tensors == from.tensors, "{:?}{:?}", to, from);
        false
    }

    fn make(egraph: &EGraph<Mdl, Self>, enode: &Mdl) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;

        // Helper function to create ffi::Tensor from Vec<i32>.
        fn map_to_tensor(vec: Vec<i32>) -> ffi::Tensor {
            ffi::Tensor {
                shape: vec.into_iter().map(|x| x as i64).collect(),
                element_type: ffi::Type::i32, // Example: assuming i32 element type, modify as needed
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
                    shape: vec![0],
                    element_type: ffi::Type::i32,
                }],
                name: Some("num"),
            },
            Mdl::Var(name) => {
                let shape: Vec<i64> = name
                    .as_str()
                    .split("@")
                    .nth(1)
                    .expect("Invalid Var name: check shape")
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
                }
            }
            Mdl::Input([node, _block_arg_number]) => x(node).clone(),
            Mdl::Index([index, input]) => {
                let index = *get_num(*index) as usize;
                let input = x(input);
                TensorData {
                    tensors: vec![input.tensors[index].clone()],
                    name: None,
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
                }
            }
            Mdl::ReturnOp(_) => TensorData {
                tensors: vec![],
                name: None,
            },
            x => TensorData {
                tensors: create_stablehlo_op(egraph, x, ffi::get_shape),
                name: None,
            },
        }
    }

    fn modify(_egraph: &mut EGraph<Mdl, Self>, _id: Id) {}
}
