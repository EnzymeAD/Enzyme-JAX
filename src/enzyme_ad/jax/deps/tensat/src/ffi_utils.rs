use crate::{
    input::ffi,
    model::*,
    rewrites::{get_num_option, get_vec_of_nums_option, get_vec_option},
};
use egg::*;

fn process_enode_args(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    enode: &Mdl,
) -> (Vec<ffi::Tensor>, Vec<ffi::Vector>, Vec<i64>) {
    let mut args: Vec<ffi::Tensor> = vec![];
    let mut other_vecs: Vec<ffi::Vector> = vec![];
    let mut int_args: Vec<i64> = vec![];

    for child in enode.children().iter() {
        if let Some(other_vec) = get_vec_of_nums_option(egraph, &egraph[*child]) {
            other_vecs.push(other_vec)
        } else if let Some(vec) = get_vec_option(&egraph[*child]) {
            vec.iter()
                .for_each(|&id| args.push(egraph[id].data.tensors[0].clone()))
        } else if let Some(num) = get_num_option(&egraph[*child]) {
            int_args.push(num as i64)
        } else {
            // TODO: throw in some assertions
            args.push(egraph[*child].data.tensors[0].clone())
        }
    }

    (args, other_vecs, int_args)
}

pub fn convert_mdl_to_ffi_op(enode: &Mdl) -> ffi::Ops {
    match enode {
        Mdl::Input(_) => ffi::Ops::Input,
        Mdl::CompareOp(_) => ffi::Ops::CompareOp,
        Mdl::BroadcastInDimOp(_) => ffi::Ops::BroadcastInDimOp,
        Mdl::ConvertOp(_) => ffi::Ops::ConvertOp,
        Mdl::ReduceOp(_) => ffi::Ops::ReduceOp,
        Mdl::ReshapeOp(_) => ffi::Ops::ReshapeOp,
        Mdl::GatherOp(_) => ffi::Ops::GatherOp,
        Mdl::SelectOp(_) => ffi::Ops::SelectOp,
        Mdl::ConcatenateOp(_) => ffi::Ops::ConcatenateOp,
        Mdl::DotGeneralOp(_) => ffi::Ops::DotGeneralOp,
        Mdl::PadOp(_) => ffi::Ops::PadOp,
        Mdl::SliceOp(_) => ffi::Ops::SliceOp,
        Mdl::TransposeOp(_) => ffi::Ops::TransposeOp,
        Mdl::MulOp(_) => ffi::Ops::MulOp,
        Mdl::AddOp(_) => ffi::Ops::AddOp,
        Mdl::DivOp(_) => ffi::Ops::DivOp,
        Mdl::SubtractOp(_) => ffi::Ops::SubtractOp,
        Mdl::MinOp(_) => ffi::Ops::MinOp,
        Mdl::MaxOp(_) => ffi::Ops::MaxOp,
        Mdl::NegOp(_) => ffi::Ops::NegOp,
        Mdl::TanhOp(_) => ffi::Ops::TanhOp,
        Mdl::ExpOp(_) => ffi::Ops::ExpOp,
        Mdl::IotaOp(_) => ffi::Ops::IotaOp,
        Mdl::DynamicUpdateSliceOp(_) => ffi::Ops::DynamicUpdateSliceOp,
        Mdl::DynamicSliceOp(_) => ffi::Ops::DynamicSliceOp,
        Mdl::ScatterOp(_) => ffi::Ops::ScatterOp,
        _ => panic!("Unsupported op for creating StableHLO op"),
    }
}

pub fn create_stablehlo_op<F, R>(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    enode: &Mdl,
    process_output: F,
) -> R
where
    F: Fn(ffi::Ops, Vec<ffi::Tensor>, Vec<ffi::Vector>, Vec<i64>) -> R,
{
    let op = convert_mdl_to_ffi_op(enode);
    let (args, other_vecs, int_args) = process_enode_args(egraph, enode);
    let res = process_output(op, args, other_vecs, int_args);
    res
}
