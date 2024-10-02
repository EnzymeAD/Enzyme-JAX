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

pub fn create_stablehlo_op<F, R>(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    enode: &Mdl,
    process_output: F,
) -> R
where
    F: Fn(ffi::Ops, Vec<ffi::Tensor>, Vec<ffi::Vector>, Vec<i64>) -> R,
{
    let op = ffi::Ops::from_mdl(enode);
    let (args, other_vecs, int_args) = process_enode_args(egraph, enode);
    let res = process_output(op, args, other_vecs, int_args);
    res
}
