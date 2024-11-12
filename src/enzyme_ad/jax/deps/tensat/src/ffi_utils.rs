use crate::{
    input::ffi,
    model::*,
    rewrites::{get_matrix_option, get_num_option, get_vec_of_nums_option, get_vec_option},
};
use egg::*;

fn process_enode_args(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    enode: &Mdl,
) -> (
    Vec<ffi::Tensor>,
    Vec<ffi::Vector>,
    Vec<i64>,
    Vec<ffi::Matrix>,
) {
    let mut args: Vec<ffi::Tensor> = vec![];
    let mut other_vecs: Vec<ffi::Vector> = vec![];
    let mut int_args: Vec<i64> = vec![];
    let mut matrix_args: Vec<ffi::Matrix> = vec![];

    for child in enode.children().iter() {
        if let Some(other_vec) = get_vec_of_nums_option(egraph, &egraph[*child]) {
            other_vecs.push(other_vec)
        } else if let Some(mat) = get_matrix_option(egraph, &egraph[*child]) {
            matrix_args.push(mat)
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

    (args, other_vecs, int_args, matrix_args)
}

pub fn create_stablehlo_op<F, R>(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    enode: &Mdl,
    process_output: F,
) -> R
where
    F: Fn(ffi::Ops, Vec<ffi::Tensor>, Vec<ffi::Vector>, Vec<i64>, Vec<ffi::Matrix>) -> R,
{
    let op = ffi::Ops::from_mdl(enode);
    let (args, other_vecs, int_args, matrix_args) = process_enode_args(egraph, enode);
    let res = process_output(op, args, other_vecs, int_args, matrix_args);
    res
}


// TODO: dedup with convert_to_node in input.rs
pub fn recexpr_to_node(
    rec_expr: &RecExpr<Mdl>
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
