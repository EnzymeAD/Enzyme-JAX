use crate::{input::ffi, model::*};
use egg::{rewrite as rw, *};
use itertools::Itertools;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::time::{Duration, Instant};

// TODO egg now provides bidirectional rules whic should cut down
// this list in half.
#[rustfmt::skip]
pub fn rules<A: Analysis<Mdl>>() -> Vec<Rewrite<Mdl, A>> { vec![
        rw!("ewadd-is-associative"            ; "(ewadd ?x (ewadd ?y ?z)) "                                             => "(ewadd (ewadd ?x ?y) ?z)"),
        rw!("ewadd-is-commutative"            ; "(ewadd ?x ?y) "                                                        => "(ewadd ?y ?x)"),
        rw!("ewmul-is-associative"            ; "(ewmul ?x (ewmul ?y ?z)) "                                             => "(ewmul (ewmul ?x ?y) ?z)"),
        rw!("ewmul-is-commutative"            ; "(ewmul ?x ?y) "                                                        => "(ewmul ?y ?x)"),
        rw!("distributivity-0"                ; "(ewmul (ewadd ?x ?y) ?z) "                                             => "(ewadd (ewmul ?x ?z) (ewmul ?y ?z))"),
        rw!("smul-is-associative"             ; "(smul (smul ?x ?y) ?w) "                                               => "(smul ?x  (smul ?y ?w))"),
        rw!("distributivity-1"                ; "(smul (ewadd ?x ?y) ?w) "                                              => "(ewadd (smul ?x ?w)  (smul ?y ?w))"),
        rw!("operator-commutativity-0"        ; "(smul (ewmul ?x ?y) ?w) "                                              => "(ewmul ?x  (smul ?y ?w))"),
        rw!("transpose-is-its-own-inverse"    ; "(transpose (transpose ?x)) "                                           => "?x"),
        rw!("operator-commutativity-1"        ; "(transpose (ewadd ?x ?y)) "                                            => "(ewadd (transpose ?x)  (transpose ?y))"),
        rw!("operator-commutativity-2"        ; "(transpose (ewmul ?x ?y)) "                                            => "(ewmul (transpose ?x)  (transpose ?y))"),
        rw!("operator-commutativity-3"        ; "(smul (transpose ?x) ?w) "                                             => "(transpose (smul ?x ?w))"),
        rw!("matmul-is-associative"           ; "(matmul ?x (matmul ?y ?z)) "                                           => "(matmul (matmul ?x ?y) ?z)"),
        rw!("matmul-is-linear-0"              ; "(smul (matmul ?x ?y) ?w) "                                             => "(matmul ?x  (smul ?y ?w))"),
        rw!("matmul-is-linear-1"              ; "(matmul ?x (ewadd ?y ?z)) "                                            => "(ewadd (matmul ?x ?y) (matmul ?x ?z))"),
        rw!("matmul-and-transpose"            ; "(transpose (matmul ?x ?y)) "                                           => "(matmul (transpose ?y)  (transpose ?x))"),
        rw!("conv-is-bilinear-0"              ; "(conv2d ?sx ?sy ?p ?c (smul ?x ?w) ?y) "                               => "(conv2d ?sx ?sy ?p ?c ?x (smul ?y ?w))"),
        rw!("conv-is-bilinear-1"              ; "(smul (conv2d ?sx ?sy ?p 0 ?x ?y) ?w) "                            => "(conv2d ?sx ?sy ?p 0 (smul ?x ?w) ?y)"),
        rw!("conv-is-bilinear-2"              ; "(conv2d ?sx ?sy ?p 0 ?x (ewadd ?y ?z)) "                           => "(ewadd (conv2d ?sx ?sy ?p 0 ?x ?y) (conv2d ?sx ?sy ?p 0 ?x ?z))"),
        rw!("conv-is-bilinear-3"              ; "(conv2d ?sx ?sy ?p 0 (ewadd ?x ?y) ?z) "                           => "(ewadd (conv2d ?sx ?sy ?p 0 ?x ?z) (conv2d ?sx ?sy ?p 0 ?y ?z))"),
        //rw!("enlarge-convolution-kernel"      ; "(conv2d ?sx ?sy 0 ?c ?x ?y) "                                      => "(conv2d ?sx ?sy 0 ?c ?x (enlarge ?kx ?ky ?y))"),
        rw!("operator-commutativity-4"        ; "(conv2d ?sx ?sy ?p 2 ?x ?y) "                                      => "(relu (conv2d ?sx ?sy ?p 0 ?x ?y))"),
        rw!("conv-with-2-applies-relu"    ; "(relu (transpose ?x)) "                                                => "(transpose (relu ?x))"),
        // rw!("pooling-by-conv.-with-Cpool"     ; "(conv2d ?sx ?sy ?p 0 ?x (Cpool ?kx ?ky)) "                              => "(poolavg ?kx ?ky ?sx ?sy ?p ?x)"),
        rw!("const_iconv-and-const_pool"      ; "(poolavg ?kx ?ky 1 1 0 (Iconv ?kx ?ky)) "                              => "(Cpool ?kx ?ky)"),
        rw!("identity-kernel"                 ; "(conv2d 1 1 0 0 ?x (Iconv ?kx ?ky)) "                               => "?x"),
        rw!("identity-matrix"                 ; "(matmul ?x   Imatmul ) "                                               => "?x"),
        rw!("ewmul-identity"                  ; "(ewmul ?x Iewmul) "                                                    => "?x"),
        rw!("split-definition-0"              ; "(split_0 ?a (concat ?a ?x ?y)) "                                       => "?x"),
        rw!("split-definition-1"              ; "(split_1 ?a (concat ?a ?x ?y)) "                                       => "?y"),
        rw!("geometry-of-concatenation"       ; "(concat 0 (concat 1 ?x ?y) (concat 1 ?z ?w)) "                         => "(concat 1 (concat 0 ?x ?z) (concat 0 ?y ?w))"),
        rw!("operator-commutativity-5"        ; "(concat ?a (smul ?x ?w) (smul ?y ?w)) "                                => "(smul (concat ?a ?x ?y) ?w)"),
        rw!("operator-commutativity-6"        ; "(concat ?a (ewadd ?x ?y) (ewadd ?z ?w)) "                              => "(ewadd (concat ?a ?x ?z) (concat ?a ?y ?w))"),
        rw!("operator-commutativity-7"        ; "(concat ?a (ewmul ?x ?y) (ewmul ?z ?w)) "                              => "(ewmul (concat ?a ?x ?z) (concat ?a ?y ?w))"),
        rw!("operator-commutativity-8"        ; "(concat ?a (relu ?x) (relu ?y)) "                                      => "(relu (concat ?a ?x ?y))"),
        rw!("concatenation-and-transpose"     ; "(concat 1 (transpose ?x) (transpose ?y)) "                             => "(transpose (concat 0 ?x ?y))"),
        rw!("concatenation-and-matrix-mul.-0" ; "(concat 1 (matmul ?x ?y) (matmul ?x ?z)) "                             => "(matmul ?x (concat 1 ?y ?z))"),
        rw!("concatenation-and-matrix-mul.-1" ; "(matmul (concat 1 ?x ?z) (concat 0 ?y ?w)) "                           => "(ewadd (matmul ?x ?y) (matmul ?z ?w))"),
        rw!("concatenation-and-conv.-0"       ; "(concat 0 (conv2d ?sx ?sy ?p ?c ?x ?z) (conv2d ?sx ?sy ?p ?c ?y ?z)) " => "(conv2d ?sx ?sy ?p ?c (concat 0 ?x ?y) ?z)"),
        rw!("concatenation-and-conv.-1"       ; "(concat 1 (conv2d ?sx ?sy ?p ?c ?x ?y) (conv2d ?sx ?sy ?p ?c ?x ?z)) " => "(conv2d ?sx ?sy ?p ?c ?x (concat 0 ?y ?z))"),
        rw!("concatenation-and-conv.-2"       ; "(conv2d ?sx ?sy ?p 0 (concat 1 ?x ?z) (concat 1 ?y ?w)) "          => "(ewadd (conv2d ?sx ?sy ?p 0 ?x ?y) (conv2d ?sx ?sy ?p 0 ?z ?w))"),
        rw!("concatenation-and-pooling-0"     ; "(concat 1 (poolavg ?kx ?ky ?sx ?sy ?p ?x) (poolavg ?kx ?ky ?sx ?sy ?p ?y)) "               => "(poolavg ?kx ?ky ?sx ?sy ?p (concat 1 ?x ?y))"),
        rw!("concatenation-and-pooling-1"     ; "(concat 0 (poolmax ?kx ?ky ?sx ?sy ?p ?x) (poolmax ?kx ?ky ?sx ?sy ?p ?y)) "               => "(poolmax ?kx ?ky ?sx ?sy ?p (concat 0 ?x ?y))"),
        rw!("concatenation-and-pooling-2"     ; "(concat 1 (poolmax ?kx ?ky ?sx ?sy ?p ?x) (poolmax ?kx ?ky ?sx ?sy ?p ?y)) "               => "(poolmax ?kx ?ky ?sx ?sy ?p (concat 1 ?x ?y))"),
        // inverse
        rw!("-ewadd-is-associative"            ;"(ewadd (ewadd ?x ?y) ?z)"                                                => "(ewadd ?x (ewadd ?y ?z)) "                                             ),
        rw!("-ewadd-is-commutative"            ;"(ewadd ?y ?x)"                                                           => "(ewadd ?x ?y) "                                                        ),
        rw!("-ewmul-is-associative"            ;"(ewmul (ewmul ?x ?y) ?z)"                                                => "(ewmul ?x (ewmul ?y ?z)) "                                             ),
        rw!("-ewmul-is-commutative"            ;"(ewmul ?y ?x)"                                                           => "(ewmul ?x ?y) "                                                        ),
        rw!("-distributivity-0"                ;"(ewadd (ewmul ?x ?z) (ewmul ?y ?z))"                                     => "(ewmul (ewadd ?x ?y) ?z) "                                             ),
        rw!("-smul-is-associative"             ;"(smul ?x  (smul ?y ?w))"                                                 => "(smul (smul ?x ?y) ?w) "                                               ),
        rw!("-distributivity-1"                ;"(ewadd (smul ?x ?w)  (smul ?y ?w))"                                      => "(smul (ewadd ?x ?y) ?w) "                                              ),
        rw!("-operator-commutativity-0"        ;"(ewmul ?x  (smul ?y ?w))"                                                => "(smul (ewmul ?x ?y) ?w) "                                              ),
        rw!("-transpose-is-its-own-inverse"    ;"?x"                                                                      => "(transpose (transpose ?x)) "                                           ),
        rw!("-operator-commutativity-1"        ;"(ewadd (transpose ?x)  (transpose ?y))"                                  => "(transpose (ewadd ?x ?y)) "                                            ),
        rw!("-operator-commutativity-2"        ;"(ewmul (transpose ?x)  (transpose ?y))"                                  => "(transpose (ewmul ?x ?y)) "                                            ),
        rw!("-operator-commutativity-3"        ;"(transpose (smul ?x ?w))"                                                => "(smul (transpose ?x) ?w) "                                             ),
        rw!("-matmul-is-associative"           ;"(matmul (matmul ?x ?y) ?z)"                                              => "(matmul ?x (matmul ?y ?z)) "                                           ),
        rw!("-matmul-is-linear-0"              ;"(matmul ?x  (smul ?y ?w))"                                               => "(smul (matmul ?x ?y) ?w) "                                             ),
        rw!("-matmul-is-linear-1"              ;"(ewadd (matmul ?x ?y) (matmul ?x ?z))"                                   => "(matmul ?x (ewadd ?y ?z)) "                                            ),
        rw!("-matmul-and-transpose"            ;"(matmul (transpose ?y)  (transpose ?x))"                                 => "(transpose (matmul ?x ?y)) "                                           ),
        rw!("-conv-is-bilinear-0"              ;"(conv2d ?sx ?sy ?p ?c ?x (smul ?y ?w))"                                  => "(conv2d ?sx ?sy ?p ?c (smul ?x ?w) ?y) "                               ),
        rw!("-conv-is-bilinear-1"              ;"(conv2d ?sx ?sy ?p 0 (smul ?x ?w) ?y)"                               => "(smul (conv2d ?sx ?sy ?p 0 ?x ?y) ?w) "                            ),
        rw!("-conv-is-bilinear-2"              ;"(ewadd (conv2d ?sx ?sy ?p 0 ?x ?y) (conv2d ?sx ?sy ?p 0 ?x ?z))" => "(conv2d ?sx ?sy ?p 0 ?x (ewadd ?y ?z)) "                           ),
        rw!("-conv-is-bilinear-3"              ;"(ewadd (conv2d ?sx ?sy ?p 0 ?x ?z) (conv2d ?sx ?sy ?p 0 ?y ?z))" => "(conv2d ?sx ?sy ?p 0 (ewadd ?x ?y) ?z) "                           ),
        rw!("-enlarge-convolution-kernel"      ;"(conv2d ?sx ?sy 0 ?c ?x (enlarge ?kx ?ky ?y))"                            => "(conv2d ?sx ?sy 0 ?c ?x ?y) "                                      ),
        rw!("-operator-commutativity-4"        ;"(relu (conv2d ?sx ?sy ?p 0 ?x ?y))"                                  => "(conv2d ?sx ?sy ?p 2 ?x ?y) "                                      ),
        rw!("-conv-with-2-applies-relu"    ;"(transpose (relu ?x))"                                                   => "(relu (transpose ?x)) "                                                ),
        rw!("-pooling-by-conv.-with-Cpool"     ;"(poolavg ?kx ?ky ?sx ?sy ?p ?x)"                                                   => "(conv2d ?sx ?sy ?p 0 ?x (Cpool ?kx ?ky)) "                              ),
        rw!("-const_iconv-and-const_pool"      ;"(Cpool ?kx ?ky)"                              => "(poolavg ?kx ?ky 1 1 0 (Iconv ?kx ?ky))"),
        // rw!("-identity-kernel"                 ;"?x"                                                                      => "(conv2d 1 1 0 0 ?x (Iconv ?k)) "                               ),
        rw!("-identity-matrix"                 ;"?x"                                                                      => "(matmul ?x   Imatmul ) "                                               ),
        rw!("-ewmul-identity"                  ;"?x"                                                                      => "(ewmul ?x Iewmul) "                                                    ),
        // rw!("-split-definition-00"              ;"?x"                                                                      => "(split_0 1 (concat 1 ?x ?y)) "                                       ),
        // rw!("-split-definition-01"              ;"?x"                                                                      => "(split_0 0 (concat 0 ?x ?y)) "                                       ),
        // rw!("-split-definition-10"              ;"?y"                                                                      => "(split_1 0 (concat 0 ?x ?y)) "                                       ),
        // rw!("-split-definition-11"              ;"?y"                                                                      => "(split_1 1 (concat 1 ?x ?y)) "                                       ),
        rw!("-geometry-of-concatenation"       ;"(concat 1 (concat 0 ?x ?z) (concat 0 ?y ?w))"                            => "(concat 0 (concat 1 ?x ?y) (concat 1 ?z ?w)) "                         ),
        rw!("-operator-commutativity-5"        ;"(smul (concat ?a ?x ?y) ?w)"                                             => "(concat ?a (smul ?x ?w) (smul ?y ?w)) "                                ),
        rw!("-operator-commutativity-6"        ;"(ewadd (concat ?a ?x ?z) (concat ?a ?y ?w))"                             => "(concat ?a (ewadd ?x ?y) (ewadd ?z ?w)) "                              ),
        rw!("-operator-commutativity-7"        ;"(ewmul (concat ?a ?x ?z) (concat ?a ?y ?w))"                             => "(concat ?a (ewmul ?x ?y) (ewmul ?z ?w)) "                              ),
        rw!("-operator-commutativity-8"        ;"(relu (concat ?a ?x ?y))"                                                => "(concat ?a (relu ?x) (relu ?y)) "                                      ),
        rw!("-concatenation-and-transpose"     ;"(transpose (concat 0 ?x ?y))"                                            => "(concat 1 (transpose ?x) (transpose ?y)) "                             ),
        rw!("-concatenation-and-matrix-mul.-0" ;"(matmul ?x (concat 1 ?y ?z))"                                            => "(concat 1 (matmul ?x ?y) (matmul ?x ?z)) "                             ),
        rw!("-concatenation-and-matrix-mul.-1" ;"(ewadd (matmul ?x ?y) (matmul ?z ?w))"                                   => "(matmul (concat 1 ?x ?z) (concat 0 ?y ?w)) "                           ),
        rw!("-concatenation-and-conv.-0"       ;"(conv2d ?sx ?sy ?p ?c (concat 0 ?x ?y) ?z)"                              => "(concat 0 (conv2d ?sx ?sy ?p ?c ?x ?z) (conv2d ?sx ?sy ?p ?c ?y ?z)) " ),
        rw!("-concatenation-and-conv.-1"       ;"(conv2d ?sx ?sy ?p ?c ?x (concat 0 ?y ?z))"                              => "(concat 1 (conv2d ?sx ?sy ?p ?c ?x ?y) (conv2d ?sx ?sy ?p ?c ?x ?z)) " ),
        rw!("-concatenation-and-conv.-2"       ;"(ewadd (conv2d ?sx ?sy ?p 0 ?x ?y) (conv2d ?sx ?sy ?p 0 ?z ?w))" => "(conv2d ?sx ?sy ?p 0 (concat 1 ?x ?z) (concat 1 ?y ?w)) "          ),
        rw!("-concatenation-and-pooling-0"     ;"(poolavg ?kx ?ky ?sx ?sy ?p (concat 1 ?x ?y))"                                     => "(concat 1 (poolavg ?kx ?ky ?sx ?sy ?p ?x) (poolavg ?kx ?ky ?sx ?sy ?p ?y)) "               ),
        rw!("-concatenation-and-pooling-1"     ;"(poolmax ?kx ?ky ?sx ?sy ?p (concat 0 ?x ?y))"                                     => "(concat 0 (poolmax ?kx ?ky ?sx ?sy ?p ?x) (poolmax ?kx ?ky ?sx ?sy ?p ?y)) "               ),
        rw!("-concatenation-and-pooling-2"     ;"(poolmax ?kx ?ky ?sx ?sy ?p (concat 1 ?x ?y))"                                     => "(concat 1 (poolmax ?kx ?ky ?sx ?sy ?p ?x) (poolmax ?kx ?ky ?sx ?sy ?p ?y)) "               ),
]}

pub fn rules_from_str(rs: Vec<&str>, filter_after: bool) -> Vec<Rewrite<Mdl, TensorAnalysis>> {
    let mut rule_vec = Vec::new();
    for (pos, rule) in rs.iter().enumerate() {
        let eqn: Vec<&str> = rule.split("=>").collect();
        let lhs: Pattern<Mdl> = eqn[0].parse().unwrap();
        let rhs: Pattern<Mdl> = eqn[1].parse().unwrap();
        let rule_name = format!("rule{}", pos);
        rule_vec.push(rw!(rule_name; { lhs.clone() } => { CheckApply {
            pat: rhs,
            src_pat: lhs,
            filter_after: filter_after,
        } }));
    }
    rule_vec
}

/// Hand specified normal rules from TASO
#[rustfmt::skip]
pub static PRE_DEFINED_RULES: &[&str] = &[
    /*
    "(conv2d 1 1 0 0 ?input_1 ?input_2)=>(conv2d 1 1 0 0 ?input_1 (merge ?input_2 2))",
    "(conv2d 1 1 0 2 ?input_1 ?input_2)=>(conv2d 1 1 0 2 ?input_1 (merge ?input_2 2))",
    "(conv2d 2 2 0 0 ?input_1 ?input_2)=>(conv2d 2 2 0 0 ?input_1 (merge ?input_2 2))",
    "(conv2d 2 2 0 2 ?input_1 ?input_2)=>(conv2d 2 2 0 2 ?input_1 (merge ?input_2 2))",
    */
];

/// Hand specified multi-pattern rules from TASO
#[rustfmt::skip]
pub static PRE_DEFINED_MULTI: &[&str] = &[
    /* 
    "(conv2d 1 1 0 0 ?input_1 ?input_2)=>(split_0 (split 1 (conv2d 1 1 0 0 ?input_1 (concat 0 4 (enlarge ?input_2 ?input_3) ?input_3))))",
    "(conv2d 1 1 0 0 ?input_1 ?input_3)=>(split_1 (split 1 (conv2d 1 1 0 0 ?input_1 (concat 0 4 (enlarge ?input_2 ?input_3) ?input_3))))",
    "(conv2d 1 1 0 2 ?input_1 ?input_2)=>(split_0 (split 1 (conv2d 1 1 0 2 ?input_1 (concat 0 4 (enlarge ?input_2 ?input_3) ?input_3))))",
    "(conv2d 1 1 0 2 ?input_1 ?input_3)=>(split_1 (split 1 (conv2d 1 1 0 2 ?input_1 (concat 0 4 (enlarge ?input_2 ?input_3) ?input_3))))",
    */
];

// TODO: We should really clean these. Just dirty hacks for now to test out conditional rewrites

pub fn get_vec_of_nums(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    eclass: &EClass<Mdl, TensorData>,
) -> Vec<i64> {
    let vec = get_vec(eclass);
    let mut result = Vec::new();

    for id in vec.iter() {
        let num_eclass = &egraph[*id];
        let num = get_num(&num_eclass);
        result.push(*num);
    }

    result
}

pub fn get_vec_of_nums_option(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    eclass: &EClass<Mdl, TensorData>,
) -> Option<ffi::Vector> {
    get_vec_option(eclass)
        .map(|vec| {
            vec.iter()
                .map(|&id| get_num_option(&egraph[id]))
                .collect::<Option<Vec<i64>>>()
        })
        .and_then(|opt_vec| opt_vec.map(|vec| ffi::Vector { vec }))
}

pub fn get_vec_option(eclass: &EClass<Mdl, TensorData>) -> Option<&Vec<Id>> {
    let mut iter = eclass.iter();
    match iter.next() {
        Some(Mdl::Vec(vec)) => match iter.next() {
            Some(_) => panic!("Malformed vec node!"),
            None => Some(vec),
        },
        _ => None,
    }
}

pub fn get_vec(eclass: &EClass<Mdl, TensorData>) -> &Vec<Id> {
    for node in eclass.iter() {
        match node {
            Mdl::Vec(vec) => return vec,
            _ => {}
        }
    }

    panic!("no vec found");
}

pub fn get_num_option(eclass: &EClass<Mdl, TensorData>) -> Option<i64> {
    for node in eclass.iter() {
        match node {
            Mdl::Num(n) => return Some(*n),
            _ => {}
        }
    }

    return None;
}

pub fn get_num(eclass: &EClass<Mdl, TensorData>) -> &i64 {
    for node in eclass.iter() {
        match node {
            Mdl::Num(n) => return n,
            _ => {}
        }
    }

    panic!("no num found");
}

fn make_num(egraph: &mut EGraph<Mdl, TensorAnalysis>, num: i64) -> Id {
    egraph.add(Mdl::Num(num))
}

fn make_vec(egraph: &mut EGraph<Mdl, TensorAnalysis>, seq: &[Id]) -> Id {
    egraph.add(Mdl::Vec((*seq).to_vec()))
}

pub fn decreasing_perm<'a>(
    var: &'a str,
) -> impl Fn(&mut EGraph<Mdl, TensorAnalysis>, Id, &Subst) -> bool + 'a {
    let var = var.parse().unwrap();
    move |egraph, _, subst: &Subst| {
        let eclass = &egraph[subst[var]];
        let vec = get_vec(eclass);
        let n = vec.len();
        for i in 1..n {
            let prev = get_num(&egraph[vec[i - 1]]);
            let cur = get_num(&egraph[vec[i]]);
            if prev < cur {
                return false;
            }
        }
        return true;
    }
}

fn finish_apply(egraph: &mut EGraph<Mdl, TensorAnalysis>, matched_id: Id, new_id: Id) -> Vec<Id> {
    if egraph.union(matched_id, new_id).1 {
        vec![new_id]
    } else {
        vec![]
    }
}

pub fn concat_dot_compatible(
    lc: &'static str,
    d1: &'static str,
    rc: &'static str,
    d2: &'static str,
) -> impl Fn(&mut EGraph<Mdl, TensorAnalysis>, Id, &Subst) -> bool {
    // The dot_general of two concats should be able to be reduced into an add of two (smaller) dot_generals
    // if the dimension of each concat is included in the corresponding contract dims.
    // TODO: I'm not confident this always holds - seemed to work for smaller examples but there might
    // be a case I'm missing.

    let lc = lc.parse().unwrap();
    let d1 = d1.parse().unwrap();
    let rc = rc.parse().unwrap();
    let d2 = d2.parse().unwrap();

    move |egraph, _, subst| {
        let lc = get_vec(&egraph[subst[lc]]);
        let d1 = &subst[d1];
        let rc = get_vec(&egraph[subst[rc]]);
        let d2 = &subst[d2];

        lc.contains(d1) && rc.contains(d2)
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlattenConcat {
    pub vec: Var,
    pub dim: Var,
}

impl Applier<Mdl, TensorAnalysis> for FlattenConcat {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Mdl, TensorAnalysis>,
        matched_id: Id,
        subst: &Subst,
    ) -> Vec<Id> {
        let vec = get_vec(&egraph[subst[self.vec]]);
        let dim = *get_num(&egraph[subst[self.dim]]);

        // Go through elements in vec, and see if there is another concat with the same dimension.
        // If so we can flatten it.
        let mut new_vec: Vec<Id> = vec![];

        'outer: for i in vec.iter() {
            for node in egraph[*i].iter() {
                match node {
                    Mdl::ConcatenateOp(c) => {
                        let inp = c[0];
                        let d = *get_num(&egraph[c[1]]);

                        if dim != d {
                            continue;
                        }
                        // We can merge this one in
                        let children = get_vec(&egraph[inp]);
                        for c in children.iter() {
                            new_vec.push(*c);
                        }
                        continue 'outer;
                    }
                    _ => {}
                }
            }

            // No concat found
            new_vec.push(*i);
        }
        let dim_id = make_num(egraph, dim);
        let vec_id = make_vec(egraph, &new_vec);
        let id = egraph.add(Mdl::ConcatenateOp([vec_id, dim_id]));

        finish_apply(egraph, matched_id, id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeSlices {
    pub x: Var,
    pub s1: Var,
    pub s2: Var,
    pub l1: Var,
    pub l2: Var,
    pub strides: Var,
    pub dim: Var,
}

impl Applier<Mdl, TensorAnalysis> for MergeSlices {
    fn apply_one(
        &self,
        egraph: &mut EGraph<Mdl, TensorAnalysis>,
        matched_id: Id,
        subst: &Subst,
    ) -> Vec<Id> {
        let x = subst[self.x];
        let starting_1 = get_vec(&egraph[subst[self.s1]]);
        let starting_2 = get_vec(&egraph[subst[self.s2]]);
        let limiting_1 = get_vec(&egraph[subst[self.l1]]);
        let limiting_2 = get_vec(&egraph[subst[self.l2]]);
        let strides_id = subst[self.strides];
        let strides = get_vec(&egraph[strides_id]);
        let dim = *get_num(&egraph[subst[self.dim]]);

        assert!(starting_1.len() == starting_2.len());
        assert!(limiting_1.len() == limiting_2.len());
        assert!(starting_1.len() == limiting_1.len());

        let n = starting_1.len();
        // Concat(Split(A), Split(A)) => Split(A)
        // For every dimension not equal to the concat dim, the starting and
        // limiting indices should be the same. The concat dim, we need to do
        // a bit of maths to ensure they are "contiguous" wrt the stride.

        let mut new_starting: Vec<i64> = vec![];
        let mut new_limiting: Vec<i64> = vec![];

        for i in 0..n {
            let s1 = *get_num(&egraph[starting_1[i]]);
            let s2 = *get_num(&egraph[starting_2[i]]);
            let l1 = *get_num(&egraph[limiting_1[i]]);
            let l2 = *get_num(&egraph[limiting_2[i]]);
            let stride = *get_num(&egraph[strides[i]]);

            if i != (dim as usize) {
                if s1 != s2 || l1 != l2 {
                    return vec![];
                }
                new_starting.push(s1);
                new_limiting.push(l1);
            } else {
                if s2 > l1 {
                    return vec![];
                } // should be non-overlapping

                // Check the next unchosen index for the first slice. This should be s2
                // For example, if stride = 3, s1 = 1, l1 = 5:
                // 1 2 3 4 5 6 7 8 9
                // x     x    [x]
                // So s2 = 7 for the two slices to be contiguous.

                let numbers_picked = (l1 - s1) / stride + i64::from((l1 - s1) % stride != 0); // ceil
                let next_unchosen_index = s1 + numbers_picked * stride;
                if next_unchosen_index != s2 {
                    return vec![];
                }
                new_starting.push(s1);
                new_limiting.push(l2);
            }
        }

        let new_starting_ids: Vec<Id> = new_starting.iter().map(|x| make_num(egraph, *x)).collect();
        let new_limiting_ids: Vec<Id> = new_limiting.iter().map(|x| make_num(egraph, *x)).collect();
        let node = Mdl::SliceOp([
            x,
            make_vec(egraph, &new_starting_ids),
            make_vec(egraph, &new_limiting_ids),
            strides_id,
        ]);
        let id = egraph.add(node);

        finish_apply(egraph, matched_id, id)
    }
}

/// Custom struct implementing the Applier trait, checking the new nodes to
/// construct are all valid before actually apply.
#[derive(Debug, Clone, PartialEq)]
struct CheckApply {
    /// the pattern of the right hand side of the rewrite rule, the one
    /// to be constructed.
    pat: Pattern<Mdl>,
    /// Source graph pattern, used in cycle filtering
    src_pat: Pattern<Mdl>,
    /// Whether we need to check if any node in matched source graph is in blacklist
    filter_after: bool,
}

impl Applier<Mdl, TensorAnalysis> for CheckApply {
    /// Apply the pattern once. Check the new nodes are valid before actually
    /// apply. See Applier trait in egg for more information.
    fn apply_one(
        &self,
        egraph: &mut EGraph<Mdl, TensorAnalysis>,
        matched_id: Id,
        subst: &Subst,
    ) -> Vec<Id> {
        if self.filter_after {
            // Check if any node in matched source graph is in blacklist. If so, stop applying
            let (contains, _) = contains_blacklist(self.src_pat.ast.as_ref(), egraph, subst);
            if contains {
                return vec![];
            }
        }
        let (valid, _, existing) =
            check_pat(self.pat.ast.as_ref(), egraph, subst, self.filter_after);
        if valid {
            let result = self.pat.apply_one(egraph, matched_id, subst);

            // Add the newly added nodes to the ordering vector
            if self.filter_after {
                let existing = existing.unwrap();
                add_newly_added(self.pat.ast.as_ref(), egraph, subst, &existing);
            }
            result
        } else {
            vec![]
        }
    }

    fn vars(&self) -> Vec<Var> {
        self.pat.vars()
    }
}

/// Check if the matched graph of the pattern contains any blacklisted nodes
///
/// # Returns
///
/// A tuple of (bool, Option<Id>) where
///
/// - bool: true if the nodes in this pattern contains some node in blacklist
/// - Option<Id>: if the nodes in this pattern do not contain blacklisted
///     nodes, then this is the Id of the matched EClass of the root of this pattern(pat.last())
fn contains_blacklist(
    pat: &[ENodeOrVar<Mdl>],
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    subst: &Subst,
) -> (bool, Option<Id>) {
    match pat.last().unwrap() {
        ENodeOrVar::Var(w) => (false, Some(subst[*w])),
        ENodeOrVar::ENode(e) => {
            let children = e.children();
            let results: Vec<(bool, Option<Id>)> = children
                .iter()
                .map(|child| contains_blacklist(&pat[..usize::from(*child) + 1], egraph, subst))
                .collect();

            let contains = results.iter().any(|res| res.0);

            if contains {
                (true, None)
            } else {
                let mut new_e = e.clone();
                let new_e_ch = new_e.children_mut();
                for (i, res) in results.iter().enumerate() {
                    if let Some(id) = res.1 {
                        new_e_ch[i] = id;
                    } else {
                        // This place shouldn't be reached in any case. The pat and subst passed
                        // in as arguments are from the results of searching pat in the Egraph.
                        // So all the nodes in pat should be present in the EGraph. But if we run
                        // bert with 1 iteration of multi and more than ~5/6 iterations of single
                        // rules, this place is reached. Seems that the searched matches returned
                        // by egg can have some nodes in pat that cannot be found in the Egraph.
                        //
                        // Right now, we simply treat the pattern as not containing blacklisted
                        // nodes if this happens. Since we do cycle filtering in the end of each
                        // iteration, this should be fine.
                        //
                        // TODO: look into the above issue
                        return (false, None);
                    }
                }
                if egraph.analysis.blacklist_nodes.contains(&new_e) {
                    (true, None)
                } else {
                    let looked = egraph.lookup(new_e);
                    // This looked should never be None. See the above issue.
                    (false, looked)
                }
            }
        }
    }
}

/// Check if all the new nodes to create in the pattern is valid.
///
/// This function does the checking recursively.
///
/// # Parameters
///
/// - `pat`: the AST representation of the pattern. See egg::Pattern for more info
/// - `egraph`: E-graph of interest
/// - `subst`: mapping variable to eclass ID. See egg::Subst for more info.
/// - `get_exist_nodes`: whether to get a set of existing nodes in this pattern
///
/// # Returns
///
/// A tuple of (bool, Option<Id>, TensorData) where
///
/// - bool: true if the nodes in this pattern are all valid
/// - Option<Id>: if the root node of this pattern (pat.last()) is in egraph,
///     then it is the Id of that eclass. Otherwise None
/// - TensorData: The TensorData for the root node of this pattern. This is read from
///     egraph if the root node is in egraph, otherwise constructed by calling
///     TASO functions.
/// - Option<HashSet<Mdl>>: if get_exist_nodes is true, this returns the set of
///     existing nodes in this pattern
fn check_pat(
    pat: &[ENodeOrVar<Mdl>],
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    subst: &Subst,
    get_exist_nodes: bool,
) -> (bool, Option<Id>, Option<HashSet<Mdl>>) {
    match pat.last().unwrap() {
        ENodeOrVar::Var(w) => {
            // The root node is a variable, then use subst to get metadata from egraph
            let cid = subst[*w];
            if get_exist_nodes {
                return (true, Some(cid), Some(HashSet::<Mdl>::new()));
            } else {
                return (true, Some(cid), None);
            }
        }
        ENodeOrVar::ENode(e) => {
            // The root is an enode. Recursively get checking results from its children
            let children = e.children();
            let results: Vec<(bool, Option<Id>, Option<HashSet<Mdl>>)> = children
                .iter()
                .map(|child| {
                    check_pat(
                        &pat[..usize::from(*child) + 1],
                        egraph,
                        subst,
                        get_exist_nodes,
                    )
                })
                .collect();

            // Check if any children contains invalid nodes
            let mut violated = false;
            for res in &results {
                if !res.0 {
                    violated = true;
                }
            }
            if violated {
                return (false, None, None);
            } else {
                // Check if all children are in egraph
                let mut all_in = true;
                for res in &results {
                    let is_in = match res.1 {
                        Some(_) => true,
                        None => false,
                    };
                    if !is_in {
                        all_in = false;
                    }
                }
                if all_in {
                    // Construct enode, check if in egraph
                    let mut new_e = e.clone();
                    let new_e_ch = new_e.children_mut();
                    for (i, res) in results.iter().enumerate() {
                        new_e_ch[i] = res.1.unwrap();
                    }
                    let looked = egraph.lookup(new_e.clone());
                    if let Some(id) = looked {
                        // Get metadata from egraph
                        if get_exist_nodes {
                            let mut existing_nodes = HashSet::<Mdl>::new();
                            for res in results.iter() {
                                for node in res.2.as_ref().unwrap().iter() {
                                    existing_nodes.insert(node.clone());
                                }
                            }
                            existing_nodes.insert(new_e);
                            return (true, looked, Some(existing_nodes));
                        } else {
                            return (true, looked, None);
                        }
                    }
                }
                let result = match e {
                    _ => (true, None),
                };
                if get_exist_nodes && result.0 {
                    let mut existing_nodes = HashSet::<Mdl>::new();
                    for res in results.iter() {
                        for node in res.2.as_ref().unwrap().iter() {
                            existing_nodes.insert(node.clone());
                        }
                    }
                    return (result.0, result.1, Some(existing_nodes));
                } else {
                    return (result.0, result.1, None);
                }
            }
        }
    };
}

/// Struct for storing information on how each pattern maps to its canonical version
#[derive(Debug)]
struct MapToCanonical {
    /// Index into MultiPatterns.canonical_src_pat. Points to the canonical version.
    index: usize,
    /// Mapping from variable in this pattern to variable in the canonical pattern.
    var_map: HashMap<egg::Var, egg::Var>,
}

/// Struct for the multi-pattern rules. In charge of searching for matches and
/// applying the rewrite.
#[derive(Debug)]
pub struct MultiPatterns {
    /// Vec of (src_1, src_2, dst_1, dst_2, symmetric)
    rules: Vec<(Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, bool)>,
    /// Vec of all unique canonical source patterns (for src_1's and src_2's)
    canonical_src_pat: Vec<Pattern<Mdl>>,
    /// Mapping information for each src pattern. The order is the same as in rules
    src_pat_maps: Vec<(MapToCanonical, MapToCanonical)>,
    /// Whether to allow cycles in EGraph
    no_cycle: bool,
    /// Whether to do cycle filtering after applying. This is always false when no_cycle is false
    filter_after: bool,
    /// Number of iterations to run multi-pattern rules
    iter_limit: usize,
    /// Maximum number of nodes to added here
    node_limit: usize,
    /// Maximum number of seconds to run
    n_sec: u64,
    /// Number of successfully applied matches
    num_applied: usize,
    /// Descendents map. Only used if filter_after is true
    descendents: Option<HashMap<Id, HashSet<Id>>>,
}

impl MultiPatterns {
    /// Construct a MultiPatterns with rules. Each multi-pattern rule contains two matched outputs.
    ///
    /// # Parameters
    ///
    /// - `rules`: every adjacent pair of entries should belong to the same multi-pattern rule.
    /// - `no_cycle`: whether or not to do cycle filtering
    /// - `iter_limit`: Number of iterations to apply multi-pattern rules
    /// - `filter_after`: if true, do efficient filtering (filter cycle after the iteration);
    ///         else, do naive filtering (check cycle before each application)
    /// - `node_limit`: Maximum number of nodes to added here
    /// - `n_sec`: Maximum number of seconds to run
    pub fn with_rules(
        rules: Vec<(&str, bool)>,
        no_cycle: bool,
        iter_limit: usize,
        filter_after: bool,
        node_limit: usize,
        n_sec: u64,
    ) -> MultiPatterns {
        assert!(rules.len() % 2 == 0);

        let mut multi_rules =
            Vec::<(Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, bool)>::new();
        let mut canonical_pats = Vec::<Pattern<Mdl>>::new();
        let mut src_pat_maps = Vec::<(MapToCanonical, MapToCanonical)>::new();

        let mut canonicalize_and_add = |pat: &Pattern<Mdl>| {
            let (pat_canonical, pat_var_map) = canonicalize(pat);

            let index_found = canonical_pats.iter().position(|x| *x == pat_canonical);
            let pat_index = index_found
                .or_else(|| {
                    canonical_pats.push(pat_canonical);
                    Some(canonical_pats.len() - 1)
                })
                .unwrap();
            MapToCanonical {
                index: pat_index,
                var_map: pat_var_map,
            }
        };

        let get_pats = |rule: &str| {
            rule.split("=>")
                .map(|x| x.parse().unwrap())
                .next_tuple()
                .unwrap()
        };

        for i in 0..(rules.len() / 2) {
            let (src_1, dst_1) = get_pats(rules[2 * i].0);
            let (src_2, dst_2) = get_pats(rules[2 * i + 1].0);

            let src_1_map = canonicalize_and_add(&src_1);
            let src_2_map = canonicalize_and_add(&src_2);

            assert!(rules[2 * i].1 == rules[2 * i + 1].1);
            let symmetric = rules[2 * i].1;

            multi_rules.push((src_1, src_2, dst_1, dst_2, symmetric));
            src_pat_maps.push((src_1_map, src_2_map));
        }

        println!("Number of canonicalized {:?}", canonical_pats.len());

        MultiPatterns {
            rules: multi_rules,
            canonical_src_pat: canonical_pats,
            src_pat_maps: src_pat_maps,
            no_cycle: no_cycle,
            iter_limit: iter_limit,
            filter_after: filter_after && no_cycle,
            descendents: None,
            node_limit: node_limit,
            num_applied: 0,
            n_sec: n_sec,
        }
    }

    /// Search and apply all multi-pattern rules for one iteration
    ///
    /// This function is used as hook function to egg::Runner. It first searches for matches
    /// of all canonicalized source patterns. Then for all compatible substitutions found,
    /// it checks and applies the dst patterns. It won't apply if src_1 and src_2 matches with
    /// the same eclass. It always returns Ok()
    pub fn run_one(&mut self, runner: &mut Runner<Mdl, TensorAnalysis, ()>) -> Result<(), String> {
        if self.filter_after {
            // This is to remove cycles introduced during the last iteration of single rules
            remove_cycle_by_order(runner);
        }

        if runner.iterations.len() < self.iter_limit && self.node_limit > 0 && self.n_sec > 0 {
            println!("Run one");
            let starting_num_nodes = runner.egraph.analysis.newly_added.len();
            let start_time = Instant::now();
            let mut num_applied = 0;

            // Construct Vec to store matches for each canonicalized pattern
            let matches: Vec<Vec<SearchMatches>> = self
                .canonical_src_pat
                .iter()
                .map(|x| x.search(&runner.egraph))
                .collect();

            if self.filter_after {
                // Make a pass to get descendents
                self.descendents = Some(compute_all_descendents(
                    &runner.egraph,
                    runner.roots[0],
                    /*check_blacklist=*/ true,
                ));
            }

            // For each multi rule
            'outer: for (i, rule) in self.rules.iter().enumerate() {
                let map_1 = &self.src_pat_maps[i].0;
                let map_2 = &self.src_pat_maps[i].1;
                // If the rule is fully symmetrical
                if map_1.index == map_2.index && rule.4 {
                    let matches_both = &matches[map_1.index];
                    for (i, match_1) in matches_both.iter().enumerate() {
                        for match_2 in (&matches_both[(i + 1)..]).iter() {
                            if match_1.eclass == match_2.eclass {
                                // We don't want to apply multi-pattern rules on the same eclass
                                continue;
                            }
                            let n_applied =
                                self.apply_match_pair(rule, match_1, match_2, map_1, map_2, runner);
                            //num_applied += n_applied;
                            //let num_nodes = runner.egraph.analysis.newly_added.len();
                            //if num_nodes - starting_num_nodes > self.node_limit {
                            //    break 'outer;
                            //}
                            if start_time.elapsed().as_secs() > self.n_sec {
                                break 'outer;
                            }
                        }
                    }
                } else {
                    let matches_1 = &matches[map_1.index];
                    let matches_2 = &matches[map_2.index];
                    for match_1 in matches_1 {
                        for match_2 in matches_2 {
                            if match_1.eclass == match_2.eclass {
                                // We don't want to apply multi-pattern rules on the same eclass
                                continue;
                            }
                            let n_applied =
                                self.apply_match_pair(rule, match_1, match_2, map_1, map_2, runner);
                            //num_applied += n_applied;
                            //let num_nodes = runner.egraph.analysis.newly_added.len();
                            //if num_nodes - starting_num_nodes > self.node_limit {
                            //    break 'outer;
                            //}
                            if start_time.elapsed().as_secs() > self.n_sec {
                                break 'outer;
                            }
                        }
                    }
                }
            }

            runner.egraph.rebuild();

            if self.filter_after {
                // This is to remove cycles introduced during this run_one
                remove_cycle_by_order(runner);
            }
            println!("Done one");

            let ending_num_nodes = runner.egraph.analysis.newly_added.len();
            self.node_limit = if ending_num_nodes - starting_num_nodes > self.node_limit {
                0
            } else {
                self.node_limit - (ending_num_nodes - starting_num_nodes)
            };
            println!(
                "Number of nodes added: {}",
                ending_num_nodes - starting_num_nodes
            );

            let time_taken = start_time.elapsed().as_secs();
            self.n_sec = if time_taken > self.n_sec {
                0
            } else {
                self.n_sec - time_taken
            };

            //println!("Number of applied: {}", num_applied);
        }

        Ok(())
    }

    /// Apply a rule with a pair of matches for its src patterns.
    /// Returns the number of successful applications
    fn apply_match_pair(
        &self,
        rule: &(Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, Pattern<Mdl>, bool),
        match_1: &SearchMatches,
        match_2: &SearchMatches,
        map_1: &MapToCanonical,
        map_2: &MapToCanonical,
        runner: &mut Runner<Mdl, TensorAnalysis, ()>,
    ) -> usize {
        let mut num_applied = 0;
        for subst_1 in &match_1.substs {
            for subst_2 in &match_2.substs {
                // De-canonicalize the substitutions
                let subst_1_dec = decanonicalize(subst_1, &map_1.var_map);
                let subst_2_dec = decanonicalize(subst_2, &map_2.var_map);
                // Check if two substitutions have matching shared variables
                if compatible(&subst_1_dec, &subst_2_dec, &map_1.var_map) {
                    // If so, merge two substitutions
                    let merged_subst = merge_subst(subst_1_dec, subst_2_dec, &map_1.var_map);
                    // Check if any source pattern contains blacklisted nodes
                    if self.filter_after {
                        let condition_tmp = contains_blacklist(
                            rule.0.ast.as_ref(),
                            &mut runner.egraph,
                            &merged_subst,
                        )
                        .0 || contains_blacklist(
                            rule.1.ast.as_ref(),
                            &mut runner.egraph,
                            &merged_subst,
                        )
                        .0;
                        if condition_tmp {
                            continue;
                        }
                    }

                    // check_pat on both dst patterns
                    let (valid_1, _, existing_1) = check_pat(
                        rule.2.ast.as_ref(),
                        &mut runner.egraph,
                        &merged_subst,
                        /*get_exist_nodes=*/ self.filter_after,
                    );
                    let (valid_2, _, existing_2) = check_pat(
                        rule.3.ast.as_ref(),
                        &mut runner.egraph,
                        &merged_subst,
                        /*get_exist_nodes=*/ self.filter_after,
                    );
                    if valid_1 && valid_2 {
                        let cycle_check_passed = if self.no_cycle {
                            if self.filter_after {
                                // Do pre-filtering using the pre-collected descendents info
                                self.check_cycle_partial(
                                    &runner.egraph,
                                    &merged_subst,
                                    &map_1.var_map,
                                    &map_2.var_map,
                                    match_1.eclass,
                                    match_2.eclass,
                                )
                            } else {
                                // Check cycle by make a pass in egraph
                                let mut descendents: HashMap<Id, HashSet<Id>> = Default::default();
                                check_cycle(
                                    &runner.egraph,
                                    &merged_subst,
                                    &map_1.var_map,
                                    &map_2.var_map,
                                    match_1.eclass,
                                    match_2.eclass,
                                    &mut descendents,
                                )
                            }
                        } else {
                            true
                        };
                        if cycle_check_passed {
                            // apply dst patterns, union
                            let id_1 =
                                rule.2
                                    .apply_one(&mut runner.egraph, match_1.eclass, &merged_subst)
                                    [0];

                            let id_2 =
                                rule.3
                                    .apply_one(&mut runner.egraph, match_2.eclass, &merged_subst)
                                    [0];

                            // Add the newly added nodes to the ordering list
                            if self.filter_after {
                                //let n_before = runner.egraph.analysis.newly_added.len();

                                let existing_1 = existing_1.unwrap();
                                let existing_2 = existing_2.unwrap();
                                let (nodes_in_1, _) = add_newly_added(
                                    rule.2.ast.as_ref(),
                                    &mut runner.egraph,
                                    &merged_subst,
                                    &existing_1,
                                );
                                let existing_2_updated: HashSet<Mdl> = existing_2
                                    .iter()
                                    .chain(nodes_in_1.iter())
                                    .map(|node| node.clone())
                                    .collect();
                                add_newly_added(
                                    rule.3.ast.as_ref(),
                                    &mut runner.egraph,
                                    &merged_subst,
                                    &existing_2_updated,
                                );

                                /*let n_after = runner.egraph.analysis.newly_added.len();
                                if n_after > n_before {
                                    num_applied += 1;
                                }*/
                            }

                            runner.egraph.union(id_1, match_1.eclass);
                            runner.egraph.union(id_2, match_2.eclass);
                        }
                    }
                }
            }
        }
        num_applied
    }

    /// Returns true if there will not be a cycle introduced by applying this rule.
    ///
    /// Checking based on descendents collected at beginning of run_one(). If any input node
    /// contains descendents that is any of the matched output class, then there can be a cycle
    /// created.
    ///
    /// # Parameters
    ///
    /// - `egraph`: egraph of interest
    /// - `input_subst`: substitution containing the input variables
    /// - `var_map_1`: keys of this map contains all the input variables in source pattern 1
    /// - `var_map_2`: keys of this map contains all the input variables in source pattern 2
    /// - `out_class_1`: Id of the matched eclass of the output of source pattern 1
    /// - `out_class_2`: Id of the matched eclass of the output of source pattern 2
    fn check_cycle_partial(
        &self,
        egraph: &EGraph<Mdl, TensorAnalysis>,
        input_subst: &Subst,
        var_map_1: &HashMap<egg::Var, egg::Var>,
        var_map_2: &HashMap<egg::Var, egg::Var>,
        out_class_1: Id,
        out_class_2: Id,
    ) -> bool {
        // Get all input eclass IDs
        let input_ids: HashSet<Id> = var_map_1
            .iter()
            .chain(var_map_2.iter())
            .map(|(var, _)| *input_subst.get(*var).unwrap())
            .collect();
        // Check descendents of the input eclasses
        return input_ids.iter().all(|id| {
            let descendents = self.descendents.as_ref().unwrap();
            let descendents_input = descendents.get(id).unwrap();
            !descendents_input.contains(&out_class_1) && !descendents_input.contains(&out_class_2)
        });
    }
}

/// Do post-processing to remove cycles in the egraph, by adding nodes to blacklist.
pub fn remove_cycle_by_order(runner: &mut Runner<Mdl, TensorAnalysis, ()>) {
    // Update blacklist (canonicalize Ids with egraph.find())
    update_blacklist(&mut runner.egraph);

    // Update newly_added (canonicalize Ids with egraph.find()) and construct hashmap
    // for newly_added
    let updated: Vec<Mdl> = runner
        .egraph
        .analysis
        .newly_added
        .iter()
        .map(|node| node.clone().map_children(|id| runner.egraph.find(id)))
        .collect();
    let mut added_node_to_order = HashMap::<Mdl, usize>::new();
    for (i, node) in updated.iter().enumerate() {
        added_node_to_order.entry(node.clone()).or_insert(i);
    }
    // Remove cycles by adding nodes to blacklist
    remove_cycles_by_add_to_blacklist(&mut runner.egraph, &added_node_to_order, runner.roots[0]);
}

/// Add newly added nodes in this pattern to the list of newly added nodes, for use in cycle
/// filtering
///
/// The newly added nodes are stored in the graph level metadata in egraph.analysis
///
/// # Parameters
///
/// - `pat`: the AST representation of the pattern. See egg::Pattern for more info
/// - `egraph`: E-graph of interest
/// - `subst`: mapping variable to eclass ID. See egg::Subst for more info.
/// - `existing_nodes`: the set of nodes within this pattern that already exists before this
///         pattern is applied
///
/// # Returns
///
/// A tuple of (HashSet<Mdl>, Id) where
///
/// - HashSet<Mdl>: The set of all nodes in this pattern
/// - Id: the Id into egraph of the matched root of this pattern
fn add_newly_added(
    pat: &[ENodeOrVar<Mdl>],
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    subst: &Subst,
    existing_nodes: &HashSet<Mdl>,
) -> (HashSet<Mdl>, Id) {
    match pat.last().unwrap() {
        ENodeOrVar::Var(w) => (HashSet::<Mdl>::new(), subst[*w]),
        ENodeOrVar::ENode(e) => {
            let children = e.children();
            let results: Vec<(HashSet<Mdl>, Id)> = children
                .iter()
                .map(|child| {
                    add_newly_added(
                        &pat[..usize::from(*child) + 1],
                        egraph,
                        subst,
                        existing_nodes,
                    )
                })
                .collect();

            let mut new_e = e.clone();
            let new_e_ch = new_e.children_mut();
            for (i, res) in results.iter().enumerate() {
                new_e_ch[i] = res.1;
            }

            let mut nodes_in_pat = HashSet::<Mdl>::new();
            for res in results.iter() {
                for node in res.0.iter() {
                    nodes_in_pat.insert(node.clone());
                }
            }
            nodes_in_pat.insert(new_e.clone());

            // Add to order list
            if !existing_nodes.contains(&new_e) {
                egraph.analysis.newly_added.push(new_e.clone());
            }

            (nodes_in_pat, egraph.lookup(new_e).unwrap())
        }
    }
}

/// Remove cycles in EGraph by adding nodes to blacklist
///
/// This function works by:
///     - Make a pass over egraph to get a set of cycles
///     - For each cycle, pick the node that got added latest and add it to blacklist
///     - Repeat until no cycles are left
fn remove_cycles_by_add_to_blacklist(
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    added_node_to_order: &HashMap<Mdl, usize>,
    root: Id,
) {
    loop {
        let mut visited = HashSet::<Id>::new();
        let mut cycles = Vec::<Vec<Mdl>>::new();

        get_cycles(egraph, root, &mut visited, &mut cycles);

        if cycles.len() == 0 {
            break;
        }
        for cycle in cycles.iter() {
            resolve_cycle(egraph, cycle, added_node_to_order);
        }
    }
}

/// Resolve cycle by adding node to blacklist
///
/// # Parameters
///
/// - `egraph`: E-graph of interest
/// - `cycle`: list of nodes within the cycle
/// - `added_node_to_order`: HashMap, map from node to the order that it was added into the egraph
fn resolve_cycle(
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    cycle: &[Mdl],
    added_node_to_order: &HashMap<Mdl, usize>,
) {
    // Check if any node in cycle is already in blacklist
    let already_solved = cycle
        .iter()
        .any(|node| egraph.analysis.blacklist_nodes.contains(node));
    if !already_solved {
        assert!(cycle.len() > 0);
        let (ord, n) = cycle
            .iter()
            .map(|node| {
                let order = added_node_to_order
                    .get(node)
                    .map_or(-1, |index| *index as i32);
                (order, node.clone())
            })
            .max_by_key(|(o, _)| *o)
            .unwrap();
        assert!(ord >= 0);
        egraph.analysis.blacklist_nodes.insert(n.clone());
    }
}

/// Traverse the EGraph and get a set of cycles (reachable from root)
///
/// # Parameters
///
/// - `egraph`: E-graph of interest
/// - `root`: Id of root eclass
/// - `visited`: HashSet storing the eclasses visited.
/// - `cycles`: list of cycles. Each cycle is a list of nodes. A cycle of 1->2->4->3->1 will be
///         stored as [1,2,4,3]
fn get_cycles(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    visited: &mut HashSet<Id>,
    cycles: &mut Vec<Vec<Mdl>>,
) {
    // Get a map from Id to the eclass objects, since egg doesn't provide accessing eclass from Id
    let id_to_class: HashMap<Id, &EClass<Mdl, TensorData>> =
        egraph.classes().map(|class| (class.id, class)).collect();

    get_cycles_rec(
        egraph,
        root,
        /*path_to_here=*/ Vec::<(Id, Mdl)>::new(),
        &id_to_class,
        visited,
        cycles,
    );
}

/// Traverse the EGraph in DFS order, update visited and cycles on the fly
///
/// # Parameters
///
/// - `egraph`: E-graph of interest
/// - `eclass`: The current eclass that we are visiting
/// - `path_to_here`: A path from root to this eclass
/// - `id_to_class`: Map from eclass ID to the eclass objects
/// - `visited`: HashSet storing the eclasses visited.
/// - `cycles`: list of cycles. Each cycle is a list of nodes. A cycle of 1->2->4->3->1 will be
///         stored as [1,2,4,3]
fn get_cycles_rec(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    eclass: Id,
    path_to_here: Vec<(Id, Mdl)>,
    id_to_class: &HashMap<Id, &EClass<Mdl, TensorData>>,
    visited: &mut HashSet<Id>,
    cycles: &mut Vec<Vec<Mdl>>,
) {
    assert!(!visited.contains(&eclass));
    visited.insert(eclass);

    let class = id_to_class.get(&eclass).unwrap();
    for node in class.iter() {
        if egraph.analysis.blacklist_nodes.contains(node) {
            continue;
        }
        for child in node.children().iter() {
            if !visited.contains(child) {
                // Haven't visited, so visit
                // TODO: Instead of cloning, you can use the same list,
                // and pop the added node after the call to get_cycles_rec()
                let mut path_to_child = path_to_here.clone();
                path_to_child.push((eclass, node.clone()));
                get_cycles_rec(egraph, *child, path_to_child, id_to_class, visited, cycles);
            } else {
                // Visited before. Check if in path_to_here
                if let Some((i, _)) = path_to_here
                    .iter()
                    .enumerate()
                    .find(|(i, (cid, n))| *cid == *child)
                {
                    let mut cycle: Vec<Mdl> =
                        path_to_here[i..].iter().map(|(cid, n)| n.clone()).collect();
                    cycle.push(node.clone());
                    cycles.push(cycle);
                }
            }
        }
    }
}

/// Update the blacklist_nodes in egraph.analysis with the new canonical EClass IDs
fn update_blacklist(egraph: &mut EGraph<Mdl, TensorAnalysis>) {
    egraph.analysis.blacklist_nodes = egraph
        .analysis
        .blacklist_nodes
        .iter()
        .map(|node| node.clone().map_children(|id| egraph.find(id)))
        .collect();
}

/// Returns true if there will not be a cycle introduced by applying this rule.
///
/// Checking based on freshly collected descendents info. If any input node
/// contains descendents that is any of the matched output class, then there can be a cycle
/// created.
///
/// # Parameters
///
/// - `egraph`: egraph of interest
/// - `input_subst`: substitution containing the input variables
/// - `var_map_1`: keys of this map contains all the input variables in source pattern 1
/// - `var_map_2`: keys of this map contains all the input variables in source pattern 2
/// - `out_class_1`: Id of the matched eclass of the output of source pattern 1
/// - `out_class_2`: Id of the matched eclass of the output of source pattern 2
/// - `descendents`: Map from each eclass ID to its set of descendents. Constructed here.
fn check_cycle(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    input_subst: &Subst,
    var_map_1: &HashMap<egg::Var, egg::Var>,
    var_map_2: &HashMap<egg::Var, egg::Var>,
    out_class_1: Id,
    out_class_2: Id,
    descendents: &mut HashMap<Id, HashSet<Id>>,
) -> bool {
    // Get all input eclass IDs
    let input_ids: HashSet<Id> = var_map_1
        .iter()
        .chain(var_map_2.iter())
        .map(|(var, _)| *input_subst.get(*var).unwrap())
        .collect();
    // Get a map from eclass IDs to eclass
    let id_to_class: HashMap<Id, &EClass<Mdl, TensorData>> =
        egraph.classes().map(|class| (class.id, class)).collect();
    // Check descendents of the input eclasses
    return input_ids.iter().all(|id| {
        get_descendents(
            egraph,
            *id,
            &id_to_class,
            /*check_blacklist=*/ false,
            descendents,
        );
        let descendents_input = descendents.get(id).unwrap();
        !descendents_input.contains(&out_class_1) && !descendents_input.contains(&out_class_2)
    });
}

/// Get a map of all eclass (reachable from root) to their descendent eclasses
///
/// # Parameters
///
/// - `egraph`: egraph of interest
/// - `root`: root class of egraph
/// - `check_blacklist: if true, check if a node is in blacklist, dependencies introduced
///         by nodes in the blacklist will not be counted into descendents.
///
/// # Returns
///
///   The hashmap from Eclass ID to the set of its descendent eclasses
fn compute_all_descendents(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    check_blacklist: bool,
) -> HashMap<Id, HashSet<Id>> {
    // Get a map from eclass IDs to eclass
    let id_to_class: HashMap<Id, &EClass<Mdl, TensorData>> =
        egraph.classes().map(|class| (class.id, class)).collect();

    let mut descendents: HashMap<Id, HashSet<Id>> = Default::default();
    get_descendents(
        egraph,
        root,
        &id_to_class,
        check_blacklist,
        &mut descendents,
    );
    descendents
}

/// Get the descendent of eclass. The result will be in descendents
///
/// # Parameters
///
/// - `egraph`: egraph of interest
/// - `eclass`: eclass to get descendents for. This will also get descendents for all its
///         descendents, since this function is doing it recursively
/// - `id_to_class`: map from eclass id to eclass objects
/// - `check_blacklist`: if true, dependencies introduced by blacklisted nodes will not be counted
/// - `descendents`: Map from each eclass ID to its set of descendents. Constructed here.
fn get_descendents(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    eclass: Id,
    id_to_class: &HashMap<Id, &EClass<Mdl, TensorData>>,
    check_blacklist: bool,
    descendents: &mut HashMap<Id, HashSet<Id>>,
) {
    match descendents.get(&eclass) {
        Some(desc) => (),
        None => {
            let class = id_to_class.get(&eclass).unwrap();
            let mut result_desc = HashSet::<Id>::new();
            for node in class.iter() {
                if check_blacklist {
                    if egraph.analysis.blacklist_nodes.contains(node) {
                        continue;
                    }
                }
                for child in node.children().iter() {
                    get_descendents(egraph, *child, id_to_class, check_blacklist, descendents);
                    let child_desc = descendents.get(child).unwrap();
                    result_desc = result_desc.union(child_desc).map(|&id| id).collect();
                    result_desc.insert(*child);
                }
            }
            descendents.insert(eclass, result_desc);
        }
    }
}

/// Canonicalize a pattern
///
/// This function constructs a canonicalized pattern for a given pattern. It sequentially
/// replaces each unique variable with ?i_0, ?i_1 ...
///
/// # Parameters
///
/// - `pat`: the pattern to canonicalize. See egg::Pattern for more info
///
/// # Returns
///
/// - Pattern<Mdl>: the canonicalized pattern
/// - HashMap<egg::Var, egg::Var>: Mapping from variable in the original pattern to
///     variable in the canonical pattern.
fn canonicalize(pat: &Pattern<Mdl>) -> (Pattern<Mdl>, HashMap<egg::Var, egg::Var>) {
    let mut var_map = HashMap::<egg::Var, egg::Var>::new();
    let mut count = 0;
    let substituted: Vec<_> = pat
        .ast
        .as_ref()
        .iter()
        .cloned()
        .map(|x| match x {
            ENodeOrVar::ENode(_) => x,
            ENodeOrVar::Var(v) => {
                let var = var_map.entry(v).or_insert_with(|| {
                    let name = format!("?i_{}", count);
                    count += 1;
                    name.parse().unwrap()
                });
                ENodeOrVar::Var(*var)
            }
        })
        .collect();

    let ast = RecExpr::from(substituted);
    let canonical_pat = Pattern::<Mdl>::from(ast);
    (canonical_pat, var_map)
}

/// Merge two substitutions
///
/// This function merges two substitutions. The merged one contains substitutions (egg::Var -> Id)
/// from both input substitutions.
///
/// # Parameters
///
/// - `subst_1`: substitution to be merged
/// - `subst_2`: substitution to be merged
/// - `var_map_1`: the keys of this map should be all Var in subst_1. It is just for providing the Vars
fn merge_subst(
    subst_1: Subst,
    mut subst_2: Subst,
    var_map_1: &HashMap<egg::Var, egg::Var>,
) -> Subst {
    for (var, _) in var_map_1.iter() {
        let id_1 = subst_1.get(*var).unwrap();
        subst_2.insert(*var, *id_1);
    }
    subst_2
}

/// Decanonicalize a substitution
///
/// Create a decanonicalized substitution by replacing the variables in the canonical substitution
/// with the original variables
///
/// # Parameters
///
/// - `subst`: The substitution using the canonicalized variables
/// - `var_map`: Mapping from variable in the original pattern to variable in the canonical pattern.
fn decanonicalize(subst: &Subst, var_map: &HashMap<egg::Var, egg::Var>) -> Subst {
    let mut new_subst: Subst = Default::default();
    for (orig_var, canonical_var) in var_map.iter() {
        new_subst.insert(*orig_var, *subst.get(*canonical_var).unwrap());
    }
    new_subst
}

/// Check if the shared variables between two substitutions point to the same eclass Id.
///
/// # Parameters
///
/// - `subst_1`: substitution to be checked
/// - `subst_2`: substitution to be checked
/// - `var_map_1`: the keys of this map should be all Var in subst_1. It is just for providing the Vars
///
/// # Returns
///
///   Return true if all corresponding shared variables between two substitutions point to the
///   same eclass Id's.
fn compatible(subst_1: &Subst, subst_2: &Subst, var_map_1: &HashMap<egg::Var, egg::Var>) -> bool {
    for (var, _) in var_map_1.iter() {
        let id_1 = subst_1.get(*var).unwrap();
        if let Some(id_2) = subst_2.get(*var) {
            if id_1 != id_2 {
                return false;
            }
        }
    }
    return true;
}
