use crate::{
    ffi_utils::*,
    input::{ffi, CppGraphConverter},
    model::*,
    rewrites::*,
};
use egg::*;
use argmin::{core::*, solver::simulatedannealing::*};
use argmin_observer_slog::SlogLogger;
// use cxx::UniquePtr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Wrapper class for egg's cost function
pub struct TensorCost<'a> {
    pub egraph: &'a EGraph<Mdl, TensorAnalysis>,
    pub cost_model: &'a CostModel,
}

impl egg::CostFunction<Mdl> for TensorCost<'_> {
    type Cost = f32;
    /// Getting total cost for the subtree rooted at enode. See egg::CostFunction
    /// trait for more information on interface.
    fn cost<C: FnMut(Id) -> Self::Cost>(&mut self, enode: &Mdl, mut costs: C) -> Self::Cost {
        let (self_cost, _) = self.cost_model.get_self_cost(self.egraph, enode);
        enode.fold(self_cost, |sum, id| sum + costs(id))
    }
}

/// Class for our cost model
pub struct CostModel {}

impl CostModel {
    pub fn new() -> Self {
        Self {}
    }

    /// Gets cost for the enode itself.
    ///
    /// This function gets the cost by calling TASO's get_or_create_{some_op}()
    /// functions with the tensor information stored in metadata. TASO side stores
    /// hashmaps for OpBase objects. So here TASO side will simply lookup previously
    /// created ops (with previously measured runtime).
    ///
    /// # Parameters
    ///
    /// - `egraph`: E-graph of interest
    /// - `enode`: enode to get cost for
    ///
    /// # Returns
    ///
    /// Cost for this enode.
    pub fn get_self_cost(&self, egraph: &EGraph<Mdl, TensorAnalysis>, enode: &Mdl) -> (f32, f32) {
        match enode {
            // NO REWRITES APPLY TO THESE SO THEY CAN HAVE ARBITRARY COST
            Mdl::Num(_)
            | Mdl::Var(_)
            | Mdl::Input(_)
            | Mdl::Vec(_)
            | Mdl::BlackBox(_)
            | Mdl::Index(_)
            | Mdl::ReturnOp(_)
            // InferReshape at most adds a ReshapeOp without changing the underlying data, which should be free
            // TODO: Gather all the "zero cost ops" assumptions together (the other bits currently being in EqualitySaturation.cpp)
            | Mdl::InferReshape(_) => (0.0, 0.0),
            x => {
              let cs = create_stablehlo_op(egraph, x, ffi::get_cost);
              (cs[0] as f32, cs[1] as f32)
            }
        }
    }
}

// extraction
pub type Candidate = HashMap<Id, usize>;

enum CostModels {
    Analytical,
    Measured,
}

pub enum OptimizationMethod {
    SimulatedAnnealing,
}

pub struct GlobalExtractor<'a> {
    pub egraph: &'a EGraph<Mdl, TensorAnalysis>,
    cost_model: &'a CostModel,
    root: Id,
}

impl<'a> GlobalExtractor<'a> {
    pub fn new(
        egraph: &'a EGraph<Mdl, TensorAnalysis>,
        cost_model: &'a CostModel,
        root: Id,
    ) -> Self {
        Self {
            egraph,
            cost_model,
            root,
        }
    }

    pub fn cost(&self, candidate: &Candidate) -> f64 {
        let (rec_expr, to_egraph) = candidate_to_recexpr(&candidate, &self.egraph, self.root);
        let cost = CppGraphConverter::get_end_to_end_cost(self.egraph, &to_egraph, &rec_expr);
        cost as f64 // Return the computed cost
    }
}


/// Starting at the root, convert the Candidate into a RecExpr, and also return a
/// mapping from the RecExpr Id to the EGraph Id
pub fn candidate_to_recexpr(
    candidate: &Candidate,
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
) -> (RecExpr<Mdl>, HashMap<Id, Id>) {
    let mut node_picked: HashMap<Id, Mdl> = HashMap::new();
    for eclass in egraph.classes() {
        let eclass_id = egraph.find(eclass.id);
        let enodes = &egraph[eclass_id].nodes;
        // Check if the candidate mapping is valid; if not, use 0 as default.
        let node_idx = match candidate.get(&eclass_id) {
            Some(&i) if i < enodes.len() => i,
            _ => 0,
        };
        let enode = enodes[node_idx].clone();
        node_picked.insert(eclass_id.clone(), enode);
    }

    let mut egraph_to_recexpr: HashMap<Id, Id> = HashMap::new();
    let mut recexpr_to_egraph: HashMap<Id, Id> = HashMap::new();
    let mut expr: RecExpr<Mdl> = RecExpr::default();

    construct_best_rec(
        &mut node_picked,
        root,
        &mut egraph_to_recexpr,
        &mut recexpr_to_egraph,
        &egraph,
        &mut expr,
    );

    (expr, recexpr_to_egraph)
}


pub fn extract_by_optimization(extractor: GlobalExtractor, method: OptimizationMethod) -> Candidate {
    match method {
        OptimizationMethod::SimulatedAnnealing => {
            // compute m_id_map and id_m_map like in prep_ilp_data
            let m_id_map: Vec<Id> = extractor.egraph.classes()
                .map(|c| extractor.egraph.find(c.id))
                .collect();
            let id_m_map: HashMap<Id, usize> = m_id_map
                .iter()
                .enumerate()
                .map(|(i, id)| (*id, i))
                .collect();

            // build g_i mapping by iterating through all nodes
            let mut g_i = Vec::new();
            let mut nodes = Vec::new();
            for class in extractor.egraph.classes() {
                let m = id_m_map[&extractor.egraph.find(class.id)];
                for node in class.iter() {
                    nodes.push(node.clone());
                    g_i.push(m);
                }
            }
            let mut e_m: Vec<Vec<usize>> = vec![Vec::new(); m_id_map.len()];
            for (i, &m) in g_i.iter().enumerate() {
                e_m[m].push(i);
            }

            let node_to_i: HashMap<_, _> = nodes
                .iter()
                .enumerate()
                .map(|(i, node)| (node.clone(), i))
                .collect();

            let tnsr_cost = TensorCost {
                egraph: extractor.egraph,
                cost_model: extractor.cost_model,
            };
            let mut greedy_extractor = Extractor::new(extractor.egraph, tnsr_cost);
            let _ = greedy_extractor.find_best(extractor.root);

            let (i_list, m_list) = get_init_solution(
                extractor.egraph,
                extractor.root,
                &greedy_extractor.costs,
                &g_i,
                &node_to_i
            );

            let mut greedy_candidate = HashMap::new();
            for (i, m) in i_list.iter().zip(m_list.iter()) {
                let eclass_id = m_id_map[*m];
                let local_idx = e_m[*m].iter().position(|x| x == i).expect("Node not found in eclass");
                greedy_candidate.insert(eclass_id, *i);
            }

            let sa = SimulatedAnnealing::new(0.1)
                .unwrap()
                .with_temp_func(SATempFunc::Boltzmann)
                .with_stall_best(1000)
                .with_stall_accepted(1000)
                .with_reannealing_fixed(1000)
                .with_reannealing_accepted(400)
                .with_reannealing_best(400);

            let solver = Executor::new(extractor, sa)
                .configure(|state| state.param(greedy_candidate).max_iters(1000))
                .add_observer(SlogLogger::term(), observers::ObserverMode::Every(10));
                
            solver.run().unwrap().state.param.unwrap()
        }
    }
}

/// Prepare the data for formulation ILP
///
/// # Returns
///
/// - `m_id_map`: list of EClass Id's each index m refers to
/// - `e_m`: each entry is the list of nodes i within eclass m
/// - `h_i`: each entry is the list of children EClass indices for node i
/// - `cost_i`: self cost for each node i
/// - `fus_i`: fusability of each node i
/// - `g_i`: which EClass index does node i belong to
/// - `root_m`: EClass index of the root eclass
/// - `i_to_nodes: Vector of enodes, ordered by index i
/// - `blacklist_i: Vector of indices of nodes that are blacklisted
pub fn prep_ilp_data(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    cost_model: &CostModel,
) -> (
    Vec<Id>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<f32>,
    Vec<f32>,
    Vec<bool>,
    Vec<usize>,
    usize,
    Vec<Mdl>,
    Vec<usize>,
) {
    let m_id_map: Vec<Id> = egraph.classes().map(|c| egraph.find(c.id)).collect();
    assert!(m_id_map.len() == egraph.number_of_classes());
    let id_m_map: HashMap<Id, usize> = m_id_map
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let num_classes = egraph.number_of_classes();
    let num_nodes = egraph.total_size();
    let mut i_to_nodes: Vec<Mdl> = Vec::with_capacity(num_nodes);
    let mut e_m: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
    let mut h_i: Vec<Vec<usize>> = Vec::with_capacity(num_nodes);
    let mut cost_i: Vec<f32> = Vec::with_capacity(num_nodes);
    let mut fus_cost_i: Vec<f32> = Vec::with_capacity(num_nodes);
    let mut g_i: Vec<usize> = Vec::with_capacity(num_nodes);
    let mut fus_i: Vec<bool> = Vec::with_capacity(num_nodes);
    let mut blacklist_i: Vec<usize> = Vec::new();

    let mut i = 0;
    for class in egraph.classes() {
        let m = *id_m_map.get(&egraph.find(class.id)).unwrap();
        for node in class.iter() {
            i_to_nodes.push(node.clone());
            if egraph.analysis.blacklist_nodes.contains(node) {
                blacklist_i.push(i);
            }
            e_m[m].push(i);
            h_i.push(
                node.children()
                    .iter()
                    .map(|id| *id_m_map.get(&egraph.find(*id)).unwrap())
                    .collect(),
            );
            let (cost, fus_cost) = cost_model.get_self_cost(egraph, node);
            cost_i.push(cost);
            fus_cost_i.push(fus_cost);
            g_i.push(m);
            use crate::model::Mdl::*;
            fus_i.push(match node {
                Input(..) => true,
                CompareOp(..) => true,
                BroadcastInDimOp(..) => true,
                ConvertOp(..) => true,
                ReshapeOp(..) => true,
                GatherOp(..) => true,
                ConcatenateOp(..) => true,
                SliceOp(..) => true,
                TransposeOp(..) => true,
                MulOp(..) => true,
                AddOp(..) => true,
                DivOp(..) => true,
                SubtractOp(..) => true,
                MinOp(..) => true,
                MinOp(..) => true,
                NegOp(..) => true,
                TanhOp(..) => true,
                ExpOp(..) => true,
                IotaOp(..) => true,
                DynamicUpdateSliceOp(..) => true,
                DynamicSliceOp(..) => true,
                ScatterOp(..) => true,
                ReturnOp(..) => true,
                BlackBox(..) => false,
                _ => false,
            });
            i += 1;
        }
    }

    let root_m = *id_m_map.get(&egraph.find(root)).unwrap();

    (
        m_id_map,
        e_m,
        h_i,
        cost_i,
        fus_cost_i,
        fus_i,
        g_i,
        root_m,
        i_to_nodes,
        blacklist_i,
    )
}

/// Struct for storing the solved results from ILP
#[derive(Debug, Serialize, Deserialize)]
pub struct SolvedResults {
    /// The solved values for the variables associated with each node
    pub solved_x: Vec<i32>,
    /// The minimum total cost found
    pub cost: f32,
    /// Time for solver
    pub time: f32,
}

/// Construct the RecExpr of the optimized graph extracted
///
/// This function does the construction recursively with memoization. Call it with eclass=root
/// will construct the whole extracted graph
///
/// # Parameters
///
/// - `node_picked`: hashmap storing which node is picked for each EClass ID
/// - `eclass`: The EClass ID that we aim to construct as root
/// - `added_memo`: Map from EClass ID to RecExpr ID. Storing the eclasses that were already added
/// - `to_egraph`: Map from RecExpr ID to EClass ID
/// - `egraph`: E-graph of interest
/// - `expr`: the RecExpr storing the optimized graph, it is constructed within this function
///
/// # Returns
///
/// - The ID (index) in the output RecExpr for the eclass passed in as argument
pub fn construct_best_rec(
    node_picked: &HashMap<Id, Mdl>,
    eclass: Id,
    added_memo: &mut HashMap<Id, Id>,
    to_egraph: &mut HashMap<Id, Id>,
    egraph: &EGraph<Mdl, TensorAnalysis>,
    expr: &mut RecExpr<Mdl>,
) -> Id {
    let id = egraph.find(eclass);

    match added_memo.get(&id) {
        Some(id_expr) => *id_expr,
        None => {
            let node = node_picked.get(&id).unwrap().clone().map_children(|child| {
                construct_best_rec(node_picked, child, added_memo, to_egraph, egraph, expr)
            });
            let id_expr = expr.add(node);
            assert!(added_memo.insert(id, id_expr).is_none());
            assert!(to_egraph.insert(id_expr, id).is_none());
            id_expr
        }
    }
}

/// Get the initial solution for ILP using the greedy extraction
///
/// This function does the construction recursively with memoization. Call it with eclass=root
/// will construct the whole extracted graph
///
/// # Parameters
///
/// - `egraph`: egraph of interest
/// - `root`: root eclass
/// - `costs`: Map from eclass ID to the node with the lowest subtree cost (cost, node).
///         Constructed by egg's Extractor
/// - `g_i`: which EClass index does node i belong to
/// - `nodes_to_i`: map from node to index i
///
/// # Returns
///
/// A tuple of (i_list, m_list), where
///
/// - `i_list`: list of i picked by greedy extraction
/// - `m_list`: list of eclass index m that i_list belongs to
pub fn get_init_solution(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    costs: &HashMap<Id, (f32, Mdl)>,
    g_i: &[usize],
    nodes_to_i: &HashMap<Mdl, usize>,
) -> (Vec<usize>, Vec<usize>) {
    let mut nodes: Vec<Mdl> = Vec::new();
    // added_memo maps eclass id to id in expr
    let mut added_memo: HashSet<Id> = Default::default();
    get_init_rec(egraph, root, &mut added_memo, costs, &mut nodes);

    let i_list: Vec<usize> = nodes
        .iter()
        .map(|node| *nodes_to_i.get(node).unwrap())
        .collect();
    let m_list: Vec<usize> = i_list.iter().map(|i| g_i[*i]).collect();

    (i_list, m_list)
}

/// Recursively get the initial solution for ILP using the greedy extraction, results stored in nodes
///
/// # Parameters
///
/// - `egraph`: egraph of interest
/// - `eclass`: get solution rooted from here
/// - `added_memo`: Stores the set of eclasses that has already been processed
/// - `costs`: Map from eclass ID to the node with the lowest subtree cost (cost, node).
///         Constructed by egg's Extractor
/// - `nodes`: List of nodes picked by greedy extraction. Constructed within this function
fn get_init_rec(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    eclass: Id,
    added_memo: &mut HashSet<Id>,
    costs: &HashMap<Id, (f32, Mdl)>,
    nodes: &mut Vec<Mdl>,
) {
    let id = egraph.find(eclass);

    if !added_memo.contains(&id) {
        let (_, best_node) = match costs.get(&id) {
            Some(result) => result.clone(),
            None => panic!("Failed to extract from eclass {}", id),
        };
        best_node.for_each(|child| get_init_rec(egraph, child, added_memo, costs, nodes));
        nodes.push(best_node);
        added_memo.insert(id);
    }
}

pub fn compute_greedy_candidate(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    cost_model: &CostModel,
) -> Candidate {
    let mut candidate = Candidate::new();
    for eclass in egraph.classes() {
        let id = egraph.find(eclass.id);
        let enodes = &egraph[id].nodes;
        let (min_idx, _) = enodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let cost_a = cost_model.get_self_cost(egraph, a).0;
                let cost_b = cost_model.get_self_cost(egraph, b).0;
                cost_a.partial_cmp(&cost_b).unwrap()
            })
            .unwrap();
        candidate.insert(id, min_idx);
    }
    candidate
}
