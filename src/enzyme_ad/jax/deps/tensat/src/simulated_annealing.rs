use crate::{
  optimize::*,
  model::*,
};
use argmin::{core::*, solver::simulatedannealing::*};
use rand::prelude::*;
use rand::seq::index::sample;
use std::collections::{HashMap, HashSet};
use egg::*;

impl<'a> GlobalExtractor<'a> {
    /// Get list of eclass IDs that the current candidate "uses".
    /// This helps ensure that the annealing function chooses one
    /// of these eclasses to perturb, meaning it actually does
    /// something (otherwise, at any given point, the vast majority
    /// of eclasses aren't being used, so perturbing them does
    /// nothing until potentially many iterations later, if that).
    fn current_eclasses(
        &self,
        candidate: &Candidate
    ) -> Vec<Id> {
        let mut result: HashSet<Id> = HashSet::new();
        self.current_eclasses_rec(&candidate, self.root, &mut result);
        result.iter().map(|id| *id).collect()
    }

    fn current_eclasses_rec(
        &self,
        candidate: &Candidate,
        root: Id,
        result: &mut HashSet<Id>, // eclass ids that have been visited + recursed into
    ) {
        match result.contains(&root) {
            true => (),
            false => {
                let enode_idx = candidate.get(&root).unwrap();
                let enode = &self.egraph[root].nodes[*enode_idx];
                for c in enode.children() {
                    let id = self.egraph.find(*c);
                    self.current_eclasses_rec(&candidate, id, result);
                }
                result.insert(root);
            }
        }
    }
}

impl<'a> Anneal for GlobalExtractor<'a> {
    type Param = Candidate;
    type Output = Candidate;
    type Float = f64;

    fn anneal(
        &self,
        param: &Candidate,
        temp: f64,
    ) -> Result<Candidate, Error> {
        let mut rng = rand::rng();
        // Start from the current candidate.
        let mut new_candidate = param.clone();
        let eclasses = self.current_eclasses(param);

        // only consider the non-trivial classes
        let nontrivial: Vec<Id> = eclasses.iter()
            .map(|id| *id)
            .filter(|id| self.egraph[*id].len() > 1)
            .collect();

        let n = nontrivial.len();
        // at higher temperatures, we make more changes.
        // use ceil so that even at low-but-nonzero temperature we change at least one
        let mut num_to_change = temp.ceil() as usize;
        // if the computed number is 0, bump it to 1?
        if num_to_change == 0 {
            num_to_change = 1;
        }
        num_to_change = num_to_change.min(n);

        // choose 'num_to_change' eclasses (without replacement) from the nontrivial ones
        let indices_to_change = sample(&mut rng, n, num_to_change).into_vec();

        println!("changing {} out of {}", indices_to_change.len(), n);

        // choose the new enode uniformly?
        for idx in indices_to_change {
            let id = nontrivial[idx];
            let ub = self.egraph[id].len();
            let choice = rng.random_range(0..ub);
            new_candidate.insert(id, choice);
        }

        Ok(new_candidate)
    }
}

impl<'a> argmin::core::CostFunction for GlobalExtractor<'a> {
    type Param = Candidate;
    type Output = f64;

    fn cost(&self, param: &Candidate) -> Result<f64, Error> {
        Ok(self.cost(param))
    }
}
