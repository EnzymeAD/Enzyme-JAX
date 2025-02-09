use crate::{
    optimize::*,
    model::*,
};
use argmin::{core::*, solver::simulatedannealing::*};
use rand::prelude::*;

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
        let mut new_candidate = param.clone();

        for eclass in self.egraph.classes() {
            let id = self.egraph.find(eclass.id);
            let should_modify = rng.random_bool(temp);
            if should_modify {
                let ub = eclass.len();
                let choice = rng.random_range(0..ub);
                new_candidate.insert(id, choice);
            }
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
