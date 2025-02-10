use crate::{
  optimize::*,
  model::*,
};
use argmin::{core::*, solver::simulatedannealing::*};
use rand::prelude::*;
use rand::seq::index::sample;

impl<'a> Anneal for GlobalExtractor<'a> {
  type Param = Candidate;
  type Output = Candidate;
  type Float = f64;

  fn anneal(
      &self,
      param: &Candidate,
      temp: f64,
  ) -> Result<Candidate, Error> {
      let mut rng = rand::thread_rng();
      // Start from the current candidate.
      let mut new_candidate = param.clone();

      // only consider the non-trivial classes
      let nontrivial: Vec<_> = self.egraph.classes()
          .filter(|eclass| eclass.len() > 1)
          .collect();

      let n = nontrivial.len();
      if n > 0 && temp > 0.0 {
          // at higher temperatures, we make more changes.
          // use ceil so that even at low-but-nonzero temperature we change at least one
          let mut num_to_change = (temp * n as f64).ceil() as usize;
          // if the computed number is 0, bump it to 1?
          if num_to_change == 0 {
              num_to_change = 1;
          }
          num_to_change = num_to_change.min(n);

          // choose 'num_to_change' eclasses (without replacement) from the nontrivial ones
          let indices_to_change = sample(&mut rng, n, num_to_change).into_vec();

          // choose the new enode uniformly?
          for idx in indices_to_change {
              let eclass = nontrivial[idx];
              let id = self.egraph.find(eclass.id);
              let ub = eclass.len();
              let choice = rng.gen_range(0..ub);
              new_candidate.insert(id, choice);
          }
      }

      Ok(new_candidate)
  }
}
