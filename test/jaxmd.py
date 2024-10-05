from absl.testing import absltest
import jax.numpy as jnp
import jax.random
import jax.lax
import enzyme_ad.jax as enzyme_jax
from enzyme_ad.jax import (
    enzyme_jax_ir,
    NewXLAPipeline,
    OldXLAPipeline,
    JaXPipeline,
    hlo_opts,
)
import numpy as np
import timeit
from test_utils import *

argv = ("-I/usr/include/c++/11", "-I/usr/include/x86_64-linux-gnu/c++/11")

import jax.numpy as np
import numpy as onp
from jax import jit
from jax import random
from jax import lax

jax.config.update("jax_enable_x64", True)

from jax_md import space
from jax_md import energy
from jax_md import simulate
from jax_md import quantity
from jax_md import partition

partialopt = (
    "inline{default-pipeline=canonicalize max-iterations=4},"
    + """canonicalize,cse,
enzyme-hlo-generate-td{
            patterns=compare_op_canon<16>;
transpose_transpose<16>;
broadcast_in_dim_op_canon<16>;
convert_op_canon<16>;
dynamic_broadcast_in_dim_op_not_actually_dynamic<16>;
chained_dynamic_broadcast_in_dim_canonicalization<16>;
dynamic_broadcast_in_dim_all_dims_non_expanding<16>;
noop_reduce_op_canon<16>;
empty_reduce_op_canon<16>;
dynamic_reshape_op_canon<16>;
get_tuple_element_op_canon<16>;
real_op_canon<16>;
imag_op_canon<16>;
get_dimension_size_op_canon<16>;
gather_op_canon<16>;
reshape_op_canon<16>;
merge_consecutive_reshapes<16>;
transpose_is_reshape<16>;
zero_extent_tensor_canon<16>;
reorder_elementwise_and_shape_op<16>;

cse_broadcast_in_dim<16>;
cse_slice<16>;
cse_transpose<16>;
cse_convert<16>;
cse_pad<16>;
cse_dot_general<16>;
cse_reshape<16>;
cse_mul<16>;
cse_div<16>;
cse_add<16>;
cse_subtract<16>;
cse_min<16>;
cse_max<16>;
cse_neg<16>;
cse_concatenate<16>;

concatenate_op_canon<16>(1024);
select_op_canon<16>(1024);
add_simplify<16>;
sub_simplify<16>;
and_simplify<16>;
max_simplify<16>;
min_simplify<16>;
or_simplify<16>;
negate_simplify<16>;
mul_simplify<16>;
div_simplify<16>;
rem_simplify<16>;
pow_simplify<16>;
sqrt_simplify<16>;
cos_simplify<16>;
sin_simplify<16>;
noop_slice<16>;
const_prop_through_barrier<16>;
slice_slice<16>;
shift_right_logical_simplify<16>;
pad_simplify<16>;
negative_pad_to_slice<16>;
tanh_simplify<16>;
exp_simplify<16>;
slice_simplify<16>;
convert_simplify<16>;
reshape_simplify<16>;
dynamic_slice_to_static<16>;
dynamic_update_slice_elim<16>;
concat_to_broadcast<16>;
reduce_to_reshape<16>;
broadcast_to_reshape<16>;
gather_simplify<16>;
iota_simplify<16>(1024);
broadcast_in_dim_simplify<16>(1024);
convert_concat<1>;
dynamic_update_to_concat<1>;
slice_of_dynamic_update<1>;
slice_elementwise<1>;
slice_pad<1>;
dot_reshape_dot<1>;
concat_const_prop<1>;
concat_fuse<1>;
pad_reshape_pad<1>;
pad_pad<1>;
concat_push_binop_add<1>;
concat_push_binop_mul<1>;
scatter_to_dynamic_update_slice<1>;
reduce_concat<1>;
slice_concat<1>;

bin_broadcast_splat_add<1>;
bin_broadcast_splat_subtract<1>;
bin_broadcast_splat_div<1>;
bin_broadcast_splat_mul<1>;
slice_reshape<1>;

dot_reshape_pad<1>;
pad_dot_general<1>(1);
broadcast_reduce<1>;
            },
            transform-interpreter,
            enzyme-hlo-remove-transform,cse"""
)

pipelines = [
    ("JaX  ", None, CurBackends),
    ("JaXPipe", JaXPipeline(), CurBackends),
    (
        "HLOOpt",
        JaXPipeline(
            "inline{default-pipeline=canonicalize max-iterations=4},"
            + "canonicalize,cse,enzyme-hlo-opt,cse"
        ),
        CurBackends,
    ),
    ("PartOpt", JaXPipeline(partialopt), CurBackends),
    ("DefOpt", JaXPipeline(hlo_opts()), CurBackends),
]


class JAXMD(EnzymeJaxTest):
    def setUp(self):
        lattice_constant = 1.37820
        N_rep = 40
        box_size = N_rep * lattice_constant
        # Using float32 for positions / velocities, but float64 for reductions.
        dtype = np.float32

        # Specify the format of the neighbor list.
        # Options are Dense, Sparse, or OrderedSparse.
        format = partition.OrderedSparse

        displacement, shift = space.periodic(box_size)

        R = []
        for i in range(N_rep):
          for j in range(N_rep):
            for k in range(N_rep):
              R += [[i, j, k]]
        R = np.array(R, dtype=dtype) * lattice_constant


        N = R.shape[0]
        phi = N / (lattice_constant * N_rep) ** 3
        print(f'Created a system of {N} LJ particles with number density {phi:.3f}')

        neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(displacement,
                                                                    box_size,
                                                                    r_cutoff=3.0,
                                                                    dr_threshold=1.,
                                                                    format=format)


        init, apply = simulate.nvt_nose_hoover(energy_fn, shift, 5e-3, kT=1.2)


        key = random.PRNGKey(0)

        # We pick an "extra capacity" to ensure ahead of time that the neighbor
        # list will have enough capacity. Since sparse neighbor lists are more
        # robust to changes in the number of particles, in this case we only
        # need to actually add more capacity for dense neighbor lists.
        if format is partition.Dense:
          nbrs = neighbor_fn.allocate(R, extra_capacity=55)
        else:
          nbrs = neighbor_fn.allocate(R)

        state = init(key, R, neighbor=nbrs)


        def step(i, state_and_nbrs):
          state, nbrs = state_and_nbrs
          nbrs = nbrs.update(state.position)
          return apply(state, neighbor=nbrs), nbrs

        iters = 10
        degrees_of_freedom = state.chain.degrees_of_freedom
        def forward(position, momentum, force, mass, c_position, c_momentum, c_mass, c_tau, c_KE):
          chain = simulate.NoseHooverChain(c_position, c_momentum, c_mass, c_tau, c_KE, degrees_of_freedom) 
          state = simulate.NVTNoseHooverState(position, momentum, force, mass, chain)
          new_state, new_nbrs = lax.fori_loop(0, 10, step, (state, nbrs))
          return (new_state.position, new_state.momentum, new_state.force, new_state.mass, new_state.chain.position, new_state.chain.momentum, new_state.chain.mass, new_state.chain.tau, new_state.chain.kinetic_energy)
          
        self.fn = forward
        self.name = "jaxmd40"
        self.count = 10
        # self.revprimal = False
        # self.AllPipelines = pipelines
        # self.AllBackends = CurBackends

        self.ins = [state.position, state.momentum, state.force, state.mass, state.chain.position, state.chain.momentum, state.chain.mass, state.chain.tau, state.chain.kinetic_energy]
        self.dins = [x.copy() for x in self.ins]
        self.douts = [x.copy() for x in self.ins]
        self.AllPipelines = pipelines
        # No support for stablehlo.while atm
        # self.revfilter = justjax
        self.mlirad = False

        self.tol = 5e-5


if __name__ == "__main__":
    absltest.main()
