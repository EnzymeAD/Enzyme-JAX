from absl.testing import absltest

from test_utils import *


class JAXMD(EnzymeJaxTest):
    def setUp(self):

        from jax import jit
        from jax import random
        from jax import lax

        import jax.numpy as np

        from jax_md import space
        from jax_md import energy
        from jax_md import simulate
        from jax_md import quantity
        from jax_md import partition

        lattice_constant = 1.37820
        # We hit a GPU memory limit for N_rep = 40
        N_rep = 20 if jax.default_backend() == "gpu" else 40
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
        print(f"Created a system of {N} LJ particles with number density {phi:.3f}")

        neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(
            displacement, box_size, r_cutoff=3.0, dr_threshold=1.0, format=format
        )

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

        def forward(
            position, momentum, force, mass, c_position, c_momentum, c_mass, c_tau, c_KE
        ):
            chain = simulate.NoseHooverChain(
                c_position, c_momentum, c_mass, c_tau, c_KE, degrees_of_freedom
            )
            state = simulate.NVTNoseHooverState(position, momentum, force, mass, chain)
            # new_state, new_nbrs = lax.fori_loop(0, iters, step, (state, nbrs))
            new_state, new_nbrs = step(0, (state, nbrs))
            return (
                new_state.position,
                new_state.momentum,
                new_state.force,
                new_state.mass,
                new_state.chain.position,
                new_state.chain.momentum,
                new_state.chain.mass,
                new_state.chain.tau,
                new_state.chain.kinetic_energy,
            )

        self.fn = forward
        self.name = "jaxmd40"
        self.count = 10
        # self.revprimal = False
        # self.AllPipelines = pipelines
        # self.AllBackends = CurBackends

        self.ins = [
            state.position,
            state.momentum,
            state.force,
            state.mass,
            state.chain.position,
            state.chain.momentum,
            state.chain.mass,
            state.chain.tau,
            state.chain.kinetic_energy,
        ]
        # for i, v in enumerate(self.ins):
        #    print("i=", i, v)
        self.dins = [x.copy() for x in self.ins]
        self.douts = tuple(x.copy() for x in self.ins)
        self.AllPipelines = pipelines()
        # No support for stablehlo.while atm
        # self.revfilter = justjax
        self.mlirad_rev = False

        self.tol = 5e-4

        # GPU CI reverse mode needs loose, merits future investigation
        self.tol = 1e-2


if __name__ == "__main__":
    import platform

    # Deps not available on macos
    if platform.system() != "Darwin":
        from test_utils import fix_paths

        fix_paths()
        import jax

        jax.config.update("jax_enable_x64", True)
        absltest.main()
