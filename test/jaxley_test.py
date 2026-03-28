from absl.testing import absltest

from test_utils import EnzymeJaxTest, pipelines


# Based on https://jaxley.readthedocs.io/en/latest/examples/00_l5pc_gradient_descent.html#defining-the-model
class Jaxley(EnzymeJaxTest):
    def setUp(self):
        import jax.numpy as jnp
        import numpy as np

        import jaxley as jx
        from jaxley.channels import Leak
        from jaxley_mech.channels import l5pc
        from jaxley.morphology import distance_direct

        import tempfile
        import requests

        response = requests.get(
            "https://raw.githubusercontent.com/jaxleyverse/jaxley/refs/heads/main/tests/swc_files/morph_l5pc_with_axon.swc"
        )
        response.raise_for_status()

        tmpfile = tempfile.NamedTemporaryFile(delete=False)
        with open(tmpfile.name, "wb") as f:
            f.write(response.content)

        cell = jx.read_swc(tmpfile.name, ncomp=1)

        cell.set("axial_resistivity", 100.0)
        cell.apical.set("capacitance", 2.0)

        # Run the d_lambda rule.
        frequency = 100.0
        d_lambda = 0.1  # Larger -> more coarse-grained.

        for branch in cell.branches:
            diameter = 2 * branch.nodes["radius"].to_numpy()[0]
            c_m = branch.nodes["capacitance"].to_numpy()[0]
            r_a = branch.nodes["axial_resistivity"].to_numpy()[0]
            length = branch.nodes["length"].to_numpy()[0]

            lambda_f = 1e5 * np.sqrt(diameter / (4 * np.pi * frequency * c_m * r_a))
            ncomp = int((length / (d_lambda * lambda_f) + 0.9) / 2) * 2 + 1
            branch.set_ncomp(ncomp, initialize=False)
        cell.initialize()

        ########## APICAL ##########
        cell.apical.insert(l5pc.NaTs2T())
        cell.apical.insert(l5pc.SKv3_1())
        cell.apical.insert(l5pc.M())
        cell.apical.insert(l5pc.H())

        ########## SOMA ##########
        cell.soma.insert(l5pc.NaTs2T())
        cell.soma.insert(l5pc.SKv3_1())
        cell.soma.insert(l5pc.SKE2())
        ca_dynamics = l5pc.CaNernstReversal()
        ca_dynamics.channel_constants["T"] = 307.15
        cell.soma.insert(ca_dynamics)
        cell.soma.insert(l5pc.CaPump())
        cell.soma.insert(l5pc.CaHVA())
        cell.soma.insert(l5pc.CaLVA())

        ########## BASAL ##########
        cell.basal.insert(l5pc.H())

        # ########## AXON ##########
        cell.insert(l5pc.CaNernstReversal())
        cell.axon.insert(l5pc.NaTaT())
        cell.axon.insert(l5pc.NapEt2())
        cell.axon.insert(l5pc.KTst())
        cell.axon.insert(l5pc.CaPump())
        cell.axon.insert(l5pc.SKE2())
        cell.axon.insert(l5pc.CaHVA())
        cell.axon.insert(l5pc.KPst())
        cell.axon.insert(l5pc.SKv3_1())
        cell.axon.insert(l5pc.CaLVA())

        ########## WHOLE CELL ##########
        cell.insert(Leak())

        dt = 0.025  # ms
        t_max = 100.0  # ms
        time_vec = np.arange(0, t_max + 2 * dt, dt)

        cell.delete_stimuli()
        cell.delete_recordings()

        i_delay = 5.0  # ms
        i_dur = 90.0  # ms
        i_amp = 1.8  # nA
        current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
        cell.soma.branch(0).loc(0.5).stimulate(current)
        cell.soma.branch(0).loc(0.5).record()

        cell.set("v", -72.0)
        cell.init_states()

        cell.set("CaCon_i", 5e-05)
        cell.set("CaCon_e", 2.0)

        cell.apical.set("NaTs2T_gNaTs2T", 0.026145)
        cell.apical.set("SKv3_1_gSKv3_1", 0.004226)
        cell.apical.set("M_gM", 0.000143)
        cell.soma.set("NaTs2T_gNaTs2T", 0.983955)
        cell.soma.set("SKv3_1_gSKv3_1", 0.303472)
        cell.soma.set("SKE2_gSKE2", 0.008407)
        cell.soma.set("CaPump_gamma", 0.000609)
        cell.soma.set("CaPump_decay", 210.485291)
        cell.soma.set("CaHVA_gCaHVA", 0.000994)
        cell.soma.set("CaLVA_gCaLVA", 0.000333)

        cell.axon.set("NaTaT_gNaTaT", 3.137968)
        cell.axon.set("NapEt2_gNapEt2", 0.006827)
        cell.axon.set("KTst_gKTst", 0.089259)
        cell.axon.set("CaPump_gamma", 0.00291)
        cell.axon.set("CaPump_decay", 287.19873)
        cell.axon.set("SKE2_gSKE2", 0.007104)
        cell.axon.set("CaHVA_gCaHVA", 0.00099)
        cell.axon.set("KPst_gKPst", 0.973538)
        cell.axon.set("SKv3_1_gSKv3_1", 1.021945)
        cell.axon.set("CaLVA_gCaLVA", 0.008752)

        # The H-conductance depends on the distance from the soma.
        cell.compute_compartment_centers()
        direct_dists = distance_direct(cell.soma.branch(0).comp(0), cell)
        cell.nodes["dist_from_soma"] = direct_dists
        gH_conductance = (
            -0.8696 + 2.087 * np.exp(cell.basal.nodes["dist_from_soma"] * 0.0031)
        ) * 8e-5
        cell.basal.set("H_gH", gH_conductance)

        cell.set("Leak_gLeak", 3e-05)
        cell.set("Leak_eLeak", -75.0)

        cell.set("eNa", 50.0)
        cell.set("eK", -85.0)

        x_o = jx.integrate(cell)[0]

        bounds = {}
        bounds["apical_NaTs2T_gNaTs2T"] = [0, 0.04]
        bounds["apical_SKv3_1_gSKv3_1"] = [0, 0.04]
        bounds["apical_M_gM"] = [0, 0.001]
        bounds["somatic_NaTs2T_gNaTs2T"] = [0.0, 1.0]
        bounds["somatic_SKv3_1_gSKv3_1"] = [0.25, 1]
        bounds["somatic_SKE2_gSKE2"] = [0, 0.1]
        bounds["somatic_CaPump_gamma"] = [0.0005, 0.01]
        bounds["somatic_CaPump_decay"] = [20, 1_000]
        bounds["somatic_CaHVA_gCaHVA"] = [0, 0.001]
        bounds["somatic_CaLVA_gCaLVA"] = [0, 0.01]
        bounds["axonal_NaTaT_gNaTaT"] = [0.0, 4.0]
        bounds["axonal_NapEt2_gNapEt2"] = [0.0, 0.01]
        bounds["axonal_KPst_gKPst"] = [0.0, 1.0]
        bounds["axonal_KTst_gKTst"] = [0.0, 0.1]
        bounds["axonal_SKE2_gSKE2"] = [0.0, 0.1]
        bounds["axonal_SKv3_1_gSKv3_1"] = [0.0, 2.0]
        bounds["axonal_CaHVA_gCaHVA"] = [0, 0.001]
        bounds["axonal_CaLVA_gCaLVA"] = [0, 0.01]
        bounds["axonal_CaPump_gamma"] = [0.0005, 0.05]
        bounds["axonal_CaPump_decay"] = [20, 1_000]

        # Extract the lower and upper bounds as an array, for convenience later.
        lower_bounds = jnp.asarray(list(bounds.values()))[:, 0]
        upper_bounds = jnp.asarray(list(bounds.values()))[:, 1]

        from jaxley.optimize.transforms import SigmoidTransform

        transform = SigmoidTransform(
            lower=lower_bounds,
            upper=upper_bounds,
        )

        # For checkpointing.
        checkpoint_levels = 2
        checkpoints = [
            int(np.ceil(len(time_vec) ** (1 / checkpoint_levels)))
            for _ in range(checkpoint_levels)
        ]

        def simulate(params):
            # Set apical parameters.
            pstate = None
            pstate = cell.apical.data_set("NaTs2T_gNaTs2T", params[0], pstate)
            pstate = cell.apical.data_set("SKv3_1_gSKv3_1", params[1], pstate)
            pstate = cell.apical.data_set("M_gM", params[2], pstate)

            # Set somatic parameters.
            pstate = cell.soma.data_set("NaTs2T_gNaTs2T", params[3], pstate)
            pstate = cell.soma.data_set("SKv3_1_gSKv3_1", params[4], pstate)
            pstate = cell.soma.data_set("SKE2_gSKE2", params[5], pstate)
            pstate = cell.soma.data_set("CaPump_gamma", params[6], pstate)
            pstate = cell.soma.data_set("CaPump_decay", params[7], pstate)
            pstate = cell.soma.data_set("CaHVA_gCaHVA", params[8], pstate)
            pstate = cell.soma.data_set("CaLVA_gCaLVA", params[9], pstate)

            # Set axonal parameters.
            pstate = cell.axon.data_set("NaTaT_gNaTaT", params[10], pstate)
            pstate = cell.axon.data_set("NapEt2_gNapEt2", params[11], pstate)
            pstate = cell.axon.data_set("KPst_gKPst", params[12], pstate)
            pstate = cell.axon.data_set("KTst_gKTst", params[13], pstate)
            pstate = cell.axon.data_set("SKE2_gSKE2", params[14], pstate)
            pstate = cell.axon.data_set("SKv3_1_gSKv3_1", params[15], pstate)
            pstate = cell.axon.data_set("CaHVA_gCaHVA", params[16], pstate)
            pstate = cell.axon.data_set("CaLVA_gCaLVA", params[17], pstate)
            pstate = cell.axon.data_set("CaPump_gamma", params[18], pstate)
            pstate = cell.axon.data_set("CaPump_decay", params[19], pstate)

            # Return [0] because the result of `jx.integrate` is of shape (1, time). Here, we
            # get rid of the batch dimension.
            return jx.integrate(
                cell, param_state=pstate, checkpoint_lengths=checkpoints
            )[0]

        def sample_randomly():
            return jnp.asarray(
                np.random.rand(len(upper_bounds)) * (upper_bounds - lower_bounds)
                + lower_bounds
            )

        window1 = jnp.arange(200, 2000)  # Unit: time steps.
        window2 = jnp.arange(2000, 3800)

        def summary_stats(v):
            mean1 = jnp.mean(v[window1])
            std1 = jnp.std(v[window1])
            mean2 = jnp.mean(v[window2])
            std2 = jnp.std(v[window2])
            return jnp.asarray([mean1, std1, mean2, std2])

        x_standard_deviation = jnp.asarray(
            [2.0, 1.0, 2.0, 1.0]
        )  # Large values downweigh the respective summary statistic.

        # Compute the summary statistics of the observation.
        x_o_ss = summary_stats(x_o)

        def loss_fn(opt_params):
            params = transform.forward(opt_params)
            v = simulate(params)
            ss = summary_stats(v)
            return jnp.mean(jnp.abs((ss - x_o_ss) / x_standard_deviation))

        _ = np.random.seed(0)
        initial_params = sample_randomly()
        opt_params = transform.inverse(initial_params)

        self.fn = loss_fn
        self.name = "jaxley_l5pc"

        self.AllPipelines = pipelines(noscattergather=True)

        self.ins = [opt_params]
        self.dins = [opt_params.copy()]
        self.douts = loss_fn(opt_params).copy()

        self.atol = 5e-3
        self.rtol = 1e-3

        # TODO: investigate. running inside xprof segfaults.
        self.repeat = 2
        self.use_xprof = False

        # currently missing some scatter derivative rule
        self.revfilter = lambda _: []


if __name__ == "__main__":
    import platform

    if platform.system() != "Darwin" and platform.machine() == "x86_64":
        from test_utils import fix_paths

        fix_paths()
        import jax

        jax.config.update("jax_enable_x64", True)
        absltest.main()
