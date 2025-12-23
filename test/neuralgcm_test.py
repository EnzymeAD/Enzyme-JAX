from absl.testing import absltest
from test_utils import EnzymeJaxTest
import numpy as np
import os

import jax.random


class NeuralGCM(EnzymeJaxTest):
    def setUp(self):
        import neuralgcm
        import gcsfs
        import pickle
        import xarray
        from dinosaur import horizontal_interpolation
        from dinosaur import spherical_harmonic
        from dinosaur import xarray_utils

        model_name = "v1/deterministic_2_8_deg.pkl"  # @param ['v1/deterministic_0_7_deg.pkl', 'v1/deterministic_1_4_deg.pkl', 'v1/deterministic_2_8_deg.pkl', 'v1/stochastic_1_4_deg.pkl', 'v1_precip/stochastic_precip_2_8_deg.pkl', 'v1_precip/stochastic_evap_2_8_deg.pkl'] {type: "string"}

        gcs = gcsfs.GCSFileSystem(token="anon")
        with gcs.open(f"gs://neuralgcm/models/{model_name}", "rb") as f:
            ckpt = pickle.load(f)

        model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

        era5_path = (
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        )
        full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)

        demo_start_time = "2020-02-14"
        demo_end_time = "2020-02-18"
        data_inner_steps = 24  # process every 24th hour

        sliced_era5 = (
            full_era5[model.input_variables + model.forcing_variables]
            .pipe(
                xarray_utils.selective_temporal_shift,
                variables=model.forcing_variables,
                time_shift="24 hours",
            )
            .sel(time=slice(demo_start_time, demo_end_time, data_inner_steps))
            .compute()
        )

        era5_grid = spherical_harmonic.Grid(
            latitude_nodes=full_era5.sizes["latitude"],
            longitude_nodes=full_era5.sizes["longitude"],
            latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
            longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
        )
        regridder = horizontal_interpolation.ConservativeRegridder(
            era5_grid, model.data_coords.horizontal, skipna=True
        )
        eval_era5 = xarray_utils.regrid(sliced_era5, regridder)
        eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)

        if os.getenv("NEURALGCM_LARGE") is not None:
            inner_steps = 24  # save model outputs once every 24 hours
            outer_steps = 4 * 24 // inner_steps  # total of 4 days
        elif os.getenv("NEURALGCM_MEDIUM") is not None:
            inner_steps = 4  # save model outputs once every 24 hours
            outer_steps = 4 * 4 // inner_steps  # total of 4 days
        elif jax.default_backend() == "gpu" or jax.default_backend() == "tpu":
            inner_steps = 24  # save model outputs once every 24 hours
            outer_steps = 4 * 24 // inner_steps  # total of 4 days
        else:
            inner_steps = 2  # save model outputs once every 24 hours
            outer_steps = 2 * 2 // inner_steps  # total of 4 days

        timedelta = np.timedelta64(1, "h") * inner_steps

        # initialize model state
        inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
        input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
        rng_key = jax.random.key(42)  # optional for deterministic models
        initial_state = model.encode(inputs, input_forcings, rng_key)

        # use persistence for forcing variables (SST and sea ice cover)
        all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

        def forward(initial_state, all_forcings):
            return model.unroll(
                initial_state,
                all_forcings,
                steps=outer_steps,
                timedelta=timedelta,
                start_with_input=True,
            )

        self.name = (
            model_name.split(".")[0]
            + "_inner_steps_"
            + str(inner_steps)
            + "_outer_steps_"
            + str(outer_steps)
        )

        self.fn = forward

        self.ins = (initial_state, all_forcings)
        self.dins = ()
        self.douts = ()

        self.mlirad_rev = False
        self.mlirad_fwd = False
        self.fwdfilter = lambda _: []
        self.revfilter = lambda _: []

        self.repeat = 2
        self.atol = 5e-2
        self.rtol = 1e-2

        # TODO: we should fix this at some point
        self.skip_test_assert = True


if __name__ == "__main__":
    from test_utils import fix_paths

    fix_paths()
    import platform

    # Deps not available on macos
    if platform.system() != "Darwin" and platform.machine() == "x86_64":
        absltest.main()
