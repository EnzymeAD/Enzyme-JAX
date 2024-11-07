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

import gcsfs
import jax
import numpy as np
import pickle
import xarray
import timeit

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm


class NeuralGCM:
    def setUp(self):
        gcs = gcsfs.GCSFileSystem(token="anon")

        model_name = "neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl"  # @param ['neural_gcm_dynamic_forcing_deterministic_0_7_deg.pkl', 'neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl', 'neural_gcm_dynamic_forcing_deterministic_2_8_deg.pkl', 'neural_gcm_dynamic_forcing_stochastic_1_4_deg.pkl'] {type: "string"}

        with gcs.open(f"gs://gresearch/neuralgcm/04_30_2024/{model_name}", "rb") as f:
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

        import os
        if os.getenv("NEURALGCM_LARGE") is not None: 
            inner_steps = 24  # save model outputs once every 24 hours
            outer_steps = 4 * 24 // inner_steps  # total of 4 days
        elif s.getenv("NEURALGCM_MEDIUM") is not None: 
            inner_steps = 4  # save model outputs once every 24 hours
            outer_steps = 4 * 4 // inner_steps  # total of 4 days
        else:
            inner_steps = 1  # save model outputs once every 24 hours
            outer_steps = 1 * 1 // inner_steps  # total of 4 days

        timedelta = np.timedelta64(1, "h") * inner_steps
        # times = (np.arange(outer_steps) * inner_steps)  # time axis in hours

        # initialize model state
        inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
        input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
        rng_key = jax.random.key(42)  # optional for deterministic models
        initial_state = model.encode(inputs, input_forcings, rng_key)

        # use persistence for forcing variables (SST and sea ice cover)
        all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

        # make forecast
        final_state, predictions = model.unroll(
            initial_state,
            all_forcings,
            steps=outer_steps,
            timedelta=timedelta,
            start_with_input=True,
        )
        # predictions_ds = model.data_to_xarray(predictions, times=times)

        def sub(initial_state, all_forcings):
            return model.unroll(
                initial_state,
                all_forcings,
                steps=outer_steps,
                timedelta=timedelta,
                start_with_input=True,
            )

        self.sub = sub
        self.model = model
        self.eval_era5 = eval_era5
        self.all_forcings = all_forcings
        self.outer_steps = outer_steps

        inputs = self.model.inputs_from_xarray(self.eval_era5.isel(time=0))
        input_forcings = self.model.forcings_from_xarray(self.eval_era5.isel(time=0))
        rng_key = jax.random.key(42)  # optional for deterministic models
        self.initial_state = self.model.encode(inputs, input_forcings, rng_key)

    def test(self):
        for name, pipe, _ in pipelines:
            print("name=", name)
            if pipe is None:
                nfn = jax.jit(self.sub)
            else:
                nfn = jax.jit(enzyme_jax_ir(pipeline_options=pipe)(self.sub))

            res = self.run_on_fn(nfn)
            print("name=", name, res)

    def run_on_fn(self, fn, steps=1):
        map(
            lambda x: x.block_until_ready(),
            fn(
                self.initial_state,
                self.all_forcings,
            ),
        )
        return timeit.Timer(
            """map(lambda x:x.block_until_ready(), fn(
        initial_state,
        all_forcings,
    ))""",
            globals={
                "fn": fn,
                "initial_state": self.initial_state,
                "all_forcings": self.all_forcings,
            },
        ).timeit(steps)


if __name__ == "__main__":
    c = NeuralGCM()
    c.setUp()
    c.test()
