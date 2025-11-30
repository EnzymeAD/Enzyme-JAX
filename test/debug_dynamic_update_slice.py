#!/usr/bin/env python3
"""
Minimal reproduction case for the dynamic_update_slice type mismatch issue.
"""

from absl.testing import absltest
from test_utils import EnzymeJaxTest
import jax.numpy as jnp
import jax
import numpy as np


class DynamicUpdateSliceDebug(EnzymeJaxTest):
    """Test to isolate the dynamic_update_slice type mismatch issue."""

    def setUp(self):
        def simple_update_slice_function(x):
            """Function that triggers dynamic_update_slice with potential type mismatch."""
            # This mimics what BlackJAX might be doing internally

            # Create some array to update
            base_array = jnp.zeros((10, 4))

            # Create update slice
            update_slice = jnp.ones((1, 4))

            # These index operations might create the i32/i64 mismatch
            i = jnp.array(2, dtype=jnp.int32)  # Explicit i32
            j = x.shape[0] - 1  # This might be i64 due to shape inference

            # This could trigger the type mismatch
            updated = base_array.at[i:i+1, :].set(update_slice)

            return updated.sum()

        # Simple input
        initial_x = jnp.array([1.0, 2.0, 3.0, 4.0])

        self.ins = [initial_x]
        self.dins = [jnp.ones_like(initial_x)]
        self.douts = simple_update_slice_function(initial_x)

        self.fn = simple_update_slice_function
        self.name = "debug_dynamic_update_slice"
        self.count = 10


class IndexTypeDebug(EnzymeJaxTest):
    """Test different index type scenarios."""

    def setUp(self):
        def index_type_function(x):
            """Function that tests various index types."""
            base = jnp.zeros((10, x.shape[0]))

            # Force different index types
            idx1 = jnp.array(0, dtype=jnp.int32)
            idx2 = jnp.array(x.shape[0] - 1, dtype=jnp.int64)  # This creates i64

            # Try to use both in slicing
            result = base.at[idx1].set(x)
            result = result.at[idx2].set(x * 2)

            return result.sum()

        initial_x = jnp.array([1.0, 2.0])

        self.ins = [initial_x]
        self.dins = [jnp.ones_like(initial_x)]
        self.douts = index_type_function(initial_x)

        self.fn = index_type_function
        self.name = "debug_index_types"
        self.count = 5


class WhileLoopIndexDebug(EnzymeJaxTest):
    """Test index types in while loops (like BlackJAX uses)."""

    def setUp(self):
        def while_loop_with_indices(x):
            """Function with while loop that might create index type issues."""

            def cond_fun(carry):
                i, arr = carry
                return i < 3

            def body_fun(carry):
                i, arr = carry
                # This indexing pattern might create the type mismatch
                new_arr = arr.at[i].set(x[i % x.shape[0]])
                return i + 1, new_arr

            init_arr = jnp.zeros(5)
            _, final_arr = jax.lax.while_loop(
                cond_fun, body_fun, (jnp.array(0), init_arr)
            )

            return final_arr.sum()

        initial_x = jnp.array([1.0, 2.0, 3.0])

        self.ins = [initial_x]
        self.dins = [jnp.ones_like(initial_x)]
        self.douts = while_loop_with_indices(initial_x)

        self.fn = while_loop_with_indices
        self.name = "debug_while_loop_indices"
        self.count = 5


if __name__ == "__main__":
    from test_utils import fix_paths
    fix_paths()
    absltest.main()