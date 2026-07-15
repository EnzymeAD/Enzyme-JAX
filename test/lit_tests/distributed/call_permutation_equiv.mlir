// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

// Trivial permutation equivalence: same factor multiset in different order.
module {
  distributed.MeshComputation @mc0 mesh @mesh0 {
    %ax = axis.getaxis tensor<12xf32> 0
    %f0, %f1, %f2 = axis.factor %ax [2, 2, 3] : !axis.shape_axis<tensor<12xf32>, 0>

    %ctx_callee = axis.product %f0, %f1 : !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 6>, !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 3>
    %ctx_rep = axis.product %f2 : !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 3, 1>
    // Permute the caller product order.
    %ctx_caller_perm = axis.product %f2, %f1, %f0 : !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 3, 1>, !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 6>

    distributed.Function @callee context %ctx_callee : !axis.factor_group<4> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      distributed.DistributedYield %arg0 i32
    }

    distributed.Function @caller context %ctx_caller_perm : !axis.factor_group<12> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      %r = distributed.DistributedCall @callee replicate_over %ctx_rep %arg0 : !axis.factor_group<3>, i32 -> i32
      distributed.DistributedYield %r i32
    }
  }
}

// -----

// Same extents but different provenance axes should fail equality.
module {
  distributed.MeshComputation @mc1 mesh @mesh0 {
    %axA = axis.getaxis tensor<6xf32> 0
    %a0, %a1 = axis.factor %axA [2, 3] : !axis.shape_axis<tensor<6xf32>, 0>
    %ctxA = axis.product %a0, %a1 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 3, 1>

    %axB = axis.getaxis tensor<6xf32> 0
    %b0, %b1 = axis.factor %axB [2, 3] : !axis.shape_axis<tensor<6xf32>, 0>
    %ctxB = axis.product %b0, %b1 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 3, 1>

    distributed.Function @callee context %ctxA : !axis.factor_group<6> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      distributed.DistributedYield %arg0 i32
    }

    distributed.Function @caller context %ctxB : !axis.factor_group<6> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      // expected-error @+1 {{requires caller execution context to equal callee execution_context x replicate_over (permutation-insensitive)}}
      %r = distributed.DistributedCall @callee replicate_over %ctxB %arg0 : !axis.factor_group<6>, i32 -> i32
      distributed.DistributedYield %r i32
    }
  }
}
