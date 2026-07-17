// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

// Null replication (modeled as axis.product with zero factors).
module {
  distributed.MeshComputation @mc0 mesh @mesh0 {
    %ax_main = axis.getaxis tensor<6xf32> 0
    %fmain = axis.factor %ax_main [6] : !axis.shape_axis<tensor<6xf32>, 0>
    %ctx = axis.product %fmain : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 6, 1>

    %rep = "axis.product"() : () -> !axis.factor_group<1>

    distributed.Function @callee context %ctx : !axis.factor_group<6> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      distributed.DistributedYield %arg0 i32
    }

    distributed.Function @caller context %ctx : !axis.factor_group<6> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      %r = distributed.DistributedCall @callee replicate_over %rep %arg0 : !axis.factor_group<1>, i32 -> i32
      distributed.DistributedYield %r i32
    }
  }
}

// -----

// Completely mismatching spaces should fail.
module {
  distributed.MeshComputation @mc1 mesh @mesh0 {
    %ax6 = axis.getaxis tensor<6xf32> 0
    %f6 = axis.factor %ax6 [6] : !axis.shape_axis<tensor<6xf32>, 0>
    %ctx6 = axis.product %f6 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 6, 1>

    %ax3 = axis.getaxis tensor<3xf32> 0
    %f3 = axis.factor %ax3 [3] : !axis.shape_axis<tensor<3xf32>, 0>
    %ctx3 = axis.product %f3 : !axis.axis_factor<!axis.shape_axis<tensor<3xf32>, 0>, 3, 1>

    distributed.Function @callee context %ctx3 : !axis.factor_group<3> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      distributed.DistributedYield %arg0 i32
    }

    distributed.Function @caller context %ctx6 : !axis.factor_group<6> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      // expected-error @+1 {{requires caller execution context to equal callee execution_context x replicate_over (permutation-insensitive)}}
      %r = distributed.DistributedCall @callee replicate_over %ctx3 %arg0 : !axis.factor_group<3>, i32 -> i32
      distributed.DistributedYield %r i32
    }
  }
}

// -----

// Null replication with mismatching caller/callee context should fail.
module {
  distributed.MeshComputation @mc2 mesh @mesh0 {
    %ax6 = axis.getaxis tensor<6xf32> 0
    %f6 = axis.factor %ax6 [6] : !axis.shape_axis<tensor<6xf32>, 0>
    %ctx6 = axis.product %f6 : !axis.axis_factor<!axis.shape_axis<tensor<6xf32>, 0>, 6, 1>

    %ax3 = axis.getaxis tensor<3xf32> 0
    %f3 = axis.factor %ax3 [3] : !axis.shape_axis<tensor<3xf32>, 0>
    %ctx3 = axis.product %f3 : !axis.axis_factor<!axis.shape_axis<tensor<3xf32>, 0>, 3, 1>

    %rep = "axis.product"() : () -> !axis.factor_group<1>

    distributed.Function @callee context %ctx3 : !axis.factor_group<3> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      distributed.DistributedYield %arg0 i32
    }

    distributed.Function @caller context %ctx6 : !axis.factor_group<6> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      // expected-error @+1 {{requires caller execution context to equal callee execution_context x replicate_over (permutation-insensitive)}}
      %r = distributed.DistributedCall @callee replicate_over %rep %arg0 : !axis.factor_group<1>, i32 -> i32
      distributed.DistributedYield %r i32
    }
  }
}
