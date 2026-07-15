// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

// Complete replication: caller context equals callee context x replicate_over.
module {
  distributed.MeshComputation @mc0 mesh @mesh0 {
    %ax = axis.getaxis tensor<12xf32> 0
    %f0 = axis.factor %ax : (!axis.shape_axis<tensor<12xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 6>
    %f1 = axis.factor %ax : (!axis.shape_axis<tensor<12xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 3>
    %f2 = axis.factor %ax : (!axis.shape_axis<tensor<12xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 3, 1>
    %ctx_callee = axis.product %f0, %f1 : !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 6>, !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 3>
    %ctx_rep = axis.product %f2 : !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 3, 1>
    %ctx_caller = axis.product %f0, %f1, %f2 : !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 6>, !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 2, 3>, !axis.axis_factor<!axis.shape_axis<tensor<12xf32>, 0>, 3, 1>

    distributed.Function @callee context %ctx_callee : !axis.factor_group<4> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      distributed.DistributedYield %arg0 i32
    }

    distributed.Function @caller context %ctx_caller : !axis.factor_group<12> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      %r = distributed.DistributedCall @callee replicate_over %ctx_rep %arg0 : !axis.factor_group<3>, i32 -> i32
      distributed.DistributedYield %r i32
    }
  }
}

// -----

// Elementwise callee context + mismatching replication should fail.
module {
  distributed.MeshComputation @mc1 mesh @mesh0 {
    %ax4 = axis.getaxis tensor<4xf32> 0
    %f4 = axis.factor %ax4 : (!axis.shape_axis<tensor<4xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
    %ctx4 = axis.product %f4 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>

    %ax2 = axis.getaxis tensor<2xf32> 0
    %f2 = axis.factor %ax2 : (!axis.shape_axis<tensor<2xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>
    %ctx2 = axis.product %f2 : !axis.axis_factor<!axis.shape_axis<tensor<2xf32>, 0>, 2, 1>

    %ax1 = axis.getaxis tensor<1xf32> 0
    %f1 = axis.factor %ax1 : (!axis.shape_axis<tensor<1xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>
    %ctx1 = axis.product %f1 : !axis.axis_factor<!axis.shape_axis<tensor<1xf32>, 0>, 1, 1>

    distributed.Function @callee context %ctx1 : !axis.factor_group<1> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      distributed.DistributedYield %arg0 i32
    }

    distributed.Function @caller context %ctx4 : !axis.factor_group<4> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      // expected-error @+1 {{requires caller execution context to equal callee execution_context x replicate_over (permutation-insensitive)}}
      %r = distributed.DistributedCall @callee replicate_over %ctx2 %arg0 : !axis.factor_group<2>, i32 -> i32
      distributed.DistributedYield %r i32
    }
  }
}
