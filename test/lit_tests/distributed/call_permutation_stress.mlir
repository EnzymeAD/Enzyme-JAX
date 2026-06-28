// RUN: enzymexlamlir-opt --verify-diagnostics %s

// Stress order-insensitive factor-space equality with many permuted factors.
module {
  distributed.MeshComputation @mc mesh @mesh0 {
    %ax = axis.getaxis tensor<360xf32> 0
    %f0, %f1, %f2, %f3, %f4, %f5 = axis.factor %ax [2, 2, 2, 3, 3, 5] : !axis.shape_axis<tensor<360xf32>, 0>

    %ctx_callee = axis.product %f0, %f1, %f2, %f3 : !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 2, 180>, !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 2, 90>, !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 2, 45>, !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 3, 15>
    %ctx_rep = axis.product %f4, %f5 : !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 3, 5>, !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 5, 1>

    // Same total factor multiset as callee x replicate_over, but heavily permuted.
    %ctx_caller = axis.product %f5, %f2, %f4, %f1, %f3, %f0 : !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 5, 1>, !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 2, 45>, !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 3, 5>, !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 2, 90>, !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 3, 15>, !axis.axis_factor<!axis.shape_axis<tensor<360xf32>, 0>, 2, 180>

    distributed.Function @callee context %ctx_callee : !axis.factor_group<24> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      distributed.DistributedYield %arg0 i32
    }

    distributed.Function @caller context %ctx_caller : !axis.factor_group<360> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      %r = distributed.DistributedCall @callee replicate_over %ctx_rep %arg0 : !axis.factor_group<15>, i32 -> i32
      distributed.DistributedYield %r i32
    }
  }
}
