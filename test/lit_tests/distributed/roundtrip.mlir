// RUN: enzymexlamlir-opt --split-input-file %s | FileCheck %s

// Roundtrip distributed.Function / distributed.DistributedCall /
// distributed.DistributedYield.
module {
  distributed.MeshComputation @mc mesh @mesh0 {
    %axis = axis.getaxis tensor<12xf32> 0
    %f0, %f1, %f2 = axis.factor %axis [2, 2, 3] : !axis.shape_axis<tensor<12xf32>, 0>
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

// CHECK-LABEL: module {
// CHECK: distributed.Function @callee context
// CHECK: distributed.Function @caller context
// CHECK: distributed.DistributedCall @callee replicate_over

// -----

// Roundtrip distributed.MeshComputation with a minimal metadata-only body.
module {
  distributed.MeshComputation @mc mesh @mesh0 {
    %a = axis.getaxis tensor<4xf32> 0
  }
}

// CHECK: distributed.MeshComputation @mc mesh @mesh0
