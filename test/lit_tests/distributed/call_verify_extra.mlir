// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

module {
  distributed.MeshComputation @mc0 mesh @mesh0 {
    %ax = axis.getaxis tensor<4xf32> 0
    %f0 = axis.factor %ax : (!axis.shape_axis<tensor<4xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
    %ctx = axis.product %f0 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>

    distributed.Function @callee context %ctx : !axis.factor_group<4> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      distributed.DistributedYield %arg0 i32
    }

    distributed.Function @caller context %ctx : !axis.factor_group<4> arg_types [i32] ret_types [f32] {
    ^bb0(%arg0: i32):
      // expected-error @+1 {{requires call result #0 to have type 'i32', but got 'f32'}}
      %r = distributed.DistributedCall @callee replicate_over %ctx %arg0 : !axis.factor_group<4>, i32 -> f32
      distributed.DistributedYield %r f32
    }
  }
}

// -----

module {
  distributed.MeshComputation @mc1 mesh @mesh0 {
    %ax = axis.getaxis tensor<4xf32> 0
    %f0 = axis.factor %ax : (!axis.shape_axis<tensor<4xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
    %ctx = axis.product %f0 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>

    distributed.Function @caller context %ctx : !axis.factor_group<4> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: i32):
      // expected-error @+1 {{references unknown distributed function symbol @missing}}
      %r = distributed.DistributedCall @missing replicate_over %ctx %arg0 : !axis.factor_group<4>, i32 -> i32
      distributed.DistributedYield %r i32
    }
  }
}

// -----

module {
  %ax = axis.getaxis tensor<4xf32> 0
  %f0 = axis.factor %ax : (!axis.shape_axis<tensor<4xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
  %ctx = axis.product %f0 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>

  distributed.MeshComputation @mc2 mesh @mesh0 {
    distributed.Function @callee context %ctx : !axis.factor_group<4> arg_types [!axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>] ret_types [!axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>] {
    ^bb0(%arg0: !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>):
      distributed.DistributedYield %arg0 !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
    }
  }

  // expected-error @+1 {{must be nested in distributed.function with a FactorGroupType execution_context}}
  %r = distributed.DistributedCall @callee replicate_over %ctx %f0 : !axis.factor_group<4>, !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1> -> !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
}
