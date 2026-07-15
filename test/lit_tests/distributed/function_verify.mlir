// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

module {
  distributed.MeshComputation @mc0 mesh @mesh0 {
    %axis = axis.getaxis tensor<8xf32> 0
    %f0 = axis.factor %axis [8] : !axis.shape_axis<tensor<8xf32>, 0>
    %ctx = axis.product %f0 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>

    // expected-error @+1 {{requires body block argument #0 to have type 'i32', but got 'f32'}}
    distributed.Function @arg_mismatch context %ctx : !axis.factor_group<8> arg_types [i32] ret_types [i32] {
    ^bb0(%arg0: f32):
      distributed.DistributedYield %arg0 f32
    }
  }
}

// -----

module {
  distributed.MeshComputation @mc1 mesh @mesh0 {
    %axis = axis.getaxis tensor<8xf32> 0
    %f0 = axis.factor %axis [8] : !axis.shape_axis<tensor<8xf32>, 0>
    %ctx = axis.product %f0 : !axis.axis_factor<!axis.shape_axis<tensor<8xf32>, 0>, 8, 1>

    // expected-error @+1 {{requires distributed.DistributedYield operand #0 to have type 'i32', but got 'f32'}}
    distributed.Function @ret_mismatch context %ctx : !axis.factor_group<8> arg_types [f32] ret_types [i32] {
    ^bb0(%arg0: f32):
      distributed.DistributedYield %arg0 f32
    }
  }
}

// -----

module {
  %ctx = "axis.product"() : () -> !axis.factor_group<1>

  // expected-error @+1 {{must be nested in distributed.MeshComputation}}
  distributed.Function @bad_parent context %ctx : !axis.factor_group<1> arg_types [i32] ret_types [i32] {
  ^bb0(%arg0: i32):
    distributed.DistributedYield %arg0 i32
  }
}

