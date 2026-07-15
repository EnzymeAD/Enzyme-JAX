// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

// Valid metadata-only mesh body.
module {
  distributed.MeshComputation @ok mesh @mesh0 {
  ^bb0:
  }
}

// -----

// Non-metadata, non-function op in mesh body should fail verification.
module {
  // expected-error @+1 {{only distributed.Function, stablehlo.constant, and static metadata ops are allowed in the mesh body; operation 'arith.constant' is neither}}
  distributed.MeshComputation @bad mesh @mesh0 {
  ^bb0:
    %axis = axis.getaxis tensor<4xf32> 0
    %f0 = axis.factor %axis : (!axis.shape_axis<tensor<4xf32>, 0>) -> !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
    %ctx = axis.product %f0 : !axis.axis_factor<!axis.shape_axis<tensor<4xf32>, 0>, 4, 1>
    %c0 = arith.constant 0 : i32
  }
}

// -----

// GetPhysicalMeshAxes verifier should reject unknown symbol references.
module {
  // expected-error @+1 {{references unknown physical mesh symbol @missing_mesh}}
  %a = distributed.GetPhysicalMeshAxes @missing_mesh : !distributed.physical_comm_axis<4>
}
