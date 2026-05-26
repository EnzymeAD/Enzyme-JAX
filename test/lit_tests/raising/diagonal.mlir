// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --arith-raise --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

module {
  func.func private @"##call__Z7gpu__k_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi2E5TupleI5OneToI5Int64ES6_EE7NDRangeILi2ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li2ELi1E8_17__17_ESE_S5_S5_#349$par7"(%arg0: memref<17x17xf64, 1>, %arg1: memref<17x17xf64, 1>) {
    %cst = arith.constant 2.000000e+00 : f64
    affine.parallel (%arg2) = (0) to (289) {
      %0 = affine.load %arg1[%arg2, %arg2] : memref<17x17xf64, 1>
      %1 = arith.mulf %0, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %1, %arg0[%arg2, %arg2] : memref<17x17xf64, 1>
    }
    return
  }
}

// The kernel is a diagonal update: both store indices are the same IV %arg2,
// so the raised scatter should zip them (one (i, i) per iteration), not
// Cartesian-product them (which would give all (i, j) pairs and write 289²
// values into a 289-cell matrix). The corrected lowering uses the same
// [289, 2] index tensor for the gather and the scatter.

// CHECK:  func.func private @"##call__Z7gpu__k_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi2E5TupleI5OneToI5Int64ES6_EE7NDRangeILi2ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li2ELi1E8_17__17_ESE_S5_S5_#349$par7_raised"(%arg0: tensor<17x17xf64>, %arg1: tensor<17x17xf64>) -> (tensor<17x17xf64>, tensor<17x17xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<2.000000e+00> : tensor<289xf64>
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.iota dim = 0 : tensor<289x2xi64>
// CHECK-NEXT:    %[[v1:.+]] = "stablehlo.gather"(%arg1, %[[v0]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<17x17xf64>, tensor<289x2xi64>) -> tensor<289xf64>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.multiply %[[v1]], %cst : tensor<289xf64>
// CHECK-NEXT:    %[[v3:.+]] = "stablehlo.scatter"(%arg0, %[[v0]], %[[v2]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<17x17xf64>, tensor<289x2xi64>, tensor<289xf64>) -> tensor<17x17xf64>
// CHECK-NEXT:    return %[[v3]], %arg1 : tensor<17x17xf64>, tensor<17x17xf64>
// CHECK-NEXT:  }
