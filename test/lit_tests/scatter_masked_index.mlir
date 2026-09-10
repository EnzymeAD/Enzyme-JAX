// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// A masked store scatters with its index sent out of bounds on the dead
// path (scatter drops it). The single-point scatter must not become a
// plain dynamic_update_slice — DUS clamps the index back in bounds — so
// the mask is resolved first: slice the original value and write it back
// when masked off.
func.func @main(%buf: tensor<50xf64>, %i: tensor<i64>, %p: tensor<i1>, %v: tensor<f64>) -> tensor<50xf64> {
  %cm1 = stablehlo.constant dense<-1> : tensor<1xi64>
  %ir = stablehlo.reshape %i : (tensor<i64>) -> tensor<1xi64>
  %pb = stablehlo.reshape %p : (tensor<i1>) -> tensor<1xi1>
  %sel = stablehlo.select %pb, %ir, %cm1 : tensor<1xi1>, tensor<1xi64>
  %r = "stablehlo.scatter"(%buf, %sel, %v) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = false}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<50xf64>, tensor<1xi64>, tensor<f64>) -> tensor<50xf64>
  return %r : tensor<50xf64>
}

// A full-grid masked store: the live indices are a linearized iota, so the
// masked-off lanes write their original values back and the scatter folds to
// a select over the whole tensor.
func.func @dense_masked(%input: tensor<100xf32>, %update: tensor<10x10xf32>, %pred: tensor<10x10x1xi1>) -> tensor<100xf32> {
  %c10 = stablehlo.constant dense<10> : tensor<10xi64>
  %off = stablehlo.constant dense<-1> : tensor<10x10x1xi64>
  %i = stablehlo.iota dim = 0 : tensor<10xi64>
  %row = stablehlo.multiply %i, %c10 : tensor<10xi64>
  %col = stablehlo.iota dim = 1 : tensor<10x10x1xi64>
  %rowb = stablehlo.broadcast_in_dim %row, dims = [0] : (tensor<10xi64>) -> tensor<10x10x1xi64>
  %lin = stablehlo.add %rowb, %col : tensor<10x10x1xi64>
  %idx = stablehlo.select %pred, %lin, %off : tensor<10x10x1xi1>, tensor<10x10x1xi64>
  %0 = "stablehlo.scatter"(%input, %idx, %update) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    stablehlo.return %b : tensor<f32>
  }) : (tensor<100xf32>, tensor<10x10x1xi64>, tensor<10x10xf32>) -> tensor<100xf32>
  return %0 : tensor<100xf32>
}

// CHECK:      func.func @main(%arg0: tensor<50xf64>, %arg1: tensor<i64>, %arg2: tensor<i1>, %arg3: tensor<f64>) -> tensor<50xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg1 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:    %1 = stablehlo.dynamic_slice %arg0, %arg1, sizes = [1] : (tensor<50xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:    %3 = stablehlo.select %arg2, %arg3, %2 : tensor<i1>, tensor<f64>
// CHECK-NEXT:    %4 = "stablehlo.scatter"(%arg0, %0, %3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg4: tensor<f64>, %arg5: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg5 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<50xf64>, tensor<1xi64>, tensor<f64>) -> tensor<50xf64>
// CHECK-NEXT:    return %4 : tensor<50xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @dense_masked(%arg0: tensor<100xf32>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10x1xi1>) -> tensor<100xf32> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<100xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:    %1 = stablehlo.reshape %arg2 : (tensor<10x10x1xi1>) -> tensor<10x10xi1>
// CHECK-NEXT:    %2 = stablehlo.select %1, %arg1, %0 : tensor<10x10xi1>, tensor<10x10xf32>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<10x10xf32>) -> tensor<100xf32>
// CHECK-NEXT:    return %3 : tensor<100xf32>
// CHECK-NEXT:  }
