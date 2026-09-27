// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The transpose carries an analysis annotation next to its permutation; its
// batched form takes the permutation up behind the batch dimension and leaves
// the annotation, which describes the unbatched value, behind.
func.func @annotated(%x: tensor<8x9xf64>, %y: tensor<8x9xf64>) -> tensor<8x9xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<8x9xf64> attributes {enzymexla.parallel}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 9] : (tensor<8x9xf64>, tensor<i64>, tensor<i64>) -> tensor<1x9xf64>
    %m = stablehlo.reshape %row : (tensor<1x9xf64>) -> tensor<3x3xf64>
    %t = stablehlo.transpose %m, dims = [1, 0] {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %u = stablehlo.reshape %t : (tensor<3x3xf64>) -> tensor<1x9xf64>
    %w = stablehlo.dynamic_update_slice %acc, %u, %i, %c0 : (tensor<8x9xf64>, tensor<1x9xf64>, tensor<i64>, tensor<i64>) -> tensor<8x9xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %w : tensor<i64>, tensor<8x9xf64>
  }
  return %0#1 : tensor<8x9xf64>
}

// CHECK:  func.func @annotated(%arg0: tensor<8x9xf64>, %arg1: tensor<8x9xf64>) -> tensor<8x9xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg0 : (tensor<8x9xf64>) -> tensor<8x1x9xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<8x1x9xf64>) -> tensor<8x3x3xf64>
// CHECK-NEXT:   %3 = stablehlo.transpose %2, dims = [0, 2, 1] : (tensor<8x3x3xf64>) -> tensor<8x3x3xf64>
// CHECK-NEXT:   %4 = stablehlo.reshape %3 : (tensor<8x3x3xf64>) -> tensor<8x1x9xf64>
// CHECK-NEXT:   %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %6 = stablehlo.clamp %c_0, %0, %c : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %7 = stablehlo.reshape %6 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %8 = stablehlo.clamp %c_0, %5, %c_0 : (tensor<i64>, tensor<8xi64>, tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:   %9 = stablehlo.reshape %8 : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:   %10 = stablehlo.concatenate %7, %9, dim = 1 : (tensor<8x1xi64>, tensor<8x1xi64>) -> tensor<8x2xi64>
// CHECK-NEXT:   %11 = "stablehlo.scatter"(%arg1, %10, %4) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<8x9xf64>, tensor<8x2xi64>, tensor<8x1x9xf64>) -> tensor<8x9xf64>
// CHECK-NEXT:   return %11 : tensor<8x9xf64>
// CHECK-NEXT: }
