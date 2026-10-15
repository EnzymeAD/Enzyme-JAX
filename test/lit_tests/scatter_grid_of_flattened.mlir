// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=scatter_op_canon" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A scatter into a flattened buffer along a grid of it: the update goes
// into the buffer it was flattened from, not a reshape back of the
// flattening.
func.func @grid_of_flattened(%x: tensor<3x4xf64>, %u: tensor<3x1xf64>) -> tensor<12xf64> {
  %flat = stablehlo.reshape %x : (tensor<3x4xf64>) -> tensor<12xf64>
  %iota = stablehlo.iota dim = 0 : tensor<3x1xi64>
  %four = stablehlo.constant dense<4> : tensor<3x1xi64>
  %one = stablehlo.constant dense<1> : tensor<3x1xi64>
  %m = stablehlo.multiply %iota, %four : tensor<3x1xi64>
  %idx = stablehlo.add %m, %one : tensor<3x1xi64>
  %0 = "stablehlo.scatter"(%flat, %idx, %u) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%a: tensor<f64>, %b: tensor<f64>):
    stablehlo.return %b : tensor<f64>
  }) : (tensor<12xf64>, tensor<3x1xi64>, tensor<3x1xf64>) -> tensor<12xf64>
  return %0 : tensor<12xf64>
}

// CHECK:  func.func @grid_of_flattened(%arg0: tensor<3x4xf64>, %arg1: tensor<3x1xf64>) -> tensor<12xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<3x1xf64>) -> tensor<3x1xf64>
// CHECK-NEXT:   %1 = stablehlo.dynamic_update_slice %arg0, %0, %c_0, %c : (tensor<3x4xf64>, tensor<3x1xf64>, tensor<i32>, tensor<i32>) -> tensor<3x4xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %1 : (tensor<3x4xf64>) -> tensor<12xf64>
// CHECK-NEXT:   return %2 : tensor<12xf64>
// CHECK-NEXT: }
