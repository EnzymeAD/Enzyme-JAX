// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=concat_reshape_gather},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

module @reactant_batched... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<5x4x3xf64>, %arg1: tensor<7xi64>) -> tensor<5x4x7xf64> {
    %c = stablehlo.constant dense<3> : tensor<i64>
    %c_0 = stablehlo.constant dense<2> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<7xi64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x4x3xf64>) -> tensor<3x4x5xf64>
    %1 = stablehlo.subtract %arg1, %c_2 : tensor<7xi64>
    %2 = stablehlo.reshape %1 : (tensor<7xi64>) -> tensor<7x1xi64>
    %idx = stablehlo.pad %2, %c_3, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
    %3 = "stablehlo.gather"(%0, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 5>}> : (tensor<3x4x5xf64>, tensor<7x2xi64>) -> tensor<7x5xf64>
    %4 = stablehlo.reshape %3 : (tensor<7x5xf64>) -> tensor<7x1x5xf64>
    %5 = stablehlo.pad %2, %c_1, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
    %6 = "stablehlo.gather"(%0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 5>}> : (tensor<3x4x5xf64>, tensor<7x2xi64>) -> tensor<7x5xf64>
    %7 = stablehlo.reshape %6 : (tensor<7x5xf64>) -> tensor<7x1x5xf64>
    %8 = stablehlo.pad %2, %c_0, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
    %9 = "stablehlo.gather"(%0, %8) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 5>}> : (tensor<3x4x5xf64>, tensor<7x2xi64>) -> tensor<7x5xf64>
    %10 = stablehlo.reshape %9 : (tensor<7x5xf64>) -> tensor<7x1x5xf64>
    %11 = stablehlo.pad %2, %c, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
    %12 = "stablehlo.gather"(%0, %11) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 5>}> : (tensor<3x4x5xf64>, tensor<7x2xi64>) -> tensor<7x5xf64>
    %13 = stablehlo.reshape %12 : (tensor<7x5xf64>) -> tensor<7x1x5xf64>
    %14 = stablehlo.concatenate %4, %7, %10, %13, dim = 1 : (tensor<7x1x5xf64>, tensor<7x1x5xf64>, tensor<7x1x5xf64>, tensor<7x1x5xf64>) -> tensor<7x4x5xf64>
    %15 = stablehlo.transpose %14, dims = [2, 1, 0] : (tensor<7x4x5xf64>) -> tensor<5x4x7xf64>
    return %15 : tensor<5x4x7xf64>
  }
}

// CHECK: func.func private [[FN_NAME:@.*]](%arg0: tensor<3x4x5xf64>, %arg1: tensor<7x2xi64>) -> tensor<7x5xf64> {
// CHECK-NEXT:     %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 5>}> : (tensor<3x4x5xf64>, tensor<7x2xi64>) -> tensor<7x5xf64>
// CHECK-NEXT:     return %0 : tensor<7x5xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @main(%arg0: tensor<5x4x3xf64>, %arg1: tensor<7xi64>) -> tensor<5x4x7xf64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<1> : tensor<7xi64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x4x3xf64>) -> tensor<3x4x5xf64>
// CHECK-NEXT:     %1 = stablehlo.subtract %arg1, %c_2 : tensor<7xi64>
// CHECK-NEXT:     %2 = stablehlo.reshape %1 : (tensor<7xi64>) -> tensor<7x1xi64>
// CHECK-NEXT:     %3 = stablehlo.pad %2, %c_3, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
// CHECK-NEXT:     %4 = stablehlo.pad %2, %c_1, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
// CHECK-NEXT:     %5 = stablehlo.pad %2, %c_0, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
// CHECK-NEXT:     %6 = stablehlo.pad %2, %c, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
// CHECK-NEXT:     %7 = stablehlo.reshape %0 : (tensor<3x4x5xf64>) -> tensor<1x3x4x5xf64>
// CHECK-NEXT:     %8 = stablehlo.reshape %0 : (tensor<3x4x5xf64>) -> tensor<1x3x4x5xf64>
// CHECK-NEXT:     %9 = stablehlo.reshape %0 : (tensor<3x4x5xf64>) -> tensor<1x3x4x5xf64>
// CHECK-NEXT:     %10 = stablehlo.reshape %0 : (tensor<3x4x5xf64>) -> tensor<1x3x4x5xf64>
// CHECK-NEXT:     %11 = stablehlo.concatenate %7, %8, %9, %10, dim = 0 : (tensor<1x3x4x5xf64>, tensor<1x3x4x5xf64>, tensor<1x3x4x5xf64>, tensor<1x3x4x5xf64>) -> tensor<4x3x4x5xf64>
// CHECK-NEXT:     %12 = stablehlo.reshape %3 : (tensor<7x2xi64>) -> tensor<1x7x2xi64>
// CHECK-NEXT:     %13 = stablehlo.reshape %4 : (tensor<7x2xi64>) -> tensor<1x7x2xi64>
// CHECK-NEXT:     %14 = stablehlo.reshape %5 : (tensor<7x2xi64>) -> tensor<1x7x2xi64>
// CHECK-NEXT:     %15 = stablehlo.reshape %6 : (tensor<7x2xi64>) -> tensor<1x7x2xi64>
// CHECK-NEXT:     %16 = stablehlo.concatenate %12, %13, %14, %15, dim = 0 : (tensor<1x7x2xi64>, tensor<1x7x2xi64>, tensor<1x7x2xi64>, tensor<1x7x2xi64>) -> tensor<4x7x2xi64>
// CHECK-NEXT:     %17 = enzyme.batch [[FN_NAME]](%11, %16) {batch_shape = array<i64: 4>} : (tensor<4x3x4x5xf64>, tensor<4x7x2xi64>) -> tensor<4x7x5xf64>
// CHECK-NEXT:     %18 = stablehlo.transpose %17, dims = [1, 0, 2] : (tensor<4x7x5xf64>) -> tensor<7x4x5xf64>
// CHECK-NEXT:     %19 = stablehlo.transpose %18, dims = [2, 1, 0] : (tensor<7x4x5xf64>) -> tensor<5x4x7xf64>
// CHECK-NEXT:     return %19 : tensor<5x4x7xf64>
// CHECK-NEXT:   }
