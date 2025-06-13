// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @gather_elementwise1(%arg0: tensor<32x1024xf32>, %arg1: tensor<32x1024xf32>, %arg2: tensor<5xi64>) -> tensor<32x5xf32> {
    %c = stablehlo.constant dense<1> : tensor<1x5xi64>
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<32x1024xf32>
    %1 = stablehlo.reshape %arg2 : (tensor<5xi64>) -> tensor<1x5xi64>
    %2 = stablehlo.subtract %1, %c : tensor<1x5xi64>
    %3 = "stablehlo.gather"(%0, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1]>, indices_are_sorted = false, slice_sizes = array<i64: 32, 1>}> : (tensor<32x1024xf32>, tensor<1x5xi64>) -> tensor<32x5xf32>
    return %3 : tensor<32x5xf32>
}

// CHECK: func.func @gather_elementwise1(%arg0: tensor<32x1024xf32>, %arg1: tensor<32x1024xf32>, %arg2: tensor<5xi64>) -> tensor<32x5xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<1x5xi64>
// CHECK-NEXT:     %0 = stablehlo.reshape %arg2 : (tensor<5xi64>) -> tensor<1x5xi64>
// CHECK-NEXT:     %1 = stablehlo.subtract %0, %c : tensor<1x5xi64>
// CHECK-NEXT:     %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1]>, indices_are_sorted = false, slice_sizes = array<i64: 32, 1>}> : (tensor<32x1024xf32>, tensor<1x5xi64>) -> tensor<32x5xf32>
// CHECK-NEXT:     %3 = "stablehlo.gather"(%arg1, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1]>, indices_are_sorted = false, slice_sizes = array<i64: 32, 1>}> : (tensor<32x1024xf32>, tensor<1x5xi64>) -> tensor<32x5xf32>
// CHECK-NEXT:     %4 = stablehlo.multiply %2, %3 : tensor<32x5xf32>
// CHECK-NEXT:     return %4 : tensor<32x5xf32>
// CHECK-NEXT: }

func.func @gather_elementwise2(%arg0: tensor<32x1024xf32>, %arg1: tensor<5xi64>) -> tensor<32x5xf16> {
    %c = stablehlo.constant dense<1> : tensor<5xi64>
    %0 = stablehlo.convert %arg0 : (tensor<32x1024xf32>) -> tensor<32x1024xf16>
    %1 = stablehlo.subtract %arg1, %c : tensor<5xi64>
    %2 = stablehlo.reshape %1 : (tensor<5xi64>) -> tensor<1x5xi64>
    %3 = "stablehlo.gather"(%0, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1]>, indices_are_sorted = false, slice_sizes = array<i64: 32, 1>}> : (tensor<32x1024xf16>, tensor<1x5xi64>) -> tensor<32x5xf16>
    return %3 : tensor<32x5xf16>
}

// CHECK: func.func @gather_elementwise2(%arg0: tensor<32x1024xf32>, %arg1: tensor<5xi64>) -> tensor<32x5xf16> {
// CHECK-NEXT:     %c = stablehlo.constant dense<1> : tensor<5xi64>
// CHECK-NEXT:     %0 = stablehlo.subtract %arg1, %c : tensor<5xi64>
// CHECK-NEXT:     %1 = stablehlo.reshape %0 : (tensor<5xi64>) -> tensor<1x5xi64>
// CHECK-NEXT:     %2 = "stablehlo.gather"(%arg0, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1]>, indices_are_sorted = false, slice_sizes = array<i64: 32, 1>}> : (tensor<32x1024xf32>, tensor<1x5xi64>) -> tensor<32x5xf32>
// CHECK-NEXT:     %3 = stablehlo.convert %2 : (tensor<32x5xf32>) -> tensor<32x5xf16>
// CHECK-NEXT:     return %3 : tensor<32x5xf16>
// CHECK-NEXT: }
