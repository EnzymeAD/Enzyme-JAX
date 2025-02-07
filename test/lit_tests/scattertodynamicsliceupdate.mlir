// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%prev : tensor<1x1x8192x16x256xbf16>, %update : tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16> {
    %163 = stablehlo.constant dense<0> : tensor<1xi32>
    %524 = "stablehlo.scatter"(%prev, %163, %update) ({
    ^bb0(%arg113: tensor<bf16>, %arg114: tensor<bf16>):
      stablehlo.return %arg114 : tensor<bf16>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2, 3, 4], scatter_dims_to_operand_dims = [2]>, unique_indices = true} : (tensor<1x1x8192x16x256xbf16>, tensor<1xi32>, tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16>
    return %524 : tensor<1x1x8192x16x256xbf16>
  }

  func.func @main2(%prev : tensor<1x1x8192x16x256xbf16>, %update : tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16> {
    %163 = stablehlo.constant dense<2048> : tensor<1xi32>
    %524 = "stablehlo.scatter"(%prev, %163, %update) ({
    ^bb0(%arg113: tensor<bf16>, %arg114: tensor<bf16>):
      stablehlo.return %arg114 : tensor<bf16>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2, 3, 4], scatter_dims_to_operand_dims = [2]>, unique_indices = true} : (tensor<1x1x8192x16x256xbf16>, tensor<1xi32>, tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16>
    return %524 : tensor<1x1x8192x16x256xbf16>
  }

  func.func @main3(%prev : tensor<1x1x8192x16x256xbf16>, %update : tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16> {
    %163 = stablehlo.constant dense<0> : tensor<1xi64>
    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<bf16>
    %524 = "stablehlo.scatter"(%prev, %163, %update) ({
    ^bb0(%arg113: tensor<bf16>, %arg114: tensor<bf16>):
      stablehlo.return %cst_1 : tensor<bf16>
    }) {indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2, 3, 4], scatter_dims_to_operand_dims = [2]>, unique_indices = true} : (tensor<1x1x8192x16x256xbf16>, tensor<1xi64>, tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16>
    return %524 : tensor<1x1x8192x16x256xbf16>
  }
}

// CHECK:  func.func @main(%arg0: tensor<1x1x8192x16x256xbf16>, %arg1: tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:1, 2048:8192, 0:16, 0:256] : (tensor<1x1x8192x16x256xbf16>) -> tensor<1x1x6144x16x256xbf16>
// CHECK-NEXT:    %1 = stablehlo.concatenate %arg1, %0, dim = 2 : (tensor<1x1x2048x16x256xbf16>, tensor<1x1x6144x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16>
// CHECK-NEXT:    return %1 : tensor<1x1x8192x16x256xbf16>
// CHECK-NEXT:  }

// CHECK:  func.func @main2(%arg0: tensor<1x1x8192x16x256xbf16>, %arg1: tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:1, 0:2048, 0:16, 0:256] : (tensor<1x1x8192x16x256xbf16>) -> tensor<1x1x2048x16x256xbf16>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [0:1, 0:1, 4096:8192, 0:16, 0:256] : (tensor<1x1x8192x16x256xbf16>) -> tensor<1x1x4096x16x256xbf16>
// CHECK-NEXT:    %2 = stablehlo.concatenate %0, %arg1, %1, dim = 2 : (tensor<1x1x2048x16x256xbf16>, tensor<1x1x2048x16x256xbf16>, tensor<1x1x4096x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16>
// CHECK-NEXT:    return %2 : tensor<1x1x8192x16x256xbf16>
// CHECK-NEXT:  }

// CHECK:  func.func @main3(%arg0: tensor<1x1x8192x16x256xbf16>, %arg1: tensor<1x1x2048x16x256xbf16>) -> tensor<1x1x8192x16x256xbf16> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<5.000000e+00> : tensor<bf16>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:1, 2048:8192, 0:16, 0:256] : (tensor<1x1x8192x16x256xbf16>) -> tensor<1x1x6144x16x256xbf16>
// CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0, 2048, 0, 0], high = [0, 0, 0, 0, 0], interior = [0, 0, 0, 0, 0] : (tensor<1x1x6144x16x256xbf16>, tensor<bf16>) -> tensor<1x1x8192x16x256xbf16>
// CHECK-NEXT:    return %1 : tensor<1x1x8192x16x256xbf16>
// CHECK-NEXT:  }
