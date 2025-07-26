// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzymexla-stablehlo-to-triton-compatible-dialect)" %s | FileCheck %s

func.func @main1() -> tensor<8xi64> {
    %c_0 = stablehlo.constant dense<8> : tensor<8xi64>
    return %c_0 : tensor<8xi64>
}

// CHECK: func.func @main1() -> tensor<8xi64> {
// CHECK-NEXT:     %cst = arith.constant dense<8> : tensor<8xi64>
// CHECK-NEXT:     return %cst : tensor<8xi64>
// CHECK-NEXT: }

tt.func @main2() -> tensor<8xi64> {
    %c = stablehlo.constant dense<[2, 3, 4, 5, 6, 7, 8, 9]> : tensor<8xi64>
    tt.return %c : tensor<8xi64>
}

// CHECK: tt.func @main2() -> tensor<8xi64> {
// CHECK-NEXT:     %0 = tt.make_range {end = 10 : i32, start = 2 : i32} : tensor<8xi32>
// CHECK-NEXT:     %1 = arith.extui %0 : tensor<8xi32> to tensor<8xi64>
// CHECK-NEXT:     tt.return %1 : tensor<8xi64>
// CHECK-NEXT: }

tt.func @main3() -> tensor<8xi64> {
    %c = stablehlo.iota dim = 0 : tensor<8xi64>
    tt.return %c : tensor<8xi64>
}

// CHECK: tt.func @main3() -> tensor<8xi64> {
// CHECK-NEXT:     %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT:     %1 = arith.extui %0 : tensor<8xi32> to tensor<8xi64>
// CHECK-NEXT:     tt.return %1 : tensor<8xi64>
// CHECK-NEXT: }
