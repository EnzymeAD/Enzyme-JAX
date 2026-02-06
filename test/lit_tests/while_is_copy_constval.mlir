// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

func.func @main1(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
  %cst = stablehlo.constant dense<5.000000e+00> : tensor<1xf32>
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  %c_1 = stablehlo.constant dense<64> : tensor<i64>
  %c_2 = stablehlo.constant dense<1> : tensor<i64>
  %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<64xf32>
  %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_4 = %cst_3) : tensor<i64>, tensor<64xf32>
  cond {
    %4 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %c_2, %iterArg : tensor<i64>
    %5 = stablehlo.convert %4 : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.subtract %5, %c : tensor<i32>
    %7 = stablehlo.dynamic_update_slice %iterArg_4, %cst, %6 : (tensor<64xf32>, tensor<1xf32>, tensor<i32>) -> tensor<64xf32>
    stablehlo.return %4, %7 : tensor<i64>, tensor<64xf32>
  }
  %1 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %2 = stablehlo.dot_general %arg0, %1, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %3 = stablehlo.add %2, %0#1 : tensor<64xf32>
  return %3 : tensor<64xf32>
}

// CHECK: func.func @main1(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<5.000000e+00> : tensor<64xf32>
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-NEXT:   %2 = stablehlo.add %1, %cst : tensor<64xf32>
// CHECK-NEXT:   return %2 : tensor<64xf32>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<32x1xf32>) -> tensor<64x32xf32> {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_zero = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<0> : tensor<i64>
  %c_1 = stablehlo.constant dense<32> : tensor<i64>
  %c_2 = stablehlo.constant dense<1> : tensor<i64>
  %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<64x32xf32>
  %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_4 = %cst_3) : tensor<i64>, tensor<64x32xf32>
  cond {
    %4 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %c_2, %iterArg : tensor<i64>
    %5 = stablehlo.convert %4 : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.subtract %5, %c : tensor<i32>
    %7 = stablehlo.dynamic_update_slice %iterArg_4, %arg2, %c_zero, %6 : (tensor<64x32xf32>, tensor<32x1xf32>, tensor<i32>, tensor<i32>) -> tensor<64x32xf32>
    stablehlo.return %4, %7 : tensor<i64>, tensor<64x32xf32>
  }
  %1 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %2 = stablehlo.dot_general %arg0, %1, contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
  %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<64xf32>) -> tensor<64x32xf32>
  %4 = stablehlo.add %3, %0#1 : tensor<64x32xf32>
  return %4 : tensor<64x32xf32>
}

// CHECK: func.func @main2(%arg0: tensor<64x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<32x1xf32>) -> tensor<64x32xf32> {
// CHECK-NEXT:   %[[a1:.+]] = stablehlo.broadcast_in_dim %[[a0]], dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x32xf32>
// CHECK-NEXT:   %[[a2:.+]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-NEXT:   %[[a3:.+]] = stablehlo.dot_general %arg0, %[[a2]], contracting_dims = [1] x [0] : (tensor<64x64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-NEXT:   %[[a4:.+]] = stablehlo.broadcast_in_dim %[[a3]], dims = [0] : (tensor<64xf32>) -> tensor<64x32xf32>
// CHECK-NEXT:   %[[a5:.+]] = stablehlo.slice %[[a4]] [0:32, 0:32] : (tensor<64x32xf32>) -> tensor<32x32xf32>
// CHECK-NEXT:   %[[a6:.+]] = stablehlo.add %[[a5]], %[[a1]] : tensor<32x32xf32>
// CHECK-NEXT:   %[[a7:.+]] = stablehlo.slice %[[a4]] [32:64, 0:32] : (tensor<64x32xf32>) -> tensor<32x32xf32>
// CHECK-NEXT:   %[[a8:.+]] = stablehlo.concatenate %[[a6]], %[[a7]], dim = 0 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
// CHECK-NEXT:   return %[[a8]] : tensor<64x32xf32>
// CHECK-NEXT: }
