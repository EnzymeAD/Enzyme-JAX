// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<32x32xf64>, %arg1: tensor<32x32xf64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [1] : (tensor<32x32xf64>, tensor<32x32xf64>) -> tensor<32x32xf64>
  %1 = stablehlo.reshape %0 : (tensor<32x32xf64>) -> tensor<1024xf64>
  %2 = stablehlo.slice %1 [0:1024:33] : (tensor<1024xf64>) -> tensor<32xf64>
  %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32xf64>, tensor<f64>) -> tensor<f64>
  return %3 : tensor<f64>
}

// CHECK: func.func @main1(%arg0: tensor<32x32xf64>, %arg1: tensor<32x32xf64>) -> tensor<f64> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1, 0] x [1, 0] : (tensor<32x32xf64>, tensor<32x32xf64>) -> tensor<f64>
// CHECK-NEXT:   return %0 : tensor<f64>
// CHECK-NEXT: }

func.func @main2(%arg0: tensor<4x32xf64>, %arg1: tensor<32x4xf64>) -> tensor<f64> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x4xf64>, tensor<4x32xf64>) -> tensor<32x32xf64>
  %1 = stablehlo.reshape %0 : (tensor<32x32xf64>) -> tensor<1024xf64>
  %2 = stablehlo.slice %1 [0:1024:33] : (tensor<1024xf64>) -> tensor<32xf64>
  %3 = stablehlo.reduce(%2 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<32xf64>, tensor<f64>) -> tensor<f64>
  return %3 : tensor<f64>
}

// CHECK: func.func @main2(%arg0: tensor<4x32xf64>, %arg1: tensor<32x4xf64>) -> tensor<f64> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1, 0] x [0, 1] : (tensor<32x4xf64>, tensor<4x32xf64>) -> tensor<f64>
// CHECK-NEXT:   return %0 : tensor<f64>
// CHECK-NEXT: }

func.func @main3(%arg0: tensor<4x32xf64>, %arg1: tensor<7x4xf64>) -> tensor<5xf64> {
  %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<7x4xf64>, tensor<4x32xf64>) -> tensor<7x32xf64>
  %1 = stablehlo.reshape %0 : (tensor<7x32xf64>) -> tensor<224xf64>
  %2 = stablehlo.slice %1 [66:224:33] : (tensor<224xf64>) -> tensor<5xf64>
  %3 = stablehlo.slice %1 [0:134:33] : (tensor<224xf64>) -> tensor<5xf64>
  %4 = stablehlo.add %2, %3 : tensor<5xf64>
  return %4 : tensor<5xf64>
}

// CHECK: func.func @main3(%arg0: tensor<4x32xf64>, %arg1: tensor<7x4xf64>) -> tensor<5xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:4, 0:7] : (tensor<4x32xf64>) -> tensor<4x7xf64>
// CHECK-NEXT:   %1 = stablehlo.dot_general %arg1, %0, batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<7x4xf64>, tensor<4x7xf64>) -> tensor<7xf64>
// CHECK-NEXT:   %2 = "stablehlo.reduce_window"(%1, %cst) <{window_dilations = array<i64: 2>, window_dimensions = array<i64: 2>}> ({
// CHECK-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:     %3 = stablehlo.add %arg2, %arg3 : tensor<f64>
// CHECK-NEXT:     stablehlo.return %3 : tensor<f64>
// CHECK-NEXT:   }) : (tensor<7xf64>, tensor<f64>) -> tensor<5xf64>
// CHECK-NEXT:   return %2 : tensor<5xf64>
// CHECK-NEXT: }

func.func @main4(%arg0: tensor<4x32xf64>, %arg1: tensor<7x4xf64>) -> tensor<3xf64> {
  %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<7x4xf64>, tensor<4x32xf64>) -> tensor<7x32xf64>
  %1 = stablehlo.reshape %0 : (tensor<7x32xf64>) -> tensor<224xf64>
  %2 = stablehlo.slice %1 [66:165:33] : (tensor<224xf64>) -> tensor<3xf64>
  return %2 : tensor<3xf64>
}

// CHECK: func.func @main4(%arg0: tensor<4x32xf64>, %arg1: tensor<7x4xf64>) -> tensor<3xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg1 [2:5, 0:4] : (tensor<7x4xf64>) -> tensor<3x4xf64>
// CHECK-NEXT:   %1 = stablehlo.slice %arg0 [0:4, 2:5] : (tensor<4x32xf64>) -> tensor<4x3xf64>
// CHECK-NEXT:   %2 = stablehlo.dot_general %0, %1, batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<3x4xf64>, tensor<4x3xf64>) -> tensor<3xf64>
// CHECK-NEXT:   return %2 : tensor<3xf64>
// CHECK-NEXT: }

func.func @main5(%arg0: tensor<4x32xf64>, %arg1: tensor<7x4xf64>) -> tensor<3xf64> {
  %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<7x4xf64>, tensor<4x32xf64>) -> tensor<7x32xf64>
  %1 = stablehlo.reshape %0 : (tensor<7x32xf64>) -> tensor<224xf64>
  %2 = stablehlo.slice %1 [33:198:66] : (tensor<224xf64>) -> tensor<3xf64>
  %3 = stablehlo.slice %1 [66:224:66] : (tensor<224xf64>) -> tensor<3xf64>
  %4 = stablehlo.add %2, %3 : tensor<3xf64>
  return %4 : tensor<3xf64>
}

// CHECK: func.func @main5(%arg0: tensor<4x32xf64>, %arg1: tensor<7x4xf64>) -> tensor<3xf64> {
// CHECK-NEXT:   %0 = stablehlo.slice %arg0 [0:4, 0:7] : (tensor<4x32xf64>) -> tensor<4x7xf64>
// CHECK-NEXT:   %1 = stablehlo.dot_general %arg1, %0, batching_dims = [0] x [1], contracting_dims = [1] x [0] : (tensor<7x4xf64>, tensor<4x7xf64>) -> tensor<7xf64>
// CHECK-NEXT:   %2 = stablehlo.slice %1 [1:6:2] : (tensor<7xf64>) -> tensor<3xf64>
// CHECK-NEXT:   %3 = stablehlo.slice %1 [2:7:2] : (tensor<7xf64>) -> tensor<3xf64>
// CHECK-NEXT:   %4 = stablehlo.add %2, %3 : tensor<3xf64>
// CHECK-NEXT:   return %4 : tensor<3xf64>
// CHECK-NEXT: }

func.func @fail1(%arg0: tensor<4x32xf64>, %arg1: tensor<7x4xf64>) -> tensor<5xf64> {
  // CHECK: stablehlo.dot_general
  %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<7x4xf64>, tensor<4x32xf64>) -> tensor<7x32xf64>
  %1 = stablehlo.reshape %0 : (tensor<7x32xf64>) -> tensor<224xf64>
  %2 = stablehlo.slice %1 [67:224:33] : (tensor<224xf64>) -> tensor<5xf64>
  return %2 : tensor<5xf64>
}

func.func @fail2(%arg0: tensor<128xf32>, %arg1: tensor<79x61xf32>) -> tensor<128x79x61xf32> {
  // CHECK: stablehlo.dot_general
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [] x [], precision = [DEFAULT, DEFAULT] : (tensor<128xf32>, tensor<79x61xf32>) -> tensor<128x79x61xf32>
  return %0 : tensor<128x79x61xf32>
}
