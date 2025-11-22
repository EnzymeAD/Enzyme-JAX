// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_symmetric_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @pass1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<2x2xf32>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK: func.func @pass1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.add %arg0, %0 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<2x2xf32>
// CHECK-NEXT:   return %1 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @pass2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.add %0, %arg0 : tensor<2x2xf32>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK: func.func @pass2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.add %0, %arg0 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<2x2xf32>
// CHECK-NEXT:   return %1 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @pass3() -> tensor<3x3xf32> {
  %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
  %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
  %2 = stablehlo.transpose %0, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
  %3 = stablehlo.transpose %1, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
  %4 = stablehlo.add %2, %3 : tensor<3x3xf32>
  return %4 : tensor<3x3xf32>
}

// CHECK: func.func @pass3() -> tensor<3x3xf32> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
// CHECK-NEXT:   %0 = stablehlo.broadcast_in_dim %cst_0, dims = [] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<f32>) -> tensor<3x3xf32>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %cst, dims = [] {enzymexla.symmetric_matrix = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<f32>) -> tensor<3x3xf32>
// CHECK-NEXT:   %2 = stablehlo.add %0, %1 : tensor<3x3xf32>
// CHECK-NEXT:   return %2 : tensor<3x3xf32>
// CHECK-NEXT: }

func.func @pass4(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK: func.func @pass4(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] {enzymexla.guaranteed_symmetric = true} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   return %1 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @pass5(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %1 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK: func.func @pass5(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.dot_general %arg0, %arg0, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] {enzymexla.guaranteed_symmetric = true} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   return %0 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @fail1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.subtract %arg0, %0 : tensor<2x2xf32>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK: func.func @fail1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.subtract %arg0, %0 {enzymexla.symmetric_matrix = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2x2xf32>
// CHECK-NEXT:   %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   return %2 : tensor<2x2xf32>
// CHECK-NEXT: }

func.func @fail2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %2 : tensor<2x2xf32>
}

// CHECK: func.func @fail2(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] {enzymexla.guaranteed_symmetric = false} : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   return %2 : tensor<2x2xf32>
// CHECK-NEXT: }
