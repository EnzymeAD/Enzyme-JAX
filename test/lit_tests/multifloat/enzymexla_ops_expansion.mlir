// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple expansion-size=2" %s | FileCheck %s

// CHECK-LABEL: func.func @wrap
func.func @wrap(%arg0: tensor<4x4xf64>) -> tensor<6x4xf64> {
  // CHECK: "enzymexla.wrap"
  // CHECK: "enzymexla.wrap"
  %0 = "enzymexla.wrap"(%arg0) <{lhs = 1 : i64, rhs = 1 : i64, dimension = 0 : i64}> : (tensor<4x4xf64>) -> tensor<6x4xf64>
  return %0 : tensor<6x4xf64>
}

// CHECK-LABEL: func.func @rotate
func.func @rotate(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
  // CHECK: "enzymexla.rotate"
  // CHECK: "enzymexla.rotate"
  %0 = "enzymexla.rotate"(%arg0) <{amount = 1 : i32, dimension = 0 : i32}> : (tensor<4x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

// CHECK-LABEL: func.func @extend
func.func @extend(%arg0: tensor<4x4xf64>) -> tensor<6x4xf64> {
  // CHECK: "enzymexla.extend"
  // CHECK: "enzymexla.extend"
  %0 = "enzymexla.extend"(%arg0) <{lhs = 1 : i64, rhs = 1 : i64, dimension = 0 : i64}> : (tensor<4x4xf64>) -> tensor<6x4xf64>
  return %0 : tensor<6x4xf64>
}

// CHECK-LABEL: func.func @update_without_corners
func.func @update_without_corners(%arg0: tensor<4x4xf64>, %arg1: tensor<2x2xf64>) -> tensor<4x4xf64> {
  // CHECK: "enzymexla.update_without_corners"
  // CHECK: "enzymexla.update_without_corners"
  %0 = "enzymexla.update_without_corners"(%arg0, %arg1) <{dimensionX = 0 : i64, x1 = 1 : i64, x2 = 3 : i64, dimensionY = 1 : i64, y1 = 1 : i64, y2 = 3 : i64}> : (tensor<4x4xf64>, tensor<2x2xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}
