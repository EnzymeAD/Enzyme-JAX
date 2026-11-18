// RUN: enzymexlamlir-opt %s --arith-raise | FileCheck %s

// stablehlo.convert reads a signless integer as signed. With the nneg flag the
// sign bit is clear and a signed convert is the unsigned one; without it the
// operand is reinterpreted as unsigned first. i1 converts as a boolean.

// CHECK-LABEL: @uitofp_nneg
// CHECK: %[[C:.+]] = stablehlo.convert %arg0 : (tensor<4x6xi64>) -> tensor<4x6xf64>
// CHECK-NOT: stablehlo.bitcast_convert
// CHECK: return %[[C]]

// CHECK-LABEL: @uitofp_wide
// CHECK: %[[U:.+]] = stablehlo.bitcast_convert %arg0 : (tensor<4x6xi64>) -> tensor<4x6xui64>
// CHECK: %[[C:.+]] = stablehlo.convert %[[U]] : (tensor<4x6xui64>) -> tensor<4x6xf64>
// CHECK: return %[[C]]

// CHECK-LABEL: @uitofp_i32
// CHECK: %[[U:.+]] = stablehlo.bitcast_convert %arg0 : (tensor<4xi32>) -> tensor<4xui32>
// CHECK: %[[C:.+]] = stablehlo.convert %[[U]] : (tensor<4xui32>) -> tensor<4xf32>
// CHECK: return %[[C]]

// CHECK-LABEL: @uitofp_bool
// CHECK: %[[C:.+]] = stablehlo.convert %arg0 : (tensor<4x6xi1>) -> tensor<4x6xf64>
// CHECK-NOT: stablehlo.bitcast_convert
// CHECK: return %[[C]]

module {
  func.func @uitofp_nneg(%arg0: tensor<4x6xi64>) -> tensor<4x6xf64> {
    %0 = arith.uitofp %arg0 nneg : tensor<4x6xi64> to tensor<4x6xf64>
    return %0 : tensor<4x6xf64>
  }
  func.func @uitofp_wide(%arg0: tensor<4x6xi64>) -> tensor<4x6xf64> {
    %0 = arith.uitofp %arg0 : tensor<4x6xi64> to tensor<4x6xf64>
    return %0 : tensor<4x6xf64>
  }
  func.func @uitofp_i32(%arg0: tensor<4xi32>) -> tensor<4xf32> {
    %0 = arith.uitofp %arg0 : tensor<4xi32> to tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func @uitofp_bool(%arg0: tensor<4x6xi1>) -> tensor<4x6xf64> {
    %0 = arith.uitofp %arg0 : tensor<4x6xi1> to tensor<4x6xf64>
    return %0 : tensor<4x6xf64>
  }
}
