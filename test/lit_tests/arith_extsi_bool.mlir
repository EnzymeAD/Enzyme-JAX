// RUN: enzymexlamlir-opt %s --arith-raise | FileCheck %s

// A sign extension of i1 means true -> -1; stablehlo.convert reads i1 as
// boolean (true -> 1), so the raised form negates the conversion.

// CHECK-LABEL: @extsi_bool
// CHECK: %[[C:.+]] = stablehlo.convert %arg0 : (tensor<4xi1>) -> tensor<4xi32>
// CHECK: %[[N:.+]] = stablehlo.negate %[[C]] : tensor<4xi32>
// CHECK: return %[[N]]

// CHECK-LABEL: @extsi_wide
// CHECK: stablehlo.convert
// CHECK-NOT: stablehlo.negate

// CHECK-LABEL: @extui_bool
// CHECK: stablehlo.convert
// CHECK-NOT: stablehlo.negate

module {
  func.func @extsi_bool(%arg0: tensor<4xi1>) -> tensor<4xi32> {
    %0 = arith.extsi %arg0 : tensor<4xi1> to tensor<4xi32>
    return %0 : tensor<4xi32>
  }
  func.func @extsi_wide(%arg0: tensor<4xi8>) -> tensor<4xi32> {
    %0 = arith.extsi %arg0 : tensor<4xi8> to tensor<4xi32>
    return %0 : tensor<4xi32>
  }
  func.func @extui_bool(%arg0: tensor<4xi1>) -> tensor<4xi32> {
    %0 = arith.extui %arg0 : tensor<4xi1> to tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}
