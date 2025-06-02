// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @one() -> (tensor<3x1520x3056xi1>, tensor<1520xi1>) {
    %c_286 = stablehlo.constant dense<true> : tensor<1519xi1>
    %c_291 = stablehlo.constant dense<false> : tensor<i1>
    %60 = stablehlo.pad %c_286, %c_291, low = [1], high = [0], interior = [0] : (tensor<1519xi1>, tensor<i1>) -> tensor<1520xi1>
    %168 = stablehlo.broadcast_in_dim %60, dims = [1] : (tensor<1520xi1>) -> tensor<3x1520x3056xi1>
    return %168, %60 : tensor<3x1520x3056xi1>, tensor<1520xi1>
}

// CHECK:  func.func @one() -> (tensor<3x1520x3056xi1>, tensor<1520xi1>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<3x1519x3056xi1>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<true> : tensor<1519xi1>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %0 = stablehlo.pad %c_0, %c_1, low = [1], high = [0], interior = [0] : (tensor<1519xi1>, tensor<i1>) -> tensor<1520xi1>
// CHECK-NEXT:    %1 = stablehlo.pad %c, %c_1, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<3x1519x3056xi1>, tensor<i1>) -> tensor<3x1520x3056xi1>
// CHECK-NEXT:    return %1, %0 : tensor<3x1520x3056xi1>, tensor<1520xi1>
// CHECK-NEXT:  }
