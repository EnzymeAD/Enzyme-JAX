// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=and_pad_pad" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s

  func.func @mid1() -> tensor<1520xi1> {
    %c_291 = stablehlo.constant dense<false> : tensor<i1>
    %c = stablehlo.constant dense<true> : tensor<1517xi1>

    %c_71 = stablehlo.constant dense<false> : tensor<2xi1>
    %c_72 = stablehlo.constant dense<true> : tensor<i1>

    // false 0-3, true 3-1520
    %43 = stablehlo.pad %c, %c_291, low = [3], high = [0], interior = [0] : (tensor<1517xi1>, tensor<i1>) -> tensor<1520xi1>

    // true 0 - 1518, false 1518-1520
    %45 = stablehlo.pad %c_71, %c_72, low = [1518], high = [0], interior = [0] : (tensor<2xi1>, tensor<i1>) -> tensor<1520xi1>

    %77 = stablehlo.and %43, %45 : tensor<1520xi1>

    return %77 : tensor<1520xi1>
  }

// CHECK:  func.func @mid1() -> tensor<1520xi1> {
// CHECK-NEXT:    %c = stablehlo.constant dense<true> : tensor<1515xi1>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %0 = stablehlo.pad %c, %c_0, low = [3], high = [2], interior = [0] : (tensor<1515xi1>, tensor<i1>) -> tensor<1520xi1>
// CHECK-NEXT:    return %0 : tensor<1520xi1>
// CHECK-NEXT:  }