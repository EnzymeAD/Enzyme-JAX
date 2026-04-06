// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {

  // convert(min(convert(x), convert(y))) -> min(x, y)
  func.func @convert_min(%a : tensor<4xf32>, %b : tensor<4xf32>) -> tensor<4xf32> {
    %ca = stablehlo.convert %a : (tensor<4xf32>) -> tensor<4xf64>
    %cb = stablehlo.convert %b : (tensor<4xf32>) -> tensor<4xf64>
    %min = stablehlo.minimum %ca, %cb : tensor<4xf64>
    %res = stablehlo.convert %min : (tensor<4xf64>) -> tensor<4xf32>
    return %res : tensor<4xf32>
  }

  // convert(max(convert(x), convert(y))) -> max(x, y)
  func.func @convert_max(%a : tensor<2x3xbf16>, %b : tensor<2x3xbf16>) -> tensor<2x3xbf16> {
    %ca = stablehlo.convert %a : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    %cb = stablehlo.convert %b : (tensor<2x3xbf16>) -> tensor<2x3xf32>
    %max = stablehlo.maximum %ca, %cb : tensor<2x3xf32>
    %res = stablehlo.convert %max : (tensor<2x3xf32>) -> tensor<2x3xbf16>
    return %res : tensor<2x3xbf16>
  }

  // No match: integer types
  func.func @convert_max_int_no_match(%a : tensor<2x3xi16>, %b : tensor<2x3xi16>) -> tensor<2x3xi16> {
    %ca = stablehlo.convert %a : (tensor<2x3xi16>) -> tensor<2x3xi32>
    %cb = stablehlo.convert %b : (tensor<2x3xi16>) -> tensor<2x3xi32>
    %max = stablehlo.maximum %ca, %cb : tensor<2x3xi32>
    %res = stablehlo.convert %max : (tensor<2x3xi32>) -> tensor<2x3xi16>
    return %res : tensor<2x3xi16>
  }

  // No match: x and y types differ from result type
  func.func @convert_min_no_match(%a : tensor<4xf32>, %b : tensor<4xf64>) -> tensor<4xf32> {
    %ca = stablehlo.convert %a : (tensor<4xf32>) -> tensor<4xf64>
    %min = stablehlo.minimum %ca, %b : tensor<4xf64>
    %res = stablehlo.convert %min : (tensor<4xf64>) -> tensor<4xf32>
    return %res : tensor<4xf32>
  }

}

// CHECK-LABEL: func.func @convert_min(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NEXT:    %0 = stablehlo.minimum %arg0, %arg1 : tensor<4xf32>
// CHECK-NEXT:    return %0 : tensor<4xf32>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @convert_max(%arg0: tensor<2x3xbf16>, %arg1: tensor<2x3xbf16>) -> tensor<2x3xbf16> {
// CHECK-NEXT:    %0 = stablehlo.maximum %arg0, %arg1 : tensor<2x3xbf16>
// CHECK-NEXT:    return %0 : tensor<2x3xbf16>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @convert_max_int_no_match(%arg0: tensor<2x3xi16>, %arg1: tensor<2x3xi16>) -> tensor<2x3xi16> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<2x3xi16>) -> tensor<2x3xi32>
// CHECK-NEXT:    %1 = stablehlo.convert %arg1 : (tensor<2x3xi16>) -> tensor<2x3xi32>
// CHECK-NEXT:    %2 = stablehlo.maximum %0, %1 : tensor<2x3xi32>
// CHECK-NEXT:    %3 = stablehlo.convert %2 : (tensor<2x3xi32>) -> tensor<2x3xi16>
// CHECK-NEXT:    return %3 : tensor<2x3xi16>
// CHECK-NEXT:  }

// CHECK-LABEL: func.func @convert_min_no_match(%arg0: tensor<4xf32>, %arg1: tensor<4xf64>) -> tensor<4xf32> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xf64>
// CHECK-NEXT:    %1 = stablehlo.minimum %0, %arg1 : tensor<4xf64>
// CHECK-NEXT:    %2 = stablehlo.convert %1 : (tensor<4xf64>) -> tensor<4xf32>
// CHECK-NEXT:    return %2 : tensor<4xf32>
// CHECK-NEXT:  }
