// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=slice_slice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @reshape_slice(%4388: tensor<2x4xi1>) -> (tensor<2x0xi1>) {
  %4396 = stablehlo.slice %4388 [0:2, 1:4:2] : (tensor<2x4xi1>) -> tensor<2x2xi1>
  %4412 = stablehlo.slice %4396 [0:2, 2:2:2] : (tensor<2x2xi1>) -> tensor<2x0xi1>
  return %4412 : tensor<2x0xi1>
}

// CHECK:  func.func @reshape_slice(%arg0: tensor<2x4xi1>) -> tensor<2x0xi1> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:2, 4:4:4] : (tensor<2x4xi1>) -> tensor<2x0xi1>
// CHECK-NEXT:    return %0 : tensor<2x0xi1>
// CHECK-NEXT:  }
