// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=strided_slice_to_reshape" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Every 36th element of a 972-vector seen as 972x1: column 5 of its 27x36 view.
func.func @column(%x: tensor<972x1xf64>) -> tensor<27x1xf64> {
  %0 = stablehlo.slice %x [5:942:36, 0:1] : (tensor<972x1xf64>) -> tensor<27x1xf64>
  return %0 : tensor<27x1xf64>
}

// CHECK:  func.func @column(%arg0: tensor<972x1xf64>) -> tensor<27x1xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<972x1xf64>) -> tensor<27x36x1xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:27, 5:6, 0:1] : (tensor<27x36x1xf64>) -> tensor<27x1x1xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<27x1x1xf64>) -> tensor<27x1xf64>
// CHECK-NEXT:    return %2 : tensor<27x1xf64>
// CHECK-NEXT:  }

// Fewer rows than the split has: the column is cut short.
func.func @short(%x: tensor<18xf64>) -> tensor<4xf64> {
  %0 = stablehlo.slice %x [2:12:3] : (tensor<18xf64>) -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// CHECK:  func.func @short(%arg0: tensor<18xf64>) -> tensor<4xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<18xf64>) -> tensor<6x3xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:4, 2:3] : (tensor<6x3xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:    return %2 : tensor<4xf64>
// CHECK-NEXT:  }

// A stride that does not divide the extent, or a start past the stride: unchanged.
func.func @odd(%x: tensor<20xf64>, %y: tensor<18xf64>) -> (tensor<7xf64>, tensor<5xf64>) {
  %0 = stablehlo.slice %x [0:20:3] : (tensor<20xf64>) -> tensor<7xf64>
  %1 = stablehlo.slice %y [4:18:3] : (tensor<18xf64>) -> tensor<5xf64>
  return %0, %1 : tensor<7xf64>, tensor<5xf64>
}

// CHECK:  func.func @odd
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:20:3] : (tensor<20xf64>) -> tensor<7xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [4:18:3] : (tensor<18xf64>) -> tensor<5xf64>
// CHECK-NEXT:    return %0, %1 : tensor<7xf64>, tensor<5xf64>
// CHECK-NEXT:  }
