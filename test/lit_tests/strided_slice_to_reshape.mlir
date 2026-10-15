// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=strided_slice_to_reshape" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Every 36th element of a 972-vector seen as 972x1, from 5 and from 6:
// columns 5 and 6 of its 27x36 view, which both take.
func.func @columns(%x: tensor<972x1xf64>) -> (tensor<27x1xf64>, tensor<27x1xf64>) {
  %0 = stablehlo.slice %x [5:942:36, 0:1] : (tensor<972x1xf64>) -> tensor<27x1xf64>
  %1 = stablehlo.slice %x [6:943:36, 0:1] : (tensor<972x1xf64>) -> tensor<27x1xf64>
  return %0, %1 : tensor<27x1xf64>, tensor<27x1xf64>
}

// CHECK:  func.func @columns(%arg0: tensor<972x1xf64>) -> (tensor<27x1xf64>, tensor<27x1xf64>) {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<972x1xf64>) -> tensor<27x36x1xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:27, 5:6, 0:1] : (tensor<27x36x1xf64>) -> tensor<27x1x1xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<27x1x1xf64>) -> tensor<27x1xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %0 [0:27, 6:7, 0:1] : (tensor<27x36x1xf64>) -> tensor<27x1x1xf64>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<27x1x1xf64>) -> tensor<27x1xf64>
// CHECK-NEXT:    return %2, %4 : tensor<27x1xf64>, tensor<27x1xf64>
// CHECK-NEXT:  }

// Fewer rows than the split has: the columns are cut short.
func.func @short(%x: tensor<18xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %0 = stablehlo.slice %x [2:12:3] : (tensor<18xf64>) -> tensor<4xf64>
  %1 = stablehlo.slice %x [0:10:3] : (tensor<18xf64>) -> tensor<4xf64>
  return %0, %1 : tensor<4xf64>, tensor<4xf64>
}

// CHECK:  func.func @short(%arg0: tensor<18xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<18xf64>) -> tensor<6x3xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %0 [0:4, 2:3] : (tensor<6x3xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:    %2 = stablehlo.reshape %1 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %0 [0:4, 0:1] : (tensor<6x3xf64>) -> tensor<4x1xf64>
// CHECK-NEXT:    %4 = stablehlo.reshape %3 : (tensor<4x1xf64>) -> tensor<4xf64>
// CHECK-NEXT:    return %2, %4 : tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:  }

// The operand is a reshape: the view folds into it.
func.func @of_reshape(%x: tensor<6x3xf64>) -> tensor<6xf64> {
  %r = stablehlo.reshape %x : (tensor<6x3xf64>) -> tensor<18xf64>
  %0 = stablehlo.slice %r [1:18:3] : (tensor<18xf64>) -> tensor<6xf64>
  return %0 : tensor<6xf64>
}

// CHECK:  func.func @of_reshape(%arg0: tensor<6x3xf64>) -> tensor<6xf64> {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<6x3xf64>) -> tensor<18xf64>
// CHECK-NEXT:    %1 = stablehlo.reshape %0 : (tensor<18xf64>) -> tensor<6x3xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %1 [0:6, 1:2] : (tensor<6x3xf64>) -> tensor<6x1xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<6x1xf64>) -> tensor<6xf64>
// CHECK-NEXT:    return %3 : tensor<6xf64>
// CHECK-NEXT:  }

// One column of an argument, read alone: the view would only add ops.
// Unchanged.
func.func @lone(%x: tensor<18xf64>) -> tensor<6xf64> {
  %0 = stablehlo.slice %x [1:18:3] : (tensor<18xf64>) -> tensor<6xf64>
  return %0 : tensor<6xf64>
}

// CHECK:  func.func @lone(%arg0: tensor<18xf64>) -> tensor<6xf64> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [1:18:3] : (tensor<18xf64>) -> tensor<6xf64>
// CHECK-NEXT:    return %0 : tensor<6xf64>
// CHECK-NEXT:  }

// A stride that does not divide the extent, or a start past the stride, even
// beside another column: unchanged.
func.func @odd(%x: tensor<20xf64>, %y: tensor<18xf64>) -> (tensor<7xf64>, tensor<7xf64>, tensor<5xf64>, tensor<6xf64>) {
  %0 = stablehlo.slice %x [0:20:3] : (tensor<20xf64>) -> tensor<7xf64>
  %1 = stablehlo.slice %x [1:20:3] : (tensor<20xf64>) -> tensor<7xf64>
  %2 = stablehlo.slice %y [4:18:3] : (tensor<18xf64>) -> tensor<5xf64>
  %3 = stablehlo.slice %y [0:18:3] : (tensor<18xf64>) -> tensor<6xf64>
  return %0, %1, %2, %3 : tensor<7xf64>, tensor<7xf64>, tensor<5xf64>, tensor<6xf64>
}

// CHECK:  func.func @odd(%arg0: tensor<20xf64>, %arg1: tensor<18xf64>) -> (tensor<7xf64>, tensor<7xf64>, tensor<5xf64>, tensor<6xf64>) {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:20:3] : (tensor<20xf64>) -> tensor<7xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg0 [1:20:3] : (tensor<20xf64>) -> tensor<7xf64>
// CHECK-NEXT:    %2 = stablehlo.slice %arg1 [4:18:3] : (tensor<18xf64>) -> tensor<5xf64>
// CHECK-NEXT:    %3 = stablehlo.slice %arg1 [0:18:3] : (tensor<18xf64>) -> tensor<6xf64>
// CHECK-NEXT:    return %0, %1, %2, %3 : tensor<7xf64>, tensor<7xf64>, tensor<5xf64>, tensor<6xf64>
// CHECK-NEXT:  }
