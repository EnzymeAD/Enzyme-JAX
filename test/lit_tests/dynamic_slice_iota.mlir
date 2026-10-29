// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dynamic_slice_iota" --transform-interpreter --enzyme-hlo-remove-transform --split-input-file | FileCheck %s

// Row k of a 2-D iota along rows: the row index k, clamped into the iota,
// at every position.
func.func @row(%k: tensor<i64>) -> tensor<1x26460x1xi64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %iota = stablehlo.iota dim = 0 : tensor<6x26460x1xi64>
  %0 = stablehlo.dynamic_slice %iota, %k, %c0, %c0, sizes = [1, 26460, 1] : (tensor<6x26460x1xi64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x26460x1xi64>
  return %0 : tensor<1x26460x1xi64>
}

// CHECK:      func.func @row(%arg0: tensor<i64>) -> tensor<1x26460x1xi64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.clamp %c, %arg0, %c_0 : tensor<i64>
// CHECK-NEXT:   %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<1x26460x1xi64>
// CHECK-NEXT:   return %1 : tensor<1x26460x1xi64>

// -----




// A window of more than one along the iota's dimension: the start plus the
// iota of the window; an i32 start widened to the iota's type.
func.func @window(%k: tensor<i32>) -> tensor<4x8xi64> {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %iota = stablehlo.iota dim = 1 : tensor<4x32xi64>
  %0 = stablehlo.dynamic_slice %iota, %c0, %k, sizes = [4, 8] : (tensor<4x32xi64>, tensor<i32>, tensor<i32>) -> tensor<4x8xi64>
  return %0 : tensor<4x8xi64>
}

// CHECK:      func.func @window(%arg0: tensor<i32>) -> tensor<4x8xi64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<24> : tensor<i32>
// CHECK-NEXT:   %0 = stablehlo.clamp %c, %arg0, %c_0 : tensor<i32>
// CHECK-NEXT:   %1 = stablehlo.convert %0 : (tensor<i32>) -> tensor<i64>
// CHECK-NEXT:   %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i64>) -> tensor<4x8xi64>
// CHECK-NEXT:   %3 = stablehlo.iota dim = 1 : tensor<4x8xi64>
// CHECK-NEXT:   %4 = stablehlo.add %2, %3 : tensor<4x8xi64>
// CHECK-NEXT:   return %4 : tensor<4x8xi64>

// -----



// Inside a loop, a slice at the loop's counter is batch fission's form of a
// lifted index computation: left alone.
func.func @in_loop(%x: tensor<6x4xi64>) -> tensor<6x4xi64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c6 = stablehlo.constant dense<6> : tensor<i64>
  %iota = stablehlo.iota dim = 0 : tensor<6x4xi64>
  %r:2 = stablehlo.while(%k = %c0, %acc = %x) : tensor<i64>, tensor<6x4xi64>
   cond {
    %c = stablehlo.compare LT, %k, %c6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %iota, %k, %c0, sizes = [1, 4] : (tensor<6x4xi64>, tensor<i64>, tensor<i64>) -> tensor<1x4xi64>
    %acc2 = stablehlo.dynamic_update_slice %acc, %row, %k, %c0 : (tensor<6x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<6x4xi64>
    %k2 = stablehlo.add %k, %c1 : tensor<i64>
    stablehlo.return %k2, %acc2 : tensor<i64>, tensor<6x4xi64>
  }
  return %r#1 : tensor<6x4xi64>
}

// CHECK:      func.func @in_loop(%arg0: tensor<6x4xi64>) -> tensor<6x4xi64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:   %0 = stablehlo.iota dim = 0 : tensor<6x4xi64>
// CHECK-NEXT:   %1:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<6x4xi64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %2 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %2 = stablehlo.dynamic_slice %0, %iterArg, %c, sizes = [1, 4] : (tensor<6x4xi64>, tensor<i64>, tensor<i64>) -> tensor<1x4xi64>
// CHECK-NEXT:     %3 = stablehlo.dynamic_update_slice %iterArg_2, %2, %iterArg, %c : (tensor<6x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<6x4xi64>
// CHECK-NEXT:     %4 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %4, %3 : tensor<i64>, tensor<6x4xi64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %1#1 : tensor<6x4xi64>
// CHECK-NEXT: }
