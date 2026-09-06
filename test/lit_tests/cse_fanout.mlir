// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// The two adds share every operand, in either order; the broadcast they share
// has other users too. CSE merges them (and the commutative twin) through
// the hashed index rather than a walk over the broadcast's use list.
func.func @fanout(%x: tensor<4xf64>, %c: tensor<f64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
  %b = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<f64>) -> tensor<4xf64>
  %m0 = stablehlo.multiply %b, %b : tensor<4xf64>
  %m1 = stablehlo.subtract %b, %x : tensor<4xf64>
  %a0 = stablehlo.add %b, %x : tensor<4xf64>
  %a1 = stablehlo.add %x, %b : tensor<4xf64>
  %s = stablehlo.sine %a1 : tensor<4xf64>
  return %m0, %m1, %a0, %s : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
}

// CHECK:    func.func @fanout(%arg0: tensor<4xf64>, %arg1: tensor<f64>) -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>) {
// CHECK-NEXT:    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f64>) -> tensor<4xf64>
// CHECK-NEXT:    %1 = stablehlo.multiply %0, %0 : tensor<4xf64>
// CHECK-NEXT:    %2 = stablehlo.subtract %0, %arg0 : tensor<4xf64>
// CHECK-NEXT:    %3 = stablehlo.add %0, %arg0 : tensor<4xf64>
// CHECK-NEXT:    %4 = stablehlo.sine %3 : tensor<4xf64>
// CHECK-NEXT:    return %1, %2, %3, %4 : tensor<4xf64>, tensor<4xf64>, tensor<4xf64>, tensor<4xf64>
// CHECK-NEXT:  }
