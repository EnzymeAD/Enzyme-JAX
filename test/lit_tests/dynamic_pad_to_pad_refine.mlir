// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dynamic_pad_to_pad" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The amounts became constants after shape refinement ran, so the pad kept
// its dynamic result and the loop yielded it to a static carried value. The
// pad's result is refined from the operand's shape as it becomes static.
func.func @yield_refined(%x: tensor<1xi8>, %n: tensor<i64>) -> tensor<1xi8> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %z = stablehlo.constant dense<0> : tensor<i8>
  %amount = stablehlo.constant dense<0> : tensor<1xi64>
  %0:2 = stablehlo.while(%i = %c0, %f = %x) : tensor<i64>, tensor<1xi8>
   cond {
    %c = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %p = stablehlo.dynamic_pad %f, %z, %amount, %amount, %amount : (tensor<1xi8>, tensor<i8>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi8>
    %m = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %m, %p : tensor<i64>, tensor<?xi8>
  }
  return %0#1 : tensor<1xi8>
}

// CHECK:  func.func @yield_refined(%arg0: tensor<1xi8>, %arg1: tensor<i64>) -> tensor<1xi8> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i8>
// CHECK-NEXT:    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0) : tensor<i64>, tensor<1xi8>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %1 = stablehlo.compare LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %1 = stablehlo.pad %iterArg_2, %c_1, low = [0], high = [0], interior = [0] : (tensor<1xi8>, tensor<i8>) -> tensor<1xi8>
// CHECK-NEXT:      %2 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %2, %1 : tensor<i64>, tensor<1xi8>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#1 : tensor<1xi8>
// CHECK-NEXT:  }
