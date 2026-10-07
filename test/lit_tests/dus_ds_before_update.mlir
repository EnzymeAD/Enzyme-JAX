// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dus_dynamic_slice_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The update is written at column 24 and the slice, taken in a loop, reads
// column -1 (clamped to 0): it lies before the update and reads none of it,
// so it reads the operand. (The slice of the update was taken with a
// negative limit.)
func.func @before(%u: tensor<40x1xf64>) -> tensor<40x1xf64> {
  %c = stablehlo.constant dense<0> : tensor<i64>
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_5 = stablehlo.constant dense<5> : tensor<i64>
  %c_m1 = stablehlo.constant dense<-1> : tensor<i64>
  %c_24 = stablehlo.constant dense<24> : tensor<i64>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<40x25xf64>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<40x1xf64>
  %d = stablehlo.dynamic_update_slice %zero, %u, %c, %c_24 : (tensor<40x25xf64>, tensor<40x1xf64>, tensor<i64>, tensor<i64>) -> tensor<40x25xf64>
  %0:2 = stablehlo.while(%iterArg = %c, %iterArg_3 = %cst) : tensor<i64>, tensor<40x1xf64>
  cond {
    %2 = stablehlo.compare  LT, %iterArg, %c_5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  } do {
    %ds = stablehlo.dynamic_slice %d, %c, %c_m1, sizes = [40, 1] : (tensor<40x25xf64>, tensor<i64>, tensor<i64>) -> tensor<40x1xf64>
    %s = stablehlo.add %ds, %iterArg_3 : tensor<40x1xf64>
    %4 = stablehlo.add %iterArg, %c_1 : tensor<i64>
    stablehlo.return %4, %s : tensor<i64>, tensor<40x1xf64>
  }
  return %0#1 : tensor<40x1xf64>
}

// CHECK:  func.func @before(%arg0: tensor<40x1xf64>) -> tensor<40x1xf64> {
// CHECK-NEXT:   %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<40x25xf64>
// CHECK-NEXT:   %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<40x1xf64>
// CHECK-NEXT:   %0:2 = stablehlo.while(%iterArg = %c, %iterArg_4 = %cst_3) : tensor<i64>, tensor<40x1xf64>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %1 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %1 = stablehlo.dynamic_slice %cst, %c, %c_2, sizes = [40, 1] : (tensor<40x25xf64>, tensor<i64>, tensor<i64>) -> tensor<40x1xf64>
// CHECK-NEXT:     %2 = stablehlo.add %1, %iterArg_4 : tensor<40x1xf64>
// CHECK-NEXT:     %3 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:     stablehlo.return %3, %2 : tensor<i64>, tensor<40x1xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %0#1 : tensor<40x1xf64>
// CHECK-NEXT: }
