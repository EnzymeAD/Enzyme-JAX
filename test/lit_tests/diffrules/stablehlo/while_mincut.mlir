// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

module {
  func.func public @main(%arg0: tensor<3xf64>) -> (tensor<3xf64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_10 = stablehlo.constant dense<10> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0) : tensor<i64>, tensor<3xf64>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_10,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %2 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %3 = stablehlo.cosine %iterArg_0 : tensor<3xf64>
      %4 = stablehlo.multiply %3, %3 : tensor<3xf64>
      stablehlo.return %2, %4 : tensor<i64>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }
}

// CHECK: module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<10x3xf64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %cst_3 = arith.constant dense<0.000000e+00> : tensor<3xf64>
// CHECK-NEXT:     %0:3 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %arg0, %iterArg_5 = %cst) : tensor<i64>, tensor<3xf64>, tensor<10x3xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %4 = stablehlo.compare LT, %iterArg, %c_1, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %4 = stablehlo.reshape %iterArg_4 : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:       %5 = stablehlo.dynamic_update_slice %iterArg_5, %4, %iterArg, %c_2 : (tensor<10x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<10x3xf64>
// CHECK-NEXT:       %6 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:       %7 = stablehlo.cosine %iterArg_4 : tensor<3xf64>
// CHECK-NEXT:       %8 = stablehlo.multiply %7, %7 : tensor<3xf64>
// CHECK-NEXT:       stablehlo.return %6, %8, %5 : tensor<i64>, tensor<3xf64>, tensor<10x3xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %1 = arith.addf %arg1, %cst_3 : tensor<3xf64>
// CHECK-NEXT:     %2:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %1) : tensor<i64>, tensor<3xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %4 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %4 = stablehlo.subtract %c, %iterArg : tensor<i64>
// CHECK-NEXT:       %5 = stablehlo.dynamic_slice %0#2, %4, %c_2, sizes = [1, 3] : (tensor<10x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// CHECK-NEXT:       %6 = stablehlo.reshape %5 : (tensor<1x3xf64>) -> tensor<3xf64>
// CHECK-NEXT:       %7 = stablehlo.cosine %6 : tensor<3xf64>
// CHECK-NEXT:       %8 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:       %9 = arith.addf %iterArg_4, %cst_3 : tensor<3xf64>
// CHECK-NEXT:       %10 = stablehlo.multiply %9, %7 : tensor<3xf64>
// CHECK-NEXT:       %11 = arith.addf %10, %cst_3 : tensor<3xf64>
// CHECK-NEXT:       %12 = stablehlo.multiply %9, %7 : tensor<3xf64>
// CHECK-NEXT:       %13 = arith.addf %11, %12 : tensor<3xf64>
// CHECK-NEXT:       %14 = stablehlo.sine %6 : tensor<3xf64>
// CHECK-NEXT:       %15 = stablehlo.negate %14 : tensor<3xf64>
// CHECK-NEXT:       %16 = stablehlo.multiply %13, %15 : tensor<3xf64>
// CHECK-NEXT:       %17 = arith.addf %16, %cst_3 : tensor<3xf64>
// CHECK-NEXT:       stablehlo.return %8, %17 : tensor<i64>, tensor<3xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %3 = arith.addf %2#1, %cst_3 : tensor<3xf64>
// CHECK-NEXT:     return %3 : tensor<3xf64>
// CHECK-NEXT:   }
// CHECK-NEXT: }
