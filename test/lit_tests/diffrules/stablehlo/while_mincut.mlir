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

// CHECK:  func.func @main(%[[ARG0:.+]]: tensor<3xf64>, %[[ARG1:.+]]: tensor<3xf64>) -> tensor<3xf64> {
// CHECK-NEXT:    %c = stablehlo.constant dense<9> : tensor<i64>
// CHECK-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<10x3xf64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %[[ZEROI:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : tensor<3xf64>
// CHECK-NEXT:    %0 = stablehlo.subtract %c_1, %c_2 : tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.divide %0, %c_0 : tensor<i64>
// CHECK-NEXT:    %[[FWD:.+]]:3 = stablehlo.while(%iterArg = %[[ZEROI]], %iterArg_4 = %[[ARG0]], %iterArg_5 = %cst) : tensor<i64>, tensor<3xf64>, tensor<10x3xf64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %6 = stablehlo.compare  LT, %iterArg, %c_1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %6 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %6 = stablehlo.reshape %iterArg_4 : (tensor<3xf64>) -> tensor<1x3xf64>
// CHECK-NEXT:      %7 = stablehlo.dynamic_update_slice %iterArg_5, %6, %iterArg, %c_2 : (tensor<10x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<10x3xf64>
// CHECK-NEXT:      %8 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.cosine %iterArg_4 : tensor<3xf64>
// CHECK-NEXT:      %10 = stablehlo.multiply %9, %9 : tensor<3xf64>
// CHECK-NEXT:      stablehlo.return %8, %10, %7 : tensor<i64>, tensor<3xf64>, tensor<10x3xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %3 = arith.addf %[[ARG1]], %[[ZERO]] : tensor<3xf64>
// CHECK-NEXT:    %[[REV:.+]]:3 = stablehlo.while(%iterArg = %[[ZEROI]], %iterArg_4 = %3, %iterArg_5 = %c) : tensor<i64>, tensor<3xf64>, tensor<i64>
// CHECK-NEXT:     cond {
// CHECK-NEXT:      %6 = stablehlo.compare  LT, %iterArg, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %6 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %[[FWD]]#2, %iterArg_5, %c_2, sizes = [1, 3] : (tensor<10x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1x3xf64>) -> tensor<3xf64>
// CHECK-NEXT:      %8 = stablehlo.cosine %7 : tensor<3xf64>
// CHECK-NEXT:      %9 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      %10 = stablehlo.multiply %iterArg_4, %8 : tensor<3xf64>
// CHECK-NEXT:      %11 = arith.addf %10, %[[ZERO]] : tensor<3xf64>
// CHECK-NEXT:      %12 = stablehlo.multiply %iterArg_4, %8 : tensor<3xf64>
// CHECK-NEXT:      %13 = arith.addf %11, %12 : tensor<3xf64>
// CHECK-NEXT:      %14 = stablehlo.sine %7 : tensor<3xf64>
// CHECK-NEXT:      %15 = stablehlo.negate %14 : tensor<3xf64>
// CHECK-NEXT:      %16 = stablehlo.multiply %13, %15 : tensor<3xf64>
// CHECK-NEXT:      %17 = arith.addf %16, %[[ZERO]] : tensor<3xf64>
// CHECK-NEXT:      %18 = stablehlo.subtract %iterArg_5, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %9, %17, %18 : tensor<i64>, tensor<3xf64>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %5 = arith.addf %[[REV]]#1, %[[ZERO]] : tensor<3xf64>
// CHECK-NEXT:    return %5 : tensor<3xf64>
// CHECK-NEXT:  }
