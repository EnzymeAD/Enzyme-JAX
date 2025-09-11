// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_const mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=REVERSE

module {
  func.func public @main(%arg0: tensor<3xf64>, %lim : tensor<i64>) -> (tensor<3xf64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0) : tensor<i64>, tensor<3xf64>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %lim,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %2 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      %3 = stablehlo.multiply %iterArg_0, %iterArg_0 : tensor<3xf64>
      stablehlo.return %2, %3 : tensor<i64>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>, %arg2: tensor<i64>) -> (tensor<3xf64>, tensor<3xf64>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// FORWARD-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// FORWARD-NEXT:    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0, %iterArg_2 = %arg1) : tensor<i64>, tensor<3xf64>, tensor<3xf64>
// FORWARD-NEXT:    cond {
// FORWARD-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %arg2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// FORWARD-NEXT:      stablehlo.return %1 : tensor<i1>
// FORWARD-NEXT:    } do {
// FORWARD-NEXT:      %1 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// FORWARD-NEXT:      %2 = stablehlo.multiply %iterArg_2, %iterArg_1 : tensor<3xf64>
// FORWARD-NEXT:      %3 = stablehlo.multiply %iterArg_2, %iterArg_1 : tensor<3xf64>
// FORWARD-NEXT:      %4 = arith.addf %2, %3 : tensor<3xf64>
// FORWARD-NEXT:      %5 = stablehlo.multiply %iterArg_1, %iterArg_1 : tensor<3xf64>
// FORWARD-NEXT:      stablehlo.return %1, %5, %4 : tensor<i64>, tensor<3xf64>, tensor<3xf64>
// FORWARD-NEXT:    }
// FORWARD-NEXT:    return %0#1, %0#2 : tensor<3xf64>, tensor<3xf64>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<3xf64>, %arg1: tensor<i64>, %arg2: tensor<3xf64>) -> tensor<3xf64> {
// REVERSE-NEXT:    %c = stablehlo.constant dense<0> : tensor<2xi64>
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// REVERSE-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<0x3xf64>
// REVERSE-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<i64>
// REVERSE-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<i64>
// REVERSE-NEXT:    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// REVERSE-NEXT:    %0 = stablehlo.subtract %arg1, %c_2 : tensor<i64>
// REVERSE-NEXT:    %1 = stablehlo.divide %0, %c_1 : tensor<i64>
// REVERSE-NEXT:    %2 = stablehlo.subtract %arg1, %c_2 : tensor<i64>
// REVERSE-NEXT:    %3 = stablehlo.divide %2, %c_1 : tensor<i64>
// REVERSE-NEXT:    %4 = stablehlo.reshape %3 : (tensor<i64>) -> tensor<1xi64>
// REVERSE-NEXT:    %5 = stablehlo.pad %4, %c_2, low = [0], high = [1], interior = [0] : (tensor<1xi64>, tensor<i64>) -> tensor<2xi64>
// REVERSE-NEXT:    %6 = stablehlo.dynamic_pad %cst_0, %cst, %c, %5, %c : (tensor<0x3xf64>, tensor<f64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x3xf64>
// REVERSE-NEXT:    %7:3 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %arg0, %iterArg_5 = %6) : tensor<i64>, tensor<3xf64>, tensor<?x3xf64>
// REVERSE-NEXT:    cond {
// REVERSE-NEXT:      %12 = stablehlo.compare  LT, %iterArg, %arg1,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:      stablehlo.return %12 : tensor<i1>
// REVERSE-NEXT:    } do {
// REVERSE-NEXT:      %12 = stablehlo.reshape %iterArg_4 : (tensor<3xf64>) -> tensor<1x3xf64>
// REVERSE-NEXT:      %13 = stablehlo.dynamic_update_slice %iterArg_5, %12, %iterArg, %c_2 : (tensor<?x3xf64>, tensor<1x3xf64>, tensor<i64>, tensor<i64>) -> tensor<?x3xf64>
// REVERSE-NEXT:      %14 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// REVERSE-NEXT:      %15 = stablehlo.multiply %iterArg_4, %iterArg_4 : tensor<3xf64>
// REVERSE-NEXT:      stablehlo.return %14, %15, %13 : tensor<i64>, tensor<3xf64>, tensor<?x3xf64>
// REVERSE-NEXT:    }
// REVERSE-NEXT:    %8 = stablehlo.add %arg2, %cst_3 : tensor<3xf64>
// REVERSE-NEXT:    %9 = stablehlo.subtract %3, %c_1 : tensor<i64>
// REVERSE-NEXT:    %10:3 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %8, %iterArg_5 = %9) : tensor<i64>, tensor<3xf64>, tensor<i64>
// REVERSE-NEXT:    cond {
// REVERSE-NEXT:      %12 = stablehlo.compare  LT, %iterArg, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:      stablehlo.return %12 : tensor<i1>
// REVERSE-NEXT:    } do {
// REVERSE-NEXT:      %12 = stablehlo.dynamic_slice %7#2, %iterArg_5, %c_2, sizes = [1, 3] : (tensor<?x3xf64>, tensor<i64>, tensor<i64>) -> tensor<1x3xf64>
// REVERSE-NEXT:      %13 = stablehlo.reshape %12 : (tensor<1x3xf64>) -> tensor<3xf64>
// REVERSE-NEXT:      %14 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// REVERSE-NEXT:      %15 = stablehlo.add %iterArg_4, %cst_3 : tensor<3xf64>
// REVERSE-NEXT:      %16 = stablehlo.multiply %15, %13 : tensor<3xf64>
// REVERSE-NEXT:      %17 = stablehlo.add %16, %cst_3 : tensor<3xf64>
// REVERSE-NEXT:      %18 = stablehlo.multiply %15, %13 : tensor<3xf64>
// REVERSE-NEXT:      %19 = stablehlo.add %17, %18 : tensor<3xf64>
// REVERSE-NEXT:      %20 = stablehlo.subtract %iterArg_5, %c_1 : tensor<i64>
// REVERSE-NEXT:      stablehlo.return %14, %19, %20 : tensor<i64>, tensor<3xf64>, tensor<i64>
// REVERSE-NEXT:    }
// REVERSE-NEXT:    %11 = stablehlo.add %10#1, %cst_3 : tensor<3xf64>
// REVERSE-NEXT:    return %11 : tensor<3xf64>
// REVERSE-NEXT:  }
