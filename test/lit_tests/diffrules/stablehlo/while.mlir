// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=REVERSE

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
      stablehlo.return %2, %iterArg_0 : tensor<i64>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// FORWARD-NEXT:    %c_0 = stablehlo.constant dense<10> : tensor<i64>
// FORWARD-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<i64>
// FORWARD-NEXT:    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0, %iterArg_3 = %arg1) : tensor<i64>, tensor<3xf64>, tensor<3xf64>
// FORWARD-NEXT:     cond {
// FORWARD-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// FORWARD-NEXT:      stablehlo.return %1 : tensor<i1>
// FORWARD-NEXT:    } do {
// FORWARD-NEXT:      %1 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// FORWARD-NEXT:      stablehlo.return %1, %iterArg_2, %iterArg_3 : tensor<i64>, tensor<3xf64>, tensor<3xf64>
// FORWARD-NEXT:    }
// FORWARD-NEXT:    return %0#1, %0#2 : tensor<3xf64>, tensor<3xf64>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
// REVERSE-NEXT:    %c = stablehlo.constant dense<1> : tensor<i64>
// REVERSE-NEXT:    %c_0 = stablehlo.constant dense<10> : tensor<i64>
// REVERSE-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<3xf64>
// REVERSE-NEXT:    %0 = stablehlo.subtract %c_0, %c_1 : tensor<i64>
// REVERSE-NEXT:    %1 = stablehlo.divide %0, %c : tensor<i64>
// REVERSE-NEXT:    %2 = stablehlo.add %arg1, %cst : tensor<3xf64>
// REVERSE-NEXT:    %3:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %2) : tensor<i64>, tensor<3xf64>
// REVERSE-NEXT:     cond {
// REVERSE-NEXT:      %5 = stablehlo.compare  LT, %iterArg, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// REVERSE-NEXT:      stablehlo.return %5 : tensor<i1>
// REVERSE-NEXT:    } do {
// REVERSE-NEXT:      %5 = stablehlo.add %iterArg, %c : tensor<i64>
// REVERSE-NEXT:      stablehlo.return %5, %iterArg_2 : tensor<i64>, tensor<3xf64>
// REVERSE-NEXT:    }
// REVERSE-NEXT:    %4 = stablehlo.add %3#1, %cst : tensor<3xf64>
// REVERSE-NEXT:    return %4 : tensor<3xf64>
// REVERSE-NEXT:  }
