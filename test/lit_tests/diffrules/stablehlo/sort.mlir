// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%arg0: tensor<8x4xf64>) -> (tensor<8x4xf64>) {
  %0 = "stablehlo.sort"(%arg0) <{dimension = 0 : i64, is_stable = false}> ({
  ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
    %1 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<f64>, tensor<f64>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  }) : (tensor<8x4xf64>) -> tensor<8x4xf64>
  return %0 : tensor<8x4xf64>
}

// FORWARD: func.func @main(%arg0: tensor<8x4xf64>, %arg1: tensor<8x4xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>) {
// FORWARD-NEXT:   %0:2 = "stablehlo.sort"(%arg0, %arg1) <{dimension = 0 : i64, is_stable = false}> ({
// FORWARD-NEXT:   ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<f64>, %arg5: tensor<f64>):
// FORWARD-NEXT:     %1 = stablehlo.compare  LT, %arg2, %arg3 : (tensor<f64>, tensor<f64>) -> tensor<i1>
// FORWARD-NEXT:     stablehlo.return %1 : tensor<i1>
// FORWARD-NEXT:   }) : (tensor<8x4xf64>, tensor<8x4xf64>) -> (tensor<8x4xf64>, tensor<8x4xf64>)
// FORWARD-NEXT:   return %0#0, %0#1 : tensor<8x4xf64>, tensor<8x4xf64>
// FORWARD-NEXT: }
