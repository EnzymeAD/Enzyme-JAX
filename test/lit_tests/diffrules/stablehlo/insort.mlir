// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%x : tensor<64000xi64>) -> tensor<64000xi64> {
  %i = stablehlo.iota dim = 0 : tensor<64000xi64>
  %r:2 = "stablehlo.sort"(%x, %i) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg170: tensor<i64>, %arg171: tensor<i64>, %arg172: tensor<i64>, %arg173: tensor<i64>):
      %1096 = "stablehlo.compare"(%arg170, %arg171) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%1096) : (tensor<i1>) -> ()
    }) : (tensor<64000xi64>, tensor<64000xi64>) -> (tensor<64000xi64>, tensor<64000xi64>)

  func.return %r#1 : tensor<64000xi64>
}

// FORWARD:   func.func @main(%arg0: tensor<64000xi64>, %arg1: tensor<64000xi64>) -> (tensor<64000xi64>, tensor<64000xi64>) {
// FORWARD-NEXT:     %0 = stablehlo.iota dim = 0 : tensor<64000xi64>
// FORWARD-NEXT:     %1:2 = "stablehlo.sort"(%arg0, %0) <{dimension = 0 : i64, is_stable = true}> ({
// FORWARD-NEXT:     ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>, %arg4: tensor<i64>, %arg5: tensor<i64>):
// FORWARD-NEXT:       %2 = stablehlo.compare  LT, %arg2, %arg3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// FORWARD-NEXT:       stablehlo.return %2 : tensor<i1>
// FORWARD-NEXT:     }) : (tensor<64000xi64>, tensor<64000xi64>) -> (tensor<64000xi64>, tensor<64000xi64>)
// FORWARD-NEXT:     %cst = arith.constant dense<0> : tensor<64000xi64>
// FORWARD-NEXT:     return %1#1, %cst : tensor<64000xi64>, tensor<64000xi64>
// FORWARD-NEXT:   }

// REVERSE:   func.func @main(%arg0: tensor<64000xi64>, %arg1: tensor<64000xi64>) -> tensor<64000xi64> {
// REVERSE-NEXT:     %cst = arith.constant dense<0> : tensor<64000xi64>
// REVERSE-NEXT:     return %cst : tensor<64000xi64>
// REVERSE-NEXT:   }
