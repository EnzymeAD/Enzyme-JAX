// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

// The call sits in one branch of a `stablehlo.if`, so its adjoint is built inside a region whose
// reverse block is only reached when the predicate selected that branch. Together with
// custom_call_while.mlir this covers the two control-flow shapes where the caches cannot simply
// fold away into the enclosing block.

func.func @scale_rev(%x: tensor<10xf32>, %y: tensor<10xf32>,
                     %dy: tensor<10xf32>) -> tensor<10xf32> {
  %three = stablehlo.constant dense<3.000000e+00> : tensor<10xf32>
  %res = stablehlo.multiply %dy, %three : tensor<10xf32>
  func.return %res : tensor<10xf32>
}

// Both functions have to sit in the same module: `enzyme.reverse` is a flat symbol reference, so
// the rule is looked up in the symbol table enclosing the call.
func.func @main(%arg0: tensor<10xf32>, %pred: tensor<i1>) -> tensor<10xf32> {
  %cst = stablehlo.constant dense<1.0> : tensor<10xf32>

  %0 = "stablehlo.if"(%pred) ({
    %1 = stablehlo.custom_call @scale(%arg0) {
      enzyme.reverse = @scale_rev,
      enzyme.active_operands = array<i64: 0>
    } : (tensor<10xf32>) -> tensor<10xf32>
    "stablehlo.return"(%1) : (tensor<10xf32>) -> ()
  }, {
    "stablehlo.return"(%cst) : (tensor<10xf32>) -> ()
  }) : (tensor<i1>) -> tensor<10xf32>

  return %0 : tensor<10xf32>
}

// REVERSE: func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<i1>, %arg2: tensor<10xf32>) -> tensor<10xf32> {
// REVERSE:   stablehlo.custom_call @scale(
// REVERSE:   call @scale_rev(
// REVERSE:   return
