// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%x : tensor<2x3xf32>) -> tensor<2xf32> {
    %y = "stablehlo.unary_einsum"(%x) {einsum_config = "ab->a"} : (tensor<2x3xf32>) -> tensor<2xf32>
    func.return %y : tensor<2xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.unary_einsum %arg1, config = "ab->a" : (tensor<2x3xf32>) -> tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.unary_einsum %arg0, config = "ab->a" : (tensor<2x3xf32>) -> tensor<2xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2xf32>) -> tensor<2x3xf32> {
// REVERSE-NEXT:    %0 = stablehlo.unary_einsum %arg1, config = "a->ab" : (tensor<2xf32>) -> tensor<2x3xf32>
// REVERSE-NEXT:    return %0 : tensor<2x3xf32>
// REVERSE-NEXT:  }
