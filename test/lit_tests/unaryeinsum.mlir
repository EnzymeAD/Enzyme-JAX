// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%x : tensor<2x3xf32>) -> tensor<2xf32> {
    %y = "stablehlo.unary_einsum"(%x) {einsum_config = "ab->a"} : (tensor<2x3xf32>) -> tensor<2xf32>
    func.return %y : tensor<2xf32>
  }
}

// FORWARD:  func.func @main(%x: tensor<2x3xf32>, %dx: tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %[[y:.+]] = "stablehlo.unary_einsum"(%x) {einsum_config = "ab->a"} : (tensor<2x3xf32>) -> tensor<2xf32>
// FORWARD-NEXT:    %[[dy:.+]] = "stablehlo.unary_einsum"(%dx) {einsum_config = "ab->a"} : (tensor<2x3xf32>) -> tensor<2xf32>
// FORWARD-NEXT:    return %[[y]], %[[dy]] : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%x: tensor<2x3xf32>, %dy: tensor<2xf32>) -> tensor<2x3xf32> {
// REVERSE-NEXT:    %[[dy_reshaped:.+]] = "stablehlo.reshape"(%dy) : (tensor<2xi32>) -> tensor<2x1xi32>
// REVERSE-NEXT:    %[[dx:.+]] = "stablehlo.broadcast_in_dim"(%[[dy_reshaped]]) {broadcast_dimensions = array<i64: 1, 3>} : (tensor<2x1xf32>) -> tensor<2x3xf32>
// REVERSE-NEXT:    return %[[dx]] : tensor<2x3xf32>
// REVERSE-NEXT:  }
