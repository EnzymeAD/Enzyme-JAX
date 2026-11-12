// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const mode=ReverseModeCombined strongZero=true" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=STRONG

func.func @main(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %y = chlo.polygamma %a, %b : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE:         %[[PG:.+]] = chlo.polygamma
// REVERSE-NEXT:    %[[MUL:.+]] = stablehlo.multiply %arg2, %[[PG]] : tensor<2xf32>
// REVERSE-NEXT:    return %[[MUL]] : tensor<2xf32>
// REVERSE-NEXT:  }

// STRONG:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> tensor<2xf32> {
// STRONG-NEXT:    %[[ZERO:.+]] = chlo.constant dense<0.000000e+00> : tensor<2xf32>
// STRONG:         %[[PG:.+]] = chlo.polygamma
// STRONG-NEXT:    %[[CMP:.+]] = stablehlo.compare EQ, %arg2, %[[ZERO]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// STRONG-NEXT:    %[[MUL:.+]] = stablehlo.multiply %arg2, %[[PG]] : tensor<2xf32>
// STRONG-NEXT:    %[[SEL:.+]] = stablehlo.select %[[CMP]], %[[ZERO]], %[[MUL]] : tensor<2xi1>, tensor<2xf32>
// STRONG-NEXT:    return %[[SEL]] : tensor<2xf32>
// STRONG-NEXT:  }
