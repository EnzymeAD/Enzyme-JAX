// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_const,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_const,enzyme_active mode=ReverseModeCombined strongZero=true" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=STRONG

func.func @main(%n : tensor<2xf32>, %x : tensor<2xf32>) -> tensor<2xf32> {
  %y = chlo.polygamma %n, %x : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE:         %[[ONE:.+]] = {{.*}}constant dense<1.000000e+00> : tensor<2xf32>
// REVERSE:         %[[NP1:.+]] = {{.*}}add %arg0, %[[ONE]] : tensor<2xf32>
// REVERSE:         %[[PG:.+]] = chlo.polygamma %[[NP1]], %arg1 : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
// REVERSE-NEXT:    %[[MUL:.+]] = stablehlo.multiply %arg2, %[[PG]] : tensor<2xf32>
// REVERSE-NEXT:    return %[[MUL]] : tensor<2xf32>
// REVERSE-NEXT:  }

// STRONG:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> tensor<2xf32> {
// STRONG-NEXT:    %[[ZERO:.+]] = chlo.constant dense<0.000000e+00> : tensor<2xf32>
// STRONG:         %[[ONE:.+]] = {{.*}}constant dense<1.000000e+00> : tensor<2xf32>
// STRONG:         %[[NP1:.+]] = {{.*}}add %arg0, %[[ONE]] : tensor<2xf32>
// STRONG:         %[[PG:.+]] = chlo.polygamma %[[NP1]], %arg1 : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
// STRONG-NEXT:    %[[CMP:.+]] = stablehlo.compare EQ, %arg2, %[[ZERO]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// STRONG-NEXT:    %[[MUL:.+]] = stablehlo.multiply %arg2, %[[PG]] : tensor<2xf32>
// STRONG-NEXT:    %[[SEL:.+]] = stablehlo.select %[[CMP]], %[[ZERO]], %[[MUL]] : tensor<2xi1>, tensor<2xf32>
// STRONG-NEXT:    return %[[SEL]] : tensor<2xf32>
// STRONG-NEXT:  }
