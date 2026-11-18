// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=erfc outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=erfc outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=erfc outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined strongZero=true" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=STRONG

func.func @erfc(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = chlo.erfc %x : tensor<2xf32> -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @erfc(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %[[CST:.*]] = chlo.constant dense<1.12837923> : tensor<2xf32>
// FORWARD-NEXT:    %[[XSQ:.*]] = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %[[NXSQ:.*]] = stablehlo.negate %[[XSQ]] : tensor<2xf32>
// FORWARD-NEXT:    %[[EXP:.*]] = stablehlo.exponential %[[NXSQ]] : tensor<2xf32>
// FORWARD-NEXT:    %[[CEXP:.*]] = stablehlo.multiply %[[CST]], %[[EXP]] : tensor<2xf32>
// FORWARD-NEXT:    %[[MUL:.*]] = stablehlo.multiply %arg1, %[[CEXP]] : tensor<2xf32>
// FORWARD-NEXT:    %[[TANGENT:.*]] = stablehlo.negate %[[MUL]] : tensor<2xf32>
// FORWARD-NEXT:    %[[PRIMAL:.*]] = chlo.erfc %arg0 : tensor<2xf32>
// FORWARD-NEXT:    return %[[PRIMAL]], %[[TANGENT]] : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @erfc(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %[[CST0:.*]] = chlo.constant dense<1.12837923> : tensor<2xf32>
// REVERSE-NEXT:    %[[XSQ:.*]] = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %[[NXSQ:.*]] = stablehlo.negate %[[XSQ]] : tensor<2xf32>
// REVERSE-NEXT:    %[[EXP:.*]] = stablehlo.exponential %[[NXSQ]] : tensor<2xf32>
// REVERSE-NEXT:    %[[CEXP:.*]] = stablehlo.multiply %[[CST0]], %[[EXP]] : tensor<2xf32>
// REVERSE-NEXT:    %[[GRAD:.*]] = stablehlo.multiply %arg1, %[[CEXP]] : tensor<2xf32>
// REVERSE-NEXT:    %[[NEG:.*]] = stablehlo.negate %[[GRAD]] : tensor<2xf32>
// REVERSE-NEXT:    return %[[NEG]] : tensor<2xf32>
// REVERSE-NEXT:  }

// STRONG:  func.func @erfc(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// STRONG-NEXT:    %[[ZERO:.+]] = chlo.constant dense<0.000000e+00> : tensor<2xf32>
// STRONG-NEXT:    %[[C:.+]] = chlo.constant dense<1.12837923> : tensor<2xf32>
// STRONG-NEXT:    %[[SQ:.+]] = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// STRONG-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[SQ]] : tensor<2xf32>
// STRONG-NEXT:    %[[EXP:.+]] = stablehlo.exponential %[[NEG]] : tensor<2xf32>
// STRONG-NEXT:    %[[DER:.+]] = stablehlo.multiply %[[C]], %[[EXP]] : tensor<2xf32>
// STRONG-NEXT:    %[[CMP:.+]] = stablehlo.compare EQ, %arg1, %[[ZERO]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// STRONG-NEXT:    %[[MUL:.+]] = stablehlo.multiply %arg1, %[[DER]] : tensor<2xf32>
// STRONG-NEXT:    %[[SEL:.+]] = stablehlo.select %[[CMP]], %[[ZERO]], %[[MUL]] : tensor<2xi1>, tensor<2xf32>
// STRONG-NEXT:    %[[RES:.+]] = stablehlo.negate %[[SEL]] : tensor<2xf32>
// STRONG-NEXT:    return %[[RES]] : tensor<2xf32>
// STRONG-NEXT:  }
