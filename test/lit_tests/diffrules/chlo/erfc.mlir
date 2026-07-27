// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=erfc outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=erfc outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

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
// REVERSE-NEXT:    %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %[[DRET:.*]] = arith.addf %arg1, %[[ZERO]] : tensor<2xf32>
// REVERSE-NEXT:    %[[XSQ:.*]] = stablehlo.multiply %arg0, %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %[[NXSQ:.*]] = stablehlo.negate %[[XSQ]] : tensor<2xf32>
// REVERSE-NEXT:    %[[EXP:.*]] = stablehlo.exponential %[[NXSQ]] : tensor<2xf32>
// REVERSE-NEXT:    %[[CEXP:.*]] = stablehlo.multiply %[[CST0]], %[[EXP]] : tensor<2xf32>
// REVERSE-NEXT:    %[[GRAD:.*]] = stablehlo.multiply %[[DRET]], %[[CEXP]] : tensor<2xf32>
// REVERSE-NEXT:    %[[NEG:.*]] = stablehlo.negate %[[GRAD]] : tensor<2xf32>
// REVERSE-NEXT:    %[[RES:.*]] = arith.addf %[[NEG]], %[[ZERO]] : tensor<2xf32>
// REVERSE-NEXT:    return %[[RES]] : tensor<2xf32>
// REVERSE-NEXT:  }
