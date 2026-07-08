// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=erfinv outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=erfinv outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @erfinv(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = chlo.erf_inv %x : tensor<2xf32> -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @erfinv(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %[[CST:.*]] = chlo.constant dense<0.886226952> : tensor<2xf32>
// FORWARD-NEXT:    %[[EI1:.*]] = chlo.erf_inv %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %[[EI2:.*]] = chlo.erf_inv %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %[[SQ:.*]] = stablehlo.multiply %[[EI1]], %[[EI2]] : tensor<2xf32>
// FORWARD-NEXT:    %[[EXP:.*]] = stablehlo.exponential %[[SQ]] : tensor<2xf32>
// FORWARD-NEXT:    %[[CEXP:.*]] = stablehlo.multiply %[[CST]], %[[EXP]] : tensor<2xf32>
// FORWARD-NEXT:    %[[TANGENT:.*]] = stablehlo.multiply %arg1, %[[CEXP]] : tensor<2xf32>
// FORWARD-NEXT:    %[[PRIMAL:.*]] = chlo.erf_inv %arg0 : tensor<2xf32>
// FORWARD-NEXT:    return %[[PRIMAL]], %[[TANGENT]] : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @erfinv(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %[[CST0:.*]] = chlo.constant dense<0.886226952> : tensor<2xf32>
// REVERSE-NEXT:    %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %[[DRET:.*]] = arith.addf %arg1, %[[ZERO]] : tensor<2xf32>
// REVERSE-NEXT:    %[[EI1:.*]] = chlo.erf_inv %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %[[EI2:.*]] = chlo.erf_inv %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %[[SQ:.*]] = stablehlo.multiply %[[EI1]], %[[EI2]] : tensor<2xf32>
// REVERSE-NEXT:    %[[EXP:.*]] = stablehlo.exponential %[[SQ]] : tensor<2xf32>
// REVERSE-NEXT:    %[[CEXP:.*]] = stablehlo.multiply %[[CST0]], %[[EXP]] : tensor<2xf32>
// REVERSE-NEXT:    %[[GRAD:.*]] = stablehlo.multiply %[[DRET]], %[[CEXP]] : tensor<2xf32>
// REVERSE-NEXT:    %[[RES:.*]] = arith.addf %[[GRAD]], %[[ZERO]] : tensor<2xf32>
// REVERSE-NEXT:    return %[[RES]] : tensor<2xf32>
// REVERSE-NEXT:  }
