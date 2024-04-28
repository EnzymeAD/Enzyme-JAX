// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {

  func.func @main(%a : tensor<2x3xf32>) -> tensor<18x27xf32> {
    %0 = stablehlo.constant dense<3.140000e+00> : tensor<f32>
    %2 = stablehlo.pad %a, %0, low = [5, 7], high = [11, 13], interior = [0, 2] : (tensor<2x3xf32>, tensor<f32>) -> tensor<18x27xf32>
    return %2 : tensor<18x27xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> (tensor<18x27xf32>, tensor<18x27xf32>) {
// FORWARD-NEXT:    %[[i0:.+]] = stablehlo.constant dense<3.140000e+00> : tensor<f32>
// FORWARD-NEXT:    %[[icst:.+]] = arith.constant dense<0.000000e+00> : tensor<f32>
// FORWARD-NEXT:    %[[i1:.+]] = stablehlo.pad %arg1, %[[icst]], low = [5, 7], high = [11, 13], interior = [0, 2] : (tensor<2x3xf32>, tensor<f32>) -> tensor<18x27xf32>
// FORWARD-NEXT:    %[[i2:.+]] = stablehlo.pad %arg0, %[[i0]], low = [5, 7], high = [11, 13], interior = [0, 2] : (tensor<2x3xf32>, tensor<f32>) -> tensor<18x27xf32>
// FORWARD-NEXT:    return %[[i2]], %[[i1]] : tensor<18x27xf32>, tensor<18x27xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<18x27xf32>) -> tensor<2x3xf32> {
// REVERSE-NEXT:    %[[icst:.+]] = arith.constant dense<0.000000e+00> : tensor<18x27xf32>
// REVERSE-NEXT:    %[[icst0:.+]] = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
// REVERSE-NEXT:    %[[i0:.+]] = arith.addf %arg1, %[[icst]] : tensor<18x27xf32>
// REVERSE-NEXT:    %[[i1:.+]] = stablehlo.slice %[[i0]] [5:7, 7:14:3] : (tensor<18x27xf32>) -> tensor<2x3xf32>
// REVERSE-NEXT:    %[[i2:.+]] = arith.addf %[[i1]], %[[icst0]] : tensor<2x3xf32>
// REVERSE-NEXT:    return %[[i2]] : tensor<2x3xf32>
// REVERSE-NEXT:  }
