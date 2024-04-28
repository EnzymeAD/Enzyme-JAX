// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {

  func.func @main(%a : tensor<2x100x14xf32>) -> tensor<100x14x2xf32> {
    %b = stablehlo.transpose %a, dims = [1, 2, 0] : (tensor<2x100x14xf32>) -> tensor<100x14x2xf32>
    return %b : tensor<100x14x2xf32>
  }
}


// FORWARD:  func.func @main(%arg0: tensor<2x100x14xf32>, %arg1: tensor<2x100x14xf32>) -> (tensor<100x14x2xf32>, tensor<100x14x2xf32>) {
// FORWARD-NEXT:  %0 = stablehlo.transpose %arg1, dims = [1, 2, 0] : (tensor<2x100x14xf32>) -> tensor<100x14x2xf32>
// FORWARD-NEXT:  %1 = stablehlo.transpose %arg0, dims = [1, 2, 0] : (tensor<2x100x14xf32>) -> tensor<100x14x2xf32>
// FORWARD-NEXT:  return %1, %0 : tensor<100x14x2xf32>, tensor<100x14x2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2x100x14xf32>, %arg1: tensor<100x14x2xf32>) -> tensor<2x100x14xf32> {
// REVERSE-NEXT:    %[[icst:.+]] = arith.constant dense<0.000000e+00> 
// REVERSE-NEXT:    %[[icst0:.+]] = arith.constant dense<0.000000e+00> 
// REVERSE-NEXT:    %[[i0:.+]] = arith.addf %arg1, %[[icst]]
// REVERSE-NEXT:    %[[i1:.+]] = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<100x14x2xf32>) -> tensor<2x100x14xf32>
// REVERSE-NEXT:    %[[i2:.+]] = arith.addf %[[i1]], %[[icst0]] 
// REVERSE-NEXT:    return %[[i2]] 
// REVERSE-NEXT:  }
