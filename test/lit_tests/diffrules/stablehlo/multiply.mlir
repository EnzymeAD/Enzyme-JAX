// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @main(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.multiply %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

// FORWARD:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.multiply %arg1, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg3, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = arith.addf %0, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg0, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    return %3, %2 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg2, %cst : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %0, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %2 = arith.addf %1, %cst : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.multiply %0, %arg0 : tensor<2xf32>
// REVERSE-NEXT:    %4 = arith.addf %3, %cst : tensor<2xf32>
// REVERSE-NEXT:    return %2, %4 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }

// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=mul_complex outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --arith-raise --verify-each=0 | FileCheck %s --check-prefix=FORWARDCOMPLEX
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=mul_complex outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --arith-raise  --canonicalize --remove-unnecessary-enzyme-ops --verify-each=0 | FileCheck %s --check-prefix=REVERSECOMPLEX

func.func @mul_complex(%a : tensor<2xcomplex<f32>>, %b : tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  %c = stablehlo.multiply %a, %b : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %c : tensor<2xcomplex<f32>>
}
//FORWARDCOMPLEX:  func.func @mul_complex(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>, %arg2: tensor<2xcomplex<f32>>, %arg3: tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
//FORWARDCOMPLEX-NEXT:    %0 = stablehlo.multiply %arg1, %arg2 : tensor<2xcomplex<f32>>
//FORWARDCOMPLEX-NEXT:    %1 = stablehlo.multiply %arg3, %arg0 : tensor<2xcomplex<f32>>
//FORWARDCOMPLEX-NEXT:    %2 = stablehlo.add %0, %1 : tensor<2xcomplex<f32>>
//FORWARDCOMPLEX-NEXT:    %3 = stablehlo.multiply %arg0, %arg2 : tensor<2xcomplex<f32>>
//FORWARDCOMPLEX-NEXT:    return %3, %2 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
//FORWARDCOMPLEX-NEXT:  }

//REVERSECOMPLEX:  func.func @mul_complex(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>, %arg2: tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
//REVERSECOMPLEX-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>
//REVERSECOMPLEX-NEXT:    %0 = stablehlo.add %cst, %arg2 : tensor<2xcomplex<f32>>
//REVERSECOMPLEX-NEXT:    %1 = chlo.conj %0 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
//REVERSECOMPLEX-NEXT:    %2 = stablehlo.multiply %1, %arg1 : tensor<2xcomplex<f32>>
//REVERSECOMPLEX-NEXT:    %3 = chlo.conj %2 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
//REVERSECOMPLEX-NEXT:    %4 = stablehlo.add %cst, %3 : tensor<2xcomplex<f32>>
//REVERSECOMPLEX-NEXT:    %5 = stablehlo.multiply %1, %arg0 : tensor<2xcomplex<f32>>
//REVERSECOMPLEX-NEXT:    %6 = chlo.conj %5 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
//REVERSECOMPLEX-NEXT:    %7 = stablehlo.add %cst, %6 : tensor<2xcomplex<f32>>
//REVERSECOMPLEX-NEXT:    return %4, %7 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
//REVERSECOMPLEX-NEXT:  }
