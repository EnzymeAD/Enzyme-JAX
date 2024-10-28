// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=einsum outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=einsum outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @einsum(%a : tensor<2x3xf32>, %b : tensor<4x3x5xf32>) -> tensor<4x2x5xf32> {
  %c = "stablehlo.einsum"(%a,%b) {einsum_config = "ab,cbd->cad"} : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
  func.return %c : tensor<4x2x5xf32>
}

// FORWARD:  func.func @einsum(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<4x3x5xf32>, %arg3: tensor<4x3x5xf32>) -> (tensor<4x2x5xf32>, tensor<4x2x5xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.einsum %arg1, %arg2, config = "ab,cbd->cad" : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
// FORWARD-NEXT:    %1 = stablehlo.einsum %arg0, %arg3, config = "ab,cbd->cad" : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
// FORWARD-NEXT:    %2 = stablehlo.add %0, %1 : tensor<4x2x5xf32>
// FORWARD-NEXT:    %3 = stablehlo.einsum %arg0, %arg2, config = "ab,cbd->cad" : (tensor<2x3xf32>, tensor<4x3x5xf32>) -> tensor<4x2x5xf32>
// FORWARD-NEXT:    return %3, %2 : tensor<4x2x5xf32>, tensor<4x2x5xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @einsum(%arg0: tensor<2x3xf32>, %arg1: tensor<4x3x5xf32>, %arg2: tensor<4x2x5xf32>) -> (tensor<2x3xf32>, tensor<4x3x5xf32>) {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<4x2x5xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
// REVERSE-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x3x5xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg2, %cst : tensor<4x2x5xf32>
// REVERSE-NEXT:    %1 = stablehlo.einsum %0, %arg1, config = "cad,cbd->ab" : (tensor<4x2x5xf32>, tensor<4x3x5xf32>) -> tensor<2x3xf32>
// REVERSE-NEXT:    %2 = arith.addf %1, %cst_0 : tensor<2x3xf32>
// REVERSE-NEXT:    %3 = stablehlo.einsum %0, %arg0, config = "cad,ab->cbd" : (tensor<4x2x5xf32>, tensor<2x3xf32>) -> tensor<4x3x5xf32>
// REVERSE-NEXT:    %4 = arith.addf %3, %cst_1 : tensor<4x3x5xf32>
// REVERSE-NEXT:    return %2, %4 : tensor<2x3xf32>, tensor<4x3x5xf32>
// REVERSE-NEXT:  }

// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=einsum_complex outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=einsum_complex outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-COMPLEX

func.func @einsum_complex(%a : tensor<2x3xcomplex<f32>>, %b : tensor<4x3x5xcomplex<f32>>) -> tensor<4x2x5xcomplex<f32>> {
  %c = "stablehlo.einsum"(%a,%b) {einsum_config = "ab,cbd->cad"} : (tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) -> tensor<4x2x5xcomplex<f32>>
  func.return %c : tensor<4x2x5xcomplex<f32>>
}

// FORWARD-COMPLEX:  func.func @einsum_complex(%arg0: tensor<2x3xcomplex<f32>>, %arg1: tensor<2x3xcomplex<f32>>, %arg2: tensor<4x3x5xcomplex<f32>>, %arg3: tensor<4x3x5xcomplex<f32>>) -> (tensor<4x2x5xcomplex<f32>>, tensor<4x2x5xcomplex<f32>>) {
// FORWARD-COMPLEX-NEXT:    %0 = stablehlo.einsum %arg1, %arg2, config = "ab,cbd->cad" : (tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) -> tensor<4x2x5xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    %1 = stablehlo.einsum %arg0, %arg3, config = "ab,cbd->cad" : (tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) -> tensor<4x2x5xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    %2 = stablehlo.add %0, %1 : tensor<4x2x5xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    %3 = stablehlo.einsum %arg0, %arg2, config = "ab,cbd->cad" : (tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) -> tensor<4x2x5xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    return %3, %2 : tensor<4x2x5xcomplex<f32>>, tensor<4x2x5xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:  }

// REVERSE-COMPLEX:  func.func @einsum_complex(%arg0: tensor<2x3xcomplex<f32>>, %arg1: tensor<4x3x5xcomplex<f32>>, %arg2: tensor<4x2x5xcomplex<f32>>) -> (tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) {
// REVERSE-COMPLEX-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x2x5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2x3xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %cst_1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x3x5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %0 = stablehlo.add %cst, %arg2 : tensor<4x2x5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %1 = chlo.conj %0 : tensor<4x2x5xcomplex<f32>> -> tensor<4x2x5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %2 = stablehlo.einsum %1, %arg1, config = "cad,cbd->ab" : (tensor<4x2x5xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %3 = chlo.conj %2 : tensor<2x3xcomplex<f32>> -> tensor<2x3xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %4 = stablehlo.add %cst_0, %3 : tensor<2x3xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %5 = stablehlo.einsum %1, %arg0, config = "cad,ab->cbd" : (tensor<4x2x5xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<4x3x5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %6 = chlo.conj %5 : tensor<4x3x5xcomplex<f32>> -> tensor<4x3x5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %7 = stablehlo.add %cst_1, %6 : tensor<4x3x5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    return %4, %7 : tensor<2x3xcomplex<f32>>, tensor<4x3x5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:  }
