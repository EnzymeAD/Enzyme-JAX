// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot_general_nobatch outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot_general_nobatch outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

func.func @dot_general_nobatch(%a : tensor<2x3xf32>, %b : tensor<4x2xf32>) -> tensor<3x4xf32> {
  %c = "stablehlo.dot_general"(%a, %b) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [1],
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x3xf32>, tensor<4x2xf32>) -> tensor<3x4xf32>
  func.return %c : tensor<3x4xf32>
}

// FORWARD:  func.func @dot_general_nobatch(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<4x2xf32>, %arg3: tensor<4x2xf32>) -> (tensor<3x4xf32>, tensor<3x4xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3xf32>, tensor<4x2xf32>) -> tensor<3x4xf32>
// FORWARD-NEXT:    %1 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3xf32>, tensor<4x2xf32>) -> tensor<3x4xf32>
// FORWARD-NEXT:    %2 = stablehlo.add %0, %1 : tensor<3x4xf32>
// FORWARD-NEXT:    %3 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3xf32>, tensor<4x2xf32>) -> tensor<3x4xf32>
// FORWARD-NEXT:    return %3, %2 : tensor<3x4xf32>, tensor<3x4xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @dot_general_nobatch(%arg0: tensor<2x3xf32>, %arg1: tensor<4x2xf32>, %arg2: tensor<3x4xf32>) -> (tensor<2x3xf32>, tensor<4x2xf32>) {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<3x4xf32>
// REVERSE-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
// REVERSE-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x2xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg2, %cst : tensor<3x4xf32>
// REVERSE-NEXT:    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32>
// REVERSE-NEXT:    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
// REVERSE-NEXT:    %3 = arith.addf %2, %cst_0 : tensor<2x3xf32>
// REVERSE-NEXT:    %4 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
// REVERSE-NEXT:    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
// REVERSE-NEXT:    %6 = arith.addf %5, %cst_1 : tensor<4x2xf32>
// REVERSE-NEXT:    return %3, %6 : tensor<2x3xf32>, tensor<4x2xf32>
// REVERSE-NEXT:  }

// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot_general_batch outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-BATCH
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot_general_batch outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-BATCH

func.func @dot_general_batch(%a : tensor<2x3x8xf32>, %b : tensor<4x2x8xf32>) -> tensor<8x3x4xf32> {
  %c = "stablehlo.dot_general"(%a, %b) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [2],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [1],
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x3x8xf32>, tensor<4x2x8xf32>) -> tensor<8x3x4xf32>
  func.return %c : tensor<8x3x4xf32>
}

// FORWARD-BATCH:  func.func @dot_general_batch(%arg0: tensor<2x3x8xf32>, %arg1: tensor<2x3x8xf32>, %arg2: tensor<4x2x8xf32>, %arg3: tensor<4x2x8xf32>) -> (tensor<8x3x4xf32>, tensor<8x3x4xf32>) {
// FORWARD-BATCH-NEXT:    %0 = stablehlo.dot_general %arg1, %arg2, batching_dims = [2] x [2], contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x8xf32>, tensor<4x2x8xf32>) -> tensor<8x3x4xf32>
// FORWARD-BATCH-NEXT:    %1 = stablehlo.dot_general %arg0, %arg3, batching_dims = [2] x [2], contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x8xf32>, tensor<4x2x8xf32>) -> tensor<8x3x4xf32>
// FORWARD-BATCH-NEXT:    %2 = stablehlo.add %0, %1 : tensor<8x3x4xf32>
// FORWARD-BATCH-NEXT:    %3 = stablehlo.dot_general %arg0, %arg2, batching_dims = [2] x [2], contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x8xf32>, tensor<4x2x8xf32>) -> tensor<8x3x4xf32>
// FORWARD-BATCH-NEXT:    return %3, %2 : tensor<8x3x4xf32>, tensor<8x3x4xf32>
// FORWARD-BATCH-NEXT:  }

// REVERSE-BATCH:  func.func @dot_general_batch(%arg0: tensor<2x3x8xf32>, %arg1: tensor<4x2x8xf32>, %arg2: tensor<8x3x4xf32>) -> (tensor<2x3x8xf32>, tensor<4x2x8xf32>) {
// REVERSE-BATCH-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<8x3x4xf32>
// REVERSE-BATCH-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<2x3x8xf32>
// REVERSE-BATCH-NEXT:    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x2x8xf32>
// REVERSE-BATCH-NEXT:    %0 = arith.addf %arg2, %cst : tensor<8x3x4xf32>
// REVERSE-BATCH-NEXT:    %1 = stablehlo.dot_general %0, %arg1, batching_dims = [0] x [2], contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x3x4xf32>, tensor<4x2x8xf32>) -> tensor<8x3x2xf32>
// REVERSE-BATCH-NEXT:    %2 = stablehlo.transpose %1, dims = [2, 1, 0] : (tensor<8x3x2xf32>) -> tensor<2x3x8xf32>
// REVERSE-BATCH-NEXT:    %3 = arith.addf %2, %cst_0 : tensor<2x3x8xf32>
// REVERSE-BATCH-NEXT:    %4 = stablehlo.dot_general %arg0, %0, batching_dims = [2] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x8xf32>, tensor<8x3x4xf32>) -> tensor<8x2x4xf32>
// REVERSE-BATCH-NEXT:    %5 = stablehlo.transpose %4, dims = [2, 1, 0] : (tensor<8x2x4xf32>) -> tensor<4x2x8xf32>
// REVERSE-BATCH-NEXT:    %6 = arith.addf %5, %cst_1 : tensor<4x2x8xf32>
// REVERSE-BATCH-NEXT:    return %3, %6 : tensor<2x3x8xf32>, tensor<4x2x8xf32>
// REVERSE-BATCH-NEXT:  }

// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot_general_nobatch_complex outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot_general_nobatch_complex outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-COMPLEX

func.func @dot_general_nobatch_complex(%a : tensor<2x3xcomplex<f32>>, %b : tensor<4x2xcomplex<f32>>) -> tensor<3x4xcomplex<f32>> {
  %c = "stablehlo.dot_general"(%a, %b) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [1],
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x3xcomplex<f32>>, tensor<4x2xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
  func.return %c : tensor<3x4xcomplex<f32>>
}

// FORWARD-COMPLEX:  func.func @dot_general_nobatch_complex(%arg0: tensor<2x3xcomplex<f32>>, %arg1: tensor<2x3xcomplex<f32>>, %arg2: tensor<4x2xcomplex<f32>>, %arg3: tensor<4x2xcomplex<f32>>) -> (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) {
// FORWARD-COMPLEX-NEXT:    %0 = stablehlo.dot_general %arg1, %arg2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3xcomplex<f32>>, tensor<4x2xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    %1 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3xcomplex<f32>>, tensor<4x2xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    %2 = stablehlo.add %0, %1 : tensor<3x4xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    %3 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3xcomplex<f32>>, tensor<4x2xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    return %3, %2 : tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:  }

// REVERSE-COMPLEX:  func.func @dot_general_nobatch_complex(%arg0: tensor<2x3xcomplex<f32>>, %arg1: tensor<4x2xcomplex<f32>>, %arg2: tensor<3x4xcomplex<f32>>) -> (tensor<2x3xcomplex<f32>>, tensor<4x2xcomplex<f32>>) {
// REVERSE-COMPLEX-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<3x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2x3xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %cst_1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %0 = stablehlo.add %cst, %arg2 : tensor<3x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %1 = chlo.conj %0 : tensor<3x4xcomplex<f32>> -> tensor<3x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>) -> tensor<3x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<3x2xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %4 = chlo.conj %3 : tensor<2x3xcomplex<f32>> -> tensor<2x3xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %5 = stablehlo.add %cst_0, %4 : tensor<2x3xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %6 = stablehlo.dot_general %arg0, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x3xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> tensor<2x4xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<2x4xcomplex<f32>>) -> tensor<4x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %8 = chlo.conj %7 : tensor<4x2xcomplex<f32>> -> tensor<4x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %9 = stablehlo.add %cst_1, %8 : tensor<4x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    return %5, %9 : tensor<2x3xcomplex<f32>>, tensor<4x2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:  }

// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot_general_batch_complex outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-BATCH-COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=dot_general_batch_complex outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE-BATCH-COMPLEX

func.func @dot_general_batch_complex(%a : tensor<2x3x8xcomplex<f32>>, %b : tensor<4x2x8xcomplex<f32>>) -> tensor<8x3x4xcomplex<f32>> {
  %c = "stablehlo.dot_general"(%a, %b) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [2],
      rhs_batching_dimensions = [2],
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [1],
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x3x8xcomplex<f32>>, tensor<4x2x8xcomplex<f32>>) -> tensor<8x3x4xcomplex<f32>>
  func.return %c : tensor<8x3x4xcomplex<f32>>
}

// FORWARD-BATCH-COMPLEX:  func.func @dot_general_batch_complex(%arg0: tensor<2x3x8xcomplex<f32>>, %arg1: tensor<2x3x8xcomplex<f32>>, %arg2: tensor<4x2x8xcomplex<f32>>, %arg3: tensor<4x2x8xcomplex<f32>>) -> (tensor<8x3x4xcomplex<f32>>, tensor<8x3x4xcomplex<f32>>) {
// FORWARD-BATCH-COMPLEX-NEXT:    %0 = stablehlo.dot_general %arg1, %arg2, batching_dims = [2] x [2], contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x8xcomplex<f32>>, tensor<4x2x8xcomplex<f32>>) -> tensor<8x3x4xcomplex<f32>>
// FORWARD-BATCH-COMPLEX-NEXT:    %1 = stablehlo.dot_general %arg0, %arg3, batching_dims = [2] x [2], contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x8xcomplex<f32>>, tensor<4x2x8xcomplex<f32>>) -> tensor<8x3x4xcomplex<f32>>
// FORWARD-BATCH-COMPLEX-NEXT:    %2 = stablehlo.add %0, %1 : tensor<8x3x4xcomplex<f32>>
// FORWARD-BATCH-COMPLEX-NEXT:    %3 = stablehlo.dot_general %arg0, %arg2, batching_dims = [2] x [2], contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x8xcomplex<f32>>, tensor<4x2x8xcomplex<f32>>) -> tensor<8x3x4xcomplex<f32>>
// FORWARD-BATCH-COMPLEX-NEXT:    return %3, %2 : tensor<8x3x4xcomplex<f32>>, tensor<8x3x4xcomplex<f32>>
// FORWARD-BATCH-COMPLEX-NEXT:  }

// REVERSE-BATCH-COMPLEX:  func.func @dot_general_batch_complex(%arg0: tensor<2x3x8xcomplex<f32>>, %arg1: tensor<4x2x8xcomplex<f32>>, %arg2: tensor<8x3x4xcomplex<f32>>) -> (tensor<2x3x8xcomplex<f32>>, tensor<4x2x8xcomplex<f32>>) {
// REVERSE-BATCH-COMPLEX-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<8x3x4xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<2x3x8xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %cst_1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<4x2x8xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %0 = stablehlo.add %cst, %arg2 : tensor<8x3x4xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %1 = chlo.conj %0 : tensor<8x3x4xcomplex<f32>> -> tensor<8x3x4xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %2 = stablehlo.dot_general %1, %arg1, batching_dims = [0] x [2], contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x3x4xcomplex<f32>>, tensor<4x2x8xcomplex<f32>>) -> tensor<8x3x2xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %3 = stablehlo.transpose %2, dims = [2, 1, 0] : (tensor<8x3x2xcomplex<f32>>) -> tensor<2x3x8xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %4 = chlo.conj %3 : tensor<2x3x8xcomplex<f32>> -> tensor<2x3x8xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %5 = stablehlo.add %cst_0, %4 : tensor<2x3x8xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %6 = stablehlo.dot_general %arg0, %1, batching_dims = [2] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x3x8xcomplex<f32>>, tensor<8x3x4xcomplex<f32>>) -> tensor<8x2x4xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %7 = stablehlo.transpose %6, dims = [2, 1, 0] : (tensor<8x2x4xcomplex<f32>>) -> tensor<4x2x8xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %8 = chlo.conj %7 : tensor<4x2x8xcomplex<f32>> -> tensor<4x2x8xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    %9 = stablehlo.add %cst_1, %8 : tensor<4x2x8xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:    return %5, %9 : tensor<2x3x8xcomplex<f32>>, tensor<4x2x8xcomplex<f32>>
// REVERSE-BATCH-COMPLEX-NEXT:  }
