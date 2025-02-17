// TODO: forward mode
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active,enzyme_active,enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE

func.func @main(%arg0: tensor<3x3x4x2xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>, %arg4: tensor<4xf32>) -> (tensor<3x3x4x2xf32>) {
    %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%arg0, %arg1, %arg2) <{epsilon = 9.99999974E-6 : f32, feature_index = 2 : i64}> : (tensor<3x3x4x2xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<3x3x4x2xf32>, tensor<4xf32>, tensor<4xf32>)
    return %output : tensor<3x3x4x2xf32>
}

// REVERSE:  func.func @main(%arg0: tensor<3x3x4x2xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>, %arg4: tensor<4xf32>, %arg5: tensor<3x3x4x2xf32>) -> (tensor<3x3x4x2xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
// REVERSE-NEXT:      %cst = stablehlo.constant dense<0.000000e+00> : tensor<3x3x4x2xf32>
// REVERSE-NEXT:      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// REVERSE-NEXT:      %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%arg0, %arg1, %arg2) <{epsilon = 9.99999974E-6 : f32, feature_index = 2 : i64}> : (tensor<3x3x4x2xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<3x3x4x2xf32>, tensor<4xf32>, tensor<4xf32>)
// REVERSE-NEXT:      %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%arg0, %arg1, %batch_mean, %batch_var, %cst) <{epsilon = 9.99999974E-6 : f32, feature_index = 2 : i64}> : (tensor<3x3x4x2xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<3x3x4x2xf32>) -> (tensor<3x3x4x2xf32>, tensor<4xf32>, tensor<4xf32>)
// REVERSE-NEXT:      return %grad_operand, %grad_scale, %grad_offset, %cst_0, %cst_0 : tensor<3x3x4x2xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
// REVERSE-NEXT:  }
