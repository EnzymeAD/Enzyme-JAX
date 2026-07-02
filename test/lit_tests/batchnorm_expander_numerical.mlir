// RUN: enzymexlamlir-opt %s --batchnorm-expander --arith-raise | stablehlo-translate - --interpret

// Adapted from XLA's batchnorm_expander_test.cc - Execution test
// Tests numerical correctness of the batch norm expansion

// Test batch_norm_inference with identity transformation
// With scale=1, offset=0, mean=0, var=1, epsilon=0, output should equal input
func.func @main() {
  // Test 1: batch_norm_inference with identity parameters
  %input1 = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  %scale1 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
  %offset1 = stablehlo.constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>
  %mean1 = stablehlo.constant dense<[0.0, 0.0, 0.0, 0.0]> : tensor<4xf32>
  %var1 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
  
  %out1 = "stablehlo.batch_norm_inference"(%input1, %scale1, %offset1, %mean1, %var1) <{
    epsilon = 0.0 : f32,
    feature_index = 1 : i64
  }> : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  
  // With identity parameters, output should equal input
  check.expect_eq_const %out1, dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
  
  // Test 2: batch_norm_inference with scale=2, offset=1
  // output = (input - mean) / sqrt(var + eps) * scale + offset
  // With mean=0, var=1, eps=0: output = input * 2 + 1
  %scale2 = stablehlo.constant dense<[2.0, 2.0, 2.0, 2.0]> : tensor<4xf32>
  %offset2 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf32>
  
  %out2 = "stablehlo.batch_norm_inference"(%input1, %scale2, %offset2, %mean1, %var1) <{
    epsilon = 0.0 : f32,
    feature_index = 1 : i64
  }> : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  
  // output = input * 2 + 1
  check.expect_eq_const %out2, dense<[[3.0, 5.0, 7.0, 9.0], [11.0, 13.0, 15.0, 17.0]]> : tensor<2x4xf32>
  
  // Test 3: batch_norm_inference with non-zero mean
  // With mean=[1,2,3,4], var=1, scale=1, offset=0, eps=0
  // output = (input - mean) / sqrt(var) * scale + offset = input - mean
  %mean3 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  
  %out3 = "stablehlo.batch_norm_inference"(%input1, %scale1, %offset1, %mean3, %var1) <{
    epsilon = 0.0 : f32,
    feature_index = 1 : i64
  }> : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  
  // output = input - mean
  // row 0: [1-1, 2-2, 3-3, 4-4] = [0, 0, 0, 0]
  // row 1: [5-1, 6-2, 7-3, 8-4] = [4, 4, 4, 4]
  check.expect_eq_const %out3, dense<[[0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0]]> : tensor<2x4xf32>
  
  return
}
