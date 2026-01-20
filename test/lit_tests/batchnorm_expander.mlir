// RUN: enzymexlamlir-opt %s --batchnorm-expander --canonicalize | FileCheck %s

// Adapted from XLA's batchnorm_expander_test.cc

// Test that we expand BatchNormTraining
// Original test: BatchNormExpanderTest.BatchNormTraining
// Shape: input_shape = [2, 2, 2, 2], scale_shape = [2], offset_shape = [2]
// feature_index = 3, epsilon = 0.001
// CHECK-LABEL: func.func @test_batch_norm_training
// CHECK-NOT: stablehlo.batch_norm_training
// CHECK-DAG: stablehlo.reduce
// CHECK-DAG: stablehlo.broadcast_in_dim
// CHECK-DAG: stablehlo.rsqrt
// CHECK-DAG: stablehlo.multiply
// CHECK-DAG: stablehlo.add
// CHECK: return
func.func @test_batch_norm_training(%activation: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) {
  %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%activation, %scale, %offset) <{
    epsilon = 0.001 : f32,
    feature_index = 3 : i64
  }> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  
  return %output, %batch_mean, %batch_var : tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
}

// Test that we expand BatchNormGrad
// Original test: BatchNormExpanderTest.BatchNormGrad  
// Shape: input_shape = [2, 2, 2, 2], scale_shape = [2], mean_shape = [2], var_shape = [2], grad_output_shape = [2, 2, 2, 2]
// feature_index = 3, epsilon = 0.001
// CHECK-LABEL: func.func @test_batch_norm_grad
// CHECK-NOT: stablehlo.batch_norm_grad
// CHECK-DAG: stablehlo.reduce
// CHECK-DAG: stablehlo.broadcast_in_dim
// CHECK-DAG: stablehlo.rsqrt
// CHECK-DAG: stablehlo.multiply
// CHECK: return
func.func @test_batch_norm_grad(%activation: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %var: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) {
  %grad_operand, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%activation, %scale, %mean, %var, %grad_output) <{
    epsilon = 0.001 : f32,
    feature_index = 3 : i64
  }> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  
  return %grad_operand, %grad_scale, %grad_offset : tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
}

// Test BatchNormTraining with different shape
// Original test: BatchNormExpanderTest.BatchNormTrainingSharding/Execution (without sharding)
// Shape: [8,4], feature_index = 1
// CHECK-LABEL: func.func @test_batch_norm_training_2d
// CHECK-NOT: stablehlo.batch_norm_training
// CHECK-DAG: stablehlo.reduce
// CHECK-DAG: stablehlo.broadcast_in_dim
// CHECK: return
func.func @test_batch_norm_training_2d(%param0: tensor<8x4xf32>, %param1: tensor<4xf32>, %param2: tensor<4xf32>) -> (tensor<8x4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%param0, %param1, %param2) <{
    epsilon = 0.001 : f32,
    feature_index = 1 : i64
  }> : (tensor<8x4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<8x4xf32>, tensor<4xf32>, tensor<4xf32>)
  
  return %output, %batch_mean, %batch_var : tensor<8x4xf32>, tensor<4xf32>, tensor<4xf32>
}

// Test batch_norm_inference expansion
// CHECK-LABEL: func.func @test_batch_norm_inference
// CHECK-NOT: stablehlo.batch_norm_inference
// CHECK-DAG: stablehlo.rsqrt
// CHECK-DAG: stablehlo.broadcast_in_dim
// CHECK-DAG: stablehlo.multiply
// CHECK-DAG: stablehlo.add
// CHECK: return
func.func @test_batch_norm_inference(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>) -> tensor<2x2x2x2xf32> {
  %output = "stablehlo.batch_norm_inference"(%input, %scale, %offset, %mean, %variance) <{
    epsilon = 0.001 : f32,
    feature_index = 3 : i64
  }> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x2x2x2xf32>
  
  return %output : tensor<2x2x2x2xf32>
}

// Test with feature_index = 0 (channels first, different from typical NHWC)
// CHECK-LABEL: func.func @test_batch_norm_training_nchw
// CHECK-NOT: stablehlo.batch_norm_training
// CHECK-DAG: stablehlo.reduce
// CHECK: return
func.func @test_batch_norm_training_nchw(%input: tensor<2x3x4x4xf32>, %scale: tensor<3xf32>, %offset: tensor<3xf32>) -> (tensor<2x3x4x4xf32>, tensor<3xf32>, tensor<3xf32>) {
  %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%input, %scale, %offset) <{
    epsilon = 1.0e-5 : f32,
    feature_index = 1 : i64
  }> : (tensor<2x3x4x4xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<2x3x4x4xf32>, tensor<3xf32>, tensor<3xf32>)
  
  return %output, %batch_mean, %batch_var : tensor<2x3x4x4xf32>, tensor<3xf32>, tensor<3xf32>
}

// Test with feature_index at end (NHWC layout like TensorFlow default)
// CHECK-LABEL: func.func @test_batch_norm_training_nhwc
// CHECK-NOT: stablehlo.batch_norm_training
// CHECK-DAG: stablehlo.reduce
// CHECK: return
func.func @test_batch_norm_training_nhwc(%input: tensor<2x4x4x3xf32>, %scale: tensor<3xf32>, %offset: tensor<3xf32>) -> (tensor<2x4x4x3xf32>, tensor<3xf32>, tensor<3xf32>) {
  %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%input, %scale, %offset) <{
    epsilon = 1.0e-5 : f32,
    feature_index = 3 : i64
  }> : (tensor<2x4x4x3xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<2x4x4x3xf32>, tensor<3xf32>, tensor<3xf32>)
  
  return %output, %batch_mean, %batch_var : tensor<2x4x4x3xf32>, tensor<3xf32>, tensor<3xf32>
}

// Test selective rewriting (only training op with rewrite-inference-op=false)
// This uses default options so all ops are expanded
// CHECK-LABEL: func.func @test_all_three_ops
// CHECK-NOT: stablehlo.batch_norm_training
// CHECK-NOT: stablehlo.batch_norm_inference
// CHECK-NOT: stablehlo.batch_norm_grad
// CHECK: return
func.func @test_all_three_ops(%input: tensor<2x4xf32>, %scale: tensor<4xf32>, %offset: tensor<4xf32>, %mean: tensor<4xf32>, %var: tensor<4xf32>, %grad: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) {
  // Training
  %out1, %bmean, %bvar = "stablehlo.batch_norm_training"(%input, %scale, %offset) <{
    epsilon = 0.001 : f32,
    feature_index = 1 : i64
  }> : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>)
  
  // Inference
  %out2 = "stablehlo.batch_norm_inference"(%input, %scale, %offset, %mean, %var) <{
    epsilon = 0.001 : f32,
    feature_index = 1 : i64
  }> : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  
  // Grad
  %grad_act, %grad_scale, %grad_offset = "stablehlo.batch_norm_grad"(%input, %scale, %mean, %var, %grad) <{
    epsilon = 0.001 : f32,
    feature_index = 1 : i64
  }> : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>)
  
  return %out1, %bmean, %bvar, %out2, %grad_act, %grad_scale, %grad_offset : tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>
}
