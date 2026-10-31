// RUN: enzymexlamlir-opt --lower-enzymexla-math %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --lower-enzymexla-math="backend=cuda" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --enzyme-batch %s | FileCheck %s --check-prefix=BATCH
// RUN: enzymexlamlir-opt %s --enzyme-batch --inline --lower-enzymexla-math --chlo-legalize-to-stablehlo --canonicalize --arith-raise | stablehlo-translate - --interpret

func.func @attn_bf16(%q: tensor<2x4x8x64xbf16>, %k: tensor<2x4x8x64xbf16>, %v: tensor<2x4x8x64xbf16>) -> (tensor<2x4x8x64xbf16>, tensor<2x8x4xf32>) {
  %out, %lse = "enzymexla.ml.attention"(%q, %k, %v) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>, scale = 0.125 : f64}> : (tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>) -> (tensor<2x4x8x64xbf16>, tensor<2x8x4xf32>)
  return %out, %lse : tensor<2x4x8x64xbf16>, tensor<2x8x4xf32>
}

// CPU-LABEL: func.func @attn_bf16
// CPU-NOT: enzymexla.ml.attention
// CPU-NOT: __cudnn$fmha
// CPU: stablehlo.dot_general
// CPU: stablehlo.exponential
// CPU: stablehlo.log

// CUDA-LABEL: func.func @attn_bf16
// CUDA-NOT: enzymexla.ml.attention
// CUDA: %[[CALL:.*]]:3 = stablehlo.custom_call @__cudnn$fmhaSoftmax(%arg0, %arg1, %arg2)
// CUDA-SAME: \22mask_type\22:\22CAUSAL\22
// CUDA-SAME: operand_layouts = [dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>]
// CUDA-SAME: result_layouts = [dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]
// CUDA: %[[OUT:.*]] = stablehlo.transpose %[[CALL]]#0, dims = [0, 2, 1, 3] : (tensor<2x8x4x64xbf16>) -> tensor<2x4x8x64xbf16>
// CUDA: return %[[OUT]], %[[CALL]]#1

func.func @attn_bias_padding_bf16(
    %q: tensor<2x4x8x64xbf16>,
    %k: tensor<2x4x2x64xbf16>,
    %v: tensor<2x4x2x64xbf16>,
    %bias: tensor<1x8x4x4xbf16>,
    %q_len: tensor<2xi32>,
    %kv_len: tensor<2xi32>) -> (tensor<2x4x8x64xbf16>, tensor<2x8x4xf32>) {
  %out, %lse = "enzymexla.ml.attention"(%q, %k, %v, %bias, %q_len, %kv_len) <{
    is_causal = true,
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 1>,
    scale = 0.125 : f64
  }> : (tensor<2x4x8x64xbf16>, tensor<2x4x2x64xbf16>, tensor<2x4x2x64xbf16>, tensor<1x8x4x4xbf16>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x4x8x64xbf16>, tensor<2x8x4xf32>)
  return %out, %lse : tensor<2x4x8x64xbf16>, tensor<2x8x4xf32>
}

// CPU-LABEL: func.func @attn_bias_padding_bf16
// CPU-NOT: enzymexla.ml.attention
// CPU: stablehlo.dot_general

// CUDA-LABEL: func.func @attn_bias_padding_bf16
// CUDA-NOT: enzymexla.ml.attention
// CUDA: %[[CALL2:.*]]:3 = stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmax(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
// CUDA-SAME: \22mask_type\22:\22PADDING_CAUSAL\22

func.func @attn_grad_bf16(
    %q: tensor<2x4x8x64xbf16>,
    %k: tensor<2x4x8x64xbf16>,
    %v: tensor<2x4x8x64xbf16>,
    %out: tensor<2x4x8x64xbf16>,
    %lse: tensor<2x8x4xf32>,
    %dout: tensor<2x4x8x64xbf16>,
    %bias: tensor<1x8x4x4xbf16>) -> (tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>, tensor<1x8x4x4xbf16>) {
  %dq, %dk, %dv, %dbias = "enzymexla.ml.attention_grad"(%q, %k, %v, %out, %lse, %dout, %bias) <{
    is_causal = false,
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>,
    scale = 0.125 : f64
  }> : (tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>, tensor<2x8x4xf32>, tensor<2x4x8x64xbf16>, tensor<1x8x4x4xbf16>) -> (tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>, tensor<1x8x4x4xbf16>)
  return %dq, %dk, %dv, %dbias : tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>, tensor<2x4x8x64xbf16>, tensor<1x8x4x4xbf16>
}

// CPU-LABEL: func.func @attn_grad_bf16
// CPU-NOT: enzymexla.ml.attention_grad
// CPU-NOT: __cudnn$fmha
// CPU: stablehlo.dot_general

// CUDA-LABEL: func.func @attn_grad_bf16
// CUDA-NOT: enzymexla.ml.attention_grad
// CUDA: %[[BWD:.*]]:5 = stablehlo.custom_call @__cudnn$fmhaScaleBiasSoftmaxBackward(%arg0, %arg1, %arg2, %arg4, %arg5, %arg6, %arg3)
// CUDA-SAME: \22mask_type\22:\22NO_MASK\22
// CUDA: %[[DQ:.*]] = stablehlo.transpose %[[BWD]]#0, dims = [0, 2, 1, 3]
// CUDA: %[[DK:.*]] = stablehlo.transpose %[[BWD]]#1, dims = [0, 2, 1, 3]
// CUDA: %[[DV:.*]] = stablehlo.transpose %[[BWD]]#2, dims = [0, 2, 1, 3]
// CUDA: return %[[DQ]], %[[DK]], %[[DV]], %[[BWD]]#3

func.func @attn_f32_fallback(%q: tensor<1x2x2x4xf32>, %k: tensor<1x2x2x4xf32>, %v: tensor<1x2x2x4xf32>) -> tensor<1x2x2x4xf32> {
  %out, %lse = "enzymexla.ml.attention"(%q, %k, %v) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>, scale = 0.5 : f64}> : (tensor<1x2x2x4xf32>, tensor<1x2x2x4xf32>, tensor<1x2x2x4xf32>) -> (tensor<1x2x2x4xf32>, tensor<1x2x2xf32>)
  return %out : tensor<1x2x2x4xf32>
}

// CUDA-LABEL: func.func @attn_f32_fallback
// CUDA-NOT: __cudnn$fmha
// CUDA: stablehlo.dot_general

func.func @batch_caller(%q: tensor<3x2x4x8x64xbf16>, %k: tensor<3x2x4x8x64xbf16>, %v: tensor<3x2x4x8x64xbf16>) -> (tensor<3x2x4x8x64xbf16>, tensor<3x2x8x4xf32>) {
  %out, %lse = enzyme.batch @attn_bf16(%q, %k, %v) <{batch_shape = array<i64: 3>}> : (tensor<3x2x4x8x64xbf16>, tensor<3x2x4x8x64xbf16>, tensor<3x2x4x8x64xbf16>) -> (tensor<3x2x4x8x64xbf16>, tensor<3x2x8x4xf32>)
  return %out, %lse : tensor<3x2x4x8x64xbf16>, tensor<3x2x8x4xf32>
}

// BATCH-LABEL: func.func private @batched_attn_bf16
// BATCH: %[[QR:.*]] = stablehlo.reshape %arg0 : (tensor<3x2x4x8x64xbf16>) -> tensor<6x4x8x64xbf16>
// BATCH: %[[KR:.*]] = stablehlo.reshape %arg1 : (tensor<3x2x4x8x64xbf16>) -> tensor<6x4x8x64xbf16>
// BATCH: %[[VR:.*]] = stablehlo.reshape %arg2 : (tensor<3x2x4x8x64xbf16>) -> tensor<6x4x8x64xbf16>
// BATCH: %[[BOUT:.*]], %[[BLSE:.*]] = "enzymexla.ml.attention"(%[[QR]], %[[KR]], %[[VR]])
// BATCH: %[[OUTR:.*]] = stablehlo.reshape %[[BOUT]] : (tensor<6x4x8x64xbf16>) -> tensor<3x2x4x8x64xbf16>
// BATCH: %[[LSER:.*]] = stablehlo.reshape %[[BLSE]] : (tensor<6x8x4xf32>) -> tensor<3x2x8x4xf32>
// BATCH: return %[[OUTR]], %[[LSER]]

func.func @batch_2d_bias_caller(
    %q: tensor<2x3x2x4x8x64xbf16>,
    %k: tensor<2x3x2x4x2x64xbf16>,
    %v: tensor<2x3x2x4x2x64xbf16>,
    %bias: tensor<2x3x1x8x4x4xbf16>,
    %q_len: tensor<2x3x2xi32>,
    %kv_len: tensor<2x3x2xi32>) -> (tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x8x4xf32>) {
  %out, %lse = enzyme.batch @attn_bias_padding_bf16(%q, %k, %v, %bias, %q_len, %kv_len) <{batch_shape = array<i64: 2, 3>}> : (tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x4x2x64xbf16>, tensor<2x3x2x4x2x64xbf16>, tensor<2x3x1x8x4x4xbf16>, tensor<2x3x2xi32>, tensor<2x3x2xi32>) -> (tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x8x4xf32>)
  return %out, %lse : tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x8x4xf32>
}

// BATCH-LABEL: func.func private @batched_attn_bias_padding_bf16
// BATCH: %[[Q2:.*]] = stablehlo.reshape %arg0 : (tensor<2x3x2x4x8x64xbf16>) -> tensor<12x4x8x64xbf16>
// BATCH: %[[K2:.*]] = stablehlo.reshape %arg1 : (tensor<2x3x2x4x2x64xbf16>) -> tensor<12x4x2x64xbf16>
// BATCH: %[[V2:.*]] = stablehlo.reshape %arg2 : (tensor<2x3x2x4x2x64xbf16>) -> tensor<12x4x2x64xbf16>
// BATCH: %[[BB:.*]] = stablehlo.broadcast_in_dim %arg3, dims = [0, 1, 2, 3, 4, 5] : (tensor<2x3x1x8x4x4xbf16>) -> tensor<2x3x2x8x4x4xbf16>
// BATCH: %[[B2:.*]] = stablehlo.reshape %[[BB]] : (tensor<2x3x2x8x4x4xbf16>) -> tensor<12x8x4x4xbf16>
// BATCH: %[[QL2:.*]] = stablehlo.reshape %arg4 : (tensor<2x3x2xi32>) -> tensor<12xi32>
// BATCH: %[[KVL2:.*]] = stablehlo.reshape %arg5 : (tensor<2x3x2xi32>) -> tensor<12xi32>
// BATCH: %[[OUT2:.*]], %[[LSE2:.*]] = "enzymexla.ml.attention"(%[[Q2]], %[[K2]], %[[V2]], %[[B2]], %[[QL2]], %[[KVL2]])
// BATCH: %[[OUT2R:.*]] = stablehlo.reshape %[[OUT2]] : (tensor<12x4x8x64xbf16>) -> tensor<2x3x2x4x8x64xbf16>
// BATCH: %[[LSE2R:.*]] = stablehlo.reshape %[[LSE2]] : (tensor<12x8x4xf32>) -> tensor<2x3x2x8x4xf32>
// BATCH: return %[[OUT2R]], %[[LSE2R]]

func.func @batch_grad_2d_caller(
    %q: tensor<2x3x2x4x8x64xbf16>,
    %k: tensor<2x3x2x4x8x64xbf16>,
    %v: tensor<2x3x2x4x8x64xbf16>,
    %out: tensor<2x3x2x4x8x64xbf16>,
    %lse: tensor<2x3x2x8x4xf32>,
    %dout: tensor<2x3x2x4x8x64xbf16>,
    %bias: tensor<2x3x1x8x4x4xbf16>) -> (tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x4x8x64xbf16>, tensor<2x3x1x8x4x4xbf16>) {
  %dq, %dk, %dv, %dbias = enzyme.batch @attn_grad_bf16(%q, %k, %v, %out, %lse, %dout, %bias) <{batch_shape = array<i64: 2, 3>}> : (tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x8x4xf32>, tensor<2x3x2x4x8x64xbf16>, tensor<2x3x1x8x4x4xbf16>) -> (tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x4x8x64xbf16>, tensor<2x3x1x8x4x4xbf16>)
  return %dq, %dk, %dv, %dbias : tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x4x8x64xbf16>, tensor<2x3x2x4x8x64xbf16>, tensor<2x3x1x8x4x4xbf16>
}

// BATCH-LABEL: func.func private @batched_attn_grad_bf16
// BATCH: %[[GBB:.*]] = stablehlo.broadcast_in_dim %arg6, dims = [0, 1, 2, 3, 4, 5] : (tensor<2x3x1x8x4x4xbf16>) -> tensor<2x3x2x8x4x4xbf16>
// BATCH: %[[GBF:.*]] = stablehlo.reshape %[[GBB]] : (tensor<2x3x2x8x4x4xbf16>) -> tensor<12x8x4x4xbf16>
// BATCH: %[[GDQ:.*]], %[[GDK:.*]], %[[GDV:.*]], %[[GDB:.*]] = "enzymexla.ml.attention_grad"
// BATCH: %[[GDBU:.*]] = stablehlo.reshape %[[GDB]] : (tensor<12x8x4x4xbf16>) -> tensor<2x3x2x8x4x4xbf16>
// BATCH: %[[GDBR:.*]] = stablehlo.reduce(%[[GDBU]] init: {{.*}}) applies stablehlo.add across dimensions = [2] : (tensor<2x3x2x8x4x4xbf16>, tensor<bf16>) -> tensor<2x3x8x4x4xbf16>
// BATCH: %[[GDBF:.*]] = stablehlo.reshape %[[GDBR]] : (tensor<2x3x8x4x4xbf16>) -> tensor<2x3x1x8x4x4xbf16>
// BATCH: return {{.*}}, {{.*}}, {{.*}}, %[[GDBF]]

func.func @attn_exec(
    %q: tensor<2x2x2x2xf32>,
    %k: tensor<2x2x1x2xf32>,
    %v: tensor<2x2x1x2xf32>,
    %bias: tensor<1x2x2x2xf32>,
    %q_len: tensor<2xi32>,
    %kv_len: tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x2xf32>) {
  %out, %lse = "enzymexla.ml.attention"(%q, %k, %v, %bias, %q_len, %kv_len) <{
    is_causal = true,
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 1>,
    scale = 0.5 : f64
  }> : (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x2xf32>)
  return %out, %lse : tensor<2x2x2x2xf32>, tensor<2x2x2xf32>
}

func.func @attn_grad_exec(
    %q: tensor<2x2x2x2xf32>,
    %k: tensor<2x2x1x2xf32>,
    %v: tensor<2x2x1x2xf32>,
    %out: tensor<2x2x2x2xf32>,
    %lse: tensor<2x2x2xf32>,
    %dout: tensor<2x2x2x2xf32>,
    %bias: tensor<1x2x2x2xf32>,
    %q_len: tensor<2xi32>,
    %kv_len: tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>) {
  %dq, %dk, %dv, %dbias = "enzymexla.ml.attention_grad"(%q, %k, %v, %out, %lse, %dout, %bias, %q_len, %kv_len) <{
    is_causal = true,
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 1, 1>,
    scale = 0.5 : f64
  }> : (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>)
  return %dq, %dk, %dv, %dbias : tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>
}

func.func @main() {
  %q0 = stablehlo.constant dense<[[[[1.0, 0.5], [-0.5, 1.5]], [[0.2, -0.8], [1.2, 0.3]]], [[[0.3, -0.4], [0.7, 0.2]], [[-0.9, 0.5], [0.4, -0.6]]]]> : tensor<2x2x2x2xf32>
  %q1 = stablehlo.constant dense<[[[[-0.2, 1.1], [0.8, -0.3]], [[0.5, 0.4], [-0.7, 0.9]]], [[[-0.6, 0.2], [0.1, -0.8]], [[0.9, -0.1], [-0.3, 0.7]]]]> : tensor<2x2x2x2xf32>
  %k0 = stablehlo.constant dense<[[[[0.4, -0.2]], [[0.9, 1.1]]], [[[-0.6, 0.7]], [[0.1, -0.5]]]]> : tensor<2x2x1x2xf32>
  %k1 = stablehlo.constant dense<[[[[-0.3, 0.8]], [[0.2, -0.6]]], [[[0.7, 0.3]], [[-0.4, 0.9]]]]> : tensor<2x2x1x2xf32>
  %v0 = stablehlo.constant dense<[[[[1.5, -0.5]], [[0.2, 0.8]]], [[[-1.0, 0.4]], [[0.6, -0.3]]]]> : tensor<2x2x1x2xf32>
  %v1 = stablehlo.constant dense<[[[[0.3, 0.9]], [[-0.7, 0.4]]], [[[0.8, -0.2]], [[-0.5, 1.2]]]]> : tensor<2x2x1x2xf32>
  %b0 = stablehlo.constant dense<[[[[0.1, -0.2], [0.3, 0.0]], [[-0.1, 0.4], [0.2, -0.3]]]]> : tensor<1x2x2x2xf32>
  %b1 = stablehlo.constant dense<[[[[-0.3, 0.2], [0.1, -0.4]], [[0.5, -0.1], [-0.2, 0.6]]]]> : tensor<1x2x2x2xf32>
  %ql0 = stablehlo.constant dense<[2, 1]> : tensor<2xi32>
  %ql1 = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
  %kvl0 = stablehlo.constant dense<[2, 1]> : tensor<2xi32>
  %kvl1 = stablehlo.constant dense<[2, 2]> : tensor<2xi32>
  %do0 = stablehlo.constant dense<[[[[0.3, -0.7], [1.1, 0.2]], [[-0.4, 0.9], [0.5, -0.6]]], [[[0.2, 0.8], [-0.3, 0.4]], [[0.6, -0.1], [-0.5, 0.7]]]]> : tensor<2x2x2x2xf32>
  %do1 = stablehlo.constant dense<[[[[-0.5, 0.4], [0.2, -0.9]], [[0.7, -0.3], [-0.1, 0.8]]], [[[0.4, -0.6], [0.9, 0.1]], [[-0.8, 0.3], [0.2, -0.4]]]]> : tensor<2x2x2x2xf32>

  %out0, %lse0 = func.call @attn_exec(%q0, %k0, %v0, %b0, %ql0, %kvl0) : (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x2xf32>)
  %out1, %lse1 = func.call @attn_exec(%q1, %k1, %v1, %b1, %ql1, %kvl1) : (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x2xf32>)

  %q0r = stablehlo.reshape %q0 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
  %q1r = stablehlo.reshape %q1 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
  %qb = stablehlo.concatenate %q0r, %q1r, dim = 0 : (tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>

  %k0r = stablehlo.reshape %k0 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
  %k1r = stablehlo.reshape %k1 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
  %kb = stablehlo.concatenate %k0r, %k1r, dim = 0 : (tensor<1x2x2x1x2xf32>, tensor<1x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>

  %v0r = stablehlo.reshape %v0 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
  %v1r = stablehlo.reshape %v1 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
  %vb = stablehlo.concatenate %v0r, %v1r, dim = 0 : (tensor<1x2x2x1x2xf32>, tensor<1x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>

  %b0r = stablehlo.reshape %b0 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
  %b1r = stablehlo.reshape %b1 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
  %bb = stablehlo.concatenate %b0r, %b1r, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

  %ql0r = stablehlo.reshape %ql0 : (tensor<2xi32>) -> tensor<1x2xi32>
  %ql1r = stablehlo.reshape %ql1 : (tensor<2xi32>) -> tensor<1x2xi32>
  %qlb = stablehlo.concatenate %ql0r, %ql1r, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>

  %kvl0r = stablehlo.reshape %kvl0 : (tensor<2xi32>) -> tensor<1x2xi32>
  %kvl1r = stablehlo.reshape %kvl1 : (tensor<2xi32>) -> tensor<1x2xi32>
  %kvlb = stablehlo.concatenate %kvl0r, %kvl1r, dim = 0 : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>

  %outb, %lseb = enzyme.batch @attn_exec(%qb, %kb, %vb, %bb, %qlb, %kvlb) <{batch_shape = array<i64: 2>}> : (tensor<2x2x2x2x2xf32>, tensor<2x2x2x1x2xf32>, tensor<2x2x2x1x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2x2x2x2xf32>, tensor<2x2x2x2xf32>)

  %out0r = stablehlo.reshape %out0 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
  %out1r = stablehlo.reshape %out1 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
  %out_ref = stablehlo.concatenate %out0r, %out1r, dim = 0 : (tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>
  check.expect_almost_eq %outb, %out_ref : tensor<2x2x2x2x2xf32>

  %dq0, %dk0, %dv0, %db0 = func.call @attn_grad_exec(%q0, %k0, %v0, %out0, %lse0, %do0, %b0, %ql0, %kvl0) : (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>)
  %dq1, %dk1, %dv1, %db1 = func.call @attn_grad_exec(%q1, %k1, %v1, %out1, %lse1, %do1, %b1, %ql1, %kvl1) : (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>)

  %do0r = stablehlo.reshape %do0 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
  %do1r = stablehlo.reshape %do1 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
  %dob = stablehlo.concatenate %do0r, %do1r, dim = 0 : (tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>

  %dqb, %dkb, %dvb, %dbb = enzyme.batch @attn_grad_exec(%qb, %kb, %vb, %outb, %lseb, %dob, %bb, %qlb, %kvlb) <{batch_shape = array<i64: 2>}> : (tensor<2x2x2x2x2xf32>, tensor<2x2x2x1x2xf32>, tensor<2x2x2x1x2xf32>, tensor<2x2x2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<2x2x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2x2x2x2xf32>, tensor<2x2x2x1x2xf32>, tensor<2x2x2x1x2xf32>, tensor<2x1x2x2x2xf32>)

  %dq0r = stablehlo.reshape %dq0 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
  %dq1r = stablehlo.reshape %dq1 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
  %dq_ref = stablehlo.concatenate %dq0r, %dq1r, dim = 0 : (tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>

  %dk0r = stablehlo.reshape %dk0 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
  %dk1r = stablehlo.reshape %dk1 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
  %dk_ref = stablehlo.concatenate %dk0r, %dk1r, dim = 0 : (tensor<1x2x2x1x2xf32>, tensor<1x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>

  %dv0r = stablehlo.reshape %dv0 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
  %dv1r = stablehlo.reshape %dv1 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
  %dv_ref = stablehlo.concatenate %dv0r, %dv1r, dim = 0 : (tensor<1x2x2x1x2xf32>, tensor<1x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>

  %db0r = stablehlo.reshape %db0 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
  %db1r = stablehlo.reshape %db1 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
  %db_ref = stablehlo.concatenate %db0r, %db1r, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

  check.expect_almost_eq %dqb, %dq_ref : tensor<2x2x2x2x2xf32>
  check.expect_almost_eq %dkb, %dk_ref : tensor<2x2x2x1x2xf32>
  check.expect_almost_eq %dvb, %dv_ref : tensor<2x2x2x1x2xf32>
  check.expect_almost_eq %dbb, %db_ref : tensor<2x1x2x2x2xf32>
  return
}
