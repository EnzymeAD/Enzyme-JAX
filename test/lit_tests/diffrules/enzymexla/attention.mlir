// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=attn_mha outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=MHA-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=attn_mha outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --lower-enzymexla-math="backend=cuda" | FileCheck %s --check-prefix=MHA-REV-CUDA
// RUN: enzymexlamlir-opt %s --enzyme="postpasses=arith-raise{stablehlo=true},enzyme-batch-to-stablehlo,canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize,arith-raise{stablehlo=true}" --enzyme-batch-to-stablehlo --canonicalize --remove-unnecessary-enzyme-ops --lower-enzymexla-math --inline --chlo-legalize-to-stablehlo --canonicalize --arith-raise | stablehlo-translate - --interpret

module {
  func.func @attn_mha(%q: tensor<1x2x2x8xbf16>, %k: tensor<1x2x2x8xbf16>, %v: tensor<1x2x2x8xbf16>) -> tensor<1x2x2x8xbf16> {
    %out, %lse = "enzymexla.ml.attention"(%q, %k, %v) <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>, scale = 0.35355339059327379 : f64}> : (tensor<1x2x2x8xbf16>, tensor<1x2x2x8xbf16>, tensor<1x2x2x8xbf16>) -> (tensor<1x2x2x8xbf16>, tensor<1x2x2xf32>)
    return %out : tensor<1x2x2x8xbf16>
  }

  func.func @attn_f32(%q: tensor<1x2x2x2xf32>, %k: tensor<1x2x2x2xf32>, %v: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
    %out, %lse = "enzymexla.ml.attention"(%q, %k, %v) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>, scale = 0.5 : f64}> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2xf32>)
    return %out : tensor<1x2x2x2xf32>
  }

  func.func @attn_f32_lse(%q: tensor<1x2x2x2xf32>, %k: tensor<1x2x2x2xf32>, %v: tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2xf32>) {
    %out, %lse = "enzymexla.ml.attention"(%q, %k, %v) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>, scale = 0.5 : f64}> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2xf32>)
    return %out, %lse : tensor<1x2x2x2xf32>, tensor<1x2x2xf32>
  }

  func.func @attn_f32_lse_ref(%q: tensor<1x2x2x2xf32>, %k: tensor<1x2x2x2xf32>, %v: tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2xf32>) {
    %scores = stablehlo.dot_general %q, %k, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %scale = stablehlo.constant dense<0.5> : tensor<1x2x2x2xf32>
    %logits = stablehlo.multiply %scores, %scale : tensor<1x2x2x2xf32>
    %exp = stablehlo.exponential %logits : tensor<1x2x2x2xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %sum = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [3] : (tensor<1x2x2x2xf32>, tensor<f32>) -> tensor<1x2x2xf32>
    %lse = stablehlo.log %sum : tensor<1x2x2xf32>
    %sum_b = stablehlo.broadcast_in_dim %sum, dims = [0, 1, 2] : (tensor<1x2x2xf32>) -> tensor<1x2x2x2xf32>
    %probs = stablehlo.divide %exp, %sum_b : tensor<1x2x2x2xf32>
    %out_bnth = stablehlo.dot_general %probs, %v, batching_dims = [0, 1] x [0, 2], contracting_dims = [3] x [1] : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %out = stablehlo.transpose %out_bnth, dims = [0, 2, 1, 3] : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    return %out, %lse : tensor<1x2x2x2xf32>, tensor<1x2x2xf32>
  }

  func.func @attn_f32_ref(%q: tensor<1x2x2x2xf32>, %k: tensor<1x2x2x2xf32>, %v: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
    %out, %lse = func.call @attn_f32_lse_ref(%q, %k, %v) : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2xf32>)
    return %out : tensor<1x2x2x2xf32>
  }

  func.func @hvp_fn(%q: tensor<1x2x2x2xf32>, %k: tensor<1x2x2x2xf32>, %v: tensor<1x2x2x2xf32>, %dout: tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) {
    %rev:4 = enzyme.autodiff @attn_f32(%q, %k, %v, %dout) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)
    return %rev#1, %rev#2, %rev#3 : tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>
  }

  func.func @hvp_fn_ref(%q: tensor<1x2x2x2xf32>, %k: tensor<1x2x2x2xf32>, %v: tensor<1x2x2x2xf32>, %dout: tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) {
    %rev:4 = enzyme.autodiff @attn_f32_ref(%q, %k, %v, %dout) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)
    return %rev#1, %rev#2, %rev#3 : tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>
  }

  func.func @attn_gqa_bias_causal_padding(
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

  func.func @attn_grad_gqa_f64(
      %q: tensor<2x2x2x2xf64>,
      %k: tensor<2x2x1x2xf64>,
      %v: tensor<2x2x1x2xf64>,
      %out: tensor<2x2x2x2xf64>,
      %lse: tensor<2x2x2xf64>,
      %dout: tensor<2x2x2x2xf64>,
      %bias: tensor<1x2x2x2xf64>,
      %q_len: tensor<2xi32>,
      %kv_len: tensor<2xi32>) -> (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<1x2x2x2xf64>) {
    %dq, %dk, %dv, %dbias = "enzymexla.ml.attention_grad"(%q, %k, %v, %out, %lse, %dout, %bias, %q_len, %kv_len) <{
      is_causal = true,
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 0, 1, 1>,
      scale = 0.5 : f64
    }> : (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<1x2x2x2xf64>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<1x2x2x2xf64>)
    return %dq, %dk, %dv, %dbias : tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<1x2x2x2xf64>
  }

  func.func @attn_all_masked(
      %q: tensor<1x2x1x2xf32>,
      %k: tensor<1x2x1x2xf32>,
      %v: tensor<1x2x1x2xf32>,
      %mask: tensor<1x1x2x2xi1>) -> tensor<1x2x1x2xf32> {
    %out, %lse = "enzymexla.ml.attention"(%q, %k, %v, %mask) <{
      is_causal = false,
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 0>,
      scale = 0.5 : f64
    }> : (tensor<1x2x1x2xf32>, tensor<1x2x1x2xf32>, tensor<1x2x1x2xf32>, tensor<1x1x2x2xi1>) -> (tensor<1x2x1x2xf32>, tensor<1x1x2xf32>)
    return %out : tensor<1x2x1x2xf32>
  }

  func.func @main() {
    %q = stablehlo.constant dense<[[[[1.0, 0.5], [-0.5, 1.5]], [[0.2, -0.8], [1.2, 0.3]]]]> : tensor<1x2x2x2xf32>
    %k = stablehlo.constant dense<[[[[0.4, -0.2], [0.9, 1.1]], [[-0.6, 0.7], [0.1, -0.5]]]]> : tensor<1x2x2x2xf32>
    %v = stablehlo.constant dense<[[[[1.5, -0.5], [0.2, 0.8]], [[-1.0, 0.4], [0.6, -0.3]]]]> : tensor<1x2x2x2xf32>
    %dout = stablehlo.constant dense<[[[[0.3, -0.7], [1.1, 0.2]], [[-0.4, 0.9], [0.5, -0.6]]]]> : tensor<1x2x2x2xf32>
    %dout_b = stablehlo.constant dense<[[[[0.8, 0.1], [-0.3, 0.7]], [[0.6, -0.2], [-0.9, 0.4]]]]> : tensor<1x2x2x2xf32>
    %dlse = stablehlo.constant dense<[[[0.25, -0.4], [-0.15, 0.35]]]> : tensor<1x2x2xf32>
    %dlse_b = stablehlo.constant dense<[[[-0.5, 0.3], [0.45, -0.2]]]> : tensor<1x2x2xf32>

    // 1. First-order unbatched reverse & forward AD on @attn_f32
    %rev:4 = enzyme.autodiff @attn_f32(%q, %k, %v, %dout) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    %rev_ref:4 = enzyme.autodiff @attn_f32_ref(%q, %k, %v, %dout) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    check.expect_almost_eq %rev#0, %rev_ref#0 : tensor<1x2x2x2xf32>
    check.expect_almost_eq %rev#1, %rev_ref#1 : tensor<1x2x2x2xf32>
    check.expect_almost_eq %rev#2, %rev_ref#2 : tensor<1x2x2x2xf32>
    check.expect_almost_eq %rev#3, %rev_ref#3 : tensor<1x2x2x2xf32>

    %tq = stablehlo.constant dense<[[[[0.1, -0.2], [0.3, 0.4]], [[-0.5, 0.6], [0.7, -0.8]]]]> : tensor<1x2x2x2xf32>
    %tk = stablehlo.constant dense<[[[[-0.3, 0.1], [0.2, -0.4]], [[0.5, -0.2], [-0.1, 0.3]]]]> : tensor<1x2x2x2xf32>
    %tv = stablehlo.constant dense<[[[[0.2, 0.3], [-0.4, 0.1]], [[0.6, -0.5], [-0.2, 0.4]]]]> : tensor<1x2x2x2xf32>
    %tq_b = stablehlo.constant dense<[[[[-0.4, 0.3], [0.1, -0.5]], [[0.2, -0.3], [-0.6, 0.5]]]]> : tensor<1x2x2x2xf32>
    %tk_b = stablehlo.constant dense<[[[[0.2, -0.5], [-0.3, 0.6]], [[-0.4, 0.1], [0.3, -0.2]]]]> : tensor<1x2x2x2xf32>
    %tv_b = stablehlo.constant dense<[[[[-0.1, 0.4], [0.5, -0.3]], [[-0.2, 0.7], [0.4, -0.6]]]]> : tensor<1x2x2x2xf32>

    %fwd:2 = enzyme.fwddiff @attn_f32(%q, %tq, %k, %tk, %v, %tv) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    %fwd_ref:2 = enzyme.fwddiff @attn_f32_ref(%q, %tq, %k, %tk, %v, %tv) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    check.expect_almost_eq %fwd#0, %fwd_ref#0 : tensor<1x2x2x2xf32>
    check.expect_almost_eq %fwd#1, %fwd_ref#1 : tensor<1x2x2x2xf32>

    // 2. Active LSE + Batched (width = 2) Forward & Reverse AD on @attn_f32_lse
    %rev_only_lse:5 = enzyme.autodiff @attn_f32_lse(%q, %k, %v, %dlse) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    %rev_only_lse_ref:5 = enzyme.autodiff @attn_f32_lse_ref(%q, %k, %v, %dlse) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    check.expect_almost_eq %rev_only_lse#1, %rev_only_lse_ref#1 : tensor<1x2x2xf32>
    check.expect_almost_eq %rev_only_lse#2, %rev_only_lse_ref#2 : tensor<1x2x2x2xf32>
    check.expect_almost_eq %rev_only_lse#3, %rev_only_lse_ref#3 : tensor<1x2x2x2xf32>
    check.expect_almost_eq %rev_only_lse#4, %rev_only_lse_ref#4 : tensor<1x2x2x2xf32>

    %dout_0 = stablehlo.reshape %dout : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %dout_1 = stablehlo.reshape %dout_b : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %dout_w2 = stablehlo.concatenate %dout_0, %dout_1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %dlse_0 = stablehlo.reshape %dlse : (tensor<1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %dlse_1 = stablehlo.reshape %dlse_b : (tensor<1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %dlse_w2 = stablehlo.concatenate %dlse_0, %dlse_1, dim = 0 : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) -> tensor<2x1x2x2xf32>

    %rev_lse_w2:5 = enzyme.autodiff @attn_f32_lse(%q, %k, %v, %dout_w2, %dlse_w2) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      width = 2 : i64
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2x1x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2x1x2x2x2xf32>)

    %rev_lse_ref0:5 = enzyme.autodiff @attn_f32_lse_ref(%q, %k, %v, %dout, %dlse) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    %rev_lse_ref1:5 = enzyme.autodiff @attn_f32_lse_ref(%q, %k, %v, %dout_b, %dlse_b) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    %rlq0 = stablehlo.reshape %rev_lse_ref0#2 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %rlq1 = stablehlo.reshape %rev_lse_ref1#2 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %rlq_ref = stablehlo.concatenate %rlq0, %rlq1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %rlk0 = stablehlo.reshape %rev_lse_ref0#3 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %rlk1 = stablehlo.reshape %rev_lse_ref1#3 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %rlk_ref = stablehlo.concatenate %rlk0, %rlk1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %rlv0 = stablehlo.reshape %rev_lse_ref0#4 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %rlv1 = stablehlo.reshape %rev_lse_ref1#4 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %rlv_ref = stablehlo.concatenate %rlv0, %rlv1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    check.expect_almost_eq %rev_lse_w2#2, %rlq_ref : tensor<2x1x2x2x2xf32>
    check.expect_almost_eq %rev_lse_w2#3, %rlk_ref : tensor<2x1x2x2x2xf32>
    check.expect_almost_eq %rev_lse_w2#4, %rlv_ref : tensor<2x1x2x2x2xf32>

    %tq_0 = stablehlo.reshape %tq : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %tq_1 = stablehlo.reshape %tq_b : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %tq_w2 = stablehlo.concatenate %tq_0, %tq_1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %tk_0 = stablehlo.reshape %tk : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %tk_1 = stablehlo.reshape %tk_b : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %tk_w2 = stablehlo.concatenate %tk_0, %tk_1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %tv_0 = stablehlo.reshape %tv : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %tv_1 = stablehlo.reshape %tv_b : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %tv_w2 = stablehlo.concatenate %tv_0, %tv_1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %fwd_lse_w2:4 = enzyme.fwddiff @attn_f32_lse(%q, %tq_w2, %k, %tk_w2, %v, %tv_w2) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      width = 2 : i64
    }> : (tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<1x2x2xf32>, tensor<2x1x2x2xf32>)

    %fwd_lse_ref0:4 = enzyme.fwddiff @attn_f32_lse_ref(%q, %tq, %k, %tk, %v, %tv) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>)

    %fwd_lse_ref1:4 = enzyme.fwddiff @attn_f32_lse_ref(%q, %tq_b, %k, %tk_b, %v, %tv_b) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>)

    %flo0 = stablehlo.reshape %fwd_lse_ref0#1 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %flo1 = stablehlo.reshape %fwd_lse_ref1#1 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %flo_ref = stablehlo.concatenate %flo0, %flo1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %fll0 = stablehlo.reshape %fwd_lse_ref0#3 : (tensor<1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %fll1 = stablehlo.reshape %fwd_lse_ref1#3 : (tensor<1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %fll_ref = stablehlo.concatenate %fll0, %fll1, dim = 0 : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) -> tensor<2x1x2x2xf32>

    check.expect_almost_eq %fwd_lse_w2#1, %flo_ref : tensor<2x1x2x2x2xf32>
    check.expect_almost_eq %fwd_lse_w2#3, %fll_ref : tensor<2x1x2x2xf32>

    // 3. Second-order AD (Batched Forward-over-Reverse HVP and Batched Reverse-over-Reverse, width = 2)
    %hvp_fwd:6 = enzyme.fwddiff @hvp_fn(%q, %tq_w2, %k, %tk_w2, %v, %tv_w2, %dout, %dout_w2) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      width = 2 : i64
    }> : (tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>)

    %hvp_fwd_ref0:6 = enzyme.fwddiff @hvp_fn_ref(%q, %tq, %k, %tk, %v, %tv, %dout, %dout) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    %hvp_fwd_ref1:6 = enzyme.fwddiff @hvp_fn_ref(%q, %tq_b, %k, %tk_b, %v, %tv_b, %dout, %dout_b) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    %hfq0 = stablehlo.reshape %hvp_fwd_ref0#1 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hfq1 = stablehlo.reshape %hvp_fwd_ref1#1 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hfq_ref = stablehlo.concatenate %hfq0, %hfq1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %hfk0 = stablehlo.reshape %hvp_fwd_ref0#3 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hfk1 = stablehlo.reshape %hvp_fwd_ref1#3 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hfk_ref = stablehlo.concatenate %hfk0, %hfk1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %hfv0 = stablehlo.reshape %hvp_fwd_ref0#5 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hfv1 = stablehlo.reshape %hvp_fwd_ref1#5 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hfv_ref = stablehlo.concatenate %hfv0, %hfv1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    check.expect_almost_eq %hvp_fwd#1, %hfq_ref : tensor<2x1x2x2x2xf32>
    check.expect_almost_eq %hvp_fwd#3, %hfk_ref : tensor<2x1x2x2x2xf32>
    check.expect_almost_eq %hvp_fwd#5, %hfv_ref : tensor<2x1x2x2x2xf32>

    %hvp_rev:7 = enzyme.autodiff @hvp_fn(%q, %k, %v, %dout, %tq_w2, %tk_w2, %tv_w2) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      width = 2 : i64
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2x1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2x1x2x2x2xf32>)

    %hvp_rev_ref0:7 = enzyme.autodiff @hvp_fn_ref(%q, %k, %v, %dout, %tq, %tk, %tv) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    %hvp_rev_ref1:7 = enzyme.autodiff @hvp_fn_ref(%q, %k, %v, %dout, %tq_b, %tk_b, %tv_b) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>)

    %hrq0 = stablehlo.reshape %hvp_rev_ref0#3 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hrq1 = stablehlo.reshape %hvp_rev_ref1#3 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hrq_ref = stablehlo.concatenate %hrq0, %hrq1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %hrk0 = stablehlo.reshape %hvp_rev_ref0#4 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hrk1 = stablehlo.reshape %hvp_rev_ref1#4 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hrk_ref = stablehlo.concatenate %hrk0, %hrk1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %hrv0 = stablehlo.reshape %hvp_rev_ref0#5 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hrv1 = stablehlo.reshape %hvp_rev_ref1#5 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hrv_ref = stablehlo.concatenate %hrv0, %hrv1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %hrdo0 = stablehlo.reshape %hvp_rev_ref0#6 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hrdo1 = stablehlo.reshape %hvp_rev_ref1#6 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %hrdo_ref = stablehlo.concatenate %hrdo0, %hrdo1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    check.expect_almost_eq %hvp_rev#3, %hrq_ref : tensor<2x1x2x2x2xf32>
    check.expect_almost_eq %hvp_rev#4, %hrk_ref : tensor<2x1x2x2x2xf32>
    check.expect_almost_eq %hvp_rev#5, %hrv_ref : tensor<2x1x2x2x2xf32>
    check.expect_almost_eq %hvp_rev#6, %hrdo_ref : tensor<2x1x2x2x2xf32>

    // 4. GQA + shared B=1 bias + causal + padding: unbatched & batched (width = 2) forward + reverse AD
    %q2 = stablehlo.constant dense<[[[[1.0, 0.5], [-0.5, 1.5]], [[0.2, -0.8], [1.2, 0.3]]], [[[0.3, -0.4], [0.7, 0.2]], [[-0.9, 0.5], [0.4, -0.6]]]]> : tensor<2x2x2x2xf32>
    %k2 = stablehlo.constant dense<[[[[0.4, -0.2]], [[0.9, 1.1]]], [[[-0.6, 0.7]], [[0.1, -0.5]]]]> : tensor<2x2x1x2xf32>
    %v2 = stablehlo.constant dense<[[[[1.5, -0.5]], [[0.2, 0.8]]], [[[-1.0, 0.4]], [[0.6, -0.3]]]]> : tensor<2x2x1x2xf32>
    %bias2 = stablehlo.constant dense<[[[[0.1, -0.2], [0.3, 0.0]], [[-0.1, 0.4], [0.2, -0.3]]]]> : tensor<1x2x2x2xf32>
    %q_len2 = stablehlo.constant dense<[2, 1]> : tensor<2xi32>
    %kv_len2 = stablehlo.constant dense<[2, 1]> : tensor<2xi32>
    %dout2 = stablehlo.constant dense<[[[[0.3, -0.7], [1.1, 0.2]], [[-0.4, 0.9], [0.5, -0.6]]], [[[0.2, 0.8], [-0.3, 0.4]], [[0.6, -0.1], [-0.5, 0.7]]]]> : tensor<2x2x2x2xf32>
    %dout2_b = stablehlo.constant dense<[[[[-0.5, 0.2], [0.4, -0.8]], [[0.7, -0.3], [-0.2, 0.9]]], [[[-0.6, 0.3], [0.8, -0.1]], [[0.1, -0.4], [0.3, 0.5]]]]> : tensor<2x2x2x2xf32>
    %dlse2 = stablehlo.constant dense<[[[0.2, -0.3], [-0.1, 0.4]], [[0.5, -0.2], [-0.4, 0.3]]]> : tensor<2x2x2xf32>
    %dlse2_b = stablehlo.constant dense<[[[-0.4, 0.1], [0.3, -0.5]], [[-0.2, 0.6], [0.1, -0.7]]]> : tensor<2x2x2xf32>

    %tq2 = stablehlo.constant dense<[[[[0.2, -0.1], [0.4, 0.3]], [[-0.2, 0.5], [0.1, -0.3]]], [[[-0.4, 0.2], [0.3, -0.5]], [[0.6, 0.1], [-0.2, 0.4]]]]> : tensor<2x2x2x2xf32>
    %tk2 = stablehlo.constant dense<[[[[-0.3, 0.2]], [[0.1, -0.4]]], [[[0.5, -0.2]], [[-0.1, 0.3]]]]> : tensor<2x2x1x2xf32>
    %tv2 = stablehlo.constant dense<[[[[0.2, 0.3]], [[-0.4, 0.1]]], [[[0.6, -0.5]], [[-0.2, 0.4]]]]> : tensor<2x2x1x2xf32>
    %tbias2 = stablehlo.constant dense<[[[[-0.2, 0.3], [0.1, -0.4]], [[0.5, -0.1], [-0.3, 0.2]]]]> : tensor<1x2x2x2xf32>

    %tq2_b = stablehlo.constant dense<[[[[-0.3, 0.4], [-0.1, 0.2]], [[0.5, -0.4], [-0.3, 0.6]]], [[[0.2, -0.5], [-0.4, 0.3]], [[-0.1, 0.7], [0.4, -0.2]]]]> : tensor<2x2x2x2xf32>
    %tk2_b = stablehlo.constant dense<[[[[0.4, -0.1]], [[-0.2, 0.5]]], [[[-0.3, 0.4]], [[0.6, -0.2]]]]> : tensor<2x2x1x2xf32>
    %tv2_b = stablehlo.constant dense<[[[[-0.5, 0.2]], [[0.3, -0.6]]], [[[-0.4, 0.1]], [[0.5, 0.2]]]]> : tensor<2x2x1x2xf32>
    %tbias2_b = stablehlo.constant dense<[[[[0.3, -0.4], [-0.2, 0.5]], [[-0.6, 0.2], [0.4, -0.1]]]]> : tensor<1x2x2x2xf32>

    %rev2:6 = enzyme.autodiff @attn_gqa_bias_causal_padding(%q2, %k2, %v2, %bias2, %q_len2, %kv_len2, %dout2, %dlse2) <{
      activity = [
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>]
    }> : (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2x2x2xf32>, tensor<2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>)

    %rev2_b:6 = enzyme.autodiff @attn_gqa_bias_causal_padding(%q2, %k2, %v2, %bias2, %q_len2, %kv_len2, %dout2_b, %dlse2_b) <{
      activity = [
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>]
    }> : (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2x2x2xf32>, tensor<2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>)

    %fwd2:4 = enzyme.fwddiff @attn_gqa_bias_causal_padding(%q2, %tq2, %k2, %tk2, %v2, %tv2, %bias2, %tbias2, %q_len2, %kv_len2) <{
      activity = [
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>]
    }> : (tensor<2x2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2xf32>)

    %fwd2_b:4 = enzyme.fwddiff @attn_gqa_bias_causal_padding(%q2, %tq2_b, %k2, %tk2_b, %v2, %tv2_b, %bias2, %tbias2_b, %q_len2, %kv_len2) <{
      activity = [
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>]
    }> : (tensor<2x2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2xf32>)

    %lse_fin = stablehlo.is_finite %fwd2#2 : (tensor<2x2x2xf32>) -> tensor<2x2x2xi1>
    %z_lse = stablehlo.constant dense<0.0> : tensor<2x2x2xf32>
    %dlse2_masked = stablehlo.select %lse_fin, %dlse2, %z_lse : tensor<2x2x2xi1>, tensor<2x2x2xf32>

    %lhs_out = stablehlo.dot_general %fwd2#1, %dout2, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x2x2xf32>, tensor<2x2x2x2xf32>) -> tensor<f32>
    %lhs_lse = stablehlo.dot_general %fwd2#3, %dlse2_masked, batching_dims = [] x [], contracting_dims = [0, 1, 2] x [0, 1, 2] : (tensor<2x2x2xf32>, tensor<2x2x2xf32>) -> tensor<f32>
    %lhs_dot = stablehlo.add %lhs_out, %lhs_lse : tensor<f32>

    %dot_q = stablehlo.dot_general %rev2#2, %tq2, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x2x2xf32>, tensor<2x2x2x2xf32>) -> tensor<f32>
    %dot_k = stablehlo.dot_general %rev2#3, %tk2, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>) -> tensor<f32>
    %dot_v = stablehlo.dot_general %rev2#4, %tv2, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>) -> tensor<f32>
    %dot_b = stablehlo.dot_general %rev2#5, %tbias2, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<f32>
    %sum1 = stablehlo.add %dot_q, %dot_k : tensor<f32>
    %sum2 = stablehlo.add %sum1, %dot_v : tensor<f32>
    %rhs_dot = stablehlo.add %sum2, %dot_b : tensor<f32>
    check.expect_almost_eq %lhs_dot, %rhs_dot : tensor<f32>

    // Batched width = 2 check on GQA + shared B=1 bias + causal + padding
    %dout2_0 = stablehlo.reshape %dout2 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %dout2_1 = stablehlo.reshape %dout2_b : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %dout2_w2 = stablehlo.concatenate %dout2_0, %dout2_1, dim = 0 : (tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>

    %dlse2_0 = stablehlo.reshape %dlse2 : (tensor<2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %dlse2_1 = stablehlo.reshape %dlse2_b : (tensor<2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %dlse2_w2 = stablehlo.concatenate %dlse2_0, %dlse2_1, dim = 0 : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<2x2x2x2xf32>

    %rev2_w2:6 = enzyme.autodiff @attn_gqa_bias_causal_padding(%q2, %k2, %v2, %bias2, %q_len2, %kv_len2, %dout2_w2, %dlse2_w2) <{
      activity = [
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      width = 2 : i64
    }> : (tensor<2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2x2x2x2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2x2x2xf32>, tensor<2x2x2x1x2xf32>, tensor<2x2x2x1x2xf32>, tensor<2x1x2x2x2xf32>)

    %rev2_dq0 = stablehlo.reshape %rev2#2 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %rev2_dq1 = stablehlo.reshape %rev2_b#2 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %exp_dq2 = stablehlo.concatenate %rev2_dq0, %rev2_dq1, dim = 0 : (tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>

    %rev2_dk0 = stablehlo.reshape %rev2#3 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
    %rev2_dk1 = stablehlo.reshape %rev2_b#3 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
    %exp_dk2 = stablehlo.concatenate %rev2_dk0, %rev2_dk1, dim = 0 : (tensor<1x2x2x1x2xf32>, tensor<1x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>

    %rev2_dv0 = stablehlo.reshape %rev2#4 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
    %rev2_dv1 = stablehlo.reshape %rev2_b#4 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
    %exp_dv2 = stablehlo.concatenate %rev2_dv0, %rev2_dv1, dim = 0 : (tensor<1x2x2x1x2xf32>, tensor<1x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>

    %rev2_db0 = stablehlo.reshape %rev2#5 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %rev2_db1 = stablehlo.reshape %rev2_b#5 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %exp_db2 = stablehlo.concatenate %rev2_db0, %rev2_db1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    check.expect_almost_eq %rev2_w2#2, %exp_dq2 : tensor<2x2x2x2x2xf32>
    check.expect_almost_eq %rev2_w2#3, %exp_dk2 : tensor<2x2x2x1x2xf32>
    check.expect_almost_eq %rev2_w2#4, %exp_dv2 : tensor<2x2x2x1x2xf32>
    check.expect_almost_eq %rev2_w2#5, %exp_db2 : tensor<2x1x2x2x2xf32>

    %tq2_0 = stablehlo.reshape %tq2 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %tq2_1 = stablehlo.reshape %tq2_b : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %tq2_w2 = stablehlo.concatenate %tq2_0, %tq2_1, dim = 0 : (tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>

    %tk2_0 = stablehlo.reshape %tk2 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
    %tk2_1 = stablehlo.reshape %tk2_b : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
    %tk2_w2 = stablehlo.concatenate %tk2_0, %tk2_1, dim = 0 : (tensor<1x2x2x1x2xf32>, tensor<1x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>

    %tv2_0 = stablehlo.reshape %tv2 : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
    %tv2_1 = stablehlo.reshape %tv2_b : (tensor<2x2x1x2xf32>) -> tensor<1x2x2x1x2xf32>
    %tv2_w2 = stablehlo.concatenate %tv2_0, %tv2_1, dim = 0 : (tensor<1x2x2x1x2xf32>, tensor<1x2x2x1x2xf32>) -> tensor<2x2x2x1x2xf32>

    %tb2_0 = stablehlo.reshape %tbias2 : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %tb2_1 = stablehlo.reshape %tbias2_b : (tensor<1x2x2x2xf32>) -> tensor<1x1x2x2x2xf32>
    %tb2_w2 = stablehlo.concatenate %tb2_0, %tb2_1, dim = 0 : (tensor<1x1x2x2x2xf32>, tensor<1x1x2x2x2xf32>) -> tensor<2x1x2x2x2xf32>

    %fwd2_w2:4 = enzyme.fwddiff @attn_gqa_bias_causal_padding(%q2, %tq2_w2, %k2, %tk2_w2, %v2, %tv2_w2, %bias2, %tb2_w2, %q_len2, %kv_len2) <{
      activity = [
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      width = 2 : i64
    }> : (tensor<2x2x2x2xf32>, tensor<2x2x2x2x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x2x1x2xf32>, tensor<2x2x1x2xf32>, tensor<2x2x2x1x2xf32>, tensor<1x2x2x2xf32>, tensor<2x1x2x2x2xf32>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf32>, tensor<2x2x2x2x2xf32>, tensor<2x2x2xf32>, tensor<2x2x2x2xf32>)

    %fwd2_out0 = stablehlo.reshape %fwd2#1 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %fwd2_out1 = stablehlo.reshape %fwd2_b#1 : (tensor<2x2x2x2xf32>) -> tensor<1x2x2x2x2xf32>
    %exp_fout2 = stablehlo.concatenate %fwd2_out0, %fwd2_out1, dim = 0 : (tensor<1x2x2x2x2xf32>, tensor<1x2x2x2x2xf32>) -> tensor<2x2x2x2x2xf32>

    %fwd2_lse0 = stablehlo.reshape %fwd2#3 : (tensor<2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %fwd2_lse1 = stablehlo.reshape %fwd2_b#3 : (tensor<2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %exp_flse2 = stablehlo.concatenate %fwd2_lse0, %fwd2_lse1, dim = 0 : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<2x2x2x2xf32>

    check.expect_almost_eq %fwd2_w2#1, %exp_fout2 : tensor<2x2x2x2x2xf32>
    check.expect_almost_eq %fwd2_w2#3, %exp_flse2 : tensor<2x2x2x2xf32>

    // 5. Direct Forward, Batched Forward, Reverse, and Batched Reverse AD on enzymexla.ml.attention_grad (off-manifold + GQA + shared B=1 bias + causal + padding)
    %gq = stablehlo.convert %q2 : (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf64>
    %gk = stablehlo.convert %k2 : (tensor<2x2x1x2xf32>) -> tensor<2x2x1x2xf64>
    %gv = stablehlo.convert %v2 : (tensor<2x2x1x2xf32>) -> tensor<2x2x1x2xf64>
    %gout = stablehlo.constant dense<[[[[0.6, -0.3], [0.2, 0.9]], [[-0.4, 0.7], [0.5, -0.1]]], [[[0.8, 0.2], [-0.5, 0.3]], [[0.1, -0.8], [0.4, 0.6]]]]> : tensor<2x2x2x2xf64>
    %glse = stablehlo.constant dense<[[[1.1, 0.8], [0.9, 1.3]], [[0.7, 1.2], [1.0, 0.6]]]> : tensor<2x2x2xf64>
    %gdout = stablehlo.convert %dout2 : (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf64>
    %gbias = stablehlo.convert %bias2 : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf64>

    %gtq = stablehlo.convert %tq2 : (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf64>
    %gtk = stablehlo.convert %tk2 : (tensor<2x2x1x2xf32>) -> tensor<2x2x1x2xf64>
    %gtv = stablehlo.convert %tv2 : (tensor<2x2x1x2xf32>) -> tensor<2x2x1x2xf64>
    %gtout = stablehlo.constant dense<[[[[-0.2, 0.4], [0.3, -0.1]], [[0.5, -0.3], [-0.4, 0.2]]], [[[0.1, -0.5], [0.4, 0.2]], [[-0.3, 0.6], [0.2, -0.4]]]]> : tensor<2x2x2x2xf64>
    %gtlse = stablehlo.constant dense<[[[-0.3, 0.2], [0.4, -0.1]], [[0.2, -0.4], [-0.2, 0.3]]]> : tensor<2x2x2xf64>
    %gtdout = stablehlo.convert %dout2_b : (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf64>
    %gtbias = stablehlo.convert %tbias2 : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf64>

    %gfwd:8 = enzyme.fwddiff @attn_grad_gqa_f64(%gq, %gtq, %gk, %gtk, %gv, %gtv, %gout, %gtout, %glse, %gtlse, %gdout, %gtdout, %gbias, %gtbias, %q_len2, %kv_len2) <{
      activity = [
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>
      ]
    }> : (tensor<2x2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2xf64>, tensor<2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<1x2x2x2xf64>, tensor<1x2x2x2xf64>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<1x2x2x2xf64>, tensor<1x2x2x2xf64>)

    // Central finite-difference verification of @attn_grad_gqa_f64 forward tangent
    %eps_q = stablehlo.constant dense<1.0e-6> : tensor<2x2x2x2xf64>
    %eps_k = stablehlo.constant dense<1.0e-6> : tensor<2x2x1x2xf64>
    %eps_l = stablehlo.constant dense<1.0e-6> : tensor<2x2x2xf64>
    %eps_b = stablehlo.constant dense<1.0e-6> : tensor<1x2x2x2xf64>
    %inv2e_q = stablehlo.constant dense<5.0e5> : tensor<2x2x2x2xf64>
    %inv2e_k = stablehlo.constant dense<5.0e5> : tensor<2x2x1x2xf64>
    %inv2e_b = stablehlo.constant dense<5.0e5> : tensor<1x2x2x2xf64>

    %dq_step = stablehlo.multiply %gtq, %eps_q : tensor<2x2x2x2xf64>
    %dk_step = stablehlo.multiply %gtk, %eps_k : tensor<2x2x1x2xf64>
    %dv_step = stablehlo.multiply %gtv, %eps_k : tensor<2x2x1x2xf64>
    %do_step = stablehlo.multiply %gtout, %eps_q : tensor<2x2x2x2xf64>
    %dl_step = stablehlo.multiply %gtlse, %eps_l : tensor<2x2x2xf64>
    %ddo_step = stablehlo.multiply %gtdout, %eps_q : tensor<2x2x2x2xf64>
    %db_step = stablehlo.multiply %gtbias, %eps_b : tensor<1x2x2x2xf64>

    %gqp = stablehlo.add %gq, %dq_step : tensor<2x2x2x2xf64>
    %gkp = stablehlo.add %gk, %dk_step : tensor<2x2x1x2xf64>
    %gvp = stablehlo.add %gv, %dv_step : tensor<2x2x1x2xf64>
    %goutp = stablehlo.add %gout, %do_step : tensor<2x2x2x2xf64>
    %glsep = stablehlo.add %glse, %dl_step : tensor<2x2x2xf64>
    %gdoutp = stablehlo.add %gdout, %ddo_step : tensor<2x2x2x2xf64>
    %gbiasp = stablehlo.add %gbias, %db_step : tensor<1x2x2x2xf64>

    %gqm = stablehlo.subtract %gq, %dq_step : tensor<2x2x2x2xf64>
    %gkm = stablehlo.subtract %gk, %dk_step : tensor<2x2x1x2xf64>
    %gvm = stablehlo.subtract %gv, %dv_step : tensor<2x2x1x2xf64>
    %goutm = stablehlo.subtract %gout, %do_step : tensor<2x2x2x2xf64>
    %glsem = stablehlo.subtract %glse, %dl_step : tensor<2x2x2xf64>
    %gdoutm = stablehlo.subtract %gdout, %ddo_step : tensor<2x2x2x2xf64>
    %gbiasm = stablehlo.subtract %gbias, %db_step : tensor<1x2x2x2xf64>

    %gp:4 = func.call @attn_grad_gqa_f64(%gqp, %gkp, %gvp, %goutp, %glsep, %gdoutp, %gbiasp, %q_len2, %kv_len2) : (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<1x2x2x2xf64>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<1x2x2x2xf64>)
    %gm:4 = func.call @attn_grad_gqa_f64(%gqm, %gkm, %gvm, %goutm, %glsem, %gdoutm, %gbiasm, %q_len2, %kv_len2) : (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<1x2x2x2xf64>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<1x2x2x2xf64>)

    %fd_dq_diff = stablehlo.subtract %gp#0, %gm#0 : tensor<2x2x2x2xf64>
    %fd_dk_diff = stablehlo.subtract %gp#1, %gm#1 : tensor<2x2x1x2xf64>
    %fd_dv_diff = stablehlo.subtract %gp#2, %gm#2 : tensor<2x2x1x2xf64>
    %fd_db_diff = stablehlo.subtract %gp#3, %gm#3 : tensor<1x2x2x2xf64>

    %fd_dq = stablehlo.multiply %fd_dq_diff, %inv2e_q : tensor<2x2x2x2xf64>
    %fd_dk = stablehlo.multiply %fd_dk_diff, %inv2e_k : tensor<2x2x1x2xf64>
    %fd_dv = stablehlo.multiply %fd_dv_diff, %inv2e_k : tensor<2x2x1x2xf64>
    %fd_db = stablehlo.multiply %fd_db_diff, %inv2e_b : tensor<1x2x2x2xf64>

    check.expect_almost_eq %gfwd#1, %fd_dq : tensor<2x2x2x2xf64>
    check.expect_almost_eq %gfwd#3, %fd_dk : tensor<2x2x1x2xf64>
    check.expect_almost_eq %gfwd#5, %fd_dv : tensor<2x2x1x2xf64>
    check.expect_almost_eq %gfwd#7, %fd_db : tensor<1x2x2x2xf64>

    // Reverse AD (width = 1 & width = 2) and Batched Forward AD (width = 2) on @attn_grad_gqa_f64
    %bar_dq = stablehlo.convert %tq2_b : (tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf64>
    %bar_dk = stablehlo.convert %tk2_b : (tensor<2x2x1x2xf32>) -> tensor<2x2x1x2xf64>
    %bar_dv = stablehlo.convert %tv2_b : (tensor<2x2x1x2xf32>) -> tensor<2x2x1x2xf64>
    %bar_db = stablehlo.convert %tbias2_b : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf64>

    %grev:11 = enzyme.autodiff @attn_grad_gqa_f64(%gq, %gk, %gv, %gout, %glse, %gdout, %gbias, %q_len2, %kv_len2, %bar_dq, %bar_dk, %bar_dv, %bar_db) <{
      activity = [
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>
      ]
    }> : (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<1x2x2x2xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<1x2x2x2xf64>) -> (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<1x2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<1x2x2x2xf64>)

    %glhs0 = stablehlo.dot_general %gfwd#1, %bar_dq, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x2x2xf64>, tensor<2x2x2x2xf64>) -> tensor<f64>
    %glhs1 = stablehlo.dot_general %gfwd#3, %bar_dk, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>) -> tensor<f64>
    %glhs2 = stablehlo.dot_general %gfwd#5, %bar_dv, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>) -> tensor<f64>
    %glhs3 = stablehlo.dot_general %gfwd#7, %bar_db, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<1x2x2x2xf64>, tensor<1x2x2x2xf64>) -> tensor<f64>
    %glhs01 = stablehlo.add %glhs0, %glhs1 : tensor<f64>
    %glhs23 = stablehlo.add %glhs2, %glhs3 : tensor<f64>
    %glhs = stablehlo.add %glhs01, %glhs23 : tensor<f64>

    %grhs0 = stablehlo.dot_general %grev#4, %gtq, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x2x2xf64>, tensor<2x2x2x2xf64>) -> tensor<f64>
    %grhs1 = stablehlo.dot_general %grev#5, %gtk, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>) -> tensor<f64>
    %grhs2 = stablehlo.dot_general %grev#6, %gtv, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>) -> tensor<f64>
    %grhs3 = stablehlo.dot_general %grev#7, %gtout, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x2x2xf64>, tensor<2x2x2x2xf64>) -> tensor<f64>
    %grhs4 = stablehlo.dot_general %grev#8, %gtlse, batching_dims = [] x [], contracting_dims = [0, 1, 2] x [0, 1, 2] : (tensor<2x2x2xf64>, tensor<2x2x2xf64>) -> tensor<f64>
    %grhs5 = stablehlo.dot_general %grev#9, %gtdout, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<2x2x2x2xf64>, tensor<2x2x2x2xf64>) -> tensor<f64>
    %grhs6 = stablehlo.dot_general %grev#10, %gtbias, batching_dims = [] x [], contracting_dims = [0, 1, 2, 3] x [0, 1, 2, 3] : (tensor<1x2x2x2xf64>, tensor<1x2x2x2xf64>) -> tensor<f64>
    %grs01 = stablehlo.add %grhs0, %grhs1 : tensor<f64>
    %grs23 = stablehlo.add %grhs2, %grhs3 : tensor<f64>
    %grs45 = stablehlo.add %grhs4, %grhs5 : tensor<f64>
    %grs0123 = stablehlo.add %grs01, %grs23 : tensor<f64>
    %grs456 = stablehlo.add %grs45, %grhs6 : tensor<f64>
    %grhs = stablehlo.add %grs0123, %grs456 : tensor<f64>
    check.expect_almost_eq %glhs, %grhs : tensor<f64>

    // Batched width = 2 Forward & Reverse AD on @attn_grad_gqa_f64
    %gtq_r = stablehlo.reshape %gtq : (tensor<2x2x2x2xf64>) -> tensor<1x2x2x2x2xf64>
    %bar_dq_r = stablehlo.reshape %bar_dq : (tensor<2x2x2x2xf64>) -> tensor<1x2x2x2x2xf64>
    %gtq_w2 = stablehlo.concatenate %gtq_r, %bar_dq_r, dim = 0 : (tensor<1x2x2x2x2xf64>, tensor<1x2x2x2x2xf64>) -> tensor<2x2x2x2x2xf64>

    %gtk_r = stablehlo.reshape %gtk : (tensor<2x2x1x2xf64>) -> tensor<1x2x2x1x2xf64>
    %bar_dk_r = stablehlo.reshape %bar_dk : (tensor<2x2x1x2xf64>) -> tensor<1x2x2x1x2xf64>
    %gtk_w2 = stablehlo.concatenate %gtk_r, %bar_dk_r, dim = 0 : (tensor<1x2x2x1x2xf64>, tensor<1x2x2x1x2xf64>) -> tensor<2x2x2x1x2xf64>

    %gtv_r = stablehlo.reshape %gtv : (tensor<2x2x1x2xf64>) -> tensor<1x2x2x1x2xf64>
    %bar_dv_r = stablehlo.reshape %bar_dv : (tensor<2x2x1x2xf64>) -> tensor<1x2x2x1x2xf64>
    %gtv_w2 = stablehlo.concatenate %gtv_r, %bar_dv_r, dim = 0 : (tensor<1x2x2x1x2xf64>, tensor<1x2x2x1x2xf64>) -> tensor<2x2x2x1x2xf64>

    %gtout_r = stablehlo.reshape %gtout : (tensor<2x2x2x2xf64>) -> tensor<1x2x2x2x2xf64>
    %gtout_w2 = stablehlo.concatenate %gtout_r, %gtout_r, dim = 0 : (tensor<1x2x2x2x2xf64>, tensor<1x2x2x2x2xf64>) -> tensor<2x2x2x2x2xf64>

    %gtlse_r = stablehlo.reshape %gtlse : (tensor<2x2x2xf64>) -> tensor<1x2x2x2xf64>
    %gtlse_w2 = stablehlo.concatenate %gtlse_r, %gtlse_r, dim = 0 : (tensor<1x2x2x2xf64>, tensor<1x2x2x2xf64>) -> tensor<2x2x2x2xf64>

    %gtdout_r = stablehlo.reshape %gtdout : (tensor<2x2x2x2xf64>) -> tensor<1x2x2x2x2xf64>
    %gtdout_w2 = stablehlo.concatenate %gtdout_r, %gtdout_r, dim = 0 : (tensor<1x2x2x2x2xf64>, tensor<1x2x2x2x2xf64>) -> tensor<2x2x2x2x2xf64>

    %gtbias_r = stablehlo.reshape %gtbias : (tensor<1x2x2x2xf64>) -> tensor<1x1x2x2x2xf64>
    %bar_db_r = stablehlo.reshape %bar_db : (tensor<1x2x2x2xf64>) -> tensor<1x1x2x2x2xf64>
    %gtbias_w2 = stablehlo.concatenate %gtbias_r, %bar_db_r, dim = 0 : (tensor<1x1x2x2x2xf64>, tensor<1x1x2x2x2xf64>) -> tensor<2x1x2x2x2xf64>

    %gfwd_w2:8 = enzyme.fwddiff @attn_grad_gqa_f64(%gq, %gtq_w2, %gk, %gtk_w2, %gv, %gtv_w2, %gout, %gtout_w2, %glse, %gtlse_w2, %gdout, %gtdout_w2, %gbias, %gtbias_w2, %q_len2, %kv_len2) <{
      activity = [
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>,
        #enzyme.activity<enzyme_dup>
      ],
      width = 2 : i64
    }> : (tensor<2x2x2x2xf64>, tensor<2x2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x1x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2x2x2xf64>, tensor<2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2x2x2xf64>, tensor<1x2x2x2xf64>, tensor<2x1x2x2x2xf64>, tensor<2xi32>, tensor<2xi32>) -> (tensor<2x2x2x2xf64>, tensor<2x2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x1x2xf64>, tensor<1x2x2x2xf64>, tensor<2x1x2x2x2xf64>)

    %gfwd_s0_dq = stablehlo.slice %gfwd_w2#1 [0:1, 0:2, 0:2, 0:2, 0:2] : (tensor<2x2x2x2x2xf64>) -> tensor<1x2x2x2x2xf64>
    %gfwd_s0_dq_4d = stablehlo.reshape %gfwd_s0_dq : (tensor<1x2x2x2x2xf64>) -> tensor<2x2x2x2xf64>
    %gfwd_s0_dk = stablehlo.slice %gfwd_w2#3 [0:1, 0:2, 0:2, 0:1, 0:2] : (tensor<2x2x2x1x2xf64>) -> tensor<1x2x2x1x2xf64>
    %gfwd_s0_dk_4d = stablehlo.reshape %gfwd_s0_dk : (tensor<1x2x2x1x2xf64>) -> tensor<2x2x1x2xf64>
    %gfwd_s0_dv = stablehlo.slice %gfwd_w2#5 [0:1, 0:2, 0:2, 0:1, 0:2] : (tensor<2x2x2x1x2xf64>) -> tensor<1x2x2x1x2xf64>
    %gfwd_s0_dv_4d = stablehlo.reshape %gfwd_s0_dv : (tensor<1x2x2x1x2xf64>) -> tensor<2x2x1x2xf64>
    %gfwd_s0_db = stablehlo.slice %gfwd_w2#7 [0:1, 0:1, 0:2, 0:2, 0:2] : (tensor<2x1x2x2x2xf64>) -> tensor<1x1x2x2x2xf64>
    %gfwd_s0_db_4d = stablehlo.reshape %gfwd_s0_db : (tensor<1x1x2x2x2xf64>) -> tensor<1x2x2x2xf64>

    check.expect_almost_eq %gfwd_s0_dq_4d, %gfwd#1 : tensor<2x2x2x2xf64>
    check.expect_almost_eq %gfwd_s0_dk_4d, %gfwd#3 : tensor<2x2x1x2xf64>
    check.expect_almost_eq %gfwd_s0_dv_4d, %gfwd#5 : tensor<2x2x1x2xf64>
    check.expect_almost_eq %gfwd_s0_db_4d, %gfwd#7 : tensor<1x2x2x2xf64>

    %grev_w2:11 = enzyme.autodiff @attn_grad_gqa_f64(%gq, %gk, %gv, %gout, %glse, %gdout, %gbias, %q_len2, %kv_len2, %gtq_w2, %gtk_w2, %gtv_w2, %gtbias_w2) <{
      activity = [
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>
      ],
      width = 2 : i64
    }> : (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<1x2x2x2xf64>, tensor<2xi32>, tensor<2xi32>, tensor<2x2x2x2x2xf64>, tensor<2x2x2x1x2xf64>, tensor<2x2x2x1x2xf64>, tensor<2x1x2x2x2xf64>) -> (tensor<2x2x2x2xf64>, tensor<2x2x1x2xf64>, tensor<2x2x1x2xf64>, tensor<1x2x2x2xf64>, tensor<2x2x2x2x2xf64>, tensor<2x2x2x1x2xf64>, tensor<2x2x2x1x2xf64>, tensor<2x2x2x2x2xf64>, tensor<2x2x2x2xf64>, tensor<2x2x2x2x2xf64>, tensor<2x1x2x2x2xf64>)

    %grev_s1_q = stablehlo.slice %grev_w2#4 [1:2, 0:2, 0:2, 0:2, 0:2] : (tensor<2x2x2x2x2xf64>) -> tensor<1x2x2x2x2xf64>
    %grev_s1_q_4d = stablehlo.reshape %grev_s1_q : (tensor<1x2x2x2x2xf64>) -> tensor<2x2x2x2xf64>
    %grev_s1_k = stablehlo.slice %grev_w2#5 [1:2, 0:2, 0:2, 0:1, 0:2] : (tensor<2x2x2x1x2xf64>) -> tensor<1x2x2x1x2xf64>
    %grev_s1_k_4d = stablehlo.reshape %grev_s1_k : (tensor<1x2x2x1x2xf64>) -> tensor<2x2x1x2xf64>
    %grev_s1_v = stablehlo.slice %grev_w2#6 [1:2, 0:2, 0:2, 0:1, 0:2] : (tensor<2x2x2x1x2xf64>) -> tensor<1x2x2x1x2xf64>
    %grev_s1_v_4d = stablehlo.reshape %grev_s1_v : (tensor<1x2x2x1x2xf64>) -> tensor<2x2x1x2xf64>
    %grev_s1_o = stablehlo.slice %grev_w2#7 [1:2, 0:2, 0:2, 0:2, 0:2] : (tensor<2x2x2x2x2xf64>) -> tensor<1x2x2x2x2xf64>
    %grev_s1_o_4d = stablehlo.reshape %grev_s1_o : (tensor<1x2x2x2x2xf64>) -> tensor<2x2x2x2xf64>
    %grev_s1_l = stablehlo.slice %grev_w2#8 [1:2, 0:2, 0:2, 0:2] : (tensor<2x2x2x2xf64>) -> tensor<1x2x2x2xf64>
    %grev_s1_l_3d = stablehlo.reshape %grev_s1_l : (tensor<1x2x2x2xf64>) -> tensor<2x2x2xf64>
    %grev_s1_do = stablehlo.slice %grev_w2#9 [1:2, 0:2, 0:2, 0:2, 0:2] : (tensor<2x2x2x2x2xf64>) -> tensor<1x2x2x2x2xf64>
    %grev_s1_do_4d = stablehlo.reshape %grev_s1_do : (tensor<1x2x2x2x2xf64>) -> tensor<2x2x2x2xf64>
    %grev_s1_b = stablehlo.slice %grev_w2#10 [1:2, 0:1, 0:2, 0:2, 0:2] : (tensor<2x1x2x2x2xf64>) -> tensor<1x1x2x2x2xf64>
    %grev_s1_b_4d = stablehlo.reshape %grev_s1_b : (tensor<1x1x2x2x2xf64>) -> tensor<1x2x2x2xf64>

    check.expect_almost_eq %grev_s1_q_4d, %grev#4 : tensor<2x2x2x2xf64>
    check.expect_almost_eq %grev_s1_k_4d, %grev#5 : tensor<2x2x1x2xf64>
    check.expect_almost_eq %grev_s1_v_4d, %grev#6 : tensor<2x2x1x2xf64>
    check.expect_almost_eq %grev_s1_o_4d, %grev#7 : tensor<2x2x2x2xf64>
    check.expect_almost_eq %grev_s1_l_3d, %grev#8 : tensor<2x2x2xf64>
    check.expect_almost_eq %grev_s1_do_4d, %grev#9 : tensor<2x2x2x2xf64>
    check.expect_almost_eq %grev_s1_b_4d, %grev#10 : tensor<1x2x2x2xf64>

    // 6. Verify all-masked query row produces zero primal output AND zero gradient (not NaN)
    %qm = stablehlo.constant dense<[[[[1.0, 2.0]], [[3.0, 4.0]]]]> : tensor<1x2x1x2xf32>
    %km = stablehlo.constant dense<[[[[0.5, -0.5]], [[1.5, 0.5]]]]> : tensor<1x2x1x2xf32>
    %vm = stablehlo.constant dense<[[[[2.0, -1.0]], [[-0.5, 3.0]]]]> : tensor<1x2x1x2xf32>
    // Query 0 attends to nobody (false, false); Query 1 attends to Key 0 only (true, false)
    %maskm = stablehlo.constant dense<[[[[false, false], [true, false]]]]> : tensor<1x1x2x2xi1>
    %doutm = stablehlo.constant dense<[[[[1.0, 1.0]], [[1.0, 1.0]]]]> : tensor<1x2x1x2xf32>
    %revm:4 = enzyme.autodiff @attn_all_masked(%qm, %km, %vm, %maskm, %doutm) <{
      activity = [
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_const>
      ],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<1x2x1x2xf32>, tensor<1x2x1x2xf32>, tensor<1x2x1x2xf32>, tensor<1x1x2x2xi1>, tensor<1x2x1x2xf32>) -> (tensor<1x2x1x2xf32>, tensor<1x2x1x2xf32>, tensor<1x2x1x2xf32>, tensor<1x2x1x2xf32>)

    check.expect_almost_eq_const %revm#0, dense<[[[[0.0, 0.0]], [[2.0, -1.0]]]]> : tensor<1x2x1x2xf32>
    check.expect_almost_eq_const %revm#1, dense<0.0> : tensor<1x2x1x2xf32>
    check.expect_almost_eq_const %revm#2, dense<0.0> : tensor<1x2x1x2xf32>
    check.expect_almost_eq_const %revm#3, dense<[[[[1.0, 1.0]], [[0.0, 0.0]]]]> : tensor<1x2x1x2xf32>

    return
  }
}

// MHA-REV-LABEL: func.func @attn_mha
// MHA-REV: %[[OUT:.*]], %[[LSE:.*]] = "enzymexla.ml.attention"(%arg0, %arg1, %arg2)
// MHA-REV: %[[DQ:.*]], %[[DK:.*]], %[[DV:.*]] = "enzymexla.ml.attention_grad"(%arg0, %arg1, %arg2, %[[OUT]], %[[LSE]], %arg3)
// MHA-REV: return %[[DQ]], %[[DK]], %[[DV]]

// MHA-REV-CUDA-LABEL: func.func @attn_mha
// MHA-REV-CUDA: stablehlo.custom_call @__cudnn$fmhaSoftmax(
// MHA-REV-CUDA: stablehlo.custom_call @__cudnn$fmhaSoftmaxBackward(
