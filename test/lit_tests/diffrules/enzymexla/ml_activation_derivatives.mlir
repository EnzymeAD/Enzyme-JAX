// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=relu_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-ml | FileCheck %s --check-prefix=RELU-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=gelu_none_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-ml | FileCheck %s --check-prefix=GELU-NONE-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=gelu_tanh_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-ml | FileCheck %s --check-prefix=GELU-TANH-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=gelu_sigmoid_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-ml | FileCheck %s --check-prefix=GELU-SIGMOID-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=softplus_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-ml | FileCheck %s --check-prefix=SOFTPLUS-REV
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --lower-enzymexla-ml --chlo-legalize-to-stablehlo --canonicalize --arith-raise | stablehlo-translate - --interpret

module {
  func.func @relu_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.ml.relu %x : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @gelu_none_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.ml.gelu %x, approximation = NONE : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @gelu_tanh_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.ml.gelu %x, approximation = TANH : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @gelu_sigmoid_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.ml.gelu %x, approximation = SIGMOID : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @softplus_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.ml.softplus %x : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @main() {
    %x_common = stablehlo.constant dense<[0.0, -1.0, 1.0, 2.0, 100.0, -100.0]> : tensor<6xf32>
    %d_common = stablehlo.constant dense<1.0> : tensor<6xf32>

    %relu_res:2 = enzyme.autodiff @relu_fn(%x_common, %d_common) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)

    // d/dx relu([0, -1, 1, 2, 100, -100]) = [0, 0, 1, 1, 1, 0]
    check.expect_eq_const %relu_res#1, dense<[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]> : tensor<6xf32>

    // GELU derivativ
    %gelu_none_res:2 = enzyme.autodiff @gelu_none_fn(%x_common, %d_common) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %gelu_none_res#1, dense<[0.5, -8.331547e-02, 1.0833155, 1.0852318, 1.0, 0.0]> : tensor<6xf32>

    %gelu_tanh_res:2 = enzyme.autodiff @gelu_tanh_fn(%x_common, %d_common) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %gelu_tanh_res#1, dense<[0.5, -8.296408e-02, 1.0829641, 1.0860993, 1.0, 0.0]> : tensor<6xf32>

    %gelu_sigmoid_res:2 = enzyme.autodiff @gelu_sigmoid_fn(%x_common, %d_common) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %gelu_sigmoid_res#1, dense<[0.5, -6.777961e-02, 1.0677796, 1.0738153, 1.0, 0.0]> : tensor<6xf32>

    // d/dx softplus(x) = sigmoid(x)
    %softplus_res:2 = enzyme.autodiff @softplus_fn(%x_common, %d_common) {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %softplus_res#1, dense<[0.5, 0.26894143, 0.7310586, 0.8807971, 1.0, 0.0]> : tensor<6xf32>

    return
  }
}

// RELU-REV-LABEL: func.func @relu_fn
// RELU-REV: %[[Z:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// RELU-REV: %[[DR:.*]] = stablehlo.add %arg1, %[[Z]] : tensor<6xf32>
// RELU-REV: %[[MASK:.*]] = stablehlo.compare  GT, %arg0, %[[Z]] : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
// RELU-REV: %[[SEL:.*]] = stablehlo.select %[[MASK]], %[[DR]], %[[Z]] : tensor<6xi1>, tensor<6xf32>
// RELU-REV: stablehlo.add %[[SEL]], %[[Z]] : tensor<6xf32>

// GELU-NONE-REV-LABEL: func.func @gelu_none_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// GELU-NONE-REV: %[[NEG_HALF:.*]] = stablehlo.constant dense<-5.000000e-01> : tensor<6xf32>
// GELU-NONE-REV: %[[INV_SQRT_2PI:.*]] = stablehlo.constant dense<0.398942292> : tensor<6xf32>
// GELU-NONE-REV: %[[SQRT2:.*]] = stablehlo.constant dense<1.41421354> : tensor<6xf32>
// GELU-NONE-REV: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
// GELU-NONE-REV: %[[HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<6xf32>
// GELU-NONE-REV: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// GELU-NONE-REV: %[[DR:.*]] = stablehlo.add %arg1, %[[ZERO]] : tensor<6xf32>
// GELU-NONE-REV: %[[DIV:.*]] = stablehlo.divide %arg0, %[[SQRT2]] : tensor<6xf32>
// GELU-NONE-REV: %[[ERF:.*]] = chlo.erf %[[DIV]] : tensor<6xf32> -> tensor<6xf32>
// GELU-NONE-REV: %[[ONE_PLUS_ERF:.*]] = stablehlo.add %[[ONE]], %[[ERF]] : tensor<6xf32>
// GELU-NONE-REV: %[[LEFT:.*]] = stablehlo.multiply %[[HALF]], %[[ONE_PLUS_ERF]] : tensor<6xf32>
// GELU-NONE-REV: %[[X2:.*]] = stablehlo.multiply %arg0, %arg0 : tensor<6xf32>
// GELU-NONE-REV: %[[NEG_HALF_X2:.*]] = stablehlo.multiply %[[NEG_HALF]], %[[X2]] : tensor<6xf32>
// GELU-NONE-REV: %[[EXP:.*]] = stablehlo.exponential %[[NEG_HALF_X2]] : tensor<6xf32>
// GELU-NONE-REV: %[[GAUSS:.*]] = stablehlo.multiply %[[INV_SQRT_2PI]], %[[EXP]] : tensor<6xf32>
// GELU-NONE-REV: %[[RIGHT:.*]] = stablehlo.multiply %arg0, %[[GAUSS]] : tensor<6xf32>
// GELU-NONE-REV: %[[SUM:.*]] = stablehlo.add %[[LEFT]], %[[RIGHT]] : tensor<6xf32>
// GELU-NONE-REV: %[[SCALED:.*]] = stablehlo.multiply %[[DR]], %[[SUM]] : tensor<6xf32>
// GELU-NONE-REV: stablehlo.add %[[SCALED]], %[[ZERO]] : tensor<6xf32>

// GELU-TANH-REV-LABEL: func.func @gelu_tanh_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// GELU-TANH-REV: %[[A134:.*]] = stablehlo.constant dense<1.341450e-01> : tensor<6xf32>
// GELU-TANH-REV: %[[THREE:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<6xf32>
// GELU-TANH-REV: %[[K044:.*]] = stablehlo.constant dense<4.471500e-02> : tensor<6xf32>
// GELU-TANH-REV: %[[C:.*]] = stablehlo.constant dense<0.797884583> : tensor<6xf32>
// GELU-TANH-REV: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
// GELU-TANH-REV: %[[HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<6xf32>
// GELU-TANH-REV: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// GELU-TANH-REV: %[[DR:.*]] = stablehlo.add %arg1, %[[ZERO]] : tensor<6xf32>
// GELU-TANH-REV: %[[X3_0:.*]] = stablehlo.power %arg0, %[[THREE]] : tensor<6xf32>
// GELU-TANH-REV: %[[KX3_0:.*]] = stablehlo.multiply %[[K044]], %[[X3_0]] : tensor<6xf32>
// GELU-TANH-REV: %[[U0_ARG:.*]] = stablehlo.add %arg0, %[[KX3_0]] : tensor<6xf32>
// GELU-TANH-REV: %[[U0:.*]] = stablehlo.multiply %[[C]], %[[U0_ARG]] : tensor<6xf32>
// GELU-TANH-REV: %[[T0:.*]] = stablehlo.tanh %[[U0]] : tensor<6xf32>
// GELU-TANH-REV: %[[TERM0_IN:.*]] = stablehlo.add %[[ONE]], %[[T0]] : tensor<6xf32>
// GELU-TANH-REV: %[[TERM0:.*]] = stablehlo.multiply %[[HALF]], %[[TERM0_IN]] : tensor<6xf32>
// GELU-TANH-REV: %[[HALF_X:.*]] = stablehlo.multiply %[[HALF]], %arg0 : tensor<6xf32>
// GELU-TANH-REV: %[[T1:.*]] = stablehlo.tanh {{.*}} : tensor<6xf32>
// GELU-TANH-REV: %[[T2:.*]] = stablehlo.tanh {{.*}} : tensor<6xf32>
// GELU-TANH-REV: %[[T1T2:.*]] = stablehlo.multiply %[[T1]], %[[T2]] : tensor<6xf32>
// GELU-TANH-REV: %[[ONE_MINUS:.*]] = stablehlo.subtract %[[ONE]], %[[T1T2]] : tensor<6xf32>
// GELU-TANH-REV: %[[X2:.*]] = stablehlo.multiply %arg0, %arg0 : tensor<6xf32>
// GELU-TANH-REV: %[[A134X2:.*]] = stablehlo.multiply %[[A134]], %[[X2]] : tensor<6xf32>
// GELU-TANH-REV: %[[ONE_PLUS:.*]] = stablehlo.add %[[ONE]], %[[A134X2]] : tensor<6xf32>
// GELU-TANH-REV: %[[C_ONE_PLUS:.*]] = stablehlo.multiply %[[C]], %[[ONE_PLUS]] : tensor<6xf32>
// GELU-TANH-REV: %[[CHAIN:.*]] = stablehlo.multiply %[[ONE_MINUS]], %[[C_ONE_PLUS]] : tensor<6xf32>
// GELU-TANH-REV: %[[TERM1:.*]] = stablehlo.multiply %[[HALF_X]], %[[CHAIN]] : tensor<6xf32>
// GELU-TANH-REV: %[[SUM:.*]] = stablehlo.add %[[TERM0]], %[[TERM1]] : tensor<6xf32>
// GELU-TANH-REV: %[[SCALED:.*]] = stablehlo.multiply %[[DR]], %[[SUM]] : tensor<6xf32>
// GELU-TANH-REV: stablehlo.add %[[SCALED]], %[[ZERO]] : tensor<6xf32>

// GELU-SIGMOID-REV-LABEL: func.func @gelu_sigmoid_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// GELU-SIGMOID-REV: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
// GELU-SIGMOID-REV: %[[K:.*]] = stablehlo.constant dense<1.702000e+00> : tensor<6xf32>
// GELU-SIGMOID-REV: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// GELU-SIGMOID-REV: %[[DR:.*]] = stablehlo.add %arg1, %[[ZERO]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[KX0:.*]] = stablehlo.multiply %[[K]], %arg0 : tensor<6xf32>
// GELU-SIGMOID-REV: %[[S0:.*]] = stablehlo.logistic %[[KX0]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[KX1:.*]] = stablehlo.multiply %[[K]], %arg0 : tensor<6xf32>
// GELU-SIGMOID-REV: %[[KX2:.*]] = stablehlo.multiply %[[K]], %arg0 : tensor<6xf32>
// GELU-SIGMOID-REV: %[[S1:.*]] = stablehlo.logistic %[[KX2]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[KX3:.*]] = stablehlo.multiply %[[K]], %arg0 : tensor<6xf32>
// GELU-SIGMOID-REV: %[[S2:.*]] = stablehlo.logistic %[[KX3]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[ONE_MINUS:.*]] = stablehlo.subtract %[[ONE]], %[[S2]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[PROD:.*]] = stablehlo.multiply %[[S1]], %[[ONE_MINUS]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[TERM1:.*]] = stablehlo.multiply %[[KX1]], %[[PROD]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[SUM:.*]] = stablehlo.add %[[S0]], %[[TERM1]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[SCALED:.*]] = stablehlo.multiply %[[DR]], %[[SUM]] : tensor<6xf32>
// GELU-SIGMOID-REV: stablehlo.add %[[SCALED]], %[[ZERO]] : tensor<6xf32>

// SOFTPLUS-REV-LABEL: func.func @softplus_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// SOFTPLUS-REV: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// SOFTPLUS-REV: %[[DR:.*]] = stablehlo.add %arg1, %[[ZERO]] : tensor<6xf32>
// SOFTPLUS-REV: %[[SIG:.*]] = stablehlo.logistic %arg0 : tensor<6xf32>
// SOFTPLUS-REV: %[[SCALED:.*]] = stablehlo.multiply %[[DR]], %[[SIG]] : tensor<6xf32>
// SOFTPLUS-REV: stablehlo.add %[[SCALED]], %[[ZERO]] : tensor<6xf32>
