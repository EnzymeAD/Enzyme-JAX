// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=relu_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math | FileCheck %s --check-prefix=RELU-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=gelu_none_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math | FileCheck %s --check-prefix=GELU-NONE-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=gelu_tanh_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math | FileCheck %s --check-prefix=GELU-TANH-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=gelu_sigmoid_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math | FileCheck %s --check-prefix=GELU-SIGMOID-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=softplus_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math | FileCheck %s --check-prefix=SOFTPLUS-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=tgamma_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math | FileCheck %s --check-prefix=TGAMMA-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=lgamma_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math | FileCheck %s --check-prefix=LGAMMA-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=hypot_fn outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math | FileCheck %s --check-prefix=HYPOT-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=sinc_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math | FileCheck %s --check-prefix=SINC-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=cosc_fn outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math | FileCheck %s --check-prefix=COSC-REV
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --lower-enzymexla-math --chlo-legalize-to-stablehlo --canonicalize --arith-raise | stablehlo-translate - --interpret

module {
  func.func @relu_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.math.relu %x : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @gelu_none_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.math.gelu %x, approximation = <NONE> : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @gelu_tanh_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.math.gelu %x, approximation = <TANH> : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @gelu_sigmoid_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.math.gelu %x, approximation = <SIGMOID> : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @softplus_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.math.softplus %x : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @tgamma_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.math.tgamma %x : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @lgamma_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.math.lgamma %x : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @hypot_fn(%x: tensor<6xf32>, %y: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.math.hypot %x, %y : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @sinc_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.math.sinc %x : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @cosc_fn(%x: tensor<6xf32>) -> tensor<6xf32> {
    %0 = enzymexla.math.cosc %x : (tensor<6xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @main() {
    %x_common = stablehlo.constant dense<[0.0, -1.0, 1.0, 2.0, 100.0, -100.0]> : tensor<6xf32>
    %d_common = stablehlo.constant dense<1.0> : tensor<6xf32>

    %relu_res:2 = enzyme.autodiff @relu_fn(%x_common, %d_common) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)

    // d/dx relu([0, -1, 1, 2, 100, -100]) = [0, 0, 1, 1, 1, 0]
    check.expect_eq_const %relu_res#1, dense<[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]> : tensor<6xf32>

    // GELU derivativ
    %gelu_none_res:2 = enzyme.autodiff @gelu_none_fn(%x_common, %d_common) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %gelu_none_res#1, dense<[0.5, -8.331547e-02, 1.0833155, 1.0852318, 1.0, 0.0]> : tensor<6xf32>

    %gelu_tanh_res:2 = enzyme.autodiff @gelu_tanh_fn(%x_common, %d_common) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %gelu_tanh_res#1, dense<[0.5, -8.296408e-02, 1.0829641, 1.0860993, 1.0, 0.0]> : tensor<6xf32>

    %gelu_sigmoid_res:2 = enzyme.autodiff @gelu_sigmoid_fn(%x_common, %d_common) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %gelu_sigmoid_res#1, dense<[0.5, -8.296408e-02, 1.0829641, 1.0860993, 1.0, 0.0]> : tensor<6xf32>

    // d/dx softplus(x) = sigmoid(x)
    %softplus_res:2 = enzyme.autodiff @softplus_fn(%x_common, %d_common) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %softplus_res#1, dense<[0.5, 0.26894143, 0.7310586, 0.8807971, 1.0, 0.0]> : tensor<6xf32>

    // d/dx tgamma(x) = tgamma(x) * polygamma(0, x)
    %x_tgamma = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32>
    %tgamma_res:2 = enzyme.autodiff @tgamma_fn(%x_tgamma, %d_common) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %tgamma_res#1, dense<[-0.577215672, 0.422784328, 1.84556866, 7.53670597, 36.1468239, 204.734329]> : tensor<6xf32>

    // d/dx lgamma(x) = polygamma(0, x) = digamma(x)
    %x_lgamma = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32>
    %lgamma_res:2 = enzyme.autodiff @lgamma_fn(%x_lgamma, %d_common) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %lgamma_res#1, dense<[-0.577215672, 0.422784328, 0.922784328, 1.25611770, 1.50611770, 1.70611768]> : tensor<6xf32>

    // d/dx hypot(x, y) = x / hypot(x, y), d/dy hypot(x, y) = y / hypot(x, y)
    %x_hypot = stablehlo.constant dense<[3.0, 5.0, 0.0, 1.0, -3.0, 8.0]> : tensor<6xf32>
    %y_hypot = stablehlo.constant dense<[4.0, 12.0, 2.0, 1.0, 4.0, 15.0]> : tensor<6xf32>
    %hypot_res:3 = enzyme.autodiff @hypot_fn(%x_hypot, %y_hypot, %d_common) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<6xf32>, tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %hypot_res#1, dense<[0.6, 0.38461538, 0.0, 0.70710678, -0.6, 0.47058823]> : tensor<6xf32>
    check.expect_almost_eq_const %hypot_res#2, dense<[0.8, 0.92307692, 1.0, 0.70710678, 0.8, 0.88235294]> : tensor<6xf32>

    // d/dx sinc(x) = cosc(x)
    %x_sinc = stablehlo.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, -0.5]> : tensor<6xf32>
    %sinc_res:2 = enzyme.autodiff @sinc_fn(%x_sinc, %d_common) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %sinc_res#1, dense<[0.0, -1.27323954, -1.0, 0.14147106, 0.5, 1.27323954]> : tensor<6xf32>

    // d/dx cosc(x) = -pi^2 * sinc(x) - 2 * cosc(x) / x
    %x_cosc = stablehlo.constant dense<[0.0, 0.5, 1.0, 1.5, 2.0, -0.5]> : tensor<6xf32>
    %cosc_res:2 = enzyme.autodiff @cosc_fn(%x_cosc, %d_common) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %cosc_res#1, dense<[-3.28986813, -1.1902271, 2.0, 1.905767, -0.5, -1.1902271]> : tensor<6xf32>

    // Forward mode: the tangent must be f'(x) * dx, not f(dx)
    %t_common = stablehlo.constant dense<[-1.0, 2.0, -3.0, 0.5, -1.0, 2.0]> : tensor<6xf32>
    %relu_fwd:2 = enzyme.fwddiff @relu_fn(%x_common, %t_common) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %relu_fwd#1, dense<[0.0, 0.0, -3.0, 0.5, -1.0, 0.0]> : tensor<6xf32>

    %gelu_none_fwd:2 = enzyme.fwddiff @gelu_none_fn(%x_common, %t_common) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %gelu_none_fwd#1, dense<[-0.5, -0.16663094, -3.2499466, 0.5426159, -1.0, 0.0]> : tensor<6xf32>

    %gelu_tanh_fwd:2 = enzyme.fwddiff @gelu_tanh_fn(%x_common, %t_common) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %gelu_tanh_fwd#1, dense<[-0.5, -0.16592816, -3.2488923, 0.54304963, -1.0, 0.0]> : tensor<6xf32>

    %gelu_sigmoid_fwd:2 = enzyme.fwddiff @gelu_sigmoid_fn(%x_common, %t_common) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %gelu_sigmoid_fwd#1, dense<[-0.5, -0.16592816, -3.2488923, 0.54304963, -1.0, 0.0]> : tensor<6xf32>

    %softplus_fwd:2 = enzyme.fwddiff @softplus_fn(%x_common, %t_common) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %softplus_fwd#1, dense<[-0.5, 0.53788286, -2.1931758, 0.44039854, -1.0, 0.0]> : tensor<6xf32>

    %sinc_fwd:2 = enzyme.fwddiff @sinc_fn(%x_sinc, %t_common) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %sinc_fwd#1, dense<[0.0, -2.546479, 3.0, 0.07073553, -0.5, 2.546479]> : tensor<6xf32>

    %cosc_fwd:2 = enzyme.fwddiff @cosc_fn(%x_cosc, %t_common) <{
      activity = [#enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    check.expect_almost_eq_const %cosc_fwd#1, dense<[3.289868, -2.3804543, -6.0, 0.9528835, 0.5, -2.3804543]> : tensor<6xf32>

    %dx_hypot = stablehlo.constant dense<[1.0, -2.0, 0.5, 1.0, 2.0, -1.0]> : tensor<6xf32>
    %dy_hypot = stablehlo.constant dense<[-1.0, 1.0, 3.0, 2.0, 0.5, 1.0]> : tensor<6xf32>
    %hypot_fwd:2 = enzyme.fwddiff @hypot_fn(%x_hypot, %dx_hypot, %y_hypot, %dy_hypot) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dup>]
    }> : (tensor<6xf32>, tensor<6xf32>, tensor<6xf32>, tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
    // (x * dx + y * dy) / hypot(x, y)
    check.expect_almost_eq_const %hypot_fwd#1, dense<[-0.2, 0.15384616, 3.0, 2.1213202, -0.8, 0.4117647]> : tensor<6xf32>

    return
  }

}

// RELU-REV-LABEL: func.func @relu_fn
// RELU-REV: %[[Z:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// RELU-REV: %[[MASK:.*]] = stablehlo.compare  GT, %arg0, %[[Z]] : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
// RELU-REV: %[[SEL:.*]] = stablehlo.select %[[MASK]], %arg1, %[[Z]] : tensor<6xi1>, tensor<6xf32>

// GELU-NONE-REV-LABEL: func.func @gelu_none_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// GELU-NONE-REV: %[[NEG_HALF:.*]] = stablehlo.constant dense<-5.000000e-01> : tensor<6xf32>
// GELU-NONE-REV: %[[INV_SQRT_2PI:.*]] = stablehlo.constant dense<0.398942292> : tensor<6xf32>
// GELU-NONE-REV: %[[SQRT2:.*]] = stablehlo.constant dense<1.41421354> : tensor<6xf32>
// GELU-NONE-REV: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
// GELU-NONE-REV: %[[HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<6xf32>
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
// GELU-NONE-REV: %[[SCALED:.*]] = stablehlo.multiply %arg1, %[[SUM]] : tensor<6xf32>

// GELU-TANH-REV-LABEL: func.func @gelu_tanh_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// GELU-TANH-REV: %[[A134:.*]] = stablehlo.constant dense<1.341450e-01> : tensor<6xf32>
// GELU-TANH-REV: %[[THREE:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<6xf32>
// GELU-TANH-REV: %[[K044:.*]] = stablehlo.constant dense<4.471500e-02> : tensor<6xf32>
// GELU-TANH-REV: %[[C:.*]] = stablehlo.constant dense<0.797884583> : tensor<6xf32>
// GELU-TANH-REV: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
// GELU-TANH-REV: %[[HALF:.*]] = stablehlo.constant dense<5.000000e-01> : tensor<6xf32>
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
// GELU-TANH-REV: %[[SCALED:.*]] = stablehlo.multiply %arg1, %[[SUM]] : tensor<6xf32>

// GELU-SIGMOID-REV-LABEL: func.func @gelu_sigmoid_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// GELU-SIGMOID-REV: %[[A134:.*]] = stablehlo.constant dense<1.341450e-01> : tensor<6xf32>
// GELU-SIGMOID-REV: %[[K044:.*]] = stablehlo.constant dense<4.471500e-02> : tensor<6xf32>
// GELU-SIGMOID-REV: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
// GELU-SIGMOID-REV: %[[C:.*]] = stablehlo.constant dense<1.59576917> : tensor<6xf32>
// GELU-SIGMOID-REV: %[[X2_0:.*]] = stablehlo.multiply %arg0, %arg0 : tensor<6xf32>
// GELU-SIGMOID-REV: %[[KX2_0:.*]] = stablehlo.multiply %[[K044]], %[[X2_0]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[POLY0:.*]] = stablehlo.add %[[ONE]], %[[KX2_0]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[XPOLY0:.*]] = stablehlo.multiply %arg0, %[[POLY0]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[V0:.*]] = stablehlo.multiply %[[C]], %[[XPOLY0]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[S0:.*]] = stablehlo.logistic %[[V0]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[S1:.*]] = stablehlo.logistic {{.*}} : tensor<6xf32>
// GELU-SIGMOID-REV: %[[S2:.*]] = stablehlo.logistic {{.*}} : tensor<6xf32>
// GELU-SIGMOID-REV: %[[ONE_MINUS:.*]] = stablehlo.subtract %[[ONE]], %[[S2]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[PROD:.*]] = stablehlo.multiply %[[S1]], %[[ONE_MINUS]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[XPROD:.*]] = stablehlo.multiply %arg0, %[[PROD]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[X2:.*]] = stablehlo.multiply %arg0, %arg0 : tensor<6xf32>
// GELU-SIGMOID-REV: %[[A134X2:.*]] = stablehlo.multiply %[[A134]], %[[X2]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[ONE_PLUS:.*]] = stablehlo.add %[[ONE]], %[[A134X2]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[C_ONE_PLUS:.*]] = stablehlo.multiply %[[C]], %[[ONE_PLUS]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[TERM1:.*]] = stablehlo.multiply %[[XPROD]], %[[C_ONE_PLUS]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[SUM:.*]] = stablehlo.add %[[S0]], %[[TERM1]] : tensor<6xf32>
// GELU-SIGMOID-REV: %[[SCALED:.*]] = stablehlo.multiply %arg1, %[[SUM]] : tensor<6xf32>

// SOFTPLUS-REV-LABEL: func.func @softplus_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// SOFTPLUS-REV: %[[SIG:.*]] = stablehlo.logistic %arg0 : tensor<6xf32>
// SOFTPLUS-REV: %[[SCALED:.*]] = stablehlo.multiply %arg1, %[[SIG]] : tensor<6xf32>

// TGAMMA-REV-LABEL: func.func @tgamma_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// TGAMMA-REV-DAG: %[[NAN:.*]] = stablehlo.constant dense<0x7FC00000> : tensor<6xf32>
// TGAMMA-REV-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// TGAMMA-REV: %[[NEG:.*]] = stablehlo.compare LT, %arg0, %[[ZERO]], FLOAT : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
// TGAMMA-REV: %[[TGAMMA:.*]] = stablehlo.select %[[NEG]], %[[NAN]], %{{.*}} : tensor<6xi1>, tensor<6xf32>
// TGAMMA-REV: %[[PG:.*]] = chlo.polygamma %[[ZERO]], %arg0
// TGAMMA-REV: %[[INNER:.*]] = stablehlo.multiply %[[TGAMMA]], %[[PG]] : tensor<6xf32>
// TGAMMA-REV: %[[OUTER:.*]] = stablehlo.multiply %arg1, %[[INNER]] : tensor<6xf32>

// LGAMMA-REV-LABEL: func.func @lgamma_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// LGAMMA-REV: %[[DG:.*]] = chlo.digamma %arg0 : tensor<6xf32> -> tensor<6xf32>
// LGAMMA-REV: %[[SCALED:.*]] = stablehlo.multiply %arg1, %[[DG]] : tensor<6xf32>

// HYPOT-REV-LABEL: func.func @hypot_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>, %arg2: tensor<6xf32>) -> (tensor<6xf32>, tensor<6xf32>)
// HYPOT-REV-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// HYPOT-REV-DAG: %[[ONE:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<6xf32>
// HYPOT-REV: %[[ABS_X:.*]] = stablehlo.abs %arg0 : tensor<6xf32>
// HYPOT-REV: %[[ABS_Y:.*]] = stablehlo.abs %arg1 : tensor<6xf32>
// HYPOT-REV: %[[MAX:.*]] = stablehlo.maximum %[[ABS_X]], %[[ABS_Y]] : tensor<6xf32>
// HYPOT-REV: %[[SQRT:.*]] = stablehlo.sqrt %{{.*}} : tensor<6xf32>
// HYPOT-REV: %[[HYP:.*]] = stablehlo.select %{{.*}}, %[[ZERO]], %{{.*}} : tensor<6xi1>, tensor<6xf32>
// HYPOT-REV: %[[DIV_X:.*]] = stablehlo.divide %arg0, %[[HYP]] : tensor<6xf32>
// HYPOT-REV: %[[MUL_X:.*]] = stablehlo.multiply %arg2, %[[DIV_X]] : tensor<6xf32>
// HYPOT-REV: %[[DIV_Y:.*]] = stablehlo.divide %arg1, %{{.*}} : tensor<6xf32>
// HYPOT-REV: %[[MUL_Y:.*]] = stablehlo.multiply %arg2, %[[DIV_Y]] : tensor<6xf32>
// HYPOT-REV: return %[[MUL_X]], %[[MUL_Y]]

// SINC-REV-LABEL: func.func @sinc_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// SINC-REV-DAG: %[[PI:.*]] = stablehlo.constant dense<3.14159274> : tensor<6xf32>
// SINC-REV-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// SINC-REV: %[[COS:.*]] = stablehlo.cosine %{{.*}} : tensor<6xf32>
// SINC-REV: %[[SIN:.*]] = stablehlo.sine %{{.*}} : tensor<6xf32>
// SINC-REV: %[[SUB:.*]] = stablehlo.subtract %{{.*}}, %{{.*}} : tensor<6xf32>
// SINC-REV: %[[SEL:.*]] = stablehlo.select %{{.*}}, %[[ZERO]], %[[SUB]] : tensor<6xi1>, tensor<6xf32>
// SINC-REV: %[[MUL:.*]] = stablehlo.multiply %arg1, %[[SEL]] : tensor<6xf32>

// COSC-REV-LABEL: func.func @cosc_fn(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<6xf32>
// COSC-REV-DAG: %[[PI:.*]] = stablehlo.constant dense<3.14159274> : tensor<6xf32>
// COSC-REV-DAG: %[[PI2_3:.*]] = stablehlo.constant dense<-3.28986812> : tensor<6xf32>
// COSC-REV-DAG: %[[PI2:.*]] = stablehlo.constant dense<-9.86960411> : tensor<6xf32>
// COSC-REV-DAG: %[[TWO:.*]] = stablehlo.constant dense<2.000000e+00> : tensor<6xf32>
// COSC-REV-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<6xf32>
// COSC-REV: %[[COS:.*]] = stablehlo.cosine %{{.*}} : tensor<6xf32>
// COSC-REV: %[[SIN:.*]] = stablehlo.sine %{{.*}} : tensor<6xf32>
// COSC-REV: stablehlo.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<6xi1>, tensor<6xf32>
