// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=irfft outfn=irfft_fwddiff retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-IRFFT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=irfft outfn=irfft_revdiff retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE-IRFFT
// RUN: enzymexlamlir-opt --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --chlo-legalize-to-stablehlo --enzyme-hlo-opt --verify-each=0 %s | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @irfft(%x : tensor<3xcomplex<f64>>) -> tensor<4xf64> {
  %y = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type IRFFT>,
    fft_length = array<i64 : 4>
  } : (tensor<3xcomplex<f64>>) -> tensor<4xf64>
  func.return %y : tensor<4xf64>
}

// FORWARD-IRFFT:  func.func private @irfft_fwddiff(%arg0: tensor<3xcomplex<f64>>, %arg1: tensor<3xcomplex<f64>>) -> (tensor<4xf64>, tensor<4xf64>) {
// FORWARD-IRFFT-NEXT:       %0 = stablehlo.fft %arg1, type =  IRFFT, length = [4] : (tensor<3xcomplex<f64>>) -> tensor<4xf64>
// FORWARD-IRFFT-NEXT:       %1 = stablehlo.fft %arg0, type =  IRFFT, length = [4] : (tensor<3xcomplex<f64>>) -> tensor<4xf64>
// FORWARD-IRFFT-NEXT:       return %1, %0 : tensor<4xf64>, tensor<4xf64>
// FORWARD-IRFFT-NEXT:   }

// REVERSE-IRFFT: func.func private @irfft_revdiff(%arg0: tensor<3xcomplex<f64>>, %arg1: tensor<4xf64>) -> tensor<3xcomplex<f64>> {
// REVERSE-IRFFT-NEXT:      %cst = stablehlo.constant dense<[(2.500000e-01,0.000000e+00), (5.000000e-01,0.000000e+00), (2.500000e-01,0.000000e+00)]> : tensor<3xcomplex<f64>>
// REVERSE-IRFFT-NEXT:      %0 = stablehlo.fft %arg1, type =  RFFT, length = [4] {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<4xf64>) -> tensor<3xcomplex<f64>>
// REVERSE-IRFFT-NEXT:      %1 = chlo.conj %0 {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<3xcomplex<f64>> -> tensor<3xcomplex<f64>>
// REVERSE-IRFFT-NEXT:      %2 = stablehlo.multiply %1, %cst {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<3xcomplex<f64>>
// REVERSE-IRFFT-NEXT:      %3 = chlo.conj %2 : tensor<3xcomplex<f64>> -> tensor<3xcomplex<f64>>
// REVERSE-IRFFT-NEXT:      return %3 : tensor<3xcomplex<f64>>
// REVERSE-IRFFT-NEXT:    }

func.func @main() {
  %input = stablehlo.constant dense<[(1.0, 2.0), (-3.0, 4.0), (-5.0, -6.0)]> : tensor<3xcomplex<f64>>
  %output = stablehlo.constant dense<[-2.5, -0.5, 0.5, 3.5]> : tensor<4xf64>

  // forward, index = 0, real
  %dinput_fwd_0_real = stablehlo.constant dense<[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]> : tensor<3xcomplex<f64>>

  %fwd_0_real:2 = enzyme.fwddiff @irfft(%input, %dinput_fwd_0_real) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> (tensor<4xf64>, tensor<4xf64>)

  check.expect_almost_eq %fwd_0_real#0, %output : tensor<4xf64>
  check.expect_almost_eq_const %fwd_0_real#1, dense<0.25> : tensor<4xf64>

  // forward, index = 0, imag
  %dinput_fwd_0_imag = stablehlo.constant dense<[(0.0, 1.0), (0.0, 0.0), (0.0, 0.0)]> : tensor<3xcomplex<f64>>

  %fwd_0_imag:2 = enzyme.fwddiff @irfft(%input, %dinput_fwd_0_imag) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> (tensor<4xf64>, tensor<4xf64>)

  check.expect_almost_eq %fwd_0_imag#0, %output : tensor<4xf64>
  check.expect_almost_eq_const %fwd_0_imag#1, dense<0.0> : tensor<4xf64>

  // forward, index = 1, real
  %dinput_fwd_1_real = stablehlo.constant dense<[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]> : tensor<3xcomplex<f64>>

  %fwd_1_real:2 = enzyme.fwddiff @irfft(%input, %dinput_fwd_1_real) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> (tensor<4xf64>, tensor<4xf64>)

  check.expect_almost_eq %fwd_1_real#0, %output : tensor<4xf64>
  check.expect_almost_eq_const %fwd_1_real#1, dense<[0.5, 0.0, -0.5, 0.0]> : tensor<4xf64>

  // forward, index = 1, imag
  %dinput_fwd_1_imag = stablehlo.constant dense<[(0.0, 0.0), (0.0, 1.0), (0.0, 0.0)]> : tensor<3xcomplex<f64>>

  %fwd_1_imag:2 = enzyme.fwddiff @irfft(%input, %dinput_fwd_1_imag) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> (tensor<4xf64>, tensor<4xf64>)

  check.expect_almost_eq %fwd_1_imag#0, %output : tensor<4xf64>
  check.expect_almost_eq_const %fwd_1_imag#1, dense<[0.0, -0.5, 0.0, 0.5]> : tensor<4xf64>

  // forward, index = 2, real
  %dinput_fwd_2_real = stablehlo.constant dense<[(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>

  %fwd_2_real:2 = enzyme.fwddiff @irfft(%input, %dinput_fwd_2_real) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> (tensor<4xf64>, tensor<4xf64>)

  check.expect_almost_eq %fwd_2_real#0, %output : tensor<4xf64>
  check.expect_almost_eq_const %fwd_2_real#1, dense<[0.25, -0.25, 0.25, -0.25]> : tensor<4xf64>

  // forward, index = 2, imag
  %dinput_fwd_2_imag = stablehlo.constant dense<[(0.0, 0.0), (0.0, 0.0), (0.0, 1.0)]> : tensor<3xcomplex<f64>>

  %fwd_2_imag:2 = enzyme.fwddiff @irfft(%input, %dinput_fwd_2_imag) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> (tensor<4xf64>, tensor<4xf64>)

  check.expect_almost_eq %fwd_2_imag#0, %output : tensor<4xf64>
  check.expect_almost_eq_const %fwd_2_imag#1, dense<0.0> : tensor<4xf64>

  // reverse, index = 0
  %dinput_rev_0 = stablehlo.constant dense<[1.0, 0.0, 0.0, 0.0]> : tensor<4xf64>

  %rev_0:2 = enzyme.autodiff @irfft(%input, %dinput_rev_0) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<3xcomplex<f64>>, tensor<4xf64>) -> (tensor<4xf64>, tensor<3xcomplex<f64>>)

  check.expect_almost_eq %rev_0#0, %output : tensor<4xf64>
  check.expect_almost_eq_const %rev_0#1, dense<[(0.25, 0.0), (0.5, 0.0), (0.25, 0.0)]> : tensor<3xcomplex<f64>>

  // reverse, index = 1
  %dinput_rev_1 = stablehlo.constant dense<[0.0, 1.0, 0.0, 0.0]> : tensor<4xf64>

  %rev_1:2 = enzyme.autodiff @irfft(%input, %dinput_rev_1) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<3xcomplex<f64>>, tensor<4xf64>) -> (tensor<4xf64>, tensor<3xcomplex<f64>>)

  check.expect_almost_eq %rev_1#0, %output : tensor<4xf64>
  check.expect_almost_eq_const %rev_1#1, dense<[(0.25, 0.0), (0.0, -0.5), (-0.25, 0.0)]> : tensor<3xcomplex<f64>>

  // reverse, index = 2
  %dinput_rev_2 = stablehlo.constant dense<[0.0, 0.0, 1.0, 0.0]> : tensor<4xf64>

  %rev_2:2 = enzyme.autodiff @irfft(%input, %dinput_rev_2) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<3xcomplex<f64>>, tensor<4xf64>) -> (tensor<4xf64>, tensor<3xcomplex<f64>>)

  check.expect_almost_eq %rev_2#0, %output : tensor<4xf64>
  check.expect_almost_eq_const %rev_2#1, dense<[(0.25, 0.0), (-0.5, 0.0), (0.25, 0.0)]> : tensor<3xcomplex<f64>>

  // reverse, index = 3
  %dinput_rev_3 = stablehlo.constant dense<[0.0, 0.0, 0.0, 1.0]> : tensor<4xf64>

  %rev_3:2 = enzyme.autodiff @irfft(%input, %dinput_rev_3) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<3xcomplex<f64>>, tensor<4xf64>) -> (tensor<4xf64>, tensor<3xcomplex<f64>>)

  check.expect_almost_eq %rev_3#0, %output : tensor<4xf64>
  check.expect_almost_eq_const %rev_3#1, dense<[(0.25, 0.0), (0.0, 0.5), (-0.25, 0.0)]> : tensor<3xcomplex<f64>>

  return
}
