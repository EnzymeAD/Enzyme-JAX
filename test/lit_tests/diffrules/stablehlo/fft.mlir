// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=fft outfn=fft_fwddiff retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-FFT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=fft outfn=fft_revdiff retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE-FFT
// RUN: enzymexlamlir-opt --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --chlo-legalize-to-stablehlo --enzyme-hlo-opt --verify-each=0 %s | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @fft(%x : tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>> {
  %0 = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type FFT>,
    fft_length = array<i64 : 4>
  } : (tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>>
  func.return %0 : tensor<4xcomplex<f64>>
}

// FORWARD-FFT:  func.func private @fft_fwddiff(%arg0: tensor<4xcomplex<f64>>, %arg1: tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) {
// FORWARD-FFT-NEXT:       %0 = stablehlo.fft %arg1, type =  FFT, length = [4] : (tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>>
// FORWARD-FFT-NEXT:       %1 = stablehlo.fft %arg0, type =  FFT, length = [4] : (tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>>
// FORWARD-FFT-NEXT:       return %1, %0 : tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>
// FORWARD-FFT-NEXT:   }

// REVERSE-FFT:  func.func private @fft_revdiff(%arg0: tensor<4xcomplex<f64>>, %arg1: tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>> {
// REVERSE-FFT-NEXT:    %0 = chlo.conj %arg1 : tensor<4xcomplex<f64>> -> tensor<4xcomplex<f64>>
// REVERSE-FFT-NEXT:    %1 = stablehlo.fft %0, type = FFT, length = [4] {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>>
// REVERSE-FFT-NEXT:    %2 = chlo.conj %1 : tensor<4xcomplex<f64>> -> tensor<4xcomplex<f64>>
// REVERSE-FFT-NEXT:    return %2 : tensor<4xcomplex<f64>>
// REVERSE-FFT-NEXT:  }

func.func @main() {
  %input = stablehlo.constant dense<[(1.0, 2.0), (3.0, -4.0), (-5.0, 6.0), (-7.0, -8.0)]> : tensor<4xcomplex<f64>>
  %output = stablehlo.constant dense<[(-8.0, -4.0), (10.0, -14.0), (0.0, 20.0), (2.0, 6.0)]> : tensor<4xcomplex<f64>>

  // index = 0, real
  %dinput_0_real = stablehlo.constant dense<[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]> : tensor<4xcomplex<f64>>

  %fwd_0_real:2 = enzyme.fwddiff @fft(%input, %dinput_0_real) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %fwd_0_real#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %fwd_0_real#1, dense<[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]> : tensor<4xcomplex<f64>>

  %rev_0_real:2 = enzyme.autodiff @fft(%input, %dinput_0_real) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %rev_0_real#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %rev_0_real#1, dense<[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]> : tensor<4xcomplex<f64>>

  // index = 0, imag
  %dinput_0_imag = stablehlo.constant dense<[(0.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]> : tensor<4xcomplex<f64>>

  %fwd_0_imag:2 = enzyme.fwddiff @fft(%input, %dinput_0_imag) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %fwd_0_imag#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %fwd_0_imag#1, dense<[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]> : tensor<4xcomplex<f64>>

  %rev_0_imag:2 = enzyme.autodiff @fft(%input, %dinput_0_imag) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %rev_0_imag#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %rev_0_imag#1, dense<[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]> : tensor<4xcomplex<f64>>

  // index = 1, real
  %dinput_1_real = stablehlo.constant dense<[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]> : tensor<4xcomplex<f64>>

  %fwd_1_real:2 = enzyme.fwddiff @fft(%input, %dinput_1_real) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %fwd_1_real#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %fwd_1_real#1, dense<[(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)]> : tensor<4xcomplex<f64>>

  %rev_1_real:2 = enzyme.autodiff @fft(%input, %dinput_1_real) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %rev_1_real#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %rev_1_real#1, dense<[(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]> : tensor<4xcomplex<f64>>

  // index = 1, imag
  %dinput_1_imag = stablehlo.constant dense<[(0.0, 0.0), (0.0, 1.0), (0.0, 0.0), (0.0, 0.0)]> : tensor<4xcomplex<f64>>

  %fwd_1_imag:2 = enzyme.fwddiff @fft(%input, %dinput_1_imag) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %fwd_1_imag#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %fwd_1_imag#1, dense<[(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]> : tensor<4xcomplex<f64>>

  %rev_1_imag:2 = enzyme.autodiff @fft(%input, %dinput_1_imag) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %rev_1_imag#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %rev_1_imag#1, dense<[(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]> : tensor<4xcomplex<f64>>

  // index = 2, real
  %dinput_2_real = stablehlo.constant dense<[(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]> : tensor<4xcomplex<f64>>

  %fwd_2_real:2 = enzyme.fwddiff @fft(%input, %dinput_2_real) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %fwd_2_real#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %fwd_2_real#1, dense<[(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)]> : tensor<4xcomplex<f64>>

  %rev_2_real:2 = enzyme.autodiff @fft(%input, %dinput_2_real) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %rev_2_real#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %rev_2_real#1, dense<[(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)]> : tensor<4xcomplex<f64>>

  // index = 2, imag
  %dinput_2_imag = stablehlo.constant dense<[(0.0, 0.0), (0.0, 0.0), (0.0, 1.0), (0.0, 0.0)]> : tensor<4xcomplex<f64>>

  %fwd_2_imag:2 = enzyme.fwddiff @fft(%input, %dinput_2_imag) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %fwd_2_imag#0, %output : tensor<4xcomplex<f64>>
  // check.expect_almost_eq_const %fwd_2_imag#1, dense<[(0.0, 1.0), (0.0, -1.0), (0.0, 1.0), (0.0, -1.0)]> : tensor<4xcomplex<f64>>

  %rev_2_imag:2 = enzyme.autodiff @fft(%input, %dinput_2_imag) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %rev_2_imag#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %rev_2_imag#1, dense<[(0.0, 1.0), (0.0, -1.0), (0.0, 1.0), (0.0, -1.0)]> : tensor<4xcomplex<f64>>

  // index = 3, real
  %dinput_3_real = stablehlo.constant dense<[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]> : tensor<4xcomplex<f64>>

  %fwd_3_real:2 = enzyme.fwddiff @fft(%input, %dinput_3_real) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %fwd_3_real#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %fwd_3_real#1, dense<[(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]> : tensor<4xcomplex<f64>>

  %rev_3_real:2 = enzyme.autodiff @fft(%input, %dinput_3_real) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %rev_3_real#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %rev_3_real#1, dense<[(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)]> : tensor<4xcomplex<f64>>

  // index = 3, imag
  %dinput_3_imag = stablehlo.constant dense<[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 1.0)]> : tensor<4xcomplex<f64>>

  %fwd_3_imag:2 = enzyme.fwddiff @fft(%input, %dinput_3_imag) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %fwd_3_imag#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %fwd_3_imag#1, dense<[(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]> : tensor<4xcomplex<f64>>

  %rev_3_imag:2 = enzyme.autodiff @fft(%input, %dinput_3_imag) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>)

  check.expect_almost_eq %rev_3_imag#0, %output : tensor<4xcomplex<f64>>
  check.expect_almost_eq_const %rev_3_imag#1, dense<[(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]> : tensor<4xcomplex<f64>>

  return
}
