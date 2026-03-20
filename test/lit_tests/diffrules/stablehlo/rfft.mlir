// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rfft outfn=rfft_fwddiff retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD-RFFT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rfft outfn=rfft_revdiff retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --verify-each=0 --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | FileCheck %s --check-prefix=REVERSE-RFFT
// RUN: enzymexlamlir-opt --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --chlo-legalize-to-stablehlo --enzyme-hlo-opt --verify-each=0 %s | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @rfft(%x : tensor<4xf64>) -> tensor<3xcomplex<f64>> {
  %y = "stablehlo.fft"(%x) {
    fft_type = #stablehlo<fft_type RFFT>,
    fft_length = array<i64 : 4>
  } : (tensor<4xf64>) -> tensor<3xcomplex<f64>>
  func.return %y : tensor<3xcomplex<f64>>
}

// FORWARD-RFFT:  func.func private @rfft_fwddiff(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) {
// FORWARD-RFFT-NEXT:       %0 = stablehlo.fft %arg1, type =  RFFT, length = [4] : (tensor<4xf64>) -> tensor<3xcomplex<f64>>
// FORWARD-RFFT-NEXT:       %1 = stablehlo.fft %arg0, type =  RFFT, length = [4] : (tensor<4xf64>) -> tensor<3xcomplex<f64>>
// FORWARD-RFFT-NEXT:       return %1, %0 : tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>
// FORWARD-RFFT-NEXT:   }

// REVERSE-RFFT:  func.func private @rfft_revdiff(%arg0: tensor<4xf64>, %arg1: tensor<3xcomplex<f64>>) -> tensor<4xf64> {
// REVERSE-RFFT-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
// REVERSE-RFFT-NEXT:    %0 = chlo.conj %arg1 : tensor<3xcomplex<f64>> -> tensor<3xcomplex<f64>>
// REVERSE-RFFT-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0], high = [1], interior = [0] : (tensor<3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<4xcomplex<f64>>
// REVERSE-RFFT-NEXT:    %2 = stablehlo.fft %1, type = FFT, length = [4] {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>>
// REVERSE-RFFT-NEXT:    %3 = stablehlo.real %2 : (tensor<4xcomplex<f64>>) -> tensor<4xf64>
// REVERSE-RFFT-NEXT:    return %3 : tensor<4xf64>
// REVERSE-RFFT-NEXT:  }

func.func @main() {
  %input = stablehlo.constant dense<[1.0, -2.0, 3.0, -4.0]> : tensor<4xf64>
  %output = stablehlo.constant dense<[(-2.0, 0.0), (-2.0, -2.0), (10.0, 0.0)]> : tensor<3xcomplex<f64>>

  // forward, index = 0
  %dinput_fwd_0 = stablehlo.constant dense<[1.0, 0.0, 0.0, 0.0]> : tensor<4xf64>

  %fwd_0:2 = enzyme.fwddiff @rfft(%input, %dinput_fwd_0) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xf64>, tensor<4xf64>) -> (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>)

  check.expect_almost_eq %fwd_0#0, %output : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %fwd_0#1, dense<[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>

  // forward, index = 1
  %dinput_fwd_1 = stablehlo.constant dense<[0.0, 1.0, 0.0, 0.0]> : tensor<4xf64>

  %fwd_1:2 = enzyme.fwddiff @rfft(%input, %dinput_fwd_1) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xf64>, tensor<4xf64>) -> (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>)

  check.expect_almost_eq %fwd_1#0, %output : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %fwd_1#1, dense<[(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]> : tensor<3xcomplex<f64>>

  // forward, index = 2
  %dinput_fwd_2 = stablehlo.constant dense<[0.0, 0.0, 1.0, 0.0]> : tensor<4xf64>

  %fwd_2:2 = enzyme.fwddiff @rfft(%input, %dinput_fwd_2) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xf64>, tensor<4xf64>) -> (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>)

  check.expect_almost_eq %fwd_2#0, %output : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %fwd_2#1, dense<[(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>

  // forward, index = 3
  %dinput_fwd_3 = stablehlo.constant dense<[0.0, 0.0, 0.0, 1.0]> : tensor<4xf64>

  %fwd_3:2 = enzyme.fwddiff @rfft(%input, %dinput_fwd_3) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<4xf64>, tensor<4xf64>) -> (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>)

  check.expect_almost_eq %fwd_3#0, %output : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %fwd_3#1, dense<[(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]> : tensor<3xcomplex<f64>>

  // reverse, index = 0, real
  %dinput_rev_0_real = stablehlo.constant dense<[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]> : tensor<3xcomplex<f64>>

  %rev_0_real:2 = enzyme.autodiff @rfft(%input, %dinput_rev_0_real) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xf64>, tensor<3xcomplex<f64>>) -> (tensor<3xcomplex<f64>>, tensor<4xf64>)

  check.expect_almost_eq %rev_0_real#0, %output : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %rev_0_real#1, dense<1.0> : tensor<4xf64>

  // reverse, index = 0, imag
  %dinput_rev_0_imag = stablehlo.constant dense<[(0.0, 1.0), (0.0, 0.0), (0.0, 0.0)]> : tensor<3xcomplex<f64>>

  %rev_0_imag:2 = enzyme.autodiff @rfft(%input, %dinput_rev_0_imag) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xf64>, tensor<3xcomplex<f64>>) -> (tensor<3xcomplex<f64>>, tensor<4xf64>)

  check.expect_almost_eq %rev_0_imag#0, %output : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %rev_0_imag#1, dense<0.0> : tensor<4xf64>

  // reverse, index = 1, real
  %dinput_rev_1_real = stablehlo.constant dense<[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]> : tensor<3xcomplex<f64>>

  %rev_1_real:2 = enzyme.autodiff @rfft(%input, %dinput_rev_1_real) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xf64>, tensor<3xcomplex<f64>>) -> (tensor<3xcomplex<f64>>, tensor<4xf64>)

  check.expect_almost_eq %rev_1_real#0, %output : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %rev_1_real#1, dense<[1.0, 0.0, -1.0, 0.0]> : tensor<4xf64>

  // reverse, index = 1, imag
  %dinput_rev_1_imag = stablehlo.constant dense<[(0.0, 0.0), (0.0, 1.0), (0.0, 0.0)]> : tensor<3xcomplex<f64>>

  %rev_1_imag:2 = enzyme.autodiff @rfft(%input, %dinput_rev_1_imag) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xf64>, tensor<3xcomplex<f64>>) -> (tensor<3xcomplex<f64>>, tensor<4xf64>)

  check.expect_almost_eq %rev_1_imag#0, %output : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %rev_1_imag#1, dense<[0.0, -1.0, 0.0, 1.0]> : tensor<4xf64>

  // reverse, index = 2, real
  %dinput_rev_2_real = stablehlo.constant dense<[(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]> : tensor<3xcomplex<f64>>

  %rev_2_real:2 = enzyme.autodiff @rfft(%input, %dinput_rev_2_real) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xf64>, tensor<3xcomplex<f64>>) -> (tensor<3xcomplex<f64>>, tensor<4xf64>)

  check.expect_almost_eq %rev_2_real#0, %output : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %rev_2_real#1, dense<[1.0, -1.0, 1.0, -1.0]> : tensor<4xf64>

  // reverse, index = 2, imag
  %dinput_rev_2_imag = stablehlo.constant dense<[(0.0, 0.0), (0.0, 0.0), (0.0, 1.0)]> : tensor<3xcomplex<f64>>

  %rev_2_imag:2 = enzyme.autodiff @rfft(%input, %dinput_rev_2_imag) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<4xf64>, tensor<3xcomplex<f64>>) -> (tensor<3xcomplex<f64>>, tensor<4xf64>)

  check.expect_almost_eq %rev_2_imag#0, %output : tensor<3xcomplex<f64>>
  check.expect_almost_eq_const %rev_2_imag#1, dense<0.0> : tensor<4xf64>

  return
}
