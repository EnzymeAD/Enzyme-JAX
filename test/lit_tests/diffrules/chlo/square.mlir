// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=square_real outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD-REAL
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=square_real outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse --drop-unsupported-attributes --verify-each=0 | FileCheck %s --check-prefix=REVERSE-REAL
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=square_complex outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --arith-raise --enzyme-hlo-opt --cse --drop-unsupported-attributes --verify-each=0 | FileCheck %s --check-prefix=FORWARD-COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=square_complex outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt --cse --drop-unsupported-attributes --verify-each=0 | FileCheck %s --check-prefix=REVERSE-COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --chlo-legalize-to-stablehlo --enzyme-hlo-opt --verify-each=0 | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @square_real(%x : tensor<5xf32>) -> tensor<5xf32> {
  %y = chlo.square %x : tensor<5xf32> -> tensor<5xf32>
  func.return %y : tensor<5xf32>
}

// FORWARD-REAL:  func.func @square_real(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// FORWARD-REAL-NEXT:   %[[CST:.*]] = chlo.constant dense<2.000000e+00> : tensor<5xf32>
// FORWARD-REAL-NEXT:   %[[DIFF1:.*]] = stablehlo.multiply %arg0, %[[CST]] : tensor<5xf32>
// FORWARD-REAL-NEXT:   %[[DIFF2:.*]] = stablehlo.multiply %arg1, %[[DIFF1]] : tensor<5xf32>
// FORWARD-REAL-NEXT:   %[[PRIMAL:.*]] = chlo.square %arg0 : tensor<5xf32>
// FORWARD-REAL-NEXT:   return %[[PRIMAL]], %[[DIFF2]] : tensor<5xf32>, tensor<5xf32>
// FORWARD-REAL-NEXT: }

// REVERSE-REAL:  func.func @square_real(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
// REVERSE-REAL-NEXT:   %[[CST:.*]] = chlo.constant dense<2.000000e+00> : tensor<5xf32>
// REVERSE-REAL-NEXT:   %[[DIFF1:.*]] = stablehlo.multiply %arg0, %[[CST]] : tensor<5xf32>
// REVERSE-REAL-NEXT:   %[[DIFF2:.*]] = stablehlo.multiply %arg1, %[[DIFF1]] : tensor<5xf32>
// REVERSE-REAL-NEXT:   return %[[DIFF2]] : tensor<5xf32>
// REVERSE-REAL-NEXT: }

func.func @square_complex(%x : tensor<5xcomplex<f32>>) -> tensor<5xcomplex<f32>> {
  %y = chlo.square %x : tensor<5xcomplex<f32>> -> tensor<5xcomplex<f32>>
  func.return %y : tensor<5xcomplex<f32>>
}

// FORWARD-COMPLEX:  func.func @square_complex(%arg0: tensor<5xcomplex<f32>>, %arg1: tensor<5xcomplex<f32>>) -> (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>) {
// FORWARD-COMPLEX-NEXT:   %[[CST:.*]] = chlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<5xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:   %[[DIFF1:.*]] = stablehlo.multiply %arg0, %[[CST]] : tensor<5xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:   %[[DIFF2:.*]] = stablehlo.multiply %arg1, %[[DIFF1]] : tensor<5xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:   %[[PRIMAL:.*]] = chlo.square %arg0 : tensor<5xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:   return %[[PRIMAL]], %[[DIFF2]] : tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>
// FORWARD-COMPLEX-NEXT: }

// REVERSE-COMPLEX:  func.func @square_complex(%arg0: tensor<5xcomplex<f32>>, %arg1: tensor<5xcomplex<f32>>) -> tensor<5xcomplex<f32>> {
// REVERSE-COMPLEX-NEXT:   %[[CST:.*]] = chlo.constant dense<(2.000000e+00,0.000000e+00)> : tensor<5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   %[[CONJ1:.*]] = chlo.conj %arg1 : tensor<5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   %[[DIFF1:.*]] = stablehlo.multiply %arg0, %[[CST]] : tensor<5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   %[[DIFF2:.*]] = stablehlo.multiply %[[CONJ1]], %[[DIFF1]] : tensor<5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   %[[CONJ2:.*]] = chlo.conj %[[DIFF2]] : tensor<5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:   return %[[CONJ2]] : tensor<5xcomplex<f32>>
// REVERSE-COMPLEX-NEXT: }

func.func @main() {
  // real
  %x = stablehlo.constant dense<[0.0, 1.0, 2.5, -3.0, 4.0]> : tensor<5xf32>
  %output = stablehlo.constant dense<[0.0, 1.0, 6.25, 9.0, 16.0]> : tensor<5xf32>
  %expected = stablehlo.constant dense<[0.0, 2.0, 5.0, -6.0, 8.0]> : tensor<5xf32>

  %d = stablehlo.constant dense<1.0> : tensor<5xf32>

  %fwd:2 = enzyme.fwddiff @square_real(%x, %d) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)

  check.expect_almost_eq %fwd#0, %output : tensor<5xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<5xf32>

  %rev:2 = enzyme.autodiff @square_real(%x, %d) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<5xf32>, tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>)

  check.expect_almost_eq %rev#0, %output : tensor<5xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<5xf32>

  // complex
  %x_c = stablehlo.constant dense<[(0.0, 0.0), (2.0, 0.0), (0.0, 3.0), (2.0, -3.0), (-2.0, 3.0)]> : tensor<5xcomplex<f32>>
  %output_c = stablehlo.constant dense<[(0.0, 0.0), (4.0, 0.0), (-9.0, 0.0), (-5.0, -12.0), (-5.0, -12.0)]> : tensor<5xcomplex<f32>>

  %d_c_re = stablehlo.constant dense<(1.0, 0.0)> : tensor<5xcomplex<f32>>

  // seed on real part
  %fwd_c_re:2 = enzyme.fwddiff @square_complex(%x_c, %d_c_re) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>) -> (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>)

  check.expect_almost_eq %fwd_c_re#0, %output_c : tensor<5xcomplex<f32>>
  check.expect_almost_eq_const %fwd_c_re#1, dense<[(0.0, 0.0), (4.0, 0.0), (0.0, 6.0), (4.0, -6.0), (-4.0, 6.0)]> : tensor<5xcomplex<f32>>

  %rev_c_re:2 = enzyme.autodiff @square_complex(%x_c, %d_c_re) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>) -> (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>)

  check.expect_almost_eq %rev_c_re#0, %output_c : tensor<5xcomplex<f32>>
  check.expect_almost_eq_const %rev_c_re#1, dense<[(0.0, 0.0), (4.0, 0.0), (0.0, -6.0), (4.0, 6.0), (-4.0, -6.0)]> : tensor<5xcomplex<f32>>

  // seed on imaginary part
  %d_c_im = stablehlo.constant dense<(0.0, 1.0)> : tensor<5xcomplex<f32>>

  %fwd_c_im:2 = enzyme.fwddiff @square_complex(%x_c, %d_c_im) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>) -> (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>)

  check.expect_almost_eq %fwd_c_im#0, %output_c : tensor<5xcomplex<f32>>
  check.expect_almost_eq_const %fwd_c_im#1, dense<[(0.0, 0.0), (0.0, 4.0), (-6.0, 0.0), (6.0, 4.0), (-6.0, -4.0)]> : tensor<5xcomplex<f32>>

  %rev_c_im:2 = enzyme.autodiff @square_complex(%x_c, %d_c_im) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>) -> (tensor<5xcomplex<f32>>, tensor<5xcomplex<f32>>)

  check.expect_almost_eq %rev_c_im#0, %output_c : tensor<5xcomplex<f32>>
  check.expect_almost_eq_const %rev_c_im#1, dense<[(0.0, 0.0), (0.0, 4.0), (6.0, 0.0), (-6.0, 4.0), (6.0, -4.0)]> : tensor<5xcomplex<f32>>

  func.return
}
