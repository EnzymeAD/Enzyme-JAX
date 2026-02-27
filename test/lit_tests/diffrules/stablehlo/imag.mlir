// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=imag outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=imag outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse --verify-each=0 | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --verify-each=0 | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @imag(%x : tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  %y = stablehlo.imag %x : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:  func.func @imag(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.imag %arg1 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.imag %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-NEXT:    return %1, %0 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE: func.func @imag(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xf32>) -> tensor<2xcomplex<f32>> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = stablehlo.complex %cst, %arg1 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:    return %0 : tensor<2xcomplex<f32>>
// REVERSE-NEXT:  }

func.func @main() {
  %input = stablehlo.constant dense<[(1.0,2.0),(-3.0,4.0)]> : tensor<2xcomplex<f32>>
  %output = stablehlo.constant dense<[2.0, 4.0]> : tensor<2xf32>

  %dreal = stablehlo.constant dense<(1.0, 0.0)> : tensor<2xcomplex<f32>>
  %dimag = stablehlo.constant dense<(0.0, 1.0)> : tensor<2xcomplex<f32>>
  %dcomplex = stablehlo.constant dense<1.0> : tensor<2xf32>

  // fwd diff
  %fwd_real:2 = enzyme.fwddiff @imag(%input, %dreal) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_real#0, %output : tensor<2xf32>
  check.expect_almost_eq_const %fwd_real#1, dense<0.0> : tensor<2xf32>

  %fwd_imag:2 = enzyme.fwddiff @imag(%input, %dimag) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_imag#0, %output : tensor<2xf32>
  check.expect_almost_eq_const %fwd_imag#1, dense<1.0> : tensor<2xf32>

  // rev diff
  %rev:2 = enzyme.autodiff @imag(%input, %dcomplex) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xcomplex<f32>>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xcomplex<f32>>)

  check.expect_almost_eq %rev#0, %output : tensor<2xf32>
  check.expect_almost_eq_const %rev#1, dense<(0.0,1.0)> : tensor<2xcomplex<f32>>

  func.return
}
