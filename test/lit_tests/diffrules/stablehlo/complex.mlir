// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=complex outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=complex outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse --verify-each=0 | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --verify-each=0 | stablehlo-translate - --interpret

func.func @complex(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xcomplex<f32>> {
  %c = stablehlo.complex %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xcomplex<f32>>
  func.return %c : tensor<2xcomplex<f32>>
}

// FORWARD:  func.func @complex(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
// FORWARD-NEXT:    %0 = stablehlo.complex %arg1, %arg3 : tensor<2xcomplex<f32>>
// FORWARD-NEXT:    %1 = stablehlo.complex %arg0, %arg2 : tensor<2xcomplex<f32>>
// FORWARD-NEXT:    return %1, %0 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
// FORWARD-NEXT:  }

// REVERSE:  func.func @complex(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xcomplex<f32>>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %0 = stablehlo.real %arg2 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.imag %arg2 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// REVERSE-NEXT:    return %0, %1 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }

func.func @main() {
  %real = stablehlo.constant dense<[1.0, -2.0]> : tensor<2xf32>
  %imag = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %out = stablehlo.constant dense<[(1.0,2.0),(-2.0,3.0)]> : tensor<2xcomplex<f32>>

  %done = stablehlo.constant dense<1.0> : tensor<2xf32>
  %dzero = stablehlo.constant dense<0.0> : tensor<2xf32>

  // fwd diff wrt real
  %fwd_real:2 = enzyme.fwddiff @complex(%real, %done, %imag, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>)

  check.expect_almost_eq %fwd_real#0, %out : tensor<2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_real#1, dense<(1.0,0.0)> : tensor<2xcomplex<f32>>

  // fwd diff wrt imag
  %fwd_b:2 = enzyme.fwddiff @complex(%real, %dzero, %imag, %done) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>)

  check.expect_almost_eq %fwd_b#0, %out : tensor<2xcomplex<f32>>
  check.expect_almost_eq_const %fwd_b#1, dense<(0.0,1.0)> : tensor<2xcomplex<f32>>

  // rev diff
  %dcomplex_real = stablehlo.constant dense<(1.0,0.0)> : tensor<2xcomplex<f32>>
  %dcomplex_imag = stablehlo.constant dense<(0.0,1.0)> : tensor<2xcomplex<f32>>

  %rev_real:3 = enzyme.autodiff @complex(%real, %imag, %dcomplex_real) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev_real#0, %out : tensor<2xcomplex<f32>>
  check.expect_almost_eq_const %rev_real#1, dense<1.0> : tensor<2xf32>
  check.expect_almost_eq_const %rev_real#2, dense<0.0> : tensor<2xf32>

  %rev_imag:3 = enzyme.autodiff @complex(%real, %imag, %dcomplex_imag) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev_imag#0, %out : tensor<2xcomplex<f32>>
  check.expect_almost_eq_const %rev_imag#1, dense<0.0> : tensor<2xf32>
  check.expect_almost_eq_const %rev_imag#2, dense<1.0> : tensor<2xf32>

  func.return
}
