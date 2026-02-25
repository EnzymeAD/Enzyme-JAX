// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=abs outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=abs outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse --verify-each=0 | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=abs_complex outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD-COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=abs_complex outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse --verify-each=0 | FileCheck %s --check-prefix=REVERSE-COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @abs(%arg0 : tensor<2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.abs %arg0 : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// FORWARD:  func.func @abs(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:    %0 = stablehlo.compare  GE, %arg0, %cst : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// FORWARD-NEXT:    %1 = stablehlo.negate %arg1 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.select %0, %arg1, %1 : tensor<2xi1>, tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.abs %arg0 : tensor<2xf32>
// FORWARD-NEXT:    return %3, %2 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @abs(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:    %0 = stablehlo.compare  GE, %arg0, %cst : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// REVERSE-NEXT:    %1 = stablehlo.negate %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.select %0, %arg1, %1 : tensor<2xi1>, tensor<2xf32>
// REVERSE-NEXT:    return %2 : tensor<2xf32>
// REVERSE-NEXT:  }

func.func @abs_complex(%arg0 : tensor<2xcomplex<f32>>) -> tensor<2xf32> {
  %0 = stablehlo.abs %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// FORWARD-COMPLEX:  func.func @abs_complex(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-COMPLEX-NEXT:    %0 = stablehlo.multiply %arg1, %arg0 {enzymexla.complex_is_purely_imaginary = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    %1 = stablehlo.real %0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-COMPLEX-NEXT:    %2 = stablehlo.abs %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// FORWARD-COMPLEX-NEXT:    %3 = stablehlo.divide %1, %2 : tensor<2xf32>
// FORWARD-COMPLEX-NEXT:    return %2, %3 : tensor<2xf32>, tensor<2xf32>
// FORWARD-COMPLEX-NEXT:  }

// REVERSE-COMPLEX:  func.func @abs_complex(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xf32>) -> tensor<2xcomplex<f32>> {
// REVERSE-COMPLEX-NEXT:    %0 = stablehlo.real %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// REVERSE-COMPLEX-NEXT:    %1 = stablehlo.abs %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// REVERSE-COMPLEX-NEXT:    %2 = stablehlo.divide %0, %1 : tensor<2xf32>
// REVERSE-COMPLEX-NEXT:    %3 = stablehlo.multiply %arg1, %2 : tensor<2xf32>
// REVERSE-COMPLEX-NEXT:    %4 = stablehlo.imag %arg0 : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
// REVERSE-COMPLEX-NEXT:    %5 = stablehlo.divide %4, %1 : tensor<2xf32>
// REVERSE-COMPLEX-NEXT:    %6 = stablehlo.multiply %arg1, %5 : tensor<2xf32>
// REVERSE-COMPLEX-NEXT:    %7 = stablehlo.complex %3, %6 {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %8 = chlo.conj %7 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    return %8 : tensor<2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:  }

func.func @main() {
  // real
  %input = stablehlo.constant dense<[-1.0, 2.0]> : tensor<2xf32>
  %out = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %expected = stablehlo.constant dense<[-1.0, 1.0]> : tensor<2xf32>

  %dinput = stablehlo.constant dense<1.0> : tensor<2xf32>

  %fwd:2 = enzyme.fwddiff @abs(%input, %dinput) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd#0, %out : tensor<2xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<2xf32>

  %rev:2 = enzyme.autodiff @abs(%input, %dinput) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev#0, %out : tensor<2xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<2xf32>

  // complex
  // TODO `chlo.conj` does not have interpreter support
  %cinput = stablehlo.constant dense<[(-1.0, 0.0), (3.0, -4.0)]> : tensor<2xcomplex<f32>>
  %cout = stablehlo.constant dense<[1.0, 5.0]> : tensor<2xf32>
  %cexpected = stablehlo.constant dense<[(-1.0, 0.0), (1.0, 0.0)]> : tensor<2xcomplex<f32>>

  %dcinput = stablehlo.constant dense<(1.0, 0.0)> : tensor<2xcomplex<f32>>

  %cfwd:2 = enzyme.fwddiff @abs_complex(%cinput, %dcinput) {
    activity=[#enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xf32>, tensor<2xcomplex<f32>>)

  check.expect_almost_eq %cfwd#0, %cout : tensor<2xf32>
  // check.expect_almost_eq %cfwd#1, %cexpected : tensor<2xcomplex<f32>>s

  %crev:2 = enzyme.autodiff @abs_complex(%cinput, %dinput) {
    activity=[#enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xcomplex<f32>>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xcomplex<f32>>)

  check.expect_almost_eq %crev#0, %cout : tensor<2xf32>
  // check.expect_almost_eq %crev#1, %cexpected : tensor<2xcomplex<f32>>

  func.return
}
