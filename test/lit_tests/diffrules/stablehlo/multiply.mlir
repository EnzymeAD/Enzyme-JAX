// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=multiply outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=multiply outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=multiply_complex outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse --verify-each=0 | FileCheck %s --check-prefix=FORWARD-COMPLEX
// RUN enzymexlamlir-opt %s --enzyme-wrap="infn=multiply_complex outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse --verify-each=0 | FileCheck %s --check-prefix=REVERSE-COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @multiply(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.multiply %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

// FORWARD:  func.func @multiply(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.multiply %arg1, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg3, %arg0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.add %0, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.multiply %arg0, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    return %3, %2 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @multiply(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %0 = stablehlo.multiply %arg2, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.multiply %arg2, %arg0 : tensor<2xf32>
// REVERSE-NEXT:    return %0, %1 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }

func.func @multiply_complex(%a : tensor<2xcomplex<f32>>, %b : tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  %c = stablehlo.multiply %a, %b : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %c : tensor<2xcomplex<f32>>
}

// FORWARD-COMPLEX:  func.func @multiply_complex(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>, %arg2: tensor<2xcomplex<f32>>, %arg3: tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
// FORWARD-COMPLEX-NEXT:    %0 = stablehlo.multiply %arg1, %arg2 {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<2xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    %1 = stablehlo.multiply %arg3, %arg0 : tensor<2xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    %2 = stablehlo.add %0, %1 : tensor<2xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    %3 = stablehlo.multiply %arg0, %arg2 : tensor<2xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:    return %3, %2 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
// FORWARD-COMPLEX-NEXT:  }

// REVERSE-COMPLEX:  func.func @multiply_complex(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>, %arg2: tensor<2xcomplex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) {
// REVERSE-COMPLEX-NEXT:    %0 = chlo.conj %arg2 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %1 = stablehlo.multiply %0, %arg1 : tensor<2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %2 = chlo.conj %1 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %3 = stablehlo.multiply %0, %arg0 : tensor<2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    %4 = chlo.conj %3 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:    return %2, %4 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
// REVERSE-COMPLEX-NEXT:  }

func.func @main() {
  %a = stablehlo.constant dense<[1.0, -2.0]> : tensor<2xf32>
  %b = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %output = stablehlo.constant dense<[2.0, -6.0]> : tensor<2xf32>

  %done = stablehlo.constant dense<1.0> : tensor<2xf32>
  %dzero = stablehlo.constant dense<0.0> : tensor<2xf32>

  // fwd diff wrt a
  %fwd_a:2 = enzyme.fwddiff @multiply(%a, %done, %b, %dzero) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_a#0, %output : tensor<2xf32>
  check.expect_almost_eq %fwd_a#1, %b : tensor<2xf32>

  // fwd diff wrt b
  %fwd_b:2 = enzyme.fwddiff @multiply(%a, %dzero, %b, %done) {
    activity=[#enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>],
    ret_activity=[#enzyme<activity enzyme_dup>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_b#0, %output : tensor<2xf32>
  check.expect_almost_eq %fwd_b#1, %a : tensor<2xf32>

  // rev diff
  %rev:3 = enzyme.autodiff @multiply(%a, %b, %done) {
    activity=[#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>],
    ret_activity=[#enzyme<activity enzyme_active>]
  } : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev#0, %output : tensor<2xf32>
  check.expect_almost_eq %rev#1, %b : tensor<2xf32>
  check.expect_almost_eq %rev#2, %a : tensor<2xf32>

  func.return
}
