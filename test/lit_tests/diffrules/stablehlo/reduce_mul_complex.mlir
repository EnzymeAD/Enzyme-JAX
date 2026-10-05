// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=prod outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse --verify-each=0 | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --chlo-legalize-to-stablehlo --enzyme-hlo-opt --verify-each=0 | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @prod(%a: tensor<2xcomplex<f32>>, %init: tensor<complex<f32>>) -> tensor<complex<f32>> {
  %0 = stablehlo.reduce(%a init: %init) applies stablehlo.multiply across dimensions = [0] : (tensor<2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
  return %0 : tensor<complex<f32>>
}

// CHECK:  func.func @prod(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<complex<f32>>, %arg2: tensor<complex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<complex<f32>>) {
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.multiply across dimensions = [0] {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<complex<f32>>) -> tensor<2xcomplex<f32>>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %0, dims = [] {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<complex<f32>>) -> tensor<2xcomplex<f32>>
// CHECK-NEXT:    %3 = stablehlo.divide %2, %arg0 {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<2xcomplex<f32>>
// CHECK-NEXT:    %4 = chlo.conj %3 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// CHECK-NEXT:    %5 = stablehlo.multiply %1, %4 : tensor<2xcomplex<f32>>
// CHECK-NEXT:    %6 = stablehlo.divide %0, %arg1 {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<complex<f32>>
// CHECK-NEXT:    %7 = chlo.conj %6 {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<complex<f32>> -> tensor<complex<f32>>
// CHECK-NEXT:    %8 = stablehlo.multiply %7, %arg2 {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<complex<f32>>
// CHECK-NEXT:    return %5, %8 : tensor<2xcomplex<f32>>, tensor<complex<f32>>
// CHECK-NEXT:  }

func.func @main() {
  %a = stablehlo.constant dense<[(1.0, 2.0), (3.0, -4.0)]> : tensor<2xcomplex<f32>>
  %init = stablehlo.constant dense<(2.0, 1.0)> : tensor<complex<f32>>
  %seed = stablehlo.constant dense<(1.0, 0.0)> : tensor<complex<f32>>

  %rev:3 = enzyme.autodiff @prod(%a, %init, %seed) {
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  } : (tensor<2xcomplex<f32>>, tensor<complex<f32>>, tensor<complex<f32>>) -> (tensor<complex<f32>>, tensor<2xcomplex<f32>>, tensor<complex<f32>>)

  check.expect_almost_eq_const %rev#0, dense<(20.0, 15.0)> : tensor<complex<f32>>
  check.expect_almost_eq_const %rev#1, dense<[(10.0, 5.0), (0.0, -5.0)]> : tensor<2xcomplex<f32>>
  check.expect_almost_eq_const %rev#2, dense<(11.0, -2.0)> : tensor<complex<f32>>

  func.return
}
