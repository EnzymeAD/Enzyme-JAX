// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_mul outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_mul_complex outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --verify-each=0 | FileCheck %s --check-prefix=COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --chlo-legalize-to-stablehlo --enzyme-hlo-opt --verify-each=0 | stablehlo-translate - --interpret --allow-unregistered-dialect

module {
  func.func @reduce_mul(%a: tensor<9xf32>, %init: tensor<f32>) -> tensor<f32> {
    %6 = "stablehlo.reduce"(%a, %init) <{dimensions = array<i64: 0>}> ({
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        %8 = "stablehlo.multiply"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%8) : (tensor<f32>) -> ()
        }) : (tensor<9xf32>, tensor<f32>) -> tensor<f32>
    return %6 : tensor<f32>
  }

  func.func @reduce_mul_complex(%a: tensor<2xcomplex<f32>>, %init: tensor<complex<f32>>) -> tensor<complex<f32>> {
    %0 = stablehlo.reduce(%a init: %init) applies stablehlo.multiply across dimensions = [0] : (tensor<2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
    return %0 : tensor<complex<f32>>
  }

  func.func @main() {
    %ca = stablehlo.constant dense<[(1.0, 2.0), (3.0, -1.0)]> : tensor<2xcomplex<f32>>
    %cinit = stablehlo.constant dense<(1.0, 0.0)> : tensor<complex<f32>>
    %cdres = stablehlo.constant dense<(1.0, 0.0)> : tensor<complex<f32>>

    %crev:3 = enzyme.autodiff @reduce_mul_complex(%ca, %cinit, %cdres) <{
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_active>]
    }> : (tensor<2xcomplex<f32>>, tensor<complex<f32>>, tensor<complex<f32>>) -> (tensor<complex<f32>>, tensor<2xcomplex<f32>>, tensor<complex<f32>>)

    check.expect_almost_eq_const %crev#0, dense<(5.0, 5.0)> : tensor<complex<f32>>
    check.expect_almost_eq_const %crev#1, dense<[(3.0, 1.0), (1.0, -2.0)]> : tensor<2xcomplex<f32>>
    check.expect_almost_eq_const %crev#2, dense<(5.0, -5.0)> : tensor<complex<f32>>
    return
  }
}

// CHECK:  func.func @reduce_mul(%arg0: tensor<9xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<9xf32>, tensor<f32>) {
// CHECK-NEXT:    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.multiply across dimensions = [0] : (tensor<9xf32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<9xf32>
// CHECK-NEXT:    %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<9xf32>
// CHECK-NEXT:    %3 = stablehlo.divide %2, %arg0 : tensor<9xf32>
// CHECK-NEXT:    %4 = stablehlo.multiply %1, %3 : tensor<9xf32>
// CHECK-NEXT:    %5 = stablehlo.divide %0, %arg1 : tensor<f32>
// CHECK-NEXT:    %6 = stablehlo.multiply %5, %arg2 : tensor<f32>
// CHECK-NEXT:    return %4, %6 : tensor<9xf32>, tensor<f32>
// CHECK-NEXT:  }

// COMPLEX:  func.func @reduce_mul_complex(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<complex<f32>>, %arg2: tensor<complex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<complex<f32>>) {
// COMPLEX-NEXT:    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.multiply across dimensions = [0] {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
// COMPLEX-NEXT:    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<complex<f32>>) -> tensor<2xcomplex<f32>>
// COMPLEX-NEXT:    %2 = stablehlo.broadcast_in_dim %0, dims = [] {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<complex<f32>>) -> tensor<2xcomplex<f32>>
// COMPLEX-NEXT:    %3 = stablehlo.divide %2, %arg0 {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<2xcomplex<f32>>
// COMPLEX-NEXT:    %4 = chlo.conj %3 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// COMPLEX-NEXT:    %5 = stablehlo.multiply %1, %4 : tensor<2xcomplex<f32>>
// COMPLEX-NEXT:    %6 = stablehlo.divide %0, %arg1 {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<complex<f32>>
// COMPLEX-NEXT:    %7 = chlo.conj %6 {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<complex<f32>> -> tensor<complex<f32>>
// COMPLEX-NEXT:    %8 = stablehlo.multiply %7, %arg2 {enzymexla.complex_is_purely_real = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<complex<f32>>
// COMPLEX-NEXT:    return %5, %8 : tensor<2xcomplex<f32>>, tensor<complex<f32>>
// COMPLEX-NEXT:  }
