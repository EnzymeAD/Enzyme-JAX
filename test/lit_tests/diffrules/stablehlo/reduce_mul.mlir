// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_mul outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_mul_complex outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --verify-each=0 | FileCheck %s --check-prefix=COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_mul outfn= argTys=enzyme_dup,enzyme_dup retTys=enzyme_dup mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme --enzyme-batch-to-stablehlo --inline --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --chlo-legalize-to-stablehlo --enzyme-hlo-opt --drop-unsupported-attributes --verify-each=0 | stablehlo-translate - --interpret --allow-unregistered-dialect

module {
  func.func @reduce_mul(%a: tensor<9xf32>, %init: tensor<f32>) -> tensor<f32> {
    %6 = "stablehlo.reduce"(%a, %init) <{dimensions = array<i64: 0>}> ({
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        %8 = "stablehlo.multiply"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%8) : (tensor<f32>) -> ()
        }) : (tensor<9xf32>, tensor<f32>) -> tensor<f32>
    return %6 : tensor<f32>
  }

  func.func @reduce_mul_2(%a: tensor<2xf64>, %init: tensor<f64>) -> tensor<f64> {
    %0 = stablehlo.reduce(%a init: %init) applies stablehlo.multiply across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
  }

  func.func @reduce_window_mul(%a: tensor<3xf64>) -> tensor<2xf64> {
    %cst = stablehlo.constant dense<1.0> : tensor<f64>
    %0 = "stablehlo.reduce_window"(%a, %cst) <{window_dimensions = array<i64: 2>}> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<f64>):
      %1 = stablehlo.multiply %arg1, %arg2 : tensor<f64>
      stablehlo.return %1 : tensor<f64>
    }) : (tensor<3xf64>, tensor<f64>) -> tensor<2xf64>
    return %0 : tensor<2xf64>
  }

  func.func @reduce_max_2(%a: tensor<3xf64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %0 = stablehlo.reduce(%a init: %cst) applies stablehlo.maximum across dimensions = [0] : (tensor<3xf64>, tensor<f64>) -> tensor<f64>
    return %0 : tensor<f64>
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

    // Forward mode width=1 (Reactant #3297 MWE + zero element + active init)
    %theta = stablehlo.constant dense<[0.3, -0.7]> : tensor<2xf64>
    %dtheta1 = stablehlo.constant dense<[1.0, 0.0]> : tensor<2xf64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %fwd1:2 = enzyme.fwddiff @reduce_mul_2(%theta, %dtheta1, %one) <{
      activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
      ret_activity=[#enzyme.activity<enzyme_dup>]
    }> : (tensor<2xf64>, tensor<2xf64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_almost_eq_const %fwd1#0, dense<-0.21> : tensor<f64>
    check.expect_almost_eq_const %fwd1#1, dense<-0.7> : tensor<f64>

    // Zero element in input + active init in both width=1 and batched width=2
    %z_in = stablehlo.constant dense<[0.0, 3.0]> : tensor<2xf64>
    %dz_w1 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
    %init7 = stablehlo.constant dense<7.0> : tensor<f64>
    %dinit_w1 = stablehlo.constant dense<10.0> : tensor<f64>
    %fwd_z1:2 = enzyme.fwddiff @reduce_mul_2(%z_in, %dz_w1, %init7, %dinit_w1) <{
      activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity=[#enzyme.activity<enzyme_dup>]
    }> : (tensor<2xf64>, tensor<2xf64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_almost_eq_const %fwd_z1#0, dense<0.0> : tensor<f64>
    check.expect_almost_eq_const %fwd_z1#1, dense<21.0> : tensor<f64>

    %dz_w2 = stablehlo.constant dense<[[1.0, 2.0], [4.0, 5.0]]> : tensor<2x2xf64>
    %dinit_w2 = stablehlo.constant dense<[10.0, 20.0]> : tensor<2xf64>
    %fwd_z2:2 = enzyme.fwddiff @reduce_mul_2(%z_in, %dz_w2, %init7, %dinit_w2) <{
      activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity=[#enzyme.activity<enzyme_dup>],
      width=2 : i64
    }> : (tensor<2xf64>, tensor<2x2xf64>, tensor<f64>, tensor<2xf64>) -> (tensor<f64>, tensor<2xf64>)
    check.expect_almost_eq_const %fwd_z2#0, dense<0.0> : tensor<f64>
    check.expect_almost_eq_const %fwd_z2#1, dense<[21.0, 84.0]> : tensor<2xf64>

    // Batched width=2 with constant init
    %dtheta_w2 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %fwd_cst2:2 = enzyme.fwddiff @reduce_mul_2(%theta, %dtheta_w2, %one) <{
      activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
      ret_activity=[#enzyme.activity<enzyme_dup>],
      width=2 : i64
    }> : (tensor<2xf64>, tensor<2x2xf64>, tensor<f64>) -> (tensor<f64>, tensor<2xf64>)
    check.expect_almost_eq_const %fwd_cst2#0, dense<-0.21> : tensor<f64>
    check.expect_almost_eq_const %fwd_cst2#1, dense<[-0.7, 0.3]> : tensor<2xf64>

    // Batched width=2 reduce_window multiply & reduce maximum
    %rw_in = stablehlo.constant dense<[2.0, 3.0, 5.0]> : tensor<3xf64>
    %drw_in = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]]> : tensor<2x3xf64>
    %fwd_rw:2 = enzyme.fwddiff @reduce_window_mul(%rw_in, %drw_in) <{
      activity=[#enzyme.activity<enzyme_dup>],
      ret_activity=[#enzyme.activity<enzyme_dup>],
      width=2 : i64
    }> : (tensor<3xf64>, tensor<2x3xf64>) -> (tensor<2xf64>, tensor<2x2xf64>)
    check.expect_almost_eq_const %fwd_rw#0, dense<[6.0, 15.0]> : tensor<2xf64>
    check.expect_almost_eq_const %fwd_rw#1, dense<[[3.0, 0.0], [2.0, 11.0]]> : tensor<2x2xf64>

    %fwd_mx:2 = enzyme.fwddiff @reduce_max_2(%rw_in, %drw_in) <{
      activity=[#enzyme.activity<enzyme_dup>],
      ret_activity=[#enzyme.activity<enzyme_dup>],
      width=2 : i64
    }> : (tensor<3xf64>, tensor<2x3xf64>) -> (tensor<f64>, tensor<2xf64>)
    check.expect_almost_eq_const %fwd_mx#0, dense<5.0> : tensor<f64>
    check.expect_almost_eq_const %fwd_mx#1, dense<[0.0, 2.0]> : tensor<2xf64>
    return
  }
}

// FORWARD-LABEL: func.func @reduce_mul(
// FORWARD: stablehlo.reduce
// FORWARD: stablehlo.multiply
// FORWARD: stablehlo.add

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
