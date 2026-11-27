// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --chlo-legalize-to-stablehlo --enzyme-hlo-opt --drop-unsupported-attributes --verify-each=0 | stablehlo-translate - --interpret --allow-unregistered-dialect

// Verify reverse-mode autodiff of stablehlo.reduce with stablehlo.multiply
// produces finite exact gradients (no 0/0 = NaN) when input elements or init
// are zero (EnzymeAD/Reactant.jl#3191).
module {
  func.func @reduce_mul_2d(%x: tensor<4x2xf64>, %init: tensor<f64>) -> tensor<4xf64> {
    %0 = stablehlo.reduce(%x init: %init) applies stablehlo.multiply across dimensions = [1] : (tensor<4x2xf64>, tensor<f64>) -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }

  func.func @sum_abs2_prod(%x: tensor<4x2xf64>) -> tensor<f64> {
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %p = stablehlo.reduce(%x init: %one) applies stablehlo.multiply across dimensions = [1] : (tensor<4x2xf64>, tensor<f64>) -> tensor<4xf64>
    %sq = stablehlo.multiply %p, %p : tensor<4xf64>
    %s = stablehlo.reduce(%sq init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<4xf64>, tensor<f64>) -> tensor<f64>
    return %s : tensor<f64>
  }

  func.func @reduce_mul_complex_zero(%x: tensor<3xcomplex<f32>>, %init: tensor<complex<f32>>) -> tensor<complex<f32>> {
    %0 = stablehlo.reduce(%x init: %init) applies stablehlo.multiply across dimensions = [0] : (tensor<3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
    return %0 : tensor<complex<f32>>
  }

  func.func @main() {
    // Reactant.jl#3191 reproducer: column 0 is 2.0, column 1 is 0.0;
    // d/dx sum((x[:,0] * x[:,1])^2) = 0 everywhere (not NaN on column 1).
    %s = stablehlo.constant dense<[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]> : tensor<4x2xf64>
    %one_f64 = stablehlo.constant dense<1.0> : tensor<f64>
    %rep:2 = enzyme.autodiff @sum_abs2_prod(%s, %one_f64) <{
      activity = [#enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<4x2xf64>, tensor<f64>) -> (tensor<f64>, tensor<4x2xf64>)
    check.expect_almost_eq_const %rep#0, dense<0.0> : tensor<f64>
    check.expect_almost_eq_const %rep#1, dense<0.0> : tensor<4x2xf64>

    // Mixed rows:
    // row 0: [3, 4] (0 zeros), init=2 -> y=24, dx=[8, 6], dinit contribution=12
    // row 1: [5, 0] (1 zero),  init=2 -> y=0,  dx=[0, 10], dinit contribution=0
    // row 2: [0, 7] (1 zero),  init=2 -> y=0,  dx=[14, 0], dinit contribution=0
    // row 3: [0, 0] (2 zeros), init=2 -> y=0,  dx=[0, 0],  dinit contribution=0
    %m = stablehlo.constant dense<[[3.0, 4.0], [5.0, 0.0], [0.0, 7.0], [0.0, 0.0]]> : tensor<4x2xf64>
    %init2 = stablehlo.constant dense<2.0> : tensor<f64>
    %seed4 = stablehlo.constant dense<[1.0, 1.0, 1.0, 1.0]> : tensor<4xf64>
    %r1:3 = enzyme.autodiff @reduce_mul_2d(%m, %init2, %seed4) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<4x2xf64>, tensor<f64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4x2xf64>, tensor<f64>)
    check.expect_almost_eq_const %r1#0, dense<[24.0, 0.0, 0.0, 0.0]> : tensor<4xf64>
    check.expect_almost_eq_const %r1#1, dense<[[8.0, 6.0], [0.0, 10.0], [14.0, 0.0], [0.0, 0.0]]> : tensor<4x2xf64>
    check.expect_almost_eq_const %r1#2, dense<12.0> : tensor<f64>

    // Zero init: init=0 -> dx=0 everywhere, dinit = prod_j x_j on non-zero rows (3*4 = 12).
    %init0 = stablehlo.constant dense<0.0> : tensor<f64>
    %r2:3 = enzyme.autodiff @reduce_mul_2d(%m, %init0, %seed4) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<4x2xf64>, tensor<f64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<4x2xf64>, tensor<f64>)
    check.expect_almost_eq_const %r2#0, dense<0.0> : tensor<4xf64>
    check.expect_almost_eq_const %r2#1, dense<0.0> : tensor<4x2xf64>
    check.expect_almost_eq_const %r2#2, dense<12.0> : tensor<f64>

    // Complex with single zero element: [(0, 0), (1, 2), (3, -1)], init=(1, 0)
    // prod of non-zero = (1+2i)*(3-i) = (5 + 5i), conj = (5 - 5i) at index 0, 0 elsewhere.
    %cx = stablehlo.constant dense<[(0.0, 0.0), (1.0, 2.0), (3.0, -1.0)]> : tensor<3xcomplex<f32>>
    %cinit = stablehlo.constant dense<(1.0, 0.0)> : tensor<complex<f32>>
    %cdres = stablehlo.constant dense<(1.0, 0.0)> : tensor<complex<f32>>
    %cr:3 = enzyme.autodiff @reduce_mul_complex_zero(%cx, %cinit, %cdres) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    }> : (tensor<3xcomplex<f32>>, tensor<complex<f32>>, tensor<complex<f32>>) -> (tensor<complex<f32>>, tensor<3xcomplex<f32>>, tensor<complex<f32>>)
    check.expect_almost_eq_const %cr#0, dense<(0.0, 0.0)> : tensor<complex<f32>>
    check.expect_almost_eq_const %cr#1, dense<[(5.0, -5.0), (0.0, 0.0), (0.0, 0.0)]> : tensor<3xcomplex<f32>>
    check.expect_almost_eq_const %cr#2, dense<(0.0, 0.0)> : tensor<complex<f32>>
    return
  }
}
