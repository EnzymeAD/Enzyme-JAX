// RUN: enzymexlamlir-opt %s --pass-pipeline="any(enzyme{postpasses=arith-raise{stablehlo=true},canonicalize,cse,canonicalize,remove-unnecessary-enzyme-ops,enzyme-simplify-math,canonicalize,cse,canonicalize,arith-raise{stablehlo=true} verifyPostPasses=true},inline{default-pipeline=canonicalize inlining-threshold=4294967295 max-iterations=4 },enzyme-hlo-opt,remove-unnecessary-enzyme-ops)" | stablehlo-translate - --interpret --allow-unregistered-dialect

// see issue #2105
module {
  func.func private @real_exp_complex(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %1 = stablehlo.complex %0, %arg0 : tensor<complex<f64>>
    %2 = stablehlo.exponential %1 : tensor<complex<f64>>
    %3 = stablehlo.real %2 : (tensor<complex<f64>>) -> tensor<f64>
    return %3, %arg0 : tensor<f64>, tensor<f64>
  }

  func.func private @imag_exp_complex(%arg0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %1 = stablehlo.complex %0, %arg0 : tensor<complex<f64>>
    %2 = stablehlo.exponential %1 : tensor<complex<f64>>
    %3 = stablehlo.imag %2 : (tensor<complex<f64>>) -> tensor<f64>
    return %3, %arg0 : tensor<f64>, tensor<f64>
  }

  func.func @main() {
    // %x = pi/3
    %x = stablehlo.constant dense<1.0471975511965976> : tensor<f64>

    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>

    %1:2 = enzyme.autodiff @real_exp_complex(%x, %cst, %0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>]} : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_almost_eq_const %1#1, dense<-0.8660254038> : tensor<f64>

    %2:2 = enzyme.autodiff @imag_exp_complex(%x, %cst, %0) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>]} : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_almost_eq_const %2#1, dense<0.5> : tensor<f64>

    return
  }
}
