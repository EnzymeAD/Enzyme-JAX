// RUN: enzymexlamlir-opt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise %s | FileCheck %s

module {
  func.func private @"Const{typeof(f)}(Main.f)_autodiff"(%arg0: tensor<2x3x5xf64>) -> (tensor<f64>, tensor<2x3x5xf64>) {
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<2x3x5xf64>) -> tensor<5x3x2xf64>
    %1 = stablehlo.multiply %0, %0 : tensor<5x3x2xf64>
    %2 = stablehlo.dot_general %1, %arg0, contracting_dims = [0, 1, 2] x [2, 1, 0] : (tensor<5x3x2xf64>, tensor<2x3x5xf64>) -> tensor<f64>
    return %2, %arg0 : tensor<f64>, tensor<2x3x5xf64>
  }
  func.func @gradient(%arg0: tensor<2x3x5xf64>) -> (tensor<2x3x5xf64>, tensor<f64>, tensor<2x3x5xf64>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0:3 = enzyme.autodiff @"Const{typeof(f)}(Main.f)_autodiff"(%arg0, %cst) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>]} : (tensor<2x3x5xf64>, tensor<f64>) -> (tensor<f64>, tensor<2x3x5xf64>, tensor<2x3x5xf64>)
    return %0#2, %0#0, %0#1 : tensor<2x3x5xf64>, tensor<f64>, tensor<2x3x5xf64>
  }
}

// CHECK-LABEL: @"diffeConst{typeof(f)}(Main.f)_autodiff"
// CHECK: %2 = stablehlo.dot_general %1, %arg0, contracting_dims = [0, 1, 2] x [2, 1, 0] : (tensor<5x3x2xf64>, tensor<2x3x5xf64>) -> tensor<f64>
// CHECK: %3 = stablehlo.add %arg1, %cst : tensor<f64>
// CHECK: %4 = stablehlo.dot_general %3, %arg0, contracting_dims = [] x [] : (tensor<f64>, tensor<2x3x5xf64>) -> tensor<2x3x5xf64>
// CHECK: %5 = stablehlo.transpose %4, dims = [2, 1, 0] : (tensor<2x3x5xf64>) -> tensor<5x3x2xf64>
// CHECK: %6 = stablehlo.add %5, %cst_1 : tensor<5x3x2xf64>
// CHECK: %7 = stablehlo.dot_general %3, %1, contracting_dims = [] x [] : (tensor<f64>, tensor<5x3x2xf64>) -> tensor<5x3x2xf64>
// CHECK: %8 = stablehlo.transpose %7, dims = [2, 1, 0] : (tensor<5x3x2xf64>) -> tensor<2x3x5xf64>
