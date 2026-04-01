// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s

module {
  func.func private @f(%arg18: tensor<1xf64>, %arg26: tensor<1xf64>, %arg33: tensor<1xf64>, %arg58: tensor<1xf64>) -> (tensor<1xf64>, tensor<1xf64>) {

    %c_203 = stablehlo.constant dense<8> : tensor<i32>
    %c_204 = stablehlo.constant dense<0> : tensor<i32>

    %c_214 = stablehlo.constant dense<16> : tensor<i64>

    %c_216 = stablehlo.constant dense<0> : tensor<i64>

    %c_316 = stablehlo.constant dense<1> : tensor<i64>

    %4 = stablehlo.dynamic_update_slice %arg26, %arg58, %c_204 : (tensor<1xf64>, tensor<1xf64>, tensor<i32>) -> tensor<1xf64>

    %23315:2 = stablehlo.while(%iterArg = %c_216, %iterArg_537 = %arg18) : tensor<i64>, tensor<1xf64> attributes {enzyme.disable_mincut, enzymexla.enable_checkpointing = true}
     cond {
      %69678 = stablehlo.compare  LT, %iterArg, %c_214 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %69678 : tensor<i1>
    } do {

      %69678 = stablehlo.dynamic_update_slice %arg33, %iterArg_537, %c_204 : (tensor<1xf64>, tensor<1xf64>, tensor<i32>) -> tensor<1xf64>
      
      %69740 = stablehlo.add %iterArg, %c_316 : tensor<i64>
      stablehlo.return %69740, %iterArg_537 : tensor<i64>, tensor<1xf64>
    }
    return %23315#1, %4 : tensor<1xf64>, tensor<1xf64>
  }

  func.func @differentiate_tracer_error(%arg18: tensor<1xf64>, %arg26: tensor<1xf64>, %arg33: tensor<1xf64>, %arg58: tensor<1xf64>, %res1 : tensor<1xf64>, %res2 : tensor<1xf64>) -> (tensor<1xf64>, tensor<1xf64>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %cst_0 = stablehlo.constant dense<3.600000e+02> : tensor<f64>
    %cst_1 = stablehlo.constant dense<3.000000e+01> : tensor<f64>
    %cst_2 = stablehlo.constant dense<3999.9999999999995> : tensor<f64>
    %cst_3 = stablehlo.constant dense<6.371000e+06> : tensor<f64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0:6 = enzyme.autodiff @f(%arg18, %arg26, %arg33, %arg58, %res1, %res2) {activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_active>]} : (tensor<1xf64>,  tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) -> (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>)
    return %0#0, %0#1 : tensor<1xf64>, tensor<1xf64>
  }
}

// This checks that we don't accidentally replace users of the %c_204 captured by the while loop for checkpointing with a pop from within checkpointing, and crash
// when computing the derivative of %4.

// CHECK: func.func private @diffef

