// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu},lower-probprog-trace-ops{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  // CPU:    func.func @test_uniform_scalar(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>) {
  // CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // CPU-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  // CPU-NEXT:    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
  // CPU-NEXT:    %c = stablehlo.constant dense<12> : tensor<ui64>
  // CPU-NEXT:    %0 = stablehlo.shift_right_logical %output, %c : tensor<ui64>
  // CPU-NEXT:    %c_1 = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
  // CPU-NEXT:    %1 = stablehlo.or %0, %c_1 : tensor<ui64>
  // CPU-NEXT:    %2 = stablehlo.bitcast_convert %1 : (tensor<ui64>) -> tensor<f64>
  // CPU-NEXT:    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  // CPU-NEXT:    %3 = stablehlo.subtract %2, %cst_2 : tensor<f64>
  // CPU-NEXT:    return %output_state, %3 : tensor<2xui64>, tensor<f64>
  // CPU-NEXT:  }
  func.func @test_uniform_scalar(%rng: tensor<2xui64>) -> (tensor<2xui64>, tensor<f64>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %new_rng, %result = enzyme.random %rng, %zero, %one {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %new_rng, %result : tensor<2xui64>, tensor<f64>
  }

  // CPU:  func.func @test_normal_vector(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10xf64>) {
  // CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // CPU-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  // CPU-NEXT:    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10xui64>)
  // CPU-NEXT:    %c = stablehlo.constant dense<12> : tensor<10xui64>
  // CPU-NEXT:    %0 = stablehlo.shift_right_logical %output, %c : tensor<10xui64>
  // CPU-NEXT:    %c_1 = stablehlo.constant dense<4607182418800017408> : tensor<10xui64>
  // CPU-NEXT:    %1 = stablehlo.or %0, %c_1 : tensor<10xui64>
  // CPU-NEXT:    %2 = stablehlo.bitcast_convert %1 : (tensor<10xui64>) -> tensor<10xf64>
  // CPU-NEXT:    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<10xf64>
  // CPU-NEXT:    %3 = stablehlo.subtract %2, %cst_2 : tensor<10xf64>
  // CPU-NEXT:    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<10xf64>
  // CPU-NEXT:    %4 = stablehlo.multiply %3, %cst_3 : tensor<10xf64>
  // CPU-NEXT:    %5 = stablehlo.subtract %4, %cst_2 : tensor<10xf64>
  // CPU-NEXT:    %6 = chlo.erf_inv %5 : tensor<10xf64> -> tensor<10xf64>
  // CPU-NEXT:    %cst_4 = stablehlo.constant dense<1.4142135623730951> : tensor<10xf64>
  // CPU-NEXT:    %7 = stablehlo.multiply %6, %cst_4 : tensor<10xf64>
  // CPU-NEXT:    return %output_state, %7 : tensor<2xui64>, tensor<10xf64>
  // CPU-NEXT:  }
  func.func @test_normal_vector(%rng: tensor<2xui64>) -> (tensor<2xui64>, tensor<10xf64>) {
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %new_rng, %result = enzyme.random %rng, %zero, %one {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<10xf64>)
    return %new_rng, %result : tensor<2xui64>, tensor<10xf64>
  }

  // CPU:  func.func @test_normal_2d(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<3x4xf64>) {
  // CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // CPU-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  // CPU-NEXT:    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<3x4xui64>)
  // CPU-NEXT:    %c = stablehlo.constant dense<12> : tensor<3x4xui64>
  // CPU-NEXT:    %0 = stablehlo.shift_right_logical %output, %c : tensor<3x4xui64>
  // CPU-NEXT:    %c_1 = stablehlo.constant dense<4607182418800017408> : tensor<3x4xui64>
  // CPU-NEXT:    %1 = stablehlo.or %0, %c_1 : tensor<3x4xui64>
  // CPU-NEXT:    %2 = stablehlo.bitcast_convert %1 : (tensor<3x4xui64>) -> tensor<3x4xf64>
  // CPU-NEXT:    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<3x4xf64>
  // CPU-NEXT:    %3 = stablehlo.subtract %2, %cst_2 : tensor<3x4xf64>
  // CPU-NEXT:    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<3x4xf64>
  // CPU-NEXT:    %4 = stablehlo.multiply %3, %cst_3 : tensor<3x4xf64>
  // CPU-NEXT:    %5 = stablehlo.subtract %4, %cst_2 : tensor<3x4xf64>
  // CPU-NEXT:    %6 = chlo.erf_inv %5 : tensor<3x4xf64> -> tensor<3x4xf64>
  // CPU-NEXT:    %cst_4 = stablehlo.constant dense<1.4142135623730951> : tensor<3x4xf64>
  // CPU-NEXT:    %7 = stablehlo.multiply %6, %cst_4 : tensor<3x4xf64>
  // CPU-NEXT:    return %output_state, %7 : tensor<2xui64>, tensor<3x4xf64>
  // CPU-NEXT:  }
  func.func @test_normal_2d(%rng: tensor<2xui64>) -> (tensor<2xui64>, tensor<3x4xf64>) {
    %mean = stablehlo.constant dense<0.0> : tensor<f64>
    %std = stablehlo.constant dense<1.0> : tensor<f64>
    %new_rng, %result = enzyme.random %rng, %mean, %std {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<3x4xf64>)
    return %new_rng, %result : tensor<2xui64>, tensor<3x4xf64>
  }

  // CPU:  func.func @test_multinormal(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<3xf64>) {
  // CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // CPU-NEXT:    %cst_0 = stablehlo.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00{{\]}}, {{\[}}0.000000e+00, 1.000000e+00, 0.000000e+00{{\]}}, {{\[}}0.000000e+00, 0.000000e+00, 1.000000e+00{{\]\]}}> : tensor<3x3xf64>
  // CPU-NEXT:    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<3xui64>)
  // CPU-NEXT:    %c = stablehlo.constant dense<12> : tensor<3xui64>
  // CPU-NEXT:    %0 = stablehlo.shift_right_logical %output, %c : tensor<3xui64>
  // CPU-NEXT:    %c_1 = stablehlo.constant dense<4607182418800017408> : tensor<3xui64>
  // CPU-NEXT:    %1 = stablehlo.or %0, %c_1 : tensor<3xui64>
  // CPU-NEXT:    %2 = stablehlo.bitcast_convert %1 : (tensor<3xui64>) -> tensor<3xf64>
  // CPU-NEXT:    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<3xf64>
  // CPU-NEXT:    %3 = stablehlo.subtract %2, %cst_2 : tensor<3xf64>
  // CPU-NEXT:    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<3xf64>
  // CPU-NEXT:    %4 = stablehlo.multiply %3, %cst_3 : tensor<3xf64>
  // CPU-NEXT:    %5 = stablehlo.subtract %4, %cst_2 : tensor<3xf64>
  // CPU-NEXT:    %6 = chlo.erf_inv %5 : tensor<3xf64> -> tensor<3xf64>
  // CPU-NEXT:    %cst_4 = stablehlo.constant dense<1.4142135623730951> : tensor<3xf64>
  // CPU-NEXT:    %7 = stablehlo.multiply %6, %cst_4 : tensor<3xf64>
  // CPU-NEXT:    %8 = stablehlo.cholesky %cst_0, lower = true : tensor<3x3xf64>
  // CPU-NEXT:    %9 = stablehlo.dot_general %8, %7, contracting_dims = [1] x [0] : (tensor<3x3xf64>, tensor<3xf64>) -> tensor<3xf64>
  // CPU-NEXT:    %10 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<3xf64>
  // CPU-NEXT:    %11 = stablehlo.add %10, %9 : tensor<3xf64>
  // CPU-NEXT:    return %output_state, %11 : tensor<2xui64>, tensor<3xf64>
  // CPU-NEXT:  }
  func.func @test_multinormal(%rng: tensor<2xui64>) -> (tensor<2xui64>, tensor<3xf64>) {
    %mean = stablehlo.constant dense<0.0> : tensor<f64>
    %cov = stablehlo.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf64>
    %new_rng, %result = enzyme.random %rng, %mean, %cov {rng_distribution = #enzyme<rng_distribution MULTINORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<3x3xf64>) -> (tensor<2xui64>, tensor<3xf64>)
    return %new_rng, %result : tensor<2xui64>, tensor<3xf64>
  }
}
