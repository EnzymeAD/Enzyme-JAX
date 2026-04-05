// RUN: enzymexlamlir-opt --pad-for-alignment --canonicalize --allow-unregistered-dialect %s | FileCheck %s

func.func @test_elementwise_abs(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.abs %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_abs(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.abs %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_cbrt(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.cbrt %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_cbrt(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.cbrt %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_ceil(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.ceil %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_ceil(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.ceil %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_convert(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf64> {
  %0 = stablehlo.convert %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf64>
  return %0 : tensor<4x760x1533xf64>
}

// CHECK: func.func @test_elementwise_convert(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.convert %0 : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf64>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf64>) -> tensor<4x760x1533xf64>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf64>
// CHECK-NEXT: }

func.func @test_elementwise_cosine(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.cosine %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_cosine(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.cosine %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_count_leading_zeros(%arg0: tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64> {
  %0 = stablehlo.count_leading_zeros %arg0 : (tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64>
  return %0 : tensor<4x760x1533xi64>
}

// CHECK: func.func @test_elementwise_count_leading_zeros(%arg0: tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %1 = stablehlo.count_leading_zeros %0 : tensor<4x768x1536xi64>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi64>) -> tensor<4x760x1533xi64>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xi64>
// CHECK-NEXT: }

func.func @test_elementwise_exponential(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.exponential %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_exponential(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.exponential %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_exponential_minus_one(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.exponential_minus_one %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_exponential_minus_one(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.exponential_minus_one %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_floor(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.floor %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_floor(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.floor %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_imag(%arg0: tensor<4x760x1533xcomplex<f32>>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.imag %arg0 : (tensor<4x760x1533xcomplex<f32>>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_imag(%arg0: tensor<4x760x1533xcomplex<f32>>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x768x1536xcomplex<f32>>
// CHECK-NEXT:     %1 = stablehlo.imag %0 : (tensor<4x768x1536xcomplex<f32>>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_is_finite(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xi1> {
  %0 = stablehlo.is_finite %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xi1>
  return %0 : tensor<4x760x1533xi1>
}

// CHECK: func.func @test_elementwise_is_finite(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xi1> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.is_finite %0 : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi1>) -> tensor<4x760x1533xi1>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xi1>
// CHECK-NEXT: }

func.func @test_elementwise_log(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.log %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_log(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_log_plus_one(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.log_plus_one %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_log_plus_one(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.log_plus_one %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_logistic(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.logistic %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_logistic(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.logistic %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_negate(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.negate %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_negate(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.negate %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_not(%arg0: tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1> {
  %0 = stablehlo.not %arg0 : (tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1>
  return %0 : tensor<4x760x1533xi1>
}

// CHECK: func.func @test_elementwise_not(%arg0: tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1> {
// CHECK-NEXT:     %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %1 = stablehlo.not %0 : tensor<4x768x1536xi1>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi1>) -> tensor<4x760x1533xi1>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xi1>
// CHECK-NEXT: }

func.func @test_elementwise_popcnt(%arg0: tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64> {
  %0 = stablehlo.popcnt %arg0 : (tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64>
  return %0 : tensor<4x760x1533xi64>
}

// CHECK: func.func @test_elementwise_popcnt(%arg0: tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %1 = stablehlo.popcnt %0 : tensor<4x768x1536xi64>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi64>) -> tensor<4x760x1533xi64>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xi64>
// CHECK-NEXT: }

func.func @test_elementwise_real(%arg0: tensor<4x760x1533xcomplex<f32>>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.real %arg0 : (tensor<4x760x1533xcomplex<f32>>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_real(%arg0: tensor<4x760x1533xcomplex<f32>>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x768x1536xcomplex<f32>>
// CHECK-NEXT:     %1 = stablehlo.real %0 : (tensor<4x768x1536xcomplex<f32>>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_reduce_precision(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.reduce_precision %arg0, format = e5m10 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_reduce_precision(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.reduce_precision %0, format = e5m10 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_round_nearest_afz(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.round_nearest_afz %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_round_nearest_afz(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.round_nearest_afz %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_round_nearest_even(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.round_nearest_even %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_round_nearest_even(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.round_nearest_even %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_rsqrt(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.rsqrt %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_rsqrt(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.rsqrt %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_sign(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.sign %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_sign(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.sign %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_sine(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.sine %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_sine(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.sine %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_sqrt(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.sqrt %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_sqrt(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.sqrt %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_tan(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.tan %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_tan(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.tan %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_tanh(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.tanh %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_elementwise_tanh(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.tanh %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_add(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.add %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_elementwise_add(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.add %[[pad0]], %[[pad1]] : tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_atan2(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.atan2 %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_elementwise_atan2(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.atan2 %[[pad0]], %[[pad1]] : tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_and(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1> {
  %0 = stablehlo.and %arg0, %arg1 : (tensor<4x760x1533xi1>, tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1>
  return %0 : tensor<4x760x1533xi1>
}

// CHECK-LABEL: func.func @test_elementwise_and(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1> {
// CHECK-NEXT:     %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %2 = stablehlo.and %[[pad0]], %[[pad1]] : tensor<4x768x1536xi1>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi1>) -> tensor<4x760x1533xi1>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xi1>
// CHECK-NEXT: }

func.func @test_elementwise_compare(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xi1> {
  %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xi1>
  return %0 : tensor<4x760x1533xi1>
}

// CHECK-LABEL: func.func @test_elementwise_compare(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xi1> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.compare LT, %[[pad0]], %[[pad1]] : (tensor<4x768x1536xf32>, tensor<4x768x1536xf32>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi1>) -> tensor<4x760x1533xi1>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xi1>
// CHECK-NEXT: }

func.func @test_elementwise_complex(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xcomplex<f32>> {
  %0 = stablehlo.complex %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xcomplex<f32>>
  return %0 : tensor<4x760x1533xcomplex<f32>>
}

// CHECK-LABEL: func.func @test_elementwise_complex(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xcomplex<f32>> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.complex %[[pad0]], %[[pad1]] : tensor<4x768x1536xcomplex<f32>>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xcomplex<f32>>) -> tensor<4x760x1533xcomplex<f32>>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xcomplex<f32>>
// CHECK-NEXT: }

func.func @test_elementwise_divide(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.divide %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_elementwise_divide(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.divide %[[pad0]], %[[pad1]] : tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_maximum(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.maximum %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_elementwise_maximum(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.maximum %[[pad0]], %[[pad1]] : tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_minimum(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.minimum %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_elementwise_minimum(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.minimum %[[pad0]], %[[pad1]] : tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_multiply(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_elementwise_multiply(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.multiply %[[pad0]], %[[pad1]] : tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_or(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1> {
  %0 = stablehlo.or %arg0, %arg1 : (tensor<4x760x1533xi1>, tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1>
  return %0 : tensor<4x760x1533xi1>
}

// CHECK-LABEL: func.func @test_elementwise_or(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1> {
// CHECK-NEXT:     %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %2 = stablehlo.or %[[pad0]], %[[pad1]] : tensor<4x768x1536xi1>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi1>) -> tensor<4x760x1533xi1>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xi1>
// CHECK-NEXT: }

func.func @test_elementwise_power(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.power %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_elementwise_power(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.power %[[pad0]], %[[pad1]] : tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_remainder(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.remainder %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_elementwise_remainder(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.remainder %[[pad0]], %[[pad1]] : tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_shift_left(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64> {
  %0 = stablehlo.shift_left %arg0, %arg1 : (tensor<4x760x1533xi64>, tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64>
  return %0 : tensor<4x760x1533xi64>
}

// CHECK-LABEL: func.func @test_elementwise_shift_left(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %2 = stablehlo.shift_left %[[pad0]], %[[pad1]] : tensor<4x768x1536xi64>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi64>) -> tensor<4x760x1533xi64>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xi64>
// CHECK-NEXT: }

func.func @test_elementwise_shift_right_arithmetic(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64> {
  %0 = stablehlo.shift_right_arithmetic %arg0, %arg1 : (tensor<4x760x1533xi64>, tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64>
  return %0 : tensor<4x760x1533xi64>
}

// CHECK-LABEL: func.func @test_elementwise_shift_right_arithmetic(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %2 = stablehlo.shift_right_arithmetic %[[pad0]], %[[pad1]] : tensor<4x768x1536xi64>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi64>) -> tensor<4x760x1533xi64>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xi64>
// CHECK-NEXT: }

func.func @test_elementwise_shift_right_logical(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64> {
  %0 = stablehlo.shift_right_logical %arg0, %arg1 : (tensor<4x760x1533xi64>, tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64>
  return %0 : tensor<4x760x1533xi64>
}

// CHECK-LABEL: func.func @test_elementwise_shift_right_logical(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %2 = stablehlo.shift_right_logical %[[pad0]], %[[pad1]] : tensor<4x768x1536xi64>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi64>) -> tensor<4x760x1533xi64>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xi64>
// CHECK-NEXT: }

func.func @test_elementwise_subtract(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.subtract %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_elementwise_subtract(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.subtract %[[pad0]], %[[pad1]] : tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_elementwise_xor(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1> {
  %0 = stablehlo.xor %arg0, %arg1 : (tensor<4x760x1533xi1>, tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1>
  return %0 : tensor<4x760x1533xi1>
}
// CHECK-LABEL: func.func @test_elementwise_xor(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1> {
// CHECK-NEXT:     %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %2 = stablehlo.xor %[[pad0]], %[[pad1]] : tensor<4x768x1536xi1>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xi1>) -> tensor<4x760x1533xi1>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xi1>
// CHECK-NEXT: }

func.func @test_elementwise_clamp(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK-LABEL: func.func @test_elementwise_clamp(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:      %[[pad2:.*]] = stablehlo.pad %arg2, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.clamp %[[pad0]], %[[pad1]], %[[pad2]] : tensor<4x768x1536xf32>
// CHECK-NEXT:     %4 = stablehlo.slice %3 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %4 : tensor<4x760x1533xf32>
// CHECK-NEXT: }
