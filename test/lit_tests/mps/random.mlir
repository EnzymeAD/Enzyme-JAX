// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @random_philox_state(
func.func @random_philox_state() {
  // CHECK: %{{.+}} = mps.random_philox_state  <seed = 42> : () -> tensor<7xi32>
  %0 = mps.random_philox_state <seed = 42> : () -> tensor<7xi32>
  // CHECK: %{{.+}} = mps.random_philox_state  <counter_low = 1, counter_high = 0, key = 2> : () -> tensor<7xi32>
  %1 = mps.random_philox_state <counter_low = 1, counter_high = 0, key = 2> : () -> tensor<7xi32>
  return
}

// -----

// CHECK-LABEL: func.func @random(
func.func @random() {
  // CHECK: %{{.+}} = mps.random  <distribution = uniform, min = 0.000000e+00, max = 1.000000e+00, min_integer = 0, max_integer = 0, mean = 0.000000e+00, standard_deviation = 1.000000e+00, sampling_method = invCDF> : () -> tensor<2x3xf32>
  %0 = mps.random <distribution = uniform, min = 0.000000e+00, max = 1.000000e+00, min_integer = 0, max_integer = 0, mean = 0.000000e+00, standard_deviation = 1.000000e+00, sampling_method = invCDF> : () -> tensor<2x3xf32>
  // CHECK: %{{.+}} = mps.random  <distribution = uniform, min = 0.000000e+00, max = 0.000000e+00, min_integer = 0, max_integer = 10, mean = 0.000000e+00, standard_deviation = 1.000000e+00, sampling_method = invCDF, seed = 7> : () -> tensor<2x3xi32>
  %1 = mps.random <distribution = uniform, min = 0.000000e+00, max = 0.000000e+00, min_integer = 0, max_integer = 10, mean = 0.000000e+00, standard_deviation = 1.000000e+00, sampling_method = invCDF, seed = 7> : () -> tensor<2x3xi32>
  return
}

// -----

// CHECK-LABEL: func.func @random_with_state(
func.func @random_with_state(%state: tensor<7xi32>) {
  // CHECK: %{{.+}}, %{{.+}} = mps.random_with_state %{{.+}} <distribution = truncatedNormal, min = -2.000000e+00, max = 2.000000e+00, min_integer = 0, max_integer = 0, mean = 0.000000e+00, standard_deviation = 1.000000e+00, sampling_method = boxMuller> : (tensor<7xi32>) -> (tensor<2x3xf32>, tensor<7xi32>)
  %0:2 = mps.random_with_state %state <distribution = truncatedNormal, min = -2.000000e+00, max = 2.000000e+00, min_integer = 0, max_integer = 0, mean = 0.000000e+00, standard_deviation = 1.000000e+00, sampling_method = boxMuller> : (tensor<7xi32>) -> (tensor<2x3xf32>, tensor<7xi32>)
  return
}

// -----

// CHECK-LABEL: func.func @dropout(
func.func @dropout(%x: tensor<4xf32>, %rate: tensor<f32>) {
  // CHECK: %{{.+}} = mps.dropout %{{.+}}, %{{.+}} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %0 = mps.dropout %x, %rate : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  return
}
