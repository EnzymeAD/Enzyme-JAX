// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=broadcast_in_dim_simplify<16>(1024);fft_simplify;broadcast_reshape" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s --check-prefix=TD
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @fft_zero() -> (tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>>, tensor<4x3xcomplex<f32>>, tensor<4x4xf32>) {
  %z_c64 = stablehlo.constant dense<(0.0, 0.0)> : tensor<4x4xcomplex<f64>>
  %f0 = stablehlo.fft %z_c64, type = FFT, length = [4, 4] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>>

  %s_c64 = stablehlo.constant dense<(0.0, 0.0)> : tensor<complex<f64>>
  %b_c64 = stablehlo.broadcast_in_dim %s_c64, dims = [] : (tensor<complex<f64>>) -> tensor<4x4xcomplex<f64>>
  %f1 = stablehlo.fft %b_c64, type = IFFT, length = [4, 4] : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>>

  %z_f32 = stablehlo.constant dense<0.0> : tensor<4x4xf32>
  %f2 = stablehlo.fft %z_f32, type = RFFT, length = [4] : (tensor<4x4xf32>) -> tensor<4x3xcomplex<f32>>

  %z_ir = stablehlo.constant dense<(0.0, 0.0)> : tensor<4x3xcomplex<f32>>
  %f3 = stablehlo.fft %z_ir, type = IRFFT, length = [4] : (tensor<4x3xcomplex<f32>>) -> tensor<4x4xf32>

  return %f0, %f1, %f2, %f3 : tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>>, tensor<4x3xcomplex<f32>>, tensor<4x4xf32>
}

func.func @bcast_reshape_inserted_singletons(%a4: tensor<2x1xf32>, %b: tensor<2x1x1xf32>) -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>) {
  %a5 = stablehlo.reshape %a4 : (tensor<2x1xf32>) -> tensor<2x1x1xf32>
  %a6 = stablehlo.broadcast_in_dim %a5, dims = [0, 2, 1] : (tensor<2x1x1xf32>) -> tensor<2x3x4xf32>
  %b2 = stablehlo.broadcast_in_dim %b, dims = [0, 2, 1] : (tensor<2x1x1xf32>) -> tensor<2x3x4xf32>
  return %a6, %b2 : tensor<2x3x4xf32>, tensor<2x3x4xf32>
}

func.func @main() {
  %f0, %f1, %f2, %f3 = func.call @fft_zero() : () -> (tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>>, tensor<4x3xcomplex<f32>>, tensor<4x4xf32>)
  %ez0 = stablehlo.constant dense<(0.0, 0.0)> : tensor<4x4xcomplex<f64>>
  %ez2 = stablehlo.constant dense<(0.0, 0.0)> : tensor<4x3xcomplex<f32>>
  %ez3 = stablehlo.constant dense<0.0> : tensor<4x4xf32>
  "check.expect_almost_eq"(%f0, %ez0) : (tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>>) -> ()
  "check.expect_almost_eq"(%f1, %ez0) : (tensor<4x4xcomplex<f64>>, tensor<4x4xcomplex<f64>>) -> ()
  "check.expect_almost_eq"(%f2, %ez2) : (tensor<4x3xcomplex<f32>>, tensor<4x3xcomplex<f32>>) -> ()
  "check.expect_almost_eq"(%f3, %ez3) : (tensor<4x4xf32>, tensor<4x4xf32>) -> ()

  %in_a = stablehlo.constant dense<[[10.0], [20.0]]> : tensor<2x1xf32>
  %in_b = stablehlo.constant dense<[[[10.0]], [[20.0]]]> : tensor<2x1x1xf32>
  %ra, %rb = func.call @bcast_reshape_inserted_singletons(%in_a, %in_b) : (tensor<2x1xf32>, tensor<2x1x1xf32>) -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>)
  "check.expect_almost_eq"(%ra, %rb) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @fft_zero()
// CHECK-NOT: stablehlo.fft
// CHECK: return

// CHECK-LABEL: func.func @bcast_reshape_inserted_singletons(
// CHECK-SAME: %[[A4:.+]]: tensor<2x1xf32>, %[[B:.+]]: tensor<2x1x1xf32>)
// CHECK-NEXT: %[[R0:.+]] = stablehlo.broadcast_in_dim %[[A4]], dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT: %[[R1:.+]] = stablehlo.broadcast_in_dim %[[B]], dims = [0, 1, 2] : (tensor<2x1x1xf32>) -> tensor<2x3x4xf32>
// CHECK-NEXT: return %[[R0]], %[[R1]]

// TD-LABEL: func.func @fft_zero()
// TD-NOT: stablehlo.fft
// TD-LABEL: func.func @bcast_reshape_inserted_singletons(
// TD-SAME: %[[TA4:.+]]: tensor<2x1xf32>
// TD: stablehlo.broadcast_in_dim %[[TA4]], dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x3x4xf32>
