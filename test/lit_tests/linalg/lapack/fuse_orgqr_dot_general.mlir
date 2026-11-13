// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=fuse_orgqr_dot_general" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=TD
// RUN: enzymexlamlir-opt --enzyme-hlo-opt --lower-enzymexla-lapack="backend=cpu blas_int_width=64" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --enzyme-hlo-opt --lower-enzymexla-lapack="backend=cuda" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --enzyme-hlo-opt --lower-enzymexla-lapack="backend=tpu" %s | FileCheck %s --check-prefix=TPU

module {
  // CHECK-LABEL: func.func @sq_left_none(
  // CHECK-NEXT:    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 <side = left> : (tensor<8x8xf32>, tensor<8xf32>, tensor<8x5xf32>) -> tensor<8x5xf32>
  // CHECK-NEXT:    return %0 : tensor<8x5xf32>
  // TD-LABEL: func.func @sq_left_none(
  // TD-NEXT:    %0 = enzymexla.lapack.ormqr %arg0, %arg1, %arg2 <side = left>
  // CPU-LABEL: func.func @sq_left_none(
  // CPU:         enzymexla.jit_call @enzymexla_wrapper_lapacke_sormqr_
  // CUDA-LABEL: func.func @sq_left_none(
  // CUDA-NEXT:   %0 = stablehlo.custom_call @cusolver_orgqr_ffi(%arg0, %arg1)
  // CUDA-NEXT:   %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0]
  // TPU-LABEL: func.func @sq_left_none(
  // TPU-NEXT:    %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1)
  // TPU-NEXT:    %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0]
  func.func @sq_left_none(%A: tensor<8x8xf32>, %tau: tensor<8xf32>, %X: tensor<8x5xf32>) -> tensor<8x5xf32> {
    %Q = enzymexla.lapack.orgqr %A, %tau : (tensor<8x8xf32>, tensor<8xf32>) -> tensor<8x8xf32>
    %Y = stablehlo.dot_general %Q, %X, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x5xf32>) -> tensor<8x5xf32>
    return %Y : tensor<8x5xf32>
  }

  // CHECK-LABEL: func.func @tall_left_none(
  // CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // CHECK-NEXT:    %0 = stablehlo.pad %arg2, %cst, low = [0, 0], high = [4, 0], interior = [0, 0] : (tensor<4x5xf64>, tensor<f64>) -> tensor<8x5xf64>
  // CHECK-NEXT:    %1 = enzymexla.lapack.ormqr %arg0, %arg1, %0 <side = left> : (tensor<8x4xf64>, tensor<4xf64>, tensor<8x5xf64>) -> tensor<8x5xf64>
  // CHECK-NEXT:    return %1 : tensor<8x5xf64>
  // CPU-LABEL: func.func @tall_left_none(
  // CPU:         enzymexla.jit_call @enzymexla_wrapper_lapacke_dormqr_
  func.func @tall_left_none(%A: tensor<8x4xf64>, %tau: tensor<4xf64>, %X: tensor<4x5xf64>) -> tensor<8x5xf64> {
    %Q = enzymexla.lapack.orgqr %A, %tau : (tensor<8x4xf64>, tensor<4xf64>) -> tensor<8x4xf64>
    %Y = stablehlo.dot_general %Q, %X, contracting_dims = [1] x [0] : (tensor<8x4xf64>, tensor<4x5xf64>) -> tensor<8x5xf64>
    return %Y : tensor<8x5xf64>
  }

  // CHECK-LABEL: func.func @tall_left_trans(
  // CHECK-NEXT:    %0 = stablehlo.transpose %arg2, dims = [1, 0] : (tensor<5x8xf32>) -> tensor<8x5xf32>
  // CHECK-NEXT:    %1 = enzymexla.lapack.ormqr %arg0, %arg1, %0 <side = left, transpose = transpose> : (tensor<8x4xf32>, tensor<4xf32>, tensor<8x5xf32>) -> tensor<8x5xf32>
  // CHECK-NEXT:    %2 = stablehlo.slice %1 [0:4, 0:5] : (tensor<8x5xf32>) -> tensor<4x5xf32>
  // CHECK-NEXT:    return %2 : tensor<4x5xf32>
  func.func @tall_left_trans(%A: tensor<8x4xf32>, %tau: tensor<4xf32>, %X: tensor<5x8xf32>) -> tensor<4x5xf32> {
    %Q = enzymexla.lapack.orgqr %A, %tau : (tensor<8x4xf32>, tensor<4xf32>) -> tensor<8x4xf32>
    %Y = stablehlo.dot_general %Q, %X, contracting_dims = [0] x [1] : (tensor<8x4xf32>, tensor<5x8xf32>) -> tensor<4x5xf32>
    return %Y : tensor<4x5xf32>
  }

  // CHECK-LABEL: func.func @tall_right_none_subset_tau(
  // CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:8, 0:2] : (tensor<8x4xf32>) -> tensor<8x2xf32>
  // CHECK-NEXT:    %1 = enzymexla.lapack.ormqr %0, %arg1, %arg2 <side = right> : (tensor<8x2xf32>, tensor<2xf32>, tensor<5x8xf32>) -> tensor<5x8xf32>
  // CHECK-NEXT:    %2 = stablehlo.slice %1 [0:5, 0:4] : (tensor<5x8xf32>) -> tensor<5x4xf32>
  // CHECK-NEXT:    return %2 : tensor<5x4xf32>
  func.func @tall_right_none_subset_tau(%A: tensor<8x4xf32>, %tau: tensor<2xf32>, %X: tensor<5x8xf32>) -> tensor<5x4xf32> {
    %Q = enzymexla.lapack.orgqr %A, %tau : (tensor<8x4xf32>, tensor<2xf32>) -> tensor<8x4xf32>
    %Y = stablehlo.dot_general %X, %Q, contracting_dims = [1] x [0] : (tensor<5x8xf32>, tensor<8x4xf32>) -> tensor<5x4xf32>
    return %Y : tensor<5x4xf32>
  }

  // CHECK-LABEL: func.func @tall_right_adjoint(
  // CHECK-NEXT:    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
  // CHECK-NEXT:    %0 = stablehlo.pad %arg2, %cst, low = [0, 0], high = [0, 4], interior = [0, 0] : (tensor<5x4xcomplex<f64>>, tensor<complex<f64>>) -> tensor<5x8xcomplex<f64>>
  // CHECK-NEXT:    %1 = enzymexla.lapack.ormqr %arg0, %arg1, %0 <side = right, transpose = adjoint> : (tensor<8x4xcomplex<f64>>, tensor<4xcomplex<f64>>, tensor<5x8xcomplex<f64>>) -> tensor<5x8xcomplex<f64>>
  // CHECK-NEXT:    return %1 : tensor<5x8xcomplex<f64>>
  // CPU-LABEL: func.func @tall_right_adjoint(
  // CPU:         enzymexla.jit_call @enzymexla_wrapper_lapacke_zunmqr_
  func.func @tall_right_adjoint(%A: tensor<8x4xcomplex<f64>>, %tau: tensor<4xcomplex<f64>>, %X: tensor<5x4xcomplex<f64>>) -> tensor<5x8xcomplex<f64>> {
    %Q = enzymexla.lapack.orgqr %A, %tau : (tensor<8x4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> tensor<8x4xcomplex<f64>>
    %Qc = chlo.conj %Q : tensor<8x4xcomplex<f64>> -> tensor<8x4xcomplex<f64>>
    %Y = stablehlo.dot_general %X, %Qc, contracting_dims = [1] x [1] : (tensor<5x4xcomplex<f64>>, tensor<8x4xcomplex<f64>>) -> tensor<5x8xcomplex<f64>>
    return %Y : tensor<5x8xcomplex<f64>>
  }

  // CHECK-LABEL: func.func @vec_left(
  // CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-NEXT:    %0 = stablehlo.reshape %arg2 : (tensor<3xf32>) -> tensor<3x1xf32>
  // CHECK-NEXT:    %1 = stablehlo.pad %0, %cst, low = [0, 0], high = [3, 0], interior = [0, 0] : (tensor<3x1xf32>, tensor<f32>) -> tensor<6x1xf32>
  // CHECK-NEXT:    %2 = enzymexla.lapack.ormqr %arg0, %arg1, %1 <side = left> : (tensor<6x3xf32>, tensor<3xf32>, tensor<6x1xf32>) -> tensor<6x1xf32>
  // CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<6x1xf32>) -> tensor<6xf32>
  // CHECK-NEXT:    return %3 : tensor<6xf32>
  func.func @vec_left(%A: tensor<6x3xf32>, %tau: tensor<3xf32>, %v: tensor<3xf32>) -> tensor<6xf32> {
    %Q = enzymexla.lapack.orgqr %A, %tau : (tensor<6x3xf32>, tensor<3xf32>) -> tensor<6x3xf32>
    %y = stablehlo.dot_general %Q, %v, contracting_dims = [1] x [0] : (tensor<6x3xf32>, tensor<3xf32>) -> tensor<6xf32>
    return %y : tensor<6xf32>
  }

  // Unconjugated complex transpose is not supported by LAPACK unmqr; must not fuse.
  // CHECK-LABEL: func.func @complex_unconj_transpose_no_fuse(
  // CHECK-NEXT:    %0 = enzymexla.lapack.orgqr %arg0, %arg1
  // CHECK-NEXT:    %1 = stablehlo.dot_general %0, %arg2
  func.func @complex_unconj_transpose_no_fuse(%A: tensor<4x4xcomplex<f32>>, %tau: tensor<4xcomplex<f32>>, %X: tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>> {
    %Q = enzymexla.lapack.orgqr %A, %tau : (tensor<4x4xcomplex<f32>>, tensor<4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
    %Y = stablehlo.dot_general %Q, %X, contracting_dims = [0] x [0] : (tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
    return %Y : tensor<4x4xcomplex<f32>>
  }

  // Multi-use orgqr already materializes Q; must not duplicate Householder application.
  // CHECK-LABEL: func.func @multi_use_no_fuse(
  // CHECK-NEXT:    %0 = enzymexla.lapack.orgqr %arg0, %arg1
  // CHECK-NEXT:    %1 = stablehlo.dot_general %0, %arg2
  func.func @multi_use_no_fuse(%A: tensor<4x4xf32>, %tau: tensor<4xf32>, %X: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
    %Q = enzymexla.lapack.orgqr %A, %tau : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
    %Y = stablehlo.dot_general %Q, %X, contracting_dims = [1] x [0] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %Q, %Y : tensor<4x4xf32>, tensor<4x4xf32>
  }
}
