// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-linalg{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s

func.func @gesvd(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
  %0:4 = enzymexla.linalg.svd %arg0 {algorithm = #enzymexla.svd_algorithm<QRIteration>, compute_uv = true} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>
}

// CHECK: func.func @gesvd(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
// CHECK-NEXT:   %U, %S, %Vt, %info = enzymexla.lapack.gesvd %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)

func.func @gesvd_vals(%arg0: tensor<64x64xf32>) -> (tensor<64xf32>) {
  %0:4 = enzymexla.linalg.svd %arg0 {algorithm = #enzymexla.svd_algorithm<QRIteration>} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
  return %0#1 : tensor<64xf32>
}

// CHECK: func.func @gesvd_vals
// CHECK-NEXT:   %[[U:.*]], %[[S:.*]], %[[V:.*]], %[[INFO:.*]] = enzymexla.lapack.gesvd %arg0 {compute_uv = false}


func.func @gesdd(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
  %0:4 = enzymexla.linalg.svd %arg0 {algorithm = #enzymexla.svd_algorithm<DivideAndConquer>, compute_uv = true} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>
}

// CHECK: func.func @gesdd(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
// CHECK-NEXT:   %U, %S, %Vt, %info = enzymexla.lapack.gesdd %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)

func.func @gesdd_vals(%arg0: tensor<64x64xf32>) -> (tensor<64xf32>) {
  %0:4 = enzymexla.linalg.svd %arg0 {algorithm = #enzymexla.svd_algorithm<DivideAndConquer>} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
  return %0#1 : tensor<64xf32>
}

// CHECK: func.func @gesdd_vals
// CHECK-NEXT:   %[[U:.*]], %[[S:.*]], %[[V:.*]], %[[INFO:.*]] = enzymexla.lapack.gesdd %arg0 {compute_uv = false}

func.func @gesvj(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
  %0:4 = enzymexla.linalg.svd %arg0 {algorithm = #enzymexla.svd_algorithm<Jacobi>, compute_uv = true} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>
}

// CHECK: func.func @gesvj(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>) {
// CHECK-NEXT:   %U, %S, %Vt, %info = enzymexla.lapack.gesvj %arg0 : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)

func.func @gesvj_vals(%arg0: tensor<64x64xf32>) -> (tensor<64xf32>) {
  %0:4 = enzymexla.linalg.svd %arg0 {algorithm = #enzymexla.svd_algorithm<Jacobi>} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64x64xf32>, tensor<i64>)
  return %0#1 : tensor<64xf32>
}

// CHECK: func.func @gesvj_vals
// CHECK-NEXT:   %[[U:.*]], %[[S:.*]], %[[V:.*]], %[[INFO:.*]] = enzymexla.lapack.gesvj %arg0 {compute_uv = false}
