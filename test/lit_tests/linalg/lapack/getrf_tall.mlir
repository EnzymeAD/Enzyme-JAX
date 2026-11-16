// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cuda},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CUDA
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=rocm},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=ROCM
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

// The permutation has one entry per row, so for a tall matrix it is longer
// than the pivots, and it takes the element type of the op's result.

module {
  func.func @main(%arg0: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>) {
    %0:4 = enzymexla.lapack.getrf %arg0 : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>
  }
}

// CUDA: func.func @main(%arg0: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>) {
// CUDA:     %[[LU:.+]]:3 = stablehlo.custom_call @cusolver_getrf_ffi(%arg0)
// CUDA-SAME:  -> (tensor<6x4xf32>, tensor<4xi32>, tensor<i32>)
// CUDA:     %[[PERM:.+]] = stablehlo.custom_call @cu_lu_pivots_to_permutation(%{{.+}}) {{.*}} : (tensor<4xi32>) -> tensor<6xi32>
// CUDA:     %[[PERM1:.+]] = stablehlo.add %[[PERM]], %{{.+}} : tensor<6xi32>
// CUDA:     %[[PERM64:.+]] = stablehlo.convert %[[PERM1]] : (tensor<6xi32>) -> tensor<6xi64>
// CUDA:     return %[[LU]]#0, %{{.+}}, %[[PERM64]], %{{.+}} : tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>

// ROCM: func.func @main(%arg0: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>) {
// ROCM:     %[[LU:.+]]:3 = stablehlo.custom_call @hipsolver_getrf_ffi(%arg0)
// ROCM-SAME:  -> (tensor<6x4xf32>, tensor<4xi32>, tensor<i32>)
// ROCM:     %[[PERM:.+]] = stablehlo.custom_call @hip_lu_pivots_to_permutation(%{{.+}}) {{.*}} : (tensor<4xi32>) -> tensor<6xi32>
// ROCM:     %[[PERM1:.+]] = stablehlo.add %[[PERM]], %{{.+}} : tensor<6xi32>
// ROCM:     %[[PERM64:.+]] = stablehlo.convert %[[PERM1]] : (tensor<6xi32>) -> tensor<6xi64>
// ROCM:     return %[[LU]]#0, %{{.+}}, %[[PERM64]], %{{.+}} : tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>

// TPU: func.func @main(%arg0: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>) {
// TPU:     %[[LU:.+]]:3 = stablehlo.custom_call @LuDecomposition(%arg0) : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi32>, tensor<6xi32>)
// TPU:     %[[PERM:.+]] = stablehlo.convert %[[LU]]#2 : (tensor<6xi32>) -> tensor<6xi64>
// TPU:     %[[PERM1:.+]] = stablehlo.add %{{.+}}, %[[PERM]] : tensor<6xi64>
// TPU:     return %[[LU]]#0, %{{.+}}, %[[PERM1]], %{{.+}} : tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>

// CPU: func.func @main(%arg0: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>) {
// CPU:     %[[IOTA:.+]] = stablehlo.constant dense<[0, 1, 2, 3, 4, 5]> : tensor<6xi64>
// CPU:     %[[NPIV:.+]] = stablehlo.constant dense<4> : tensor<i32>
// CPU:     %[[LOOP:.+]]:2 = stablehlo.while(%{{.+}} = %{{.+}}, %{{.+}} = %[[IOTA]]) : tensor<i32>, tensor<6xi64>
// CPU:       stablehlo.compare LT, %{{.+}}, %[[NPIV]]
// CPU:     %[[PERM:.+]] = stablehlo.add %[[LOOP]]#1, %{{.+}} : tensor<6xi64>
// CPU:     return %{{.+}}, %{{.+}}, %[[PERM]], %{{.+}} : tensor<6x4xf32>, tensor<4xi64>, tensor<6xi64>, tensor<i64>
