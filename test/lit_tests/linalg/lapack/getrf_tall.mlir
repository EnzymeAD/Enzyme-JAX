// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=cpu blas_int_width=64},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=CPU
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-lapack{backend=tpu},enzyme-hlo-opt)" %s | FileCheck %s --check-prefix=TPU

module {
  func.func @main(%arg0: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi32>, tensor<6xi32>, tensor<i32>) {
    %0:4 = enzymexla.lapack.getrf %arg0 : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi32>, tensor<6xi32>, tensor<i32>)
    return %0#0, %0#1, %0#2, %0#3 : tensor<6x4xf32>, tensor<4xi32>, tensor<6xi32>, tensor<i32>
  }
}

// The permutation has one entry per row, the pivots one per elimination step.
// CPU:  func.func @main(%arg0: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi32>, tensor<6xi32>, tensor<i32>) {
// CPU-DAG:    %[[IOTA:.+]] = stablehlo.constant dense<{{\[}}0, 1, 2, 3, 4, 5]> : tensor<6xi64>
// CPU-DAG:    %[[STEPS:.+]] = stablehlo.constant dense<4> : tensor<i32>
// CPU:    %[[LU:.+]]:3 = call @enzymexla_lapack_sgetrf_{{[0-9]+}}(%arg0) : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi64>, tensor<i64>)
// CPU:    %[[LOOP:.+]]:2 = stablehlo.while(%iterArg = %{{.+}}, %iterArg_{{[0-9]+}} = %[[IOTA]]) : tensor<i32>, tensor<6xi64>
// CPU:      stablehlo.compare LT, %iterArg, %[[STEPS]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CPU:    %[[PIVOTS:.+]] = stablehlo.convert %[[LU]]#1 : (tensor<4xi64>) -> tensor<4xi32>
// CPU:    %[[PERM:.+]] = stablehlo.convert %{{.+}} : (tensor<6xi64>) -> tensor<6xi32>
// CPU:    return %[[LU]]#0, %[[PIVOTS]], %[[PERM]], %{{.+}} : tensor<6x4xf32>, tensor<4xi32>, tensor<6xi32>, tensor<i32>

// TPU:  func.func @main(%arg0: tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi32>, tensor<6xi32>, tensor<i32>) {
// TPU:    %[[LU:.+]]:3 = stablehlo.custom_call @LuDecomposition(%arg0) : (tensor<6x4xf32>) -> (tensor<6x4xf32>, tensor<4xi32>, tensor<6xi32>)
// TPU:    %[[PIVOTS:.+]] = stablehlo.add %{{.+}}, %[[LU]]#1 : tensor<4xi32>
// TPU:    %[[PERM:.+]] = stablehlo.add %{{.+}}, %[[LU]]#2 : tensor<6xi32>
// TPU:    return %[[LU]]#0, %[[PIVOTS]], %[[PERM]], %{{.+}} : tensor<6x4xf32>, tensor<4xi32>, tensor<6xi32>, tensor<i32>
