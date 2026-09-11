// RUN: enzymexlamlir-opt --split-input-file --pass-pipeline="any(enzyme-hlo-generate-td{patterns=concat_insert_dim_elementwise},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --pass-pipeline="builtin.module(enzyme-batch,inline,canonicalize,cse,enzyme-hlo-opt,cse,enzyme-hlo-generate-td{patterns=concat_insert_dim_elementwise},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=REACTANT

module {
  func.func @mapped_sub(%arg0: tensor<3x5x10xf32>, %arg1: tensor<3x5x10xf32>) -> (tensor<5x3x10xf32>, tensor<3x5x10xf32>, tensor<3x5x10xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<3x5x10xf32>) -> tensor<10x5x3xf32>
    %1 = stablehlo.transpose %arg1, dims = [2, 1, 0] : (tensor<3x5x10xf32>) -> tensor<10x5x3xf32>
    %2 = stablehlo.slice %0 [0:10, 0:1, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %3 = stablehlo.transpose %2, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %4 = stablehlo.reshape %3 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %6 = stablehlo.convert %5 : tensor<10x3xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %9 = stablehlo.slice %1 [0:10, 0:1, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %10 = stablehlo.transpose %9, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %11 = stablehlo.reshape %10 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %12 = stablehlo.transpose %11, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %13 = stablehlo.convert %12 : tensor<10x3xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %16 = stablehlo.subtract %8, %15 : tensor<10x3xf32>
    %17 = stablehlo.slice %0 [0:10, 1:2, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %18 = stablehlo.transpose %17, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %19 = stablehlo.reshape %18 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %20 = stablehlo.transpose %19, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %21 = stablehlo.convert %20 : tensor<10x3xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %24 = stablehlo.slice %1 [0:10, 1:2, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %25 = stablehlo.transpose %24, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %26 = stablehlo.reshape %25 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %27 = stablehlo.transpose %26, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %28 = stablehlo.convert %27 : tensor<10x3xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %31 = stablehlo.subtract %23, %30 : tensor<10x3xf32>
    %32 = stablehlo.slice %0 [0:10, 2:3, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %33 = stablehlo.transpose %32, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %34 = stablehlo.reshape %33 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %35 = stablehlo.transpose %34, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %36 = stablehlo.convert %35 : tensor<10x3xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %39 = stablehlo.slice %1 [0:10, 2:3, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %40 = stablehlo.transpose %39, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %41 = stablehlo.reshape %40 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %42 = stablehlo.transpose %41, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %43 = stablehlo.convert %42 : tensor<10x3xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %46 = stablehlo.subtract %38, %45 : tensor<10x3xf32>
    %47 = stablehlo.slice %0 [0:10, 3:4, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %48 = stablehlo.transpose %47, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %49 = stablehlo.reshape %48 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %50 = stablehlo.transpose %49, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %51 = stablehlo.convert %50 : tensor<10x3xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %54 = stablehlo.slice %1 [0:10, 3:4, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %55 = stablehlo.transpose %54, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %56 = stablehlo.reshape %55 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %57 = stablehlo.transpose %56, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %58 = stablehlo.convert %57 : tensor<10x3xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %61 = stablehlo.subtract %53, %60 : tensor<10x3xf32>
    %62 = stablehlo.slice %0 [0:10, 4:5, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %63 = stablehlo.transpose %62, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %64 = stablehlo.reshape %63 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %65 = stablehlo.transpose %64, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %66 = stablehlo.convert %65 : tensor<10x3xf32>
    %67 = stablehlo.broadcast_in_dim %66, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %68 = stablehlo.broadcast_in_dim %67, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %69 = stablehlo.slice %1 [0:10, 4:5, 0:3] : (tensor<10x5x3xf32>) -> tensor<10x1x3xf32>
    %70 = stablehlo.transpose %69, dims = [2, 1, 0] : (tensor<10x1x3xf32>) -> tensor<3x1x10xf32>
    %71 = stablehlo.reshape %70 : (tensor<3x1x10xf32>) -> tensor<3x10xf32>
    %72 = stablehlo.transpose %71, dims = [1, 0] : (tensor<3x10xf32>) -> tensor<10x3xf32>
    %73 = stablehlo.convert %72 : tensor<10x3xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1] : (tensor<10x3xf32>) -> tensor<10x3xf32>
    %76 = stablehlo.subtract %68, %75 : tensor<10x3xf32>
    %77 = stablehlo.broadcast_in_dim %16, dims = [2, 1] : (tensor<10x3xf32>) -> tensor<1x3x10xf32>
    %78 = stablehlo.broadcast_in_dim %31, dims = [2, 1] : (tensor<10x3xf32>) -> tensor<1x3x10xf32>
    %79 = stablehlo.broadcast_in_dim %46, dims = [2, 1] : (tensor<10x3xf32>) -> tensor<1x3x10xf32>
    %80 = stablehlo.broadcast_in_dim %61, dims = [2, 1] : (tensor<10x3xf32>) -> tensor<1x3x10xf32>
    %81 = stablehlo.broadcast_in_dim %76, dims = [2, 1] : (tensor<10x3xf32>) -> tensor<1x3x10xf32>
    %82 = stablehlo.concatenate %77, %78, %79, %80, %81, dim = 0 : (tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>, tensor<1x3x10xf32>) -> tensor<5x3x10xf32>
    // CHECK: stablehlo.concatenate
    %83 = stablehlo.transpose %0, dims = [2, 1, 0] : (tensor<10x5x3xf32>) -> tensor<3x5x10xf32>
    %84 = stablehlo.transpose %1, dims = [2, 1, 0] : (tensor<10x5x3xf32>) -> tensor<3x5x10xf32>
    return %82, %83, %84 : tensor<5x3x10xf32>, tensor<3x5x10xf32>, tensor<3x5x10xf32>
  }
}

// -----

// Regression test: the batched op repeats the same SSA value in two operand
// slots (`multiply %1, %1`). Each slot must be batched independently; the
// wrapper must not collapse both slots onto the same argument, which turned
// `hcat(z .* z, y .* z)` into `hcat(z .* z, z .* z)`.
module {
  // CHECK-LABEL: func.func @hcat_mul_repeated_operand
  // CHECK-DAG: %[[Z:.+]] = stablehlo.slice %arg0 [2:3, 0:2]
  // CHECK-DAG: %[[Y:.+]] = stablehlo.slice %arg0 [1:2, 0:2]
  // CHECK-DAG: %[[ZR:.+]] = stablehlo.reshape %[[Z]] : (tensor<1x2xf64>) -> tensor<2xf64>
  // CHECK-DAG: %[[YR:.+]] = stablehlo.reshape %[[Y]] : (tensor<1x2xf64>) -> tensor<2xf64>
  // CHECK-DAG: %[[ZR1:.+]] = stablehlo.reshape %[[ZR]] : (tensor<2xf64>) -> tensor<1x2xf64>
  // CHECK-DAG: %[[YR1:.+]] = stablehlo.reshape %[[YR]] : (tensor<2xf64>) -> tensor<1x2xf64>
  // CHECK-DAG: %[[LHS:.+]] = stablehlo.concatenate %[[ZR1]], %[[YR1]], dim = 0
  // CHECK-DAG: %[[ZR2:.+]] = stablehlo.reshape %[[ZR]] : (tensor<2xf64>) -> tensor<1x2xf64>
  // CHECK-DAG: %[[ZR3:.+]] = stablehlo.reshape %[[ZR]] : (tensor<2xf64>) -> tensor<1x2xf64>
  // CHECK-DAG: %[[RHS:.+]] = stablehlo.concatenate %[[ZR2]], %[[ZR3]], dim = 0
  // CHECK: %[[MUL:.+]] = stablehlo.multiply %[[LHS]], %[[RHS]] : tensor<2x2xf64>
  // CHECK: stablehlo.transpose %[[MUL]], dims = [0, 1]
  func.func @hcat_mul_repeated_operand(%arg0: tensor<3x2xf64>) -> tensor<2x2xf64> {
    %0 = stablehlo.slice %arg0 [2:3, 0:2] : (tensor<3x2xf64>) -> tensor<1x2xf64>
    %1 = stablehlo.reshape %0 : (tensor<1x2xf64>) -> tensor<2xf64>
    %2 = stablehlo.multiply %1, %1 : tensor<2xf64>
    %3 = stablehlo.slice %arg0 [1:2, 0:2] : (tensor<3x2xf64>) -> tensor<1x2xf64>
    %4 = stablehlo.reshape %3 : (tensor<1x2xf64>) -> tensor<2xf64>
    %5 = stablehlo.multiply %4, %1 : tensor<2xf64>
    %6 = stablehlo.reshape %2 : (tensor<2xf64>) -> tensor<1x2xf64>
    %7 = stablehlo.reshape %5 : (tensor<2xf64>) -> tensor<1x2xf64>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<2x2xf64>
    return %8 : tensor<2x2xf64>
  }
}

// -----

// Verbatim `@code_hlo optimize=false` dump from Reactant.jl (0.2.285) for
//
//   f(u) = hcat(u[:, 3] .* u[:, 3], u[:, 2] .* u[:, 3])
//   u = [1.0 2.0 3.0; 4.0 5.0 6.0]
//
// After CSE the first multiply becomes `multiply %z, %z`, i.e. the same SSA
// value in both operand slots. Batching the two multiplies via
// concat_insert_dim_elementwise used to map both slots of the wrapper function
// onto the same argument, so the compiled result was hcat(z.*z, z.*z) instead
// of hcat(z.*z, y.*z). The second column (u[:, 2]) must survive in the output.

// REACTANT-LABEL: func.func @main
// REACTANT-DAG: %[[Z:.+]] = stablehlo.slice %arg0 [2:3, 0:2] : (tensor<3x2xf64>) -> tensor<1x2xf64>
// REACTANT-DAG: %[[Y:.+]] = stablehlo.slice %arg0 [1:2, 0:2] : (tensor<3x2xf64>) -> tensor<1x2xf64>
// REACTANT-DAG: %[[ZT:.+]] = stablehlo.reshape %[[Z]] : (tensor<1x2xf64>) -> tensor<2x1xf64>
// REACTANT-DAG: %[[YT:.+]] = stablehlo.reshape %[[Y]] : (tensor<1x2xf64>) -> tensor<2x1xf64>
// REACTANT-DAG: %[[ZY:.+]] = stablehlo.concatenate %[[ZT]], %[[YT]], dim = 1
// REACTANT-DAG: %[[ZB:.+]] = stablehlo.broadcast_in_dim %[[Z]], dims = [1, 0] : (tensor<1x2xf64>) -> tensor<2x2xf64>
// REACTANT: stablehlo.multiply %[[ZY]], %[[ZB]]

module @reactant_f attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"*_broadcast_scalar"(%arg0: tensor<f64> {enzymexla.memory_effects = []}, %arg1: tensor<f64> {enzymexla.memory_effects = []}) -> (tensor<f64>, tensor<f64>, tensor<f64>) attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
    return %0, %arg0, %arg1 : tensor<f64>, tensor<f64>, tensor<f64>
  }
  func.func private @"*_broadcast_scalar_1"(%arg0: tensor<f64> {enzymexla.memory_effects = []}, %arg1: tensor<f64> {enzymexla.memory_effects = []}) -> (tensor<f64>, tensor<f64>, tensor<f64>) attributes {enzymexla.memory_effects = []} {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
    return %0, %arg0, %arg1 : tensor<f64>, tensor<f64>, tensor<f64>
  }
  func.func @main(%arg0: tensor<3x2xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}) -> (tensor<2x2xf64>, tensor<3x2xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<3x2xf64>) -> tensor<2x3xf64>
    %1 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x3xf64>) -> tensor<2x1xf64>
    %2 = stablehlo.transpose %1, dims = [1, 0] : (tensor<2x1xf64>) -> tensor<1x2xf64>
    %3 = stablehlo.reshape %2 : (tensor<1x2xf64>) -> tensor<2xf64>
    %4 = stablehlo.transpose %3, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %5 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x3xf64>) -> tensor<2x1xf64>
    %6 = stablehlo.transpose %5, dims = [1, 0] : (tensor<2x1xf64>) -> tensor<1x2xf64>
    %7 = stablehlo.reshape %6 : (tensor<1x2xf64>) -> tensor<2xf64>
    %8 = stablehlo.transpose %7, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
    %9 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %11 = stablehlo.broadcast_in_dim %8, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %13:3 = enzyme.batch @"*_broadcast_scalar"(%10, %12) {batch_shape = array<i64: 2>} : (tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>)
    %14 = stablehlo.slice %0 [0:2, 1:2] : (tensor<2x3xf64>) -> tensor<2x1xf64>
    %15 = stablehlo.transpose %14, dims = [1, 0] : (tensor<2x1xf64>) -> tensor<1x2xf64>
    %16 = stablehlo.reshape %15 : (tensor<1x2xf64>) -> tensor<2xf64>
    %17 = stablehlo.transpose %16, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %18 = stablehlo.slice %0 [0:2, 2:3] : (tensor<2x3xf64>) -> tensor<2x1xf64>
    %19 = stablehlo.transpose %18, dims = [1, 0] : (tensor<2x1xf64>) -> tensor<1x2xf64>
    %20 = stablehlo.reshape %19 : (tensor<1x2xf64>) -> tensor<2xf64>
    %21 = stablehlo.transpose %20, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
    %22 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %24 = stablehlo.broadcast_in_dim %21, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %26:3 = enzyme.batch @"*_broadcast_scalar_1"(%23, %25) {batch_shape = array<i64: 2>} : (tensor<2xf64>, tensor<2xf64>) -> (tensor<2xf64>, tensor<2xf64>, tensor<2xf64>)
    %27 = stablehlo.transpose %13#0, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %28 = stablehlo.reshape %27 : (tensor<2xf64>) -> tensor<1x2xf64>
    %29 = stablehlo.transpose %28, dims = [1, 0] : (tensor<1x2xf64>) -> tensor<2x1xf64>
    %30 = stablehlo.convert %29 : tensor<2x1xf64>
    %31 = stablehlo.transpose %26#0, dims = [0] : (tensor<2xf64>) -> tensor<2xf64>
    %32 = stablehlo.reshape %31 : (tensor<2xf64>) -> tensor<1x2xf64>
    %33 = stablehlo.transpose %32, dims = [1, 0] : (tensor<1x2xf64>) -> tensor<2x1xf64>
    %34 = stablehlo.convert %33 : tensor<2x1xf64>
    %35 = stablehlo.concatenate %30, %34, dim = 1 : (tensor<2x1xf64>, tensor<2x1xf64>) -> tensor<2x2xf64>
    %36 = stablehlo.transpose %35, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %37 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xf64>) -> tensor<3x2xf64>
    return %36, %37 : tensor<2x2xf64>, tensor<3x2xf64>
  }
}
