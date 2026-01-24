// RUN: enzymexlamlir-opt --transform-interpreter %s | FileCheck %s

module @"reactant_\E2\88\87potential" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64, transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.dynamic_slice_elementwise
    } : !transform.any_op
    transform.yield 
  }
  func.func @main(%arg0: tensor<5x5xf32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}, %arg1: tensor<5xf32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 2 : i32}, %arg2: tensor<3x5xf32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 3 : i32}) -> (tensor<3x5xf32>, tensor<5x5xf32>, tensor<5xf32>, tensor<3x5xf32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<3x5xf32>
    %cst_0 = stablehlo.constant dense<4.471500e-02> : tensor<3x5xf32>
    %c = stablehlo.constant dense<3> : tensor<i64>
    %c_1 = stablehlo.constant dense<5> : tensor<i64>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<3x5xf32>
    %cst_3 = stablehlo.constant dense<32.00000e+00> : tensor<3x5x3x5xf32>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %c_5 = stablehlo.constant dense<0> : tensor<i64>
    %c_6 = stablehlo.constant dense<1> : tensor<i32>
    %c_7 = stablehlo.constant dense<0> : tensor<i32>
    %cst_8 = stablehlo.constant dense<5.000000e-01> : tensor<3x5xf32>
    %cst_9 = stablehlo.constant dense<0.797884583> : tensor<3x5xf32>
    %0 = stablehlo.transpose %cst_9, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
    %1 = stablehlo.transpose %cst_0, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
    %2 = stablehlo.transpose %cst, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
    %3 = stablehlo.transpose %cst_8, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
    %4:5 = stablehlo.while(%iterArg = %c_5, %iterArg_10 = %cst_2, %iterArg_11 = %arg0, %iterArg_12 = %arg1, %iterArg_13 = %arg2) : tensor<i64>, tensor<3x5xf32>, tensor<5x5xf32>, tensor<5xf32>, tensor<3x5xf32> attributes {enzyme.disable_mincut, enzymexla.symmetric_matrix = [#enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed NOTGUARANTEED>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>]}
    cond {
      %5 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    } do {
      %5 = stablehlo.add %c_4, %iterArg {enzymexla.bounds = [[1, 3]]} : tensor<i64>
      %6 = stablehlo.convert %5 {enzymexla.bounds = [[1, 3]]} : (tensor<i64>) -> tensor<i32>
      %7 = stablehlo.subtract %6, %c_6 {enzymexla.bounds = [[0, 2]]} : tensor<i32>
      %8:5 = stablehlo.while(%iterArg_14 = %c_5, %iterArg_15 = %iterArg_13, %iterArg_16 = %iterArg_10, %iterArg_17 = %iterArg_11, %iterArg_18 = %iterArg_12) : tensor<i64>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<5x5xf32>, tensor<5xf32> attributes {enzyme.disable_mincut}
      cond {
        %9 = stablehlo.compare  LT, %iterArg_14, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %9 : tensor<i1>
      } do {
        %9 = stablehlo.add %c_4, %iterArg_14 {enzymexla.bounds = [[1, 5]]} : tensor<i64>
        %10 = stablehlo.convert %9 {enzymexla.bounds = [[1, 5]]} : (tensor<i64>) -> tensor<i32>
        %11 = stablehlo.subtract %10, %c_6 {enzymexla.bounds = [[0, 4]]} : tensor<i32>
        %12 = stablehlo.dynamic_slice %cst_3, %7, %11, %c_7, %c_7, sizes = [1, 1, 3, 5] : (tensor<3x5x3x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x3x5xf32>
        %13 = stablehlo.reshape %12 : (tensor<1x1x3x5xf32>) -> tensor<3x5xf32>
        %14 = stablehlo.dot_general %13, %iterArg_17, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
        %15 = stablehlo.transpose %14, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
        %16 = stablehlo.dot_general %iterArg_15, %iterArg_17, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
        %17 = stablehlo.transpose %16, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
        %18 = stablehlo.broadcast_in_dim %iterArg_18, dims = [1] : (tensor<5xf32>) -> tensor<3x5xf32>
        %19 = stablehlo.transpose %18, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
        %20 = stablehlo.add %17, %19 : tensor<5x3xf32>
        %21 = stablehlo.multiply %15, %3 : tensor<5x3xf32>
        %22 = stablehlo.multiply %20, %3 : tensor<5x3xf32>
        %23 = stablehlo.multiply %15, %20 : tensor<5x3xf32>
        %24 = stablehlo.add %23, %23 : tensor<5x3xf32>
        %25 = stablehlo.multiply %20, %20 : tensor<5x3xf32>
        %26 = stablehlo.multiply %15, %25 : tensor<5x3xf32>
        %27 = stablehlo.multiply %24, %20 : tensor<5x3xf32>
        %28 = stablehlo.add %26, %27 : tensor<5x3xf32>
        %29 = stablehlo.multiply %20, %25 : tensor<5x3xf32>
        %30 = stablehlo.multiply %28, %1 : tensor<5x3xf32>
        %31 = stablehlo.multiply %1, %29 : tensor<5x3xf32>
        %32 = stablehlo.add %15, %30 : tensor<5x3xf32>
        %33 = stablehlo.add %20, %31 : tensor<5x3xf32>
        %34 = stablehlo.multiply %32, %0 : tensor<5x3xf32>
        %35 = stablehlo.multiply %0, %33 : tensor<5x3xf32>
        %36 = stablehlo.tanh %35 : tensor<5x3xf32>
        %37 = stablehlo.multiply %36, %36 : tensor<5x3xf32>
        %38 = stablehlo.subtract %2, %37 : tensor<5x3xf32>
        %39 = stablehlo.multiply %34, %38 : tensor<5x3xf32>
        %40 = stablehlo.add %2, %36 : tensor<5x3xf32>
        %41 = stablehlo.multiply %21, %40 : tensor<5x3xf32>
        %42 = stablehlo.multiply %39, %22 : tensor<5x3xf32>
        %43 = stablehlo.add %41, %42 : tensor<5x3xf32>
        %44 = stablehlo.dynamic_slice %43, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %45 = stablehlo.dynamic_update_slice %iterArg_16, %44, %7, %11 : (tensor<3x5xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<3x5xf32>
        stablehlo.return %9, %iterArg_15, %45, %iterArg_17, %iterArg_18 : tensor<i64>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<5x5xf32>, tensor<5xf32>
      }
      stablehlo.return %5, %8#2, %8#3, %8#4, %8#1 : tensor<i64>, tensor<3x5xf32>, tensor<5x5xf32>, tensor<5xf32>, tensor<3x5xf32>
    }
    return %4#1, %4#2, %4#3, %4#4 : tensor<3x5xf32>, tensor<5x5xf32>, tensor<5xf32>, tensor<3x5xf32>
  }
}

// CHECK:  %12 = stablehlo.dynamic_slice %cst_3, %7, %11, %c_7, %c_7, sizes = [1, 1, 3, 5] : (tensor<3x5x3x5xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x3x5xf32>
// CHECK-NEXT:  %13 = stablehlo.reshape %12 : (tensor<1x1x3x5xf32>) -> tensor<3x5xf32>
// CHECK-NEXT:  %14 = stablehlo.dot_general %13, %iterArg_17, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
// CHECK-NEXT:  %15 = stablehlo.transpose %14, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
// CHECK-NEXT:  %16 = stablehlo.dot_general %iterArg_15, %iterArg_17, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
// CHECK-NEXT:  %17 = stablehlo.transpose %16, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
// CHECK-NEXT:  %18 = stablehlo.broadcast_in_dim %iterArg_18, dims = [1] : (tensor<5xf32>) -> tensor<3x5xf32>
// CHECK-NEXT:  %19 = stablehlo.transpose %18, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
// CHECK-NEXT:  %20 = stablehlo.dynamic_slice %15, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %21 = stablehlo.dynamic_slice %3, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %22 = stablehlo.multiply %20, %21 : tensor<1x1xf32>
// CHECK-NEXT:  %23 = stablehlo.dynamic_slice %2, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %24 = stablehlo.dynamic_slice %0, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %25 = stablehlo.dynamic_slice %17, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %26 = stablehlo.dynamic_slice %19, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %27 = stablehlo.add %25, %26 : tensor<1x1xf32>
// CHECK-NEXT:  %28 = stablehlo.dynamic_slice %1, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %29 = stablehlo.multiply %27, %27 : tensor<1x1xf32>
// CHECK-NEXT:  %30 = stablehlo.multiply %27, %29 : tensor<1x1xf32>
// CHECK-NEXT:  %31 = stablehlo.multiply %28, %30 : tensor<1x1xf32>
// CHECK-NEXT:  %32 = stablehlo.add %27, %31 : tensor<1x1xf32>
// CHECK-NEXT:  %33 = stablehlo.multiply %24, %32 : tensor<1x1xf32>
// CHECK-NEXT:  %34 = stablehlo.tanh %33 : tensor<1x1xf32>
// CHECK-NEXT:  %35 = stablehlo.add %23, %34 : tensor<1x1xf32>
// CHECK-NEXT:  %36 = stablehlo.multiply %22, %35 : tensor<1x1xf32>
// CHECK-NEXT:  %37 = stablehlo.dynamic_slice %15, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %38 = stablehlo.dynamic_slice %15, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %39 = stablehlo.multiply %38, %29 : tensor<1x1xf32>
// CHECK-NEXT:  %40 = stablehlo.dynamic_slice %15, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %41 = stablehlo.multiply %40, %27 : tensor<1x1xf32>
// CHECK-NEXT:  %42 = stablehlo.add %41, %41 : tensor<1x1xf32>
// CHECK-NEXT:  %43 = stablehlo.multiply %42, %27 : tensor<1x1xf32>
// CHECK-NEXT:  %44 = stablehlo.add %39, %43 : tensor<1x1xf32>
// CHECK-NEXT:  %45 = stablehlo.dynamic_slice %1, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %46 = stablehlo.multiply %44, %45 : tensor<1x1xf32>
// CHECK-NEXT:  %47 = stablehlo.add %37, %46 : tensor<1x1xf32>
// CHECK-NEXT:  %48 = stablehlo.dynamic_slice %0, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %49 = stablehlo.multiply %47, %48 : tensor<1x1xf32>
// CHECK-NEXT:  %50 = stablehlo.dynamic_slice %2, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %51 = stablehlo.multiply %34, %34 : tensor<1x1xf32>
// CHECK-NEXT:  %52 = stablehlo.subtract %50, %51 : tensor<1x1xf32>
// CHECK-NEXT:  %53 = stablehlo.multiply %49, %52 : tensor<1x1xf32>
// CHECK-NEXT:  %54 = stablehlo.dynamic_slice %3, %11, %7, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
// CHECK-NEXT:  %55 = stablehlo.multiply %27, %54 : tensor<1x1xf32>
// CHECK-NEXT:  %56 = stablehlo.multiply %53, %55 : tensor<1x1xf32>
// CHECK-NEXT:  %57 = stablehlo.add %36, %56 : tensor<1x1xf32>
// CHECK-NEXT:  %58 = stablehlo.dynamic_update_slice %iterArg_16, %57, %7, %11 : (tensor<3x5xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<3x5xf32>
// CHECK-NEXT:  stablehlo.return %9, %iterArg_15, %58, %iterArg_17, %iterArg_18 : tensor<i64>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<5x5xf32>, tensor<5xf32>
