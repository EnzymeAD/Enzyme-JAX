// RUN: enzymexlamlir-opt --transform-interpreter %s | FileCheck %s
// RUN: enzymexlamlir-opt --transform-interpreter --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s --check-prefix=FULLRAISE

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.greedy_while_loop_batch_fission
      transform.apply_patterns.enzyme_hlo.dynamic_slice_reshape_dynamic_slice
    } : !transform.any_op
    transform.yield 
  }
  func.func @"\E2\88\87potential"(%arg0: tensor<5x5xf32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: tensor<5xf32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg2: tensor<3x5xf32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> (tensor<3x5xf32>, tensor<5x5xf32>, tensor<5xf32>, tensor<3x5xf32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = stablehlo.constant dense<0.797884583> : tensor<3x5x5x3xf32>
    %cst_0 = stablehlo.constant dense<4.471500e-02> : tensor<3x5x5x3xf32>
    %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<3x5x5x3xf32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<5x3xf32>
    %cst_3 = stablehlo.constant dense<4.471500e-02> : tensor<5x3xf32>
    %cst_4 = stablehlo.constant dense<0.797884583> : tensor<5x3xf32>
    %cst_5 = stablehlo.constant dense<5.000000e-01> : tensor<5x3xf32>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %c_7 = stablehlo.constant dense<1> : tensor<i64>
    %cst_8 = stablehlo.constant dense<"0x0000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F"> : tensor<3x5x3x5xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<3x5xf32>
    %c_10 = stablehlo.constant dense<5> : tensor<i64>
    %c_11 = stablehlo.constant dense<3> : tensor<i64>
    %0 = stablehlo.dot_general %cst_8, %arg0, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5x3x5xf32>, tensor<5x5xf32>) -> tensor<3x5x3x5xf32>
    %1 = stablehlo.transpose %0, dims = [0, 1, 3, 2] : (tensor<3x5x3x5xf32>) -> tensor<3x5x5x3xf32>
    %2 = stablehlo.add %1, %1 : tensor<3x5x5x3xf32>
    %3 = stablehlo.dot_general %arg2, %arg0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
    %4 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<5xf32>) -> tensor<3x5xf32>
    %5 = stablehlo.add %3, %4 : tensor<3x5xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<3x5xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [3, 2] : (tensor<3x5xf32>) -> tensor<3x5x5x3xf32>
    %8 = stablehlo.multiply %1, %7 : tensor<3x5x5x3xf32>
    %9 = stablehlo.multiply %1, %cst_1 : tensor<3x5x5x3xf32>
    %10 = stablehlo.transpose %5, dims = [1, 0] : (tensor<3x5xf32>) -> tensor<5x3xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<5x3xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<5x3xf32>
    %13 = stablehlo.multiply %cst_3, %12 : tensor<5x3xf32>
    %14 = stablehlo.add %10, %13 : tensor<5x3xf32>
    %15 = stablehlo.multiply %cst_4, %14 : tensor<5x3xf32>
    %16 = stablehlo.tanh %15 : tensor<5x3xf32>
    %17 = stablehlo.add %cst_2, %16 : tensor<5x3xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [2, 3] : (tensor<5x3xf32>) -> tensor<3x5x5x3xf32>
    %19 = stablehlo.multiply %9, %18 : tensor<3x5x5x3xf32>
    %20 = stablehlo.broadcast_in_dim %5, dims = [3, 2] : (tensor<3x5xf32>) -> tensor<3x5x5x3xf32>
    %21 = stablehlo.multiply %20, %2 : tensor<3x5x5x3xf32>
    %22 = stablehlo.multiply %21, %20 : tensor<3x5x5x3xf32>
    %23 = stablehlo.add %8, %22 : tensor<3x5x5x3xf32>
    %24 = stablehlo.multiply %23, %cst_0 : tensor<3x5x5x3xf32>
    %25 = stablehlo.add %1, %24 : tensor<3x5x5x3xf32>
    %26 = stablehlo.multiply %25, %cst : tensor<3x5x5x3xf32>
    %27 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [0] x [1], precision = [DEFAULT, DEFAULT] : (tensor<5x5xf32>, tensor<3x5xf32>) -> tensor<5x3xf32>
    %28 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<5xf32>) -> tensor<5x3xf32>
    %29 = stablehlo.add %27, %28 : tensor<5x3xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<5x3xf32>
    %31 = stablehlo.multiply %29, %30 : tensor<5x3xf32>
    %32 = stablehlo.multiply %cst_3, %31 : tensor<5x3xf32>
    %33 = stablehlo.add %29, %32 : tensor<5x3xf32>
    %34 = stablehlo.multiply %cst_4, %33 : tensor<5x3xf32>
    %35 = stablehlo.tanh %34 : tensor<5x3xf32>
    %36 = stablehlo.multiply %35, %35 : tensor<5x3xf32>
    %37 = stablehlo.subtract %cst_2, %36 : tensor<5x3xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [2, 3] : (tensor<5x3xf32>) -> tensor<3x5x5x3xf32>
    %39 = stablehlo.multiply %26, %38 : tensor<3x5x5x3xf32>
    %40 = stablehlo.multiply %29, %cst_5 : tensor<5x3xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [2, 3] : (tensor<5x3xf32>) -> tensor<3x5x5x3xf32>
    %42 = stablehlo.multiply %39, %41 : tensor<3x5x5x3xf32>
    %43:2 = stablehlo.while(%iterArg = %c_6, %iterArg_12 = %cst_9) : tensor<i64>, tensor<3x5xf32>
    cond {
      %44 = stablehlo.compare  LT, %iterArg, %c_11 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %44 : tensor<i1>
    } do {
      %44 = stablehlo.add %c_7, %iterArg {enzymexla.bounds = [[1, 3]]} : tensor<i64>
      %45 = stablehlo.convert %44 {enzymexla.bounds = [[1, 3]]} : (tensor<i64>) -> tensor<i32>
      %46 = stablehlo.subtract %45, %c {enzymexla.bounds = [[0, 2]]} : tensor<i32>
      %47 = stablehlo.dynamic_slice %19, %iterArg, %c_6, %c_6, %c_6, sizes = [1, 5, 5, 3] : (tensor<3x5x5x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x5x5x3xf32>
      %48 = stablehlo.reshape %47 : (tensor<1x5x5x3xf32>) -> tensor<5x5x3xf32>
      %49 = stablehlo.dynamic_slice %42, %iterArg, %c_6, %c_6, %c_6, sizes = [1, 5, 5, 3] : (tensor<3x5x5x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x5x5x3xf32>
      %50 = stablehlo.reshape %49 : (tensor<1x5x5x3xf32>) -> tensor<5x5x3xf32>
      %51:2 = stablehlo.while(%iterArg_13 = %c_6, %iterArg_14 = %iterArg_12) : tensor<i64>, tensor<3x5xf32> attributes {enzyme.disable_mincut}
      cond {
        %52 = stablehlo.compare  LT, %iterArg_13, %c_10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %52 : tensor<i1>
      } do {
        %52 = stablehlo.add %c_7, %iterArg_13 {enzymexla.bounds = [[1, 5]]} : tensor<i64>
        %53 = stablehlo.convert %52 {enzymexla.bounds = [[1, 5]]} : (tensor<i64>) -> tensor<i32>
        %54 = stablehlo.subtract %53, %c {enzymexla.bounds = [[0, 4]]} : tensor<i32>
        %55 = stablehlo.dynamic_slice %48, %iterArg_13, %c_6, %c_6, sizes = [1, 5, 3] : (tensor<5x5x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x5x3xf32>
        %56 = stablehlo.reshape %55 : (tensor<1x5x3xf32>) -> tensor<5x3xf32>
        %57 = stablehlo.dynamic_slice %50, %iterArg_13, %c_6, %c_6, sizes = [1, 5, 3] : (tensor<5x5x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x5x3xf32>
        %58 = stablehlo.reshape %57 : (tensor<1x5x3xf32>) -> tensor<5x3xf32>
        %59 = stablehlo.dynamic_slice %56, %54, %46, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %60 = stablehlo.dynamic_slice %58, %54, %46, sizes = [1, 1] : (tensor<5x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x1xf32>
        %61 = stablehlo.add %59, %60 : tensor<1x1xf32>
        %62 = stablehlo.dynamic_update_slice %iterArg_14, %61, %46, %54 : (tensor<3x5xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<3x5xf32>
        stablehlo.return %52, %62 : tensor<i64>, tensor<3x5xf32>
      }
      stablehlo.return %44, %51#1 : tensor<i64>, tensor<3x5xf32>
    }
    return %43#1, %arg0, %arg1, %arg2 : tensor<3x5xf32>, tensor<5x5xf32>, tensor<5xf32>, tensor<3x5xf32>
  }
}

// CHECK: %79 = "stablehlo.gather"(%19, %78) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [1, 2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2, 3], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<3x5x5x3xf32>, tensor<3x5x3xi64>) -> tensor<3x1x5xf32>
// CHECK-NEXT: %80 = "stablehlo.gather"(%42, %66) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [1, 2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2, 3], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 1>}> : (tensor<3x5x5x3xf32>, tensor<3x5x3xi64>) -> tensor<3x1x5xf32>

// FULLRAISE-NOT: stablehlo.while
