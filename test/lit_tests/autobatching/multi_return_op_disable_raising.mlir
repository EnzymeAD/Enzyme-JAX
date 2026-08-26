// RUN: enzymexlamlir-opt --auto-batching %s | FileCheck %s

module {
  llvm.func @enzymexla_lapack_sgetrf_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func @main(%arg0: tensor<32x10x10xf32> {enzymexla.memory_effects = []}, %arg1: tensor<32x10xf32> {enzymexla.memory_effects = []}) -> tensor<32x10xf32> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c = stablehlo.constant dense<0> : tensor<32x10xi64>
    %c_0 = stablehlo.constant dense<-1> : tensor<i64>
    %c_1 = stablehlo.constant dense<-1> : tensor<10xi64>
    %c_2 = stablehlo.constant dense<10> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x10x10xf32>
    %cst_4 = arith.constant dense<0> : tensor<32x10xi64>
    %c_5 = stablehlo.constant dense<32> : tensor<i64>
    %c_6 = stablehlo.constant dense<1> : tensor<i64>
    %c_7 = stablehlo.constant dense<1> : tensor<i32>
    %c_8 = stablehlo.constant dense<10> : tensor<i32>
    %c_9 = stablehlo.constant dense<1> : tensor<32x10xi64>
    %c_10 = stablehlo.constant dense<0> : tensor<i32>
    %c_11 = stablehlo.constant dense<1> : tensor<32x10x1xi64>
    %0 = stablehlo.transpose %arg0, dims = [0, 2, 1] : (tensor<32x10x10xf32>) -> tensor<32x10x10xf32>
    %1:3 = stablehlo.while(%iterArg = %c_3, %iterArg_12 = %cst, %iterArg_13 = %cst_4) : tensor<i64>, tensor<32x10x10xf32>, tensor<32x10xi64>
    cond {
      %14 = stablehlo.compare  LT, %iterArg, %c_5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %14 : tensor<i1>
    } do {
      %14 = stablehlo.add %iterArg, %c_6 : tensor<i64>
      %15 = stablehlo.dynamic_slice %0, %iterArg, %c_3, %c_3, sizes = [1, 10, 10] : (tensor<32x10x10xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x10x10xf32>
      %16 = stablehlo.reshape %15 : (tensor<1x10x10xf32>) -> tensor<10x10xf32>
      // CHECK: stablehlo.custom_call
      %17:3 = stablehlo.custom_call @enzymexla_compile_cpu(%c_2, %c_2, %16, %c_2, %c_1, %c_0) {api_version = 3 : i32, backend_config = "\00\90CB\CFq\00\00\00\00\00\00\00\00\00\00", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 5, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<i64>, tensor<i64>, tensor<10x10xf32>, tensor<i64>, tensor<10xi64>, tensor<i64>) -> (tensor<10x10xf32>, tensor<10xi64>, tensor<i64>)
      %18 = stablehlo.reshape %17#0 : (tensor<10x10xf32>) -> tensor<1x10x10xf32>
      %19 = stablehlo.dynamic_update_slice %iterArg_12, %18, %iterArg, %c_3, %c_3 : (tensor<32x10x10xf32>, tensor<1x10x10xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<32x10x10xf32>
      %20 = stablehlo.reshape %17#1 : (tensor<10xi64>) -> tensor<1x10xi64>
      %21 = stablehlo.dynamic_update_slice %iterArg_13, %20, %iterArg, %c_3 : (tensor<32x10xi64>, tensor<1x10xi64>, tensor<i64>, tensor<i64>) -> tensor<32x10xi64>
      stablehlo.return %14, %19, %21 : tensor<i64>, tensor<32x10x10xf32>, tensor<32x10xi64>
    }
    %2 = stablehlo.subtract %1#2, %c_9 : tensor<32x10xi64>
    %3:2 = stablehlo.while(%iterArg = %c_10, %iterArg_12 = %c) : tensor<i32>, tensor<32x10xi64>
    cond {
      %14 = stablehlo.compare  LT, %iterArg, %c_8 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %14 : tensor<i1>
    } do {
      %14 = stablehlo.add %iterArg, %c_7 : tensor<i32>
      %15 = stablehlo.dynamic_slice %2, %c_10, %iterArg, sizes = [32, 1] : (tensor<32x10xi64>, tensor<i32>, tensor<i32>) -> tensor<32x1xi64>
      %16 = stablehlo.dynamic_slice %iterArg_12, %c_10, %iterArg, sizes = [32, 1] : (tensor<32x10xi64>, tensor<i32>, tensor<i32>) -> tensor<32x1xi64>
      %17 = "stablehlo.gather"(%iterArg_12, %15) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<32x10xi64>, tensor<32x1xi64>) -> tensor<32x1xi64>
      %18 = stablehlo.dynamic_update_slice %iterArg_12, %17, %c_10, %iterArg : (tensor<32x10xi64>, tensor<32x1xi64>, tensor<i32>, tensor<i32>) -> tensor<32x10xi64>
      %19 = stablehlo.reshape %16 : (tensor<32x1xi64>) -> tensor<32xi64>
      %20 = "stablehlo.scatter"(%18, %15, %19) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false}> ({
      ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
        stablehlo.return %arg3 : tensor<i64>
      }) : (tensor<32x10xi64>, tensor<32x1xi64>, tensor<32xi64>) -> tensor<32x10xi64>
      stablehlo.return %14, %20 : tensor<i32>, tensor<32x10xi64>
    }
    %4 = stablehlo.add %3#1, %c_9 : tensor<32x10xi64>
    %5 = stablehlo.convert %4 : (tensor<32x10xi64>) -> tensor<32x10xi32>
    %6 = stablehlo.reshape %arg1 : (tensor<32x10xf32>) -> tensor<32x10x1xf32>
    %7 = stablehlo.convert %5 : (tensor<32x10xi32>) -> tensor<32x10xi64>
    %8 = stablehlo.reshape %7 : (tensor<32x10xi64>) -> tensor<32x10x1xi64>
    %9 = stablehlo.subtract %8, %c_11 : tensor<32x10x1xi64>
    %10 = "stablehlo.gather"(%6, %9) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1>}> : (tensor<32x10x1xf32>, tensor<32x10x1xi64>) -> tensor<32x10x1xf32>
    %11 = "stablehlo.triangular_solve"(%1#1, %10) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<32x10x10xf32>, tensor<32x10x1xf32>) -> tensor<32x10x1xf32>
    %12 = "stablehlo.triangular_solve"(%1#1, %11) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<32x10x10xf32>, tensor<32x10x1xf32>) -> tensor<32x10x1xf32>
    %13 = stablehlo.reshape %12 : (tensor<32x10x1xf32>) -> tensor<32x10xf32>
    return %13 : tensor<32x10xf32>
  }
}

