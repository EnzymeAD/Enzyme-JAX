// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(inline{default-pipeline=canonicalize max-iterations=4},fuse-jit)" %s | FileCheck %s

// GETRF -> GETRF; inline generated helpers so fuse-jit can see the call chain.

module {
  func.func private @enzymexla_lapack_dgetrf_1(%arg0: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xi64>, tensor<i64>) {
    %c = stablehlo.constant dense<-1> : tensor<i64>
    %c_0 = stablehlo.constant dense<-1> : tensor<4xi64>
    %c_1 = stablehlo.constant dense<4> : tensor<i64>
    %0:3 = enzymexla.jit_call @enzymexla_lapack_dgetrf_wrapper (%c_1, %c_1, %arg0, %c_1, %c_0, %c) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 5, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<4x4xf64>, tensor<i64>, tensor<4xi64>, tensor<i64>) -> (tensor<4x4xf64>, tensor<4xi64>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<4x4xf64>, tensor<4xi64>, tensor<i64>
  }

  func.func private @enzymexla_lapack_dgetrf_0(%arg0: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xi64>, tensor<i64>) {
    %c = stablehlo.constant dense<-1> : tensor<i64>
    %c_0 = stablehlo.constant dense<-1> : tensor<4xi64>
    %c_1 = stablehlo.constant dense<4> : tensor<i64>
    %0:3 = enzymexla.jit_call @enzymexla_lapack_dgetrf_wrapper (%c_1, %c_1, %arg0, %c_1, %c_0, %c) <operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 5, operand_tuple_indices = []>], xla_side_effect_free> : (tensor<i64>, tensor<i64>, tensor<4x4xf64>, tensor<i64>, tensor<4xi64>, tensor<i64>) -> (tensor<4x4xf64>, tensor<4xi64>, tensor<i64>)
    return %0#0, %0#1, %0#2 : tensor<4x4xf64>, tensor<4xi64>, tensor<i64>
  }

  llvm.func private @enzymexla_lapack_dgetrf_wrapper(%arg0: !llvm.ptr {llvm.nofree, llvm.readonly}, %arg1: !llvm.ptr {llvm.nofree, llvm.readonly}, %arg2: !llvm.ptr {llvm.nofree}, %arg3: !llvm.ptr {llvm.nofree, llvm.readonly}, %arg4: !llvm.ptr {llvm.nofree, llvm.writeonly}, %arg5: !llvm.ptr {llvm.nofree, llvm.writeonly}) {
    llvm.call @enzymexla_lapack_dgetrf_(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @enzymexla_lapack_dgetrf_(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)

  func.func @lapack_getrf_chain(%arg0: tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xi64>, tensor<4xi64>, tensor<i64>) {
    %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi64>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<4> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<4xi64>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %0:3 = call @enzymexla_lapack_dgetrf_1(%arg0) : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xi64>, tensor<i64>)
    %1:3 = call @enzymexla_lapack_dgetrf_0(%0#0) : (tensor<4x4xf64>) -> (tensor<4x4xf64>, tensor<4xi64>, tensor<i64>)
    %2 = stablehlo.subtract %1#1, %c_2 : tensor<4xi64>
    %3:2 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %c) : tensor<i32>, tensor<4xi64>
    cond {
      %5 = stablehlo.compare LT, %iterArg, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    } do {
      %5 = stablehlo.add %iterArg, %c_0 {enzymexla.bounds = [[1 : i32, 4 : i32]]} : tensor<i32>
      %6 = stablehlo.dynamic_slice %2, %iterArg, sizes = [1] : (tensor<4xi64>, tensor<i32>) -> tensor<1xi64>
      %7 = stablehlo.dynamic_slice %iterArg_4, %iterArg, sizes = [1] : (tensor<4xi64>, tensor<i32>) -> tensor<1xi64>
      %8 = stablehlo.reshape %6 : (tensor<1xi64>) -> tensor<i64>
      %9 = stablehlo.dynamic_slice %iterArg_4, %8, sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
      %10 = stablehlo.dynamic_update_slice %iterArg_4, %9, %iterArg : (tensor<4xi64>, tensor<1xi64>, tensor<i32>) -> tensor<4xi64>
      %11 = stablehlo.dynamic_update_slice %10, %7, %8 : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
      stablehlo.return %5, %11 : tensor<i32>, tensor<4xi64>
    }
    %4 = stablehlo.add %3#1, %c_2 : tensor<4xi64>
    return %1#0, %1#1, %4, %1#2 : tensor<4x4xf64>, tensor<4xi64>, tensor<4xi64>, tensor<i64>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_lapack_dgetrf_wrapper_enzymexla_lapack_dgetrf_wrapper(
// CHECK-COUNT-2: llvm.call @enzymexla_lapack_dgetrf_(
// CHECK: llvm.return
// CHECK-LABEL: func.func @lapack_getrf_chain(
// CHECK-NOT: enzymexla.jit_call @enzymexla_
// CHECK: enzymexla.jit_call @fused__enzymexla_lapack_dgetrf_wrapper_enzymexla_lapack_dgetrf_wrapper
// CHECK-SAME: operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2
// CHECK-SAME: #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 9
// CHECK-NOT: enzymexla.jit_call @enzymexla_
