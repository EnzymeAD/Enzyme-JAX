// RUN: enzymexlamlir-opt --fuse-jit %s | FileCheck %s

module {
  llvm.mlir.global external constant @MPI_STATUSES_IGNORE() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @MPI_Waitall(i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @enzymexla_jitwrap_MPI_Waitall_2(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %0 = llvm.mlir.constant(2 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.ptr : (i32) -> !llvm.ptr
    %2 = llvm.load %arg0 : !llvm.ptr -> !llvm.ptr
    %3 = llvm.getelementptr %1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %2, %3 : !llvm.ptr, !llvm.ptr
    %4 = llvm.load %arg1 : !llvm.ptr -> !llvm.ptr
    %5 = llvm.getelementptr %1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %4, %5 : !llvm.ptr, !llvm.ptr
    %7 = llvm.mlir.addressof @MPI_STATUSES_IGNORE : !llvm.ptr
    %8 = llvm.call @MPI_Waitall(%0, %1, %7) : (i32, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
  llvm.func @MPI_Irecv(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @enzymexla_jitwrap_MPI_Irecv(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> i32
    %1 = llvm.load %arg2 : !llvm.ptr -> !llvm.ptr
    %2 = llvm.load %arg3 : !llvm.ptr -> i32
    %3 = llvm.load %arg4 : !llvm.ptr -> i32
    %4 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
    %5 = llvm.call @MPI_Irecv(%arg0, %0, %1, %2, %3, %4, %arg6) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
  llvm.func @MPI_Isend(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @enzymexla_jitwrap_MPI_Isend(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr) {
    %0 = llvm.load %arg1 : !llvm.ptr -> i32
    %1 = llvm.load %arg2 : !llvm.ptr -> !llvm.ptr
    %2 = llvm.load %arg3 : !llvm.ptr -> i32
    %3 = llvm.load %arg4 : !llvm.ptr -> i32
    %4 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
    %5 = llvm.call @MPI_Isend(%arg0, %0, %1, %2, %3, %4, %arg6) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }
  func.func @main(%arg0: tensor<4xf64>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i64>) -> tensor<4xf64> {
    %c = stablehlo.constant dense<4> : tensor<i32>
    %c_0 = stablehlo.constant dense<-1> : tensor<i64>
    %c_1 = stablehlo.constant dense<3> : tensor<i64>
    %0 = enzymexla.jit_call @enzymexla_jitwrap_MPI_Isend (%arg0, %c, %c_1, %arg3, %arg4, %arg6, %c_0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 6, operand_tuple_indices = []>]> : (tensor<4xf64>, tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>)-> tensor<i64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
    %1:2 = enzymexla.jit_call @enzymexla_jitwrap_MPI_Irecv (%cst, %c, %c_1, %arg1, %arg5, %arg6, %c_0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 6, operand_tuple_indices = []>]> : (tensor<4xf64>, tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>) -> (tensor<4xf64>, tensor<i64>)
    enzymexla.jit_call @enzymexla_jitwrap_MPI_Waitall_2 (%1#1, %0) : (tensor<i64>, tensor<i64>) -> ()
    return %1#0 : tensor<4xf64>
  }
}

// CHECK-LABEL: llvm.func @fused__enzymexla_jitwrap_MPI_Isend_enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Waitall_2
// CHECK-SAME:                                                                                                           (%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: !llvm.ptr, %arg10: !llvm.ptr) {
// CHECK-NEXT:   %0 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:   %1 = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:   %2 = llvm.mlir.addressof @MPI_STATUSES_IGNORE : !llvm.ptr
// CHECK-NEXT:   %3 = llvm.load %arg1 : !llvm.ptr -> i32
// CHECK-NEXT:   %4 = llvm.load %arg2 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:   %5 = llvm.load %arg3 : !llvm.ptr -> i32
// CHECK-NEXT:   %6 = llvm.load %arg4 : !llvm.ptr -> i32
// CHECK-NEXT:   %7 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:   %8 = llvm.call @MPI_Isend(%arg0, %3, %4, %5, %6, %7, %arg6) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:   %9 = llvm.load %arg1 : !llvm.ptr -> i32
// CHECK-NEXT:   %10 = llvm.load %arg2 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:   %11 = llvm.load %arg8 : !llvm.ptr -> i32
// CHECK-NEXT:   %12 = llvm.load %arg9 : !llvm.ptr -> i32
// CHECK-NEXT:   %13 = llvm.load %arg5 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:   %14 = llvm.call @MPI_Irecv(%arg7, %9, %10, %11, %12, %13, %arg10) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:   %15 = llvm.alloca %0 x !llvm.array<2 x ptr> : (i32) -> !llvm.ptr
// CHECK-NEXT:   %16 = llvm.load %arg10 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:   llvm.store %16, %15 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:   %17 = llvm.load %arg6 : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:   %18 = llvm.getelementptr %15[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:   llvm.store %17, %18 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:   %19 = llvm.call @MPI_Waitall(%1, %15, %2) : (i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:   llvm.return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @main
// CHECK-SAME:                 (%arg0: tensor<4xf64>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i64>) -> tensor<4xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<0.000000e+00> : tensor<4xf64>
// CHECK-NEXT:   %c = stablehlo.constant dense<4> : tensor<i32>
// CHECK-NEXT:   %c_0 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:   %0:3 = enzymexla.jit_call @fused__enzymexla_jitwrap_MPI_Isend_enzymexla_jitwrap_MPI_Irecv_enzymexla_jitwrap_MPI_Waitall_2 (%arg0, %c, %c_1, %arg3, %arg4, %arg6, %c_0, %cst, %arg1, %arg5, %c_0) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 7, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 6, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [2], operand_index = 10, operand_tuple_indices = []>]> : (tensor<4xf64>, tensor<i32>, tensor<i64>, tensor<i32>, tensor<i32>, tensor<i64>, tensor<i64>, tensor<4xf64>, tensor<i32>, tensor<i32>, tensor<i64>) -> (tensor<4xf64>, tensor<i64>, tensor<i64>)
// CHECK-NEXT:   return %0#0 : tensor<4xf64>
// CHECK-NEXT: }
