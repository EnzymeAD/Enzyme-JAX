// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%arg0: tensor<5xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<42> : tensor<i32>
    %c_1 = stablehlo.constant dense<5> : tensor<i32>
    %c_2 = stablehlo.constant dense<-1> : tensor<i64>
    %outbuf, %outrequest = enzymexla.irecv(%0, %c_1, %c, %c_0, %c_2) {datatype = "MPI_DOUBLE"} : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i64>) -> (tensor<5xf64>, tensor<i64>)
    enzymexla.wait(%outrequest) : tensor<i64>
    %1 = stablehlo.transpose %outbuf, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    return %1 : tensor<5xf64>
  }
}

// CPU:  module {
// CPU-NEXT:    llvm.mlir.global external constant @MPI_DOUBLE() {addr_space = 0 : i32} : !llvm.ptr
// CPU-NEXT:    llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
// CPU-NEXT:    llvm.func @MPI_Irecv(!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:    llvm.func @enzymexla_wrapper_MPI_Irecv_MPI_DOUBLE(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg2: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg3: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg4: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
// CPU-NEXT:      %1 = llvm.mlir.addressof @MPI_DOUBLE : !llvm.ptr
// CPU-NEXT:      %2 = llvm.load %arg1 : !llvm.ptr -> i32
// CPU-NEXT:      %3 = llvm.load %arg2 : !llvm.ptr -> i32
// CPU-NEXT:      %4 = llvm.load %arg3 : !llvm.ptr -> i32
// CPU-NEXT:      %5 = llvm.call @MPI_Irecv(%arg0, %2, %1, %3, %4, %0, %arg4) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:      llvm.return
// CPU-NEXT:    }
// CPU-NEXT:    llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:    llvm.func @enzymexla_wrapper_MPI_Wait(%arg0: !llvm.ptr {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %c1_i32 = arith.constant 1 : i32
// CPU-NEXT:      %0 = llvm.alloca %c1_i32 x !llvm.array<6 x i32> : (i32) -> !llvm.ptr
// CPU-NEXT:      %1 = llvm.call @MPI_Wait(%arg0, %0) : (!llvm.ptr, !llvm.ptr) -> i32
// CPU-NEXT:      llvm.return
// CPU-NEXT:    }
// CPU-NEXT:    func.func @main(%arg0: tensor<5xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CPU-NEXT:      %c = stablehlo.constant dense<-1> : tensor<i64>
// CPU-NEXT:      %c_0 = stablehlo.constant dense<5> : tensor<i32>
// CPU-NEXT:      %c_1 = stablehlo.constant dense<42> : tensor<i32>
// CPU-NEXT:      %c_2 = stablehlo.constant dense<1> : tensor<i32>
// CPU-NEXT:      %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
// CPU-NEXT:      %1:2 = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_DOUBLE (%0, %c_0, %c_2, %c_1, %c) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>, #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 4, operand_tuple_indices = []>]} : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i64>) -> (tensor<5xf64>, tensor<i64>)
// CPU-NEXT:      enzymexla.jit_call @enzymexla_wrapper_MPI_Wait (%1#1) : (tensor<i64>) -> ()
// CPU-NEXT:      %2 = stablehlo.transpose %1#0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
// CPU-NEXT:      return %2 : tensor<5xf64>
// CPU-NEXT:    }
// CPU-NEXT:  }
