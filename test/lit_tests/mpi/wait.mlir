// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s --check-prefix=CPU

module {
  func.func @main(%arg0: tensor<5xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<42> : tensor<i32>
    %c_1 = stablehlo.constant dense<5> : tensor<i32>
    %c_2 = stablehlo.constant dense<-1> : tensor<i64>
    %outbuf, %outrequest = enzymexla.irecv(%0, %c_1, %c, %c_0, %c_2) : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i64>) -> (tensor<5xf64>, tensor<i64>)
    enzymexla.wait(%outrequest) : tensor<i64>
    %1 = stablehlo.transpose %outbuf, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    return %1 : tensor<5xf64>
  }
}

// CPU:  module {
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
// CPU-NEXT:      %outbuf, %outrequest = enzymexla.irecv(%0, %c_0, %c_2, %c_1, %c) : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i64>) -> (tensor<5xf64>, tensor<i64>)
// CPU-NEXT:      enzymexla.jit_call @enzymexla_wrapper_MPI_Wait (%outrequest) : (tensor<i64>) -> ()
// CPU-NEXT:      %1 = stablehlo.transpose %outbuf, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
// CPU-NEXT:      return %1 : tensor<5xf64>
// CPU-NEXT:    }
// CPU-NEXT:  }
