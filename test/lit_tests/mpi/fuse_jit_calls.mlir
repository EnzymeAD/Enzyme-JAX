
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit-calls)" %s | FileCheck %s --check-prefix=CPU

// It's the same code of /irecv-wait.mlir,
// just changed the CPU-LABEL to match the fuse pass

module {
  func.func @main(%arg0: tensor<5xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<42> : tensor<i32>
    %c_1 = stablehlo.constant dense<5> : tensor<i32>
    %c_2 = stablehlo.constant dense<-1> : tensor<i32>
    %outbuf, %request = enzymexla.mpi.irecv(%0, %c_1, %c, %c_0) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<5xf64>, tensor<i32>)
    enzymexla.mpi.wait(%request) : tensor<i32>
    %1 = stablehlo.transpose %outbuf, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    return %1 : tensor<5xf64>
  }
}

// CPU-LABEL: llvm.func @__enzyme_fused_enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait
// CPU-SAME: %[[REQ:[^ ,)]+]]: !llvm.ptr)
// CPU: llvm.call @MPI_Irecv({{.*}}, %[[REQ]])
// CPU: %[[STATUS:.*]] = llvm.alloca
// CPU: llvm.call @MPI_Wait(%[[REQ]], %[[STATUS]])
// CPU: llvm.return

// CPU-LABEL: func.func @main
// CPU: %[[FUSED:.*]] = enzymexla.jit_call @__enzyme_fused_enzymexla_wrapper_MPI_Irecv_MPI_INT_enzymexla_wrapper_MPI_Wait
// CPU-NOT: enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT
// CPU-NOT: enzymexla.jit_call @enzymexla_wrapper_MPI_Wait
// CPU: %[[OUT:.*]] = stablehlo.transpose %[[FUSED]]
// CPU: return %[[OUT]]
