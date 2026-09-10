// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu})" %s | FileCheck %s

// Ensure that requests received by each isend/irecv are backed by unique storage.
module {
  func.func @main(%arg0: tensor<5xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 0 : i32}) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<5xf64>) -> tensor<5xf64>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<42> : tensor<i32>
    %c_1 = stablehlo.constant dense<5> : tensor<i32>
    %outbuf, %request = enzymexla.mpi.irecv(%0, %c_1, %c, %c_0) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<5xf64>, tensor<i32>)
    %request_1 = enzymexla.mpi.isend(%outbuf, %c_1, %c, %c_0) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    enzymexla.mpi.waitall(%request, %request_1) : tensor<i32>, tensor<i32>
    return %outbuf : tensor<5xf64>
  }
}

// CHECK-LABEL: func.func @main
// CHECK:       stablehlo.constant dense<-1> : tensor<i32>
// CHECK:       stablehlo.constant dense<-2> : tensor<i32>
// CHECK:       %{{.*}}:2 = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT ({{.*}}, %[[IRECV_STORAGE:[^ ,)]+]])
// CHECK-NOT:   enzymexla.jit_call @enzymexla_wrapper_MPI_Isend_MPI_INT ({{.*}}, %[[IRECV_STORAGE]])
// CHECK:       %{{.*}} = enzymexla.jit_call @enzymexla_wrapper_MPI_Isend_MPI_INT ({{.*}}, %[[ISEND_STORAGE:[^ ,)]+]])
