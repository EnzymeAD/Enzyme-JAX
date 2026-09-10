// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(lower-enzymexla-mpi{backend=cpu},fuse-jit)" %s | FileCheck %s

// The control send must remain between the nonblocking receive and its wait.
// Moving the wait before the send can deadlock when the peer waits for the
// control message before sending the receive payload.

module {
  func.func @rank0(%sendbuf: tensor<5xf64>, %recvbuf: tensor<5xf64>, %control: tensor<1xi32>) -> tensor<5xf64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %peer = stablehlo.constant dense<1> : tensor<i32>
    %count = stablehlo.constant dense<5> : tensor<i32>
    %control_count = stablehlo.constant dense<1> : tensor<i32>
    %tag_a = stablehlo.constant dense<10> : tensor<i32>
    %tag_b = stablehlo.constant dense<20> : tensor<i32>
    %tag_control = stablehlo.constant dense<30> : tensor<i32>

    %send_request = enzymexla.mpi.isend(%sendbuf, %count, %peer, %tag_a) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %outbuf, %recv_request = enzymexla.mpi.irecv(%recvbuf, %count, %peer, %tag_b) {datatype = #enzymexla.datatype<MPI_INT>} : (tensor<5xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<5xf64>, tensor<i32>)
    enzymexla.mpi.wait(%send_request) : tensor<i32>
    enzymexla.mpi.send(%control, %control_count, %peer, %tag_control) {datatype = #enzymexla.datatype<MPI_INT>} : tensor<1xi32>, tensor<i32>, tensor<i32>, tensor<i32>
    enzymexla.mpi.wait(%recv_request) : tensor<i32>
    return %outbuf : tensor<5xf64>
  }
}

// CHECK-NOT: llvm.func @fused__
// CHECK-LABEL: func.func @rank0
// CHECK: %[[SEND_REQUEST:.*]] = enzymexla.jit_call @enzymexla_wrapper_MPI_Isend_MPI_INT
// CHECK-NEXT: %[[RECV:.*]]:2 = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_MPI_INT
// CHECK-NEXT: enzymexla.jit_call @enzymexla_wrapper_MPI_Wait (%[[SEND_REQUEST]])
// CHECK-NEXT: enzymexla.jit_call @enzymexla_wrapper_MPI_Send_MPI_INT
// CHECK-NEXT: enzymexla.jit_call @enzymexla_wrapper_MPI_Wait (%[[RECV]]#1)
// CHECK-NEXT: return %[[RECV]]#0
