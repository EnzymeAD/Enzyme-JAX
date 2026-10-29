// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl %s | FileCheck %s

func.func @main(%comm: !comm.mpi.comm, %color: tensor<i32>,
                %key: tensor<i32>) -> !comm.mpi.comm {
  %newcomm = comm.mpi.comm_split %comm, %color, %key :
      (!comm.mpi.comm, tensor<i32>, tensor<i32>) -> !comm.mpi.comm
  return %newcomm : !comm.mpi.comm
}

// CHECK-LABEL: func.func @main
// CHECK-SAME: (%[[COMM:.*]]: !comm.nccl.comm, %[[COLOR:.*]]: tensor<i32>, %[[KEY:.*]]: tensor<i32>) -> !comm.nccl.comm {
// CHECK-NEXT: %[[NEW_COMM:.*]] = comm.nccl.comm_split %[[COMM]], %[[COLOR]], %[[KEY]] : (!comm.nccl.comm, tensor<i32>, tensor<i32>) -> !comm.nccl.comm
// CHECK-NEXT: return %[[NEW_COMM]] : !comm.nccl.comm
// CHECK-NEXT: }
