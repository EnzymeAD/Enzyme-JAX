// RUN: enzymexlamlir-opt --legalize-mpi-to-nccl %s | FileCheck %s

func.func @main() -> !comm.mpi.comm {
  %comm = comm.mpi.constant #comm.mpi.comm<MPI_COMM_WORLD> : !comm.mpi.comm
  return %comm : !comm.mpi.comm
}

// CHECK-LABEL: func.func @main() -> !comm.nccl.comm {
// CHECK-NEXT: %[[COMM:.*]] = comm.nccl.constant : !comm.nccl.comm
// CHECK-NEXT: return %[[COMM]] : !comm.nccl.comm
// CHECK-NEXT: }
