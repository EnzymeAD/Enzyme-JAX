// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main() {
    // CHECK-NEXT: [[v0:%.*]] = comm.mpi.predefined_comm {value = #comm.communicator<MPI_COMM_WORLD>} : !comm.mpi.comm
    %0 = comm.mpi.predefined_comm {value = #comm.communicator<MPI_COMM_WORLD>} : !comm.mpi.comm

    // CHECK-NEXT: [[v1:%.*]] = comm.mpi.predefined_comm {value = #comm.communicator<MPI_COMM_SELF>} : !comm.mpi.comm
    %1 = comm.mpi.predefined_comm {value = #comm.communicator<MPI_COMM_SELF>} : !comm.mpi.comm

    // CHECK-NEXT: [[v2:%.*]] = comm.mpi.predefined_comm {value = #comm.communicator<MPI_COMM_NULL>} : !comm.mpi.comm
    %2 = comm.mpi.predefined_comm {value = #comm.communicator<MPI_COMM_NULL>} : !comm.mpi.comm

    return
}
