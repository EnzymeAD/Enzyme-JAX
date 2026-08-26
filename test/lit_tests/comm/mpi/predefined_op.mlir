// RUN: enzymexlamlir-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main() {
    // CHECK-NEXT: [[v0:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_OP_NULL>} : !comm.mpi.op
    %0 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_OP_NULL>} : !comm.mpi.op

    // CHECK-NEXT: [[v1:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_SUM>} : !comm.mpi.op
    %1 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_SUM>} : !comm.mpi.op

    // CHECK-NEXT: [[v2:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_MIN>} : !comm.mpi.op
    %2 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_MIN>} : !comm.mpi.op

    // CHECK-NEXT: [[v3:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_MAX>} : !comm.mpi.op
    %3 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_MAX>} : !comm.mpi.op

    // CHECK-NEXT: [[v4:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_PROD>} : !comm.mpi.op
    %4 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_PROD>} : !comm.mpi.op

    // CHECK-NEXT: [[v5:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_BAND>} : !comm.mpi.op
    %5 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_BAND>} : !comm.mpi.op

    // CHECK-NEXT: [[v6:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_BOR>} : !comm.mpi.op
    %6 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_BOR>} : !comm.mpi.op

    // CHECK-NEXT: [[v7:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_BXOR>} : !comm.mpi.op
    %7 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_BXOR>} : !comm.mpi.op

    // CHECK-NEXT: [[v8:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_LAND>} : !comm.mpi.op
    %8 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_LAND>} : !comm.mpi.op

    // CHECK-NEXT: [[v9:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_LOR>} : !comm.mpi.op
    %9 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_LOR>} : !comm.mpi.op

    // CHECK-NEXT: [[v10:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_LXOR>} : !comm.mpi.op
    %10 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_LXOR>} : !comm.mpi.op

    // CHECK-NEXT: [[v11:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_MINLOC>} : !comm.mpi.op
    %11 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_MINLOC>} : !comm.mpi.op

    // CHECK-NEXT: [[v12:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_MAXLOC>} : !comm.mpi.op
    %12 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_MAXLOC>} : !comm.mpi.op

    // CHECK-NEXT: [[v13:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_REPLACE>} : !comm.mpi.op
    %13 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_REPLACE>} : !comm.mpi.op

    // CHECK-NEXT: [[v14:%.*]] = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_NO_OP>} : !comm.mpi.op
    %14 = comm.mpi.predefined_op {value = #comm.mpi.op<MPI_NO_OP>} : !comm.mpi.op


    return
}
