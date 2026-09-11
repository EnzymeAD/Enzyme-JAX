// XFAIL: *
// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// NOTE disabled because we cannot register constants in the JIT symbol table with enzymexlamlir-opt
// TODO fix whenever the constant registration allows or we move to MPI 5 ABI

// CHECK: func.func @main() -> (tensor<i64>, tensor<i64>) {
func.func @main() -> (!comm.mpi.comm, !comm.mpi.op) {
    // CHECK-NEXT: %[[COMM:.*]] = stablehlo.constant dense<X> : tensor<i64>
    %comm = comm.mpi.constant #comm.mpi.comm<MPI_COMM_WORLD> : !comm.mpi.comm
    // CHECK-NEXT: %[[OP:.*]] = stablehlo.constant dense<X> : tensor<i64>
    %op = comm.mpi.constant #comm.mpi.op<MPI_SUM> : !comm.mpi.op
    // CHECK-NEXT: return %[[COMM]], %[[OP]] : tensor<i64>, tensor<i64>
    return %comm, %op : !comm.mpi.comm, !comm.mpi.op
}
