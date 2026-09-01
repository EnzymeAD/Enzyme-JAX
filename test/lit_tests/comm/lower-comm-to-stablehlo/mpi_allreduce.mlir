// RUN: enzymexlamlir-opt --lower-comm-to-stablehlo %s | FileCheck %s

// CHECK: func.func @main(%[[BUF:.*]]: tensor<4xf64>, %[[COMM:.*]]: tensor<i64>)  -> tensor<4xf64> {
func.func @main(%buf : tensor<4xf64>, %comm : !comm.mpi_comm) -> tensor<4xf64> {
    // CHECK-NEXT: %[[v0:.*]] = stablehlo.custom_call @MpiAllreduce(%[[BUF]], %[[COMM]]) {api_version = 4 : i32, backend_config = {op = "MPI_OP_NULL"}, has_side_effect = true} : (tensor<4xf64>, tensor<i64>) -> tensor<4xf64>
    %0 = comm.mpi.allreduce %buf, #comm.mpi.op<MPI_OP_NULL>, %comm : (tensor<4xf64>, !comm.mpi_comm) -> tensor<4xf64>
    return %0 : tensor<4xf64>
}
