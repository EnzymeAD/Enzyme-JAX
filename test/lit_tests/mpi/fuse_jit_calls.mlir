// RUN: enzymexlamlir-opt --fuse-jit-calls %s | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @enzymexla_wrapper_MPI_Irecv
  llvm.func @enzymexla_wrapper_MPI_Irecv(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    llvm.return
  }
  
  // CHECK-LABEL: llvm.func @enzymexla_wrapper_MPI_Wait
  llvm.func @enzymexla_wrapper_MPI_Wait(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }
  
  // Verify the new fused function is emitted with 4 pointer arguments (deduplicated inputs)
  // CHECK-LABEL: llvm.func @enzymexla_wrapper_MPI_Irecv_enzymexla_wrapper_MPI_Waitall
  // CHECK-SAME: (%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr) {

  // CHECK-LABEL: llvm.func @enzymexla_wrapper_MPI_Waitall
  llvm.func @enzymexla_wrapper_MPI_Waitall(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }

  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<5xf64>) -> tensor<5xf64> {
    %c_0 = stablehlo.constant dense<5> : tensor<i32>
    // Note: Fusion currently leaves the original calls as they have multiple results.
    // We check that the fused call is created and used.
    // CHECK: %[[FUSED:.*]] = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_enzymexla_wrapper_MPI_Wait
    %1:2 = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv (%arg0, %c_0) : (tensor<5xf64>, tensor<i32>) -> (tensor<5xf64>, tensor<i32>)
    enzymexla.jit_call @enzymexla_wrapper_MPI_Wait (%1#1) : (tensor<i32>) -> ()
    // CHECK-NEXT: return %[[FUSED]]
    return %1#0 : tensor<5xf64>
  }

  // CHECK-LABEL: func.func @test_waitall
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<5xf64>, %[[ARG1:.*]]: tensor<5xf64>)
  func.func @test_waitall(%arg0: tensor<5xf64>, %arg1: tensor<5xf64>) -> (tensor<5xf64>, tensor<5xf64>) {
    // CHECK-DAG: %[[C0:.*]] = stablehlo.constant dense<5>
    %c_0 = stablehlo.constant dense<5> : tensor<i32>
    
    // First Irecv
    %1:2 = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv (%arg0, %c_0) : (tensor<5xf64>, tensor<i32>) -> (tensor<5xf64>, tensor<i32>)
    
    // Second Irecv
    %2:2 = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv (%arg1, %c_0) : (tensor<5xf64>, tensor<i32>) -> (tensor<5xf64>, tensor<i32>)
    
    // Concat requests (as done in JAX lowering)
    %req1_bcast = stablehlo.broadcast_in_dim %1#1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %req2_bcast = stablehlo.broadcast_in_dim %2#1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %req_concat = stablehlo.concatenate %req1_bcast, %req2_bcast, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    
    // CHECK-DAG: %[[C2:.*]] = stablehlo.constant dense<2>
    %c_2 = stablehlo.constant dense<2> : tensor<i32>
    
    // Waitall takes count and requests
    enzymexla.jit_call @enzymexla_wrapper_MPI_Waitall (%c_2, %req_concat) : (tensor<i32>, tensor<2xi32>) -> ()
    
    // Ensure dead ops are folded away (these should be removed by the greedy driver if their uses are gone)
    // CHECK-NOT: stablehlo.concatenate
    // CHECK-NOT: stablehlo.broadcast_in_dim

    // CHECK: %[[RES:.*]]:2 = enzymexla.jit_call @enzymexla_wrapper_MPI_Irecv_enzymexla_wrapper_MPI_Waitall (%[[ARG0]], %[[C0]], %[[ARG1]], %[[C2]])
    // CHECK-NEXT: return %[[RES]]#0, %[[RES]]#1
    
    return %1#0, %2#0 : tensor<5xf64>, tensor<5xf64>
  }

}
