// RUN: enzymexlamlir-opt %s --canonicalize | FileCheck %s

module @nosideeffect {
  llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @MPI_Barrier(!llvm.ptr) -> i32
  func.func @enzymexla_wrapper_MPI_Barrier() {
    %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
    %1 = llvm.call @MPI_Barrier(%0) : (!llvm.ptr) -> i32
    return
  }
  func.func @main() {
    // CHECK-NOT: enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () : () -> ()
    enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () : () -> ()
    return
  }
}

module @sideeffect {
  llvm.mlir.global external constant @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @MPI_Barrier(!llvm.ptr) -> i32
  func.func @enzymexla_wrapper_MPI_Barrier() {
    %0 = llvm.mlir.addressof @MPI_COMM_WORLD : !llvm.ptr
    %1 = llvm.call @MPI_Barrier(%0) : (!llvm.ptr) -> i32
    return
  }
  func.func @main() {
    // CHECK: enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () {has_side_effect = true} : () -> ()
    enzymexla.jit_call @enzymexla_wrapper_MPI_Barrier () {has_side_effect = true} : () -> ()
    return
  }
}

module @nosideeffect_nouse {
  llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {llvm.align = 32, llvm.nocapture, llvm.nofree}) {
    llvm.return
  }

  func.func @main(%arg0: tensor<64xi64>, %arg1: tensor<32xi64>) -> (tensor<32xi64>) {
    // CHECK-NOT: enzymexla.jit_call @foo (%arg0) {
    %0 = enzymexla.jit_call @foo (%arg0) {
        output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], 
        operand_index = 0, operand_tuple_indices = []>]
      } : (tensor<64xi64>) -> tensor<64xi64>
    // CHECK: enzymexla.jit_call @foo (%arg1) {
    %1 = enzymexla.jit_call @foo (%arg1) {
        output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], 
        operand_index = 0, operand_tuple_indices = []>]
      } : (tensor<32xi64>) -> tensor<32xi64>
    return %1 : tensor<32xi64>
  }
}
