// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(mark-func-memory-effects{assume_no_memory_effects=true},canonicalize)" %s | FileCheck %s --check-prefix=ASSUME
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(mark-func-memory-effects{assume_no_memory_effects=false},canonicalize)" %s | FileCheck %s --check-prefix=NOASSUME

module {
  // ASSUME: llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {enzymexla.memory_effects = ["read", "write"], llvm.align = 32 : i64, llvm.nocapture, llvm.nofree}) attributes {enzymexla.memory_effects = ["read", "write"]} {
  // NOASSUME: llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {enzymexla.memory_effects = ["read", "write"], llvm.align = 32 : i64, llvm.nocapture, llvm.nofree}) attributes {enzymexla.memory_effects = ["read", "write"]} {
  llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {llvm.align = 32, llvm.nocapture, llvm.nofree}) {
    %c1 = llvm.mlir.constant(1 : index) : i64
    %ptr = llvm.getelementptr %arg0[%c1, %c1] : (!llvm.ptr<1>, i64, i64) -> !llvm.ptr<1>, !llvm.array<8 x i64>
    %val = llvm.load %ptr : !llvm.ptr<1> -> i64
    %ptr_str = llvm.getelementptr %arg0[%c1, %c1] : (!llvm.ptr<1>, i64, i64) -> !llvm.ptr<1>, !llvm.array<8 x i64>
    llvm.store %val, %ptr_str : i64, !llvm.ptr<1>
    llvm.return
  }

  func.func @main(%arg0: tensor<64xi64>, %arg1: tensor<32xi64>) -> (tensor<32xi64>) {
    // ASSUME: enzymexla.jit_call @foo (%arg0) {
    // NOASSUME: enzymexla.jit_call @foo (%arg0) {
    %0 = enzymexla.jit_call @foo (%arg0) {
        output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [],
        operand_index = 0, operand_tuple_indices = []>]
      } : (tensor<64xi64>) -> tensor<64xi64>
    // ASSUME: enzymexla.jit_call @foo (%arg1) {
    // NOASSUME: enzymexla.jit_call @foo (%arg1) {
    %1 = enzymexla.jit_call @foo (%arg1) {
        output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [],
        operand_index = 0, operand_tuple_indices = []>]
      } : (tensor<32xi64>) -> tensor<32xi64>
    return %1 : tensor<32xi64>
  }
}
