// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(mark-func-memory-effects{assume_no_memory_effects=true})" %s | FileCheck %s --check-prefix=ASSUME
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(mark-func-memory-effects{assume_no_memory_effects=false})" %s | FileCheck %s --check-prefix=NOASSUME

// ASSUME: func.func @single_dim(%arg0: memref<3xi64> {enzymexla.memory_effects = ["write"], llvm.nofree, llvm.writeonly}, %arg1: memref<3xi64> {enzymexla.memory_effects = ["read"], llvm.nofree, llvm.readonly}) attributes {enzymexla.memory_effects = ["read", "write"]} {
// NOASSUME: func.func @single_dim(%arg0: memref<3xi64> {enzymexla.memory_effects = ["write"], llvm.nofree, llvm.writeonly}, %arg1: memref<3xi64> {enzymexla.memory_effects = ["read"], llvm.nofree, llvm.readonly}) attributes {enzymexla.memory_effects = ["read", "write"]} {
func.func @single_dim(%output: memref<3xi64>, %values: memref<3xi64>) {
    affine.parallel (%i) = (0) to (3) {
        %val = memref.load %values[%i] : memref<3xi64>
        affine.store %val, %output[%i] : memref<3xi64>
    }
    return
}
