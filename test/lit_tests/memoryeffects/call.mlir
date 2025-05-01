// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(mark-func-memory-effects{assume_no_memory_effects=true})" %s | FileCheck %s --check-prefix=ASSUME
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(mark-func-memory-effects{assume_no_memory_effects=false})" %s | FileCheck %s --check-prefix=NOASSUME

// ASSUME: func.func @single_dim(%arg0: memref<3xi64>, %arg1: memref<3xi64>) attributes {enzymexla.memory_effects = ["read", "write"]} {
// NOASSUME: func.func @single_dim(%arg0: memref<3xi64>, %arg1: memref<3xi64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
func.func @single_dim(%output: memref<3xi64>, %values: memref<3xi64>) {
    affine.parallel (%i) = (0) to (3) {
        %val = memref.load %values[%i] : memref<3xi64>
        affine.store %val, %output[%i] : memref<3xi64>
    }
    return
}
// ASSUME: func.func @main(%arg0: memref<3xi64>, %arg1: memref<3xi64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate"]} {
// NOASSUME: func.func @main(%arg0: memref<3xi64>, %arg1: memref<3xi64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
func.func @main(%output: memref<3xi64>, %values: memref<3xi64>) {
    %0 = memref.alloc() {alignment = 8 : i64} : memref<3xi64>
    func.call @single_dim(%output, %values) : (memref<3xi64>, memref<3xi64>) -> ()
    return
}
