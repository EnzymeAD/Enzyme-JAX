// RUN: enzymexlamlir-opt --mark-func-memory-effects %s | FileCheck %s

// CHECK: func.func @single_dim(%arg0: memref<3xi64>, %arg1: memref<3xi64>) attributes {enzymexla.memory_effects = ["read", "write"]} {
func.func @single_dim(%output: memref<3xi64>, %values: memref<3xi64>) {
    affine.parallel (%i) = (0) to (3) {
        %val = memref.load %values[%i] : memref<3xi64>
        affine.store %val, %output[%i] : memref<3xi64>
    }
    return
}
