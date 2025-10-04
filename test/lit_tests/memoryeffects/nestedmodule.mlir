// RUN: enzymexlamlir-opt --mark-func-memory-effects %s | FileCheck %s

module {
    module @nested {
        func.func public @single_dim(%output: memref<3xi64>, %values: memref<3xi64>) {
        // CHECK: func.func public @single_dim(%arg0: memref<3xi64> {enzymexla.memory_effects = ["write"], llvm.nofree, llvm.writeonly}, %arg1: memref<3xi64> {enzymexla.memory_effects = ["read"], llvm.nofree, llvm.readonly}) attributes {enzymexla.memory_effects = ["read", "write"]}
            affine.parallel (%i) = (0) to (3) {
                %val = memref.load %values[%i] : memref<3xi64>
                affine.store %val, %output[%i] : memref<3xi64>
            }
            return
        }
    }
    func.func @main(%output: memref<3xi64>, %values: memref<3xi64>) {
    // CHECK: func.func @main(%arg0: memref<3xi64> {enzymexla.memory_effects = ["write"], llvm.nofree, llvm.writeonly}, %arg1: memref<3xi64> {enzymexla.memory_effects = ["read"], llvm.nofree, llvm.readonly}) attributes {enzymexla.memory_effects = ["read", "write"]}
        affine.parallel (%i) = (0) to (3) {
            %val = memref.load %values[%i] : memref<3xi64>
            affine.store %val, %output[%i] : memref<3xi64>
        }
        return
    }
}
