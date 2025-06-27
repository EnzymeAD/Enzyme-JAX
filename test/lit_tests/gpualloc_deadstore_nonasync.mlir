// RUN: enzymexlamlir-opt --llvm-to-affine-access %s | FileCheck %s

func.func @deadstore(%val: i8) {
    %c0 = arith.constant 0 : index
    %memref = gpu.alloc() : memref<16xi8, 1>
    memref.store %val, %memref[%c0] : memref<16xi8, 1>
    return
}

// CHECK: func.func @deadstore(%arg0: i8) {
// CHECK-NEXT:   return
// CHECK-NEXT: }
