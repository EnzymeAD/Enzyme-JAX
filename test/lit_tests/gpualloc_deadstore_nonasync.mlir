// RUN: enzymexlamlir-opt --llvm-to-affine-access --split-input-file %s | FileCheck %s

func.func @deadstore(%val: i8) {
    %c0 = arith.constant 0 : index
    %memref = gpu.alloc() : memref<16xi8, 1>
    memref.store %val, %memref[%c0] : memref<16xi8, 1>
    return
}

// CHECK: func.func @deadstore(%arg0: i8) {
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

func.func @deadstore(%host_memref: memref<16xi8>) {
    %c16 = arith.constant 16 : index
    %memref = gpu.alloc() : memref<16xi8, 1>
    enzymexla.memcpy %memref, %host_memref, %c16 : memref<16xi8, 1>, memref<16xi8>
    return
}

// CHECK: func.func @deadstore(%arg0: memref<16xi8>) {
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

func.func @notdeadstore(%host_memref: memref<16xi8>) {
    %c16 = arith.constant 16 : index
    %memref = gpu.alloc() : memref<16xi8, 1>
    enzymexla.memcpy %host_memref, %memref, %c16 : memref<16xi8>, memref<16xi8, 1>
    return
}

// CHECK: func.func @notdeadstore(%arg0: memref<16xi8>) {
// CHECK-NEXT:   %c16 = arith.constant 16 : index
// CHECK-NEXT:   %memref = gpu.alloc () : memref<16xi8, 1>
// CHECK-NEXT:   enzymexla.memcpy %arg0, %memref, %c16 : memref<16xi8>, memref<16xi8, 1>
// CHECK-NEXT:   return
// CHECK-NEXT: }
