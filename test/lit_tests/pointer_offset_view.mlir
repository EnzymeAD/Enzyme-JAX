// RUN: enzymexlamlir-opt %s --test-print-base-object -o /dev/null | FileCheck %s --check-prefix=BASE
// RUN: enzymexlamlir-opt %s --test-print-base-object="offset-allowed=false" -o /dev/null | FileCheck %s --check-prefix=ADDRESS

// Pointer conversions keep the starting address, even when strides differ.
// BASE: @pointer_to_memref return 0: %{{[^ ]+}} -> %arg0
// ADDRESS: @pointer_to_memref return 0: %{{[^ ]+}} -> %arg0
func.func @pointer_to_memref(%base: !llvm.ptr) -> memref<4xf32, strided<[2]>> {
  %view = "enzymexla.pointer2memref"(%base) : (!llvm.ptr) -> memref<4xf32, strided<[2]>>
  return %view : memref<4xf32, strided<[2]>>
}

// BASE: @memref_to_pointer return 0: %{{[^ ]+}} -> %arg0
// ADDRESS: @memref_to_pointer return 0: %{{[^ ]+}} -> %arg0
func.func @memref_to_pointer(%base: memref<4xf32, strided<[2]>>) -> !llvm.ptr {
  %pointer = "enzymexla.memref2pointer"(%base) : (memref<4xf32, strided<[2]>>) -> !llvm.ptr
  return %pointer : !llvm.ptr
}

// Follow both conversions, but keep the offset from the nonzero GEP.
// BASE: @pointer_offset return 0: %{{[^ ]+}} -> %arg0
// BASE-NEXT: @pointer_offset return 1: %{{[^ ]+}} -> %arg0
// BASE-NEXT: @pointer_offset return 2: %{{[^ ]+}} -> %arg0
// ADDRESS: @pointer_offset return 0: %[[OFFSET:[^ ]+]] -> %[[OFFSET]]
// ADDRESS-NEXT: @pointer_offset return 1: %{{[^ ]+}} -> %[[OFFSET]]
// ADDRESS-NEXT: @pointer_offset return 2: %{{[^ ]+}} -> %[[OFFSET]]
func.func @pointer_offset(%base: !llvm.ptr) -> (!llvm.ptr, memref<4xf32>, !llvm.ptr) {
  %offset = llvm.getelementptr %base[4] : (!llvm.ptr) -> !llvm.ptr, f32
  %view = "enzymexla.pointer2memref"(%offset) : (!llvm.ptr) -> memref<4xf32>
  %pointer = "enzymexla.memref2pointer"(%view) : (memref<4xf32>) -> !llvm.ptr
  return %offset, %view, %pointer : !llvm.ptr, memref<4xf32>, !llvm.ptr
}

// Follow memref2pointer, but keep the offset from the subview.
// BASE: @memref_offset return 0: %{{[^ ]+}} -> %arg0
// BASE-NEXT: @memref_offset return 1: %{{[^ ]+}} -> %arg0
// ADDRESS: @memref_offset return 0: %[[VIEW:[^ ]+]] -> %[[VIEW]]
// ADDRESS-NEXT: @memref_offset return 1: %{{[^ ]+}} -> %[[VIEW]]
func.func @memref_offset(%base: memref<8xf32>) -> (memref<4xf32, strided<[1], offset: 2>>, !llvm.ptr) {
  %view = memref.subview %base[2] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: 2>>
  %pointer = "enzymexla.memref2pointer"(%view) : (memref<4xf32, strided<[1], offset: 2>>) -> !llvm.ptr
  return %view, %pointer : memref<4xf32, strided<[1], offset: 2>>, !llvm.ptr
}
