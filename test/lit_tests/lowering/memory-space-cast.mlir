// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm | FileCheck %s

// With C-style memrefs a memref is a bare pointer in its memory space, so a
// memory space cast is exactly the LLVM addrspacecast.

module {
  func.func @cast(%a: memref<64xf64, 5>, %v: f64, %i: index) -> f64 {
    %m = memref.memory_space_cast %a : memref<64xf64, 5> to memref<64xf64>
    memref.store %v, %m[%i] : memref<64xf64>
    %x = memref.load %m[%i] : memref<64xf64>
    return %x : f64
  }
}

// CHECK-LABEL: llvm.func @cast(
// CHECK-SAME: %[[a:.+]]: !llvm.ptr<5>, %[[v:.+]]: f64, %[[i:.+]]: i64) -> f64
// CHECK: %[[m:.+]] = llvm.addrspacecast %[[a]] : !llvm.ptr<5> to !llvm.ptr
// CHECK: %[[gep:.+]] = llvm.getelementptr %[[m]][%[[i]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK: llvm.store %[[v]], %[[gep]] : f64, !llvm.ptr
