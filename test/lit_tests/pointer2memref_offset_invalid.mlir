// RUN: enzymexlamlir-opt %s --split-input-file --verify-diagnostics

func.func @nonzero_offset(%pointer: !llvm.ptr) {
  // expected-error @+1 {{requires a result layout with zero offset}}
  %view = "enzymexla.pointer2memref"(%pointer) : (!llvm.ptr) -> memref<4xf32, strided<[1], offset: 1>>
  return
}

// -----

func.func @unknown_offset(%pointer: !llvm.ptr) {
  // expected-error @+1 {{requires a result layout with zero offset}}
  %view = "enzymexla.pointer2memref"(%pointer) : (!llvm.ptr) -> memref<4xf32, strided<[1], offset: ?>>
  return
}
