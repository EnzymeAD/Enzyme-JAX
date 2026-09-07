// RUN: enzymexlamlir-opt --llvm-to-affine-access %s | FileCheck %s

// A constant-sized llvm.alloca gives the raised memref a static extent instead
// of the `?` it used to always get. The extent counts bytes, because the memref
// convertToMemref builds is i8-typed; a later reinterpretation to a wider
// element type scales it down. Everything else stays dynamic.

module {
  llvm.func @sink(!llvm.ptr)

  llvm.func @constant_alloca() {
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %a = llvm.alloca %c4 x i32 : (i32) -> !llvm.ptr
    %p = llvm.getelementptr %a[%c2] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %c0, %p : i32, !llvm.ptr
    llvm.call @sink(%a) : (!llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @byte_alloca() {
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %z = llvm.mlir.constant(0 : i8) : i8
    %a = llvm.alloca %c4 x i32 : (i32) -> !llvm.ptr
    %p = llvm.getelementptr %a[%c8] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    llvm.store %z, %p : i8, !llvm.ptr
    llvm.call @sink(%a) : (!llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @uneven_alloca() {
    %c3 = llvm.mlir.constant(3 : i32) : i32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %a = llvm.alloca %c3 x i8 : (i32) -> !llvm.ptr
    llvm.store %c0, %a : i32, !llvm.ptr
    llvm.call @sink(%a) : (!llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @variable_alloca(%n: i32) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %a = llvm.alloca %n x i32 : (i32) -> !llvm.ptr
    %p = llvm.getelementptr %a[%c2] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %c0, %p : i32, !llvm.ptr
    llvm.call @sink(%a) : (!llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @from_argument(%arg: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %p = llvm.getelementptr %arg[%c2] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %c0, %p : i32, !llvm.ptr
    llvm.return
  }
}

// 4 x i32 is 16 bytes, reinterpreted for the i32 store as 4 elements.
// CHECK-LABEL: llvm.func @constant_alloca()
// CHECK:         "enzymexla.pointer2memref"(%{{.*}}) : (!llvm.ptr) -> memref<4xi32>
// CHECK:         affine.store %{{.*}}, %{{.*}}[2] {{.*}} : memref<4xi32>

// The same alloca accessed bytewise keeps the byte extent, so the store at byte
// 8 stays in bounds.
// CHECK-LABEL: llvm.func @byte_alloca()
// CHECK:         "enzymexla.pointer2memref"(%{{.*}}) : (!llvm.ptr) -> memref<16xi8>
// CHECK:         affine.store %{{.*}}, %{{.*}}[8] {{.*}} : memref<16xi8>

// 3 bytes is not a whole number of i32s, so the reinterpretation goes dynamic
// rather than rounding down.
// CHECK-LABEL: llvm.func @uneven_alloca()
// CHECK:         "enzymexla.pointer2memref"(%{{.*}}) : (!llvm.ptr) -> memref<?xi32>

// A VLA has no constant extent, so the memref stays dynamic.
// CHECK-LABEL: llvm.func @variable_alloca(
// CHECK:         "enzymexla.pointer2memref"(%{{.*}}) : (!llvm.ptr) -> memref<?xi32>

// Neither does an incoming pointer.
// CHECK-LABEL: llvm.func @from_argument(
// CHECK:         "enzymexla.pointer2memref"(%{{.*}}) : (!llvm.ptr) -> memref<?xi32>
