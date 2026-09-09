// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `#pragma omp flush` and `#pragma omp taskyield` become omp.flush and
// omp.taskyield.  __kmpc_omp_taskyield returns an i32, so the leaf rewrite has
// to replace the call rather than erase it.
module {
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_0(";unknown;unknown;0;0;;\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_1() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)> {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %c22_i32 = arith.constant 22 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %2 = llvm.insertvalue %c0_i32, %1[0] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %3 = llvm.insertvalue %c2_i32, %2[1] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %4 = llvm.insertvalue %c0_i32, %3[2] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %5 = llvm.insertvalue %c22_i32, %4[3] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %6 = llvm.insertvalue %0, %5[4] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    llvm.return %6 : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
  }
  llvm.func @flush_region(%arg0: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @flush_region.omp_outlined : !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1x!llvm.ptr>
    memref.store %arg0, %3[%c0] : memref<1x!llvm.ptr>
    llvm.call @__kmpc_fork_call(%0, %c1_i32, %1, %2) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @flush_region.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    llvm.call tail @__kmpc_flush(%0) : (!llvm.ptr) -> ()
    %1 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
    %2 = memref.load %1[%c0] : memref<?xi32>
    %3 = llvm.call tail @__kmpc_omp_taskyield(%0, %2, %c0_i32) : (!llvm.ptr, i32, i32) -> i32
    llvm.return
  }
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_flush(!llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_omp_taskyield(!llvm.ptr, i32, i32) -> i32 attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @flush_region(
// CHECK: omp.parallel {
// CHECK:   omp.flush
// CHECK:   omp.taskyield
// CHECK:   omp.terminator
