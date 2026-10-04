// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `#pragma omp masked`: the __kmpc_masked / __kmpc_end_masked pair becomes an
// omp.masked with the thread filter it was given.
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
  llvm.func @masked_region() {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %1 = llvm.mlir.addressof @masked_region.omp_outlined : !llvm.ptr
    llvm.call tail @__kmpc_fork_call(%0, %c0_i32, %1) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @masked_region.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %1 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
    %2 = memref.load %1[%c0] : memref<?xi32>
    %3 = llvm.call tail @__kmpc_masked(%0, %2, %c0_i32) : (!llvm.ptr, i32, i32) -> i32
    %4 = arith.cmpi ne, %3, %c0_i32 : i32
    scf.if %4 {
      llvm.call tail @body(%c0_i32) : (i32) -> ()
      llvm.call tail @__kmpc_end_masked(%0, %2) : (!llvm.ptr, i32) -> ()
    }
    llvm.return
  }
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_masked(!llvm.ptr, i32, i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @body(i32 {llvm.noundef}) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_end_masked(!llvm.ptr, i32) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @masked_region(
// CHECK: omp.parallel {
// CHECK:   omp.masked filter(%{{.*}} : i32) {
// CHECK:     llvm.call tail @body(%{{.*}}) : (i32) -> ()
// CHECK:     omp.terminator
