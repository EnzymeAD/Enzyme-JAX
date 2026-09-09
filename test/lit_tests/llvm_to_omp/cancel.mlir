// RUN: enzymexlamlir-opt --llvm-to-omp --symbol-dce %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `#pragma omp cancel parallel` and `#pragma omp cancellation point parallel`.
// Both runtime calls return an i32 the surrounding code branches on, while
// omp.cancel and omp.cancellation_point are result-less and carry that
// transfer of control themselves — so the branch is fed a constant zero.
// symbol-dce drops the dead outlined copy, where the two constructs would have
// no enclosing region to cancel.
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
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_2() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)> {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %c22_i32 = arith.constant 22 : i32
    %c0_i32 = arith.constant 0 : i32
    %c66_i32 = arith.constant 66 : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %2 = llvm.insertvalue %c0_i32, %1[0] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %3 = llvm.insertvalue %c66_i32, %2[1] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %4 = llvm.insertvalue %c0_i32, %3[2] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %5 = llvm.insertvalue %c22_i32, %4[3] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %6 = llvm.insertvalue %0, %5[4] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    llvm.return %6 : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
  }
  llvm.func @cancel_region() {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %1 = llvm.mlir.addressof @cancel_region.omp_outlined : !llvm.ptr
    llvm.call tail @__kmpc_fork_call(%0, %c0_i32, %1) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @cancel_region.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_2 : !llvm.ptr
    %c1_i32 = arith.constant 1 : i32
    %1 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %2 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
    %3 = memref.load %2[%c0] : memref<?xi32>
    %4 = llvm.call tail @__kmpc_cancellationpoint(%1, %3, %c1_i32) : (!llvm.ptr, i32, i32) -> i32
    %5 = arith.cmpi eq, %4, %c0_i32 : i32
    scf.if %5 {
      %6 = llvm.call tail @__kmpc_cancel(%1, %3, %c1_i32) : (!llvm.ptr, i32, i32) -> i32
      %7 = arith.cmpi ne, %6, %c0_i32 : i32
      scf.if %7 {
        %8 = llvm.call tail @__kmpc_cancel_barrier(%0, %3) : (!llvm.ptr, i32) -> i32
      }
    } else {
      %6 = llvm.call tail @__kmpc_cancel_barrier(%0, %3) : (!llvm.ptr, i32) -> i32
    }
    llvm.return
  }
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_cancellationpoint(!llvm.ptr, i32, i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @__kmpc_cancel(!llvm.ptr, i32, i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @__kmpc_cancel_barrier(!llvm.ptr, i32) -> i32 attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @cancel_region(
// CHECK: omp.parallel {
// CHECK:   omp.cancellation_point cancellation_construct_type(parallel)
// CHECK:   omp.cancel cancellation_construct_type(parallel)
// CHECK:   omp.terminator
