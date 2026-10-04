// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// A worksharing loop over a 64-bit induction variable: Clang selects the
// __kmpc_for_static_init_8 variant, and the whole loop nest has to come out
// typed i64 rather than i32.
module {
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_0(";unknown;unknown;0;0;;\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local}
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_1() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)> {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %c22_i32 = arith.constant 22 : i32
    %c0_i32 = arith.constant 0 : i32
    %c514_i32 = arith.constant 514 : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %2 = llvm.insertvalue %c0_i32, %1[0] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %3 = llvm.insertvalue %c514_i32, %2[1] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %4 = llvm.insertvalue %c0_i32, %3[2] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %5 = llvm.insertvalue %c22_i32, %4[3] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %6 = llvm.insertvalue %0, %5[4] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    llvm.return %6 : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
  }
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_2() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)> {
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
  llvm.func @loop_i64(%arg0: i64) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_2 : !llvm.ptr
    %1 = llvm.mlir.addressof @loop_i64.omp_outlined : !llvm.ptr
    %2 = llvm.alloca %c1_i32 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi64>
    memref.store %arg0, %3[%c0] : memref<1xi64>
    llvm.call @__kmpc_fork_call(%0, %c1_i32, %1, %2) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @loop_i64.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0 = arith.constant 0 : index
    %c34_i32 = arith.constant 34 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c1_i64 = arith.constant 1 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i64 = arith.constant 2 : i64
    %1 = llvm.alloca %c1_i32 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %c1_i32 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xi64>
    %6 = memref.load %5[%c0] : memref<?xi64>
    %7 = arith.cmpi sgt, %6, %c0_i64 : i64
    scf.if %7 {
      %8 = arith.addi %6, %c-1_i64 overflow<nsw> : i64
      %9 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<1xi64>
      memref.store %c0_i64, %9[%c0] : memref<1xi64>
      %10 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi64>
      memref.store %8, %10[%c0] : memref<1xi64>
      %11 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<1xi64>
      memref.store %c1_i64, %11[%c0] : memref<1xi64>
      %12 = "enzymexla.pointer2memref"(%4) : (!llvm.ptr) -> memref<1xi32>
      memref.store %c0_i32, %12[%c0] : memref<1xi32>
      %13 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
      %14 = memref.load %13[%c0] : memref<?xi32>
      llvm.call @__kmpc_for_static_init_8(%0, %14, %c34_i32, %4, %1, %2, %3, %c1_i64, %c1_i64) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
      %15 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi64>
      %16 = memref.load %15[%c0] : memref<1xi64>
      %17 = arith.minsi %16, %8 : i64
      %18 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi64>
      memref.store %17, %18[%c0] : memref<1xi64>
      %19 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<1xi64>
      %20 = memref.load %19[%c0] : memref<1xi64>
      %21 = arith.cmpi sle, %20, %17 : i64
      scf.if %21 {
        %22 = arith.addi %17, %c1_i64 : i64
        %23 = arith.cmpi slt, %20, %22 : i64
        scf.if %23 {
          %24 = arith.addi %20, %c1_i64 : i64
          %25 = arith.addi %17, %c2_i64 : i64
          scf.for %arg3 = %24 to %25 step %c1_i64  : i64 {
            %26 = arith.subi %arg3, %24 : i64
            %27 = arith.addi %20, %26 : i64
            llvm.call @body_i64(%27) : (i64) -> ()
          }
        }
      }
      llvm.call @__kmpc_for_static_fini(%0, %14) : (!llvm.ptr, i32) -> ()
    }
    llvm.return
  }
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_for_static_init_8(!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) attributes {sym_visibility = "private"}
  llvm.func @body_i64(i64 {llvm.noundef}) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_for_static_fini(!llvm.ptr, i32) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @loop_i64(
// CHECK: omp.parallel {
// CHECK:   omp.wsloop nowait schedule(static = %{{.*}} : i64) {
// CHECK:     omp.loop_nest (%{{.*}}) : i64 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
// CHECK:       llvm.call @body_i64(%{{.*}}) : (i64) -> ()
