// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `#pragma omp taskloop grainsize(2)` inside a `single`: __kmpc_taskloop and
// the kmp_task_t it is handed become an omp.taskloop.context / .wrapper pair
// around an omp.loop_nest, with the loop bounds recovered from the stores that
// fill the task struct rather than loaded back out of it.
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
    %c322_i32 = arith.constant 322 : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %2 = llvm.insertvalue %c0_i32, %1[0] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %3 = llvm.insertvalue %c322_i32, %2[1] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %4 = llvm.insertvalue %c0_i32, %3[2] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %5 = llvm.insertvalue %c22_i32, %4[3] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    %6 = llvm.insertvalue %0, %5[4] : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
    llvm.return %6 : !llvm.struct<"struct.ident_t.1.1", (i32, i32, i32, i32, ptr)>
  }
  llvm.func @taskloop_region(%arg0: i32) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @taskloop_region.omp_outlined : !llvm.ptr
    %2 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi32>
    memref.store %arg0, %3[%c0] : memref<1xi32>
    llvm.call @__kmpc_fork_call(%0, %c1_i32, %1, %2) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @taskloop_region.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c9 = arith.constant 9 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_2 : !llvm.ptr
    %1 = llvm.mlir.zero : !llvm.ptr
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %2 = llvm.mlir.addressof @".omp_task_entry." : !llvm.ptr
    %c8_i64 = arith.constant 8 : i64
    %c80_i64 = arith.constant 80 : i64
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %3 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %4 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
    %5 = memref.load %4[%c0] : memref<?xi32>
    %6 = llvm.call tail @__kmpc_single(%3, %5) : (!llvm.ptr, i32) -> i32
    %7 = arith.cmpi ne, %6, %c0_i32 : i32
    scf.if %7 {
      llvm.call tail @__kmpc_taskgroup(%3, %5) : (!llvm.ptr, i32) -> ()
      %8 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xi32>
      %9 = memref.load %8[%c0] : memref<?xi32>
      %10 = arith.addi %9, %c-1_i32 overflow<nsw> : i32
      %11 = llvm.call tail @__kmpc_omp_task_alloc(%3, %5, %c1_i32, %c80_i64, %c8_i64, %2) : (!llvm.ptr, i32, i32, i64, i64, !llvm.ptr) -> !llvm.ptr
      %12 = "enzymexla.pointer2memref"(%11) : (!llvm.ptr) -> memref<?x!llvm.ptr>
      %13 = memref.load %12[%c0] : memref<?x!llvm.ptr>
      %14 = "enzymexla.pointer2memref"(%13) : (!llvm.ptr) -> memref<?x!llvm.ptr>
      memref.store %arg2, %14[%c0] : memref<?x!llvm.ptr>
      %15 = llvm.getelementptr inbounds %11[40] : (!llvm.ptr) -> !llvm.ptr, i8
      %16 = "enzymexla.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi64>
      memref.store %c0_i64, %16[%c5] : memref<?xi64>
      %17 = llvm.getelementptr inbounds %11[48] : (!llvm.ptr) -> !llvm.ptr, i8
      %18 = arith.extsi %10 : i32 to i64
      %19 = "enzymexla.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi64>
      memref.store %18, %19[%c6] : memref<?xi64>
      %20 = "enzymexla.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi64>
      memref.store %c1_i64, %20[%c7] : memref<?xi64>
      %21 = "enzymexla.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi64>
      memref.store %c0_i64, %21[%c9] : memref<?xi64>
      llvm.call tail @__kmpc_taskloop(%3, %5, %11, %c1_i32, %15, %17, %c1_i64, %c1_i32, %c1_i32, %c2_i64, %1) : (!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i64, i32, i32, i64, !llvm.ptr) -> ()
      llvm.call tail @__kmpc_end_taskgroup(%3, %5) : (!llvm.ptr, i32) -> ()
      llvm.call tail @__kmpc_end_single(%3, %5) : (!llvm.ptr, i32) -> ()
    }
    llvm.call tail @__kmpc_barrier(%0, %5) : (!llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_single(!llvm.ptr, i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @__kmpc_taskgroup(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func internal @".omp_task_entry."(%arg0: i32, %arg1: !llvm.ptr) -> (i32 {llvm.noundef}) attributes {dso_local, sym_visibility = "private"} {
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c-1_i64 = arith.constant -1 : i64
    %0 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %1 = memref.load %0[%c0] : memref<?x!llvm.ptr>
    %2 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xi64>
    %3 = memref.load %2[%c5] : memref<?xi64>
    %4 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xi64>
    %5 = memref.load %4[%c6] : memref<?xi64>
    %6 = arith.addi %5, %c1_i64 : i64
    %7 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %8 = memref.load %7[%c0] : memref<?x!llvm.ptr>
    %9 = "enzymexla.pointer2memref"(%8) : (!llvm.ptr) -> memref<?xi32>
    %10 = memref.load %9[%c0] : memref<?xi32>
    %11 = arith.cmpi sgt, %10, %c0_i32 : i32
    %12 = arith.shli %3, %c32_i64 : i64
    %13 = arith.shrsi %12, %c32_i64 exact : i64
    %14 = arith.addi %13, %c1_i64 overflow<nsw> : i64
    %15 = arith.maxsi %6, %14 : i64
    %16 = arith.subi %15, %13 : i64
    %17 = arith.cmpi uge, %5, %13 : i64
    %18 = arith.andi %11, %17 : i1
    %19 = arith.cmpi sgt, %16, %c0_i64 : i64
    %20 = arith.andi %18, %19 : i1
    scf.if %20 {
      %21 = arith.maxsi %16, %c1_i64 : i64
      %22 = arith.addi %21, %c1_i64 : i64
      scf.for %arg2 = %c1_i64 to %22 step %c1_i64  : i64 {
        %23 = arith.addi %arg2, %c-1_i64 : i64
        %24 = arith.addi %3, %23 : i64
        %25 = arith.trunci %24 : i64 to i32
        llvm.call tail @body(%25) : (i32) -> ()
      }
    }
    llvm.return %c0_i32 : i32
  }
  llvm.func @__kmpc_omp_task_alloc(!llvm.ptr, i32, i32, i64, i64, !llvm.ptr) -> (!llvm.ptr {llvm.noalias}) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_taskloop(!llvm.ptr, i32, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr, i64, i32, i32, i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_end_taskgroup(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_end_single(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_barrier(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @body(i32 {llvm.noundef}) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @taskloop_region(
// CHECK: omp.parallel {
// CHECK:   omp.single {
// CHECK:     omp.taskgroup {
// CHECK:       omp.taskloop.context grainsize(%{{.*}}: i64) nogroup {
// CHECK:         omp.taskloop.wrapper {
// CHECK:           omp.loop_nest (%[[IV:.*]]) : i64 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CHECK:             llvm.call tail @body(%{{.*}}) : (i32) -> ()
// CHECK:             omp.yield
