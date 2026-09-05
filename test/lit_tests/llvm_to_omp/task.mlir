// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `#pragma omp task` and `#pragma omp taskwait`: __kmpc_omp_task_alloc plus
// __kmpc_omp_task become an omp.task, and __kmpc_omp_taskwait becomes omp.taskwait.
#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = ".omp_outlined.">
#alias_scope = #llvm.alias_scope<id = distinct[1]<>, domain = #alias_scope_domain, description = ".omp_outlined.: argument 0">
module {
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_0(";unknown;unknown;0;0;;\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_1() {addr_space = 0 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %c22_i32 = arith.constant 22 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
    %2 = llvm.insertvalue %c0_i32, %1[0] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %3 = llvm.insertvalue %c2_i32, %2[1] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %4 = llvm.insertvalue %c0_i32, %3[2] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %5 = llvm.insertvalue %c22_i32, %4[3] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %6 = llvm.insertvalue %0, %5[4] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    llvm.return %6 : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
  }
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_2() {addr_space = 0 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %c22_i32 = arith.constant 22 : i32
    %c0_i32 = arith.constant 0 : i32
    %c322_i32 = arith.constant 322 : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
    %2 = llvm.insertvalue %c0_i32, %1[0] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %3 = llvm.insertvalue %c322_i32, %2[1] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %4 = llvm.insertvalue %c0_i32, %3[2] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %5 = llvm.insertvalue %c22_i32, %4[3] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %6 = llvm.insertvalue %0, %5[4] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    llvm.return %6 : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
  }
  llvm.func @task(%arg0: !llvm.ptr, %arg1: i32) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c2_i32 = arith.constant 2 : i32
    %1 = llvm.mlir.addressof @task.omp_outlined : !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1x!llvm.ptr>
    memref.store %arg0, %4[%c0] : memref<1x!llvm.ptr>
    %5 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<1xi32>
    memref.store %arg1, %5[%c0] : memref<1xi32>
    llvm.call @__kmpc_fork_call(%0, %c2_i32, %1, %2, %3) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @task.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_2 : !llvm.ptr
    %1 = llvm.mlir.addressof @".omp_task_entry." : !llvm.ptr
    %c16_i64 = arith.constant 16 : i64
    %c40_i64 = arith.constant 40 : i64
    %c1_i32 = arith.constant 1 : i32
    %2 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %3 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
    %4 = memref.load %3[%c0] : memref<?xi32>
    %5 = llvm.call tail @__kmpc_single(%2, %4) {convergent, no_unwind} : (!llvm.ptr, i32) -> i32
    %6 = arith.cmpi ne, %5, %c0_i32 : i32
    scf.if %6 {
      %7 = llvm.call tail @__kmpc_omp_task_alloc(%2, %4, %c1_i32, %c40_i64, %c16_i64, %1) : (!llvm.ptr, i32, i32, i64, i64, !llvm.ptr) -> !llvm.ptr
      %8 = "enzymexla.pointer2memref"(%7) : (!llvm.ptr) -> memref<?x!llvm.ptr>
      %9 = memref.load %8[%c0] : memref<?x!llvm.ptr>
      %10 = "enzymexla.pointer2memref"(%9) : (!llvm.ptr) -> memref<?x!llvm.ptr>
      memref.store %arg2, %10[%c0] : memref<?x!llvm.ptr>
      %11 = "enzymexla.pointer2memref"(%9) : (!llvm.ptr) -> memref<?x!llvm.ptr>
      memref.store %arg3, %11[%c1] : memref<?x!llvm.ptr>
      %12 = llvm.call tail @__kmpc_omp_task(%2, %4, %7) : (!llvm.ptr, i32, !llvm.ptr) -> i32
      %13 = llvm.call tail @__kmpc_omp_taskwait(%2, %4) {convergent, no_unwind} : (!llvm.ptr, i32) -> i32
      llvm.call tail @__kmpc_end_single(%2, %4) {convergent, no_unwind} : (!llvm.ptr, i32) -> ()
    }
    llvm.call tail @__kmpc_barrier(%0, %4) {convergent, no_unwind} : (!llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @__kmpc_single(!llvm.ptr, i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @__kmpc_end_single(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @body(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func internal @".omp_task_entry."(%arg0: i32, %arg1: !llvm.ptr) -> i32 attributes {dso_local, sym_visibility = "private"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %1 = memref.load %0[%c0] : memref<?x!llvm.ptr>
    llvm.intr.experimental.noalias.scope.decl #alias_scope
    %2 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %3 = memref.load %2[%c0] : memref<?x!llvm.ptr>
    %4 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %5 = memref.load %4[%c0] : memref<?x!llvm.ptr>
    %6 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %7 = memref.load %6[%c1] : memref<?x!llvm.ptr>
    %8 = "enzymexla.pointer2memref"(%7) : (!llvm.ptr) -> memref<?xi32>
    %9 = memref.load %8[%c0] : memref<?xi32>
    llvm.call tail @body(%5, %9) {no_unwind, noalias_scopes = [#alias_scope]} : (!llvm.ptr, i32) -> ()
    llvm.return %c0_i32 : i32
  }
  llvm.func @__kmpc_omp_task_alloc(!llvm.ptr, i32, i32, i64, i64, !llvm.ptr) -> !llvm.ptr attributes {sym_visibility = "private"}
  llvm.func @__kmpc_omp_task(!llvm.ptr, i32, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @__kmpc_omp_taskwait(!llvm.ptr, i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @__kmpc_barrier(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @task(
// CHECK: omp.parallel {
// CHECK:  omp.single {
// CHECK:    omp.task {
// CHECK:      llvm.call tail @body(%{{.*}}, %{{.*}}) {no_unwind, noalias_scopes = [#alias_scope]} : (!llvm.ptr, i32) -> ()
// CHECK:      omp.terminator
// CHECK:    omp.taskwait
// CHECK:    omp.terminator
// CHECK:  omp.barrier
// CHECK:  omp.terminator
