// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `#pragma omp teams num_teams(4)`: __kmpc_fork_teams becomes an omp.teams.
// __kmpc_push_num_teams carries the single count of `num_teams(N)`, which is
// the league's UPPER bound — omp.teams rejects a lower bound given on its own.
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
  llvm.func @teams_region() {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %1 = llvm.mlir.addressof @teams_region.omp_outlined : !llvm.ptr
    %2 = llvm.call tail @__kmpc_global_thread_num(%0) : (!llvm.ptr) -> i32
    llvm.call tail @__kmpc_push_num_teams(%0, %2, %c4_i32, %c0_i32) : (!llvm.ptr, i32, i32, i32) -> ()
    llvm.call tail @__kmpc_fork_teams(%0, %c0_i32, %1) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func @__kmpc_global_thread_num(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @__kmpc_push_num_teams(!llvm.ptr, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func internal @teams_region.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0_i32 = arith.constant 0 : i32
    llvm.call tail @body(%c0_i32) : (i32) -> ()
    llvm.return
  }
  llvm.func @__kmpc_fork_teams(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
  llvm.func @body(i32 {llvm.noundef}) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @teams_region(
// CHECK: omp.teams num_teams( to %{{.*}} : i32)
// CHECK:   llvm.call tail @body(%{{.*}}) : (i32) -> ()
// CHECK:   omp.terminator
