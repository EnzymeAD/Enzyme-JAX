// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `schedule(guided, 4)`: a non-static schedule kind reaches omp.wsloop
// together with its chunk size.
#access_group = #llvm.access_group<id = distinct[0]<>>
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
  llvm.func @guided_loop(%arg0: i32) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @guided_loop.omp_outlined : !llvm.ptr
    %2 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi32>
    memref.store %arg0, %3[%c0] : memref<1xi32>
    llvm.call @__kmpc_fork_call(%0, %c1_i32, %1, %2) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @guided_loop.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0 = arith.constant 0 : index
    %c4_i32 = arith.constant 4 : i32
    %c1073741860_i32 = arith.constant 1073741860 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %1 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xi32>
    %6 = memref.load %5[%c0] : memref<?xi32>
    %7 = arith.cmpi sgt, %6, %c0_i32 : i32
    scf.if %7 {
      %8 = arith.addi %6, %c-1_i32 overflow<nsw> : i32
      %9 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<1xi32>
      memref.store %c0_i32, %9[%c0] : memref<1xi32>
      %10 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi32>
      memref.store %8, %10[%c0] : memref<1xi32>
      %11 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<1xi32>
      memref.store %c1_i32, %11[%c0] : memref<1xi32>
      %12 = "enzymexla.pointer2memref"(%4) : (!llvm.ptr) -> memref<1xi32>
      memref.store %c0_i32, %12[%c0] : memref<1xi32>
      %13 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
      %14 = memref.load %13[%c0] : memref<?xi32>
      llvm.call tail @__kmpc_dispatch_init_4(%0, %14, %c1073741860_i32, %c0_i32, %8, %c1_i32, %c4_i32) : (!llvm.ptr, i32, i32, i32, i32, i32, i32) -> ()
      %15 = llvm.call @__kmpc_dispatch_next_4(%0, %14, %4, %1, %2, %3) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %16 = arith.cmpi ne, %15, %c0_i32 : i32
      scf.if %16 {
        scf.while : () -> () {
          %17 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<1xi32>
          %18 = memref.load %17[%c0] : memref<1xi32>
          %19 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi32>
          %20 = memref.load %19[%c0] : memref<1xi32>
          %21 = arith.cmpi sle, %18, %20 : i32
          scf.if %21 {
            %24 = arith.addi %20, %c1_i32 : i32
            %25 = arith.cmpi slt, %18, %24 : i32
            scf.if %25 {
              %26 = arith.addi %18, %c1_i32 : i32
              %27 = arith.addi %20, %c2_i32 : i32
              scf.for %arg3 = %26 to %27 step %c1_i32  : i32 {
                %28 = arith.subi %arg3, %26 : i32
                %29 = arith.addi %18, %28 : i32
                llvm.call @body(%29) {access_groups = [#access_group], no_unwind} : (i32) -> ()
              }
            }
          }
          %22 = llvm.call @__kmpc_dispatch_next_4(%0, %14, %4, %1, %2, %3) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
          %23 = arith.cmpi ne, %22, %c0_i32 : i32
          scf.condition(%23)
        } do {
          scf.yield
        }
      }
      llvm.call @__kmpc_dispatch_deinit(%0, %14) : (!llvm.ptr, i32) -> ()
    }
    llvm.return
  }
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_dispatch_init_4(!llvm.ptr, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_dispatch_next_4(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @body(i32 {llvm.noundef}) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_dispatch_deinit(!llvm.ptr, i32) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @guided_loop(
// CHECK: omp.parallel {
// CHECK:   omp.wsloop nowait schedule(guided = %{{.*}} : i32) {
// CHECK:     omp.loop_nest (%{{.*}}) : i32 =
