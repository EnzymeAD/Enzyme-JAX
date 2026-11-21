// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `schedule(dynamic)`: the __kmpc_dispatch_init / _next / _fini
// dispatch loop becomes an omp.wsloop with a dynamic schedule.
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
  llvm.func @wsloop_dynamic(%arg0: !llvm.ptr, %arg1: i32) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c2_i32 = arith.constant 2 : i32
    %1 = llvm.mlir.addressof @wsloop_dynamic.omp_outlined : !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1x!llvm.ptr>
    memref.store %arg0, %4[%c0] : memref<1x!llvm.ptr>
    %5 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<1xi32>
    memref.store %arg1, %5[%c0] : memref<1xi32>
    llvm.call @__kmpc_fork_call(%0, %c2_i32, %1, %3, %2) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @wsloop_dynamic.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %c1073741859_i32 = arith.constant 1073741859 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xi32>
    %6 = memref.load %5[%c0] : memref<?xi32>
    %7 = arith.cmpi sgt, %6, %c0_i32 : i32
    scf.if %7 {
      %8 = arith.addi %6, %c-1_i32 : i32
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
      llvm.call tail @__kmpc_dispatch_init_4(%0, %14, %c1073741859_i32, %c0_i32, %8, %c1_i32, %c1_i32) : (!llvm.ptr, i32, i32, i32, i32, i32, i32) -> ()
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
            %24 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?x!llvm.ptr>
            %25 = memref.load %24[%c0] : memref<?x!llvm.ptr>
            %26 = arith.extsi %18 : i32 to i64
            %27 = scf.while (%arg4 = %26) : (i64) -> i64 {
              %28 = arith.trunci %arg4 : i64 to i32
              %29 = "enzymexla.pointer2memref"(%25) : (!llvm.ptr) -> memref<?xi32>
              %30 = arith.index_cast %arg4 : i64 to index
              memref.store %28, %29[%30] : memref<?xi32>
              %31 = arith.addi %arg4, %c1_i64 : i64
              %32 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi32>
              %33 = memref.load %32[%c0] : memref<1xi32>
              %34 = arith.extsi %33 : i32 to i64
              %35 = arith.cmpi slt, %arg4, %34 : i64
              scf.condition(%35) %31 : i64
            } do {
            ^bb0(%arg4: i64):
              scf.yield %arg4 : i64
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
  llvm.func @__kmpc_dispatch_init_4(!llvm.ptr, i32, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_dispatch_next_4(!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @__kmpc_dispatch_deinit(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @wsloop_dynamic(
// CHECK: omp.parallel {
// CHECK:    omp.wsloop nowait schedule(dynamic = %{{.*}} : i32) {
// CHECK:      omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
// CHECK:        omp.yield
// CHECK:  omp.terminator
