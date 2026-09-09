// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `reduction(+ : s)`: the __kmpc_reduce_nowait dispatch and the combiner
// in its case-1 block are lifted into an omp.declare_reduction symbol, referenced
// from the reduction clause of the enclosing omp.parallel.
module {
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_0(";unknown;unknown;0;0;;\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_1() {addr_space = 0 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %c22_i32 = arith.constant 22 : i32
    %c0_i32 = arith.constant 0 : i32
    %c514_i32 = arith.constant 514 : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
    %2 = llvm.insertvalue %c0_i32, %1[0] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %3 = llvm.insertvalue %c514_i32, %2[1] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %4 = llvm.insertvalue %c0_i32, %3[2] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %5 = llvm.insertvalue %c22_i32, %4[3] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %6 = llvm.insertvalue %0, %5[4] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    llvm.return %6 : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
  }
  llvm.mlir.global common @".gomp_critical_user_.reduction.var"(dense<0> : tensor<8xi32>) {addr_space = 0 : i32, alignment = 8 : i64, sym_visibility = "private"} : !llvm.array<8 x i32>
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_2() {addr_space = 0 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %c22_i32 = arith.constant 22 : i32
    %c0_i32 = arith.constant 0 : i32
    %c18_i32 = arith.constant 18 : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
    %2 = llvm.insertvalue %c0_i32, %1[0] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %3 = llvm.insertvalue %c18_i32, %2[1] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %4 = llvm.insertvalue %c0_i32, %3[2] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %5 = llvm.insertvalue %c22_i32, %4[3] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %6 = llvm.insertvalue %0, %5[4] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    llvm.return %6 : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
  }
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_3() {addr_space = 0 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> {
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
  llvm.func @reduction(%arg0: !llvm.ptr, %arg1: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_3 : !llvm.ptr
    %c3_i32 = arith.constant 3 : i32
    %1 = llvm.mlir.addressof @reduction.omp_outlined : !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1x!llvm.ptr>
    memref.store %arg0, %5[%c0] : memref<1x!llvm.ptr>
    %6 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<1xi32>
    memref.store %arg1, %6[%c0] : memref<1xi32>
    %7 = "enzymexla.pointer2memref"(%4) : (!llvm.ptr) -> memref<1xi32>
    memref.store %c0_i32, %7[%c0] : memref<1xi32>
    llvm.call @__kmpc_fork_call(%0, %c3_i32, %1, %3, %4, %2) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %8 = "enzymexla.pointer2memref"(%4) : (!llvm.ptr) -> memref<1xi32>
    %9 = memref.load %8[%c0] : memref<1xi32>
    llvm.return %9 : i32
  }
  llvm.func internal @reduction.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @".gomp_critical_user_.reduction.var" : !llvm.ptr
    %1 = llvm.mlir.addressof @reduction.omp_outlined.omp.reduction.reduction_func : !llvm.ptr
    %c8_i64 = arith.constant 8 : i64
    %2 = llvm.mlir.addressof @mlir.llvm.nameless_global_2 : !llvm.ptr
    %c1_i64 = arith.constant 1 : i64
    %c34_i32 = arith.constant 34 : i32
    %3 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %4 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %9 = llvm.alloca %c1_i32 x !llvm.array<1 x ptr> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %10 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?xi32>
    %11 = memref.load %10[%c0] : memref<?xi32>
    %12 = arith.cmpi sgt, %11, %c0_i32 : i32
    scf.if %12 {
      %13 = arith.addi %11, %c-1_i32 : i32
      %14 = "enzymexla.pointer2memref"(%4) : (!llvm.ptr) -> memref<1xi32>
      memref.store %c0_i32, %14[%c0] : memref<1xi32>
      %15 = "enzymexla.pointer2memref"(%5) : (!llvm.ptr) -> memref<1xi32>
      memref.store %13, %15[%c0] : memref<1xi32>
      %16 = "enzymexla.pointer2memref"(%6) : (!llvm.ptr) -> memref<1xi32>
      memref.store %c1_i32, %16[%c0] : memref<1xi32>
      %17 = "enzymexla.pointer2memref"(%7) : (!llvm.ptr) -> memref<1xi32>
      memref.store %c0_i32, %17[%c0] : memref<1xi32>
      %18 = "enzymexla.pointer2memref"(%8) : (!llvm.ptr) -> memref<1xi32>
      memref.store %c0_i32, %18[%c0] : memref<1xi32>
      %19 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
      %20 = memref.load %19[%c0] : memref<?xi32>
      llvm.call @__kmpc_for_static_init_4(%3, %20, %c34_i32, %7, %4, %5, %6, %c1_i32, %c1_i32) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> ()
      %21 = "enzymexla.pointer2memref"(%5) : (!llvm.ptr) -> memref<1xi32>
      %22 = memref.load %21[%c0] : memref<1xi32>
      %23 = arith.minsi %22, %13 : i32
      %24 = "enzymexla.pointer2memref"(%5) : (!llvm.ptr) -> memref<1xi32>
      memref.store %23, %24[%c0] : memref<1xi32>
      %25 = "enzymexla.pointer2memref"(%4) : (!llvm.ptr) -> memref<1xi32>
      %26 = memref.load %25[%c0] : memref<1xi32>
      %27 = arith.cmpi sle, %26, %23 : i32
      scf.if %27 {
        %32 = "enzymexla.pointer2memref"(%8) : (!llvm.ptr) -> memref<1xi32>
        %33 = memref.load %32[%c0] : memref<1xi32>
        %34 = "enzymexla.pointer2memref"(%arg4) : (!llvm.ptr) -> memref<?x!llvm.ptr>
        %35 = memref.load %34[%c0] : memref<?x!llvm.ptr>
        %36 = arith.extsi %26 : i32 to i64
        %37 = arith.addi %23, %c1_i32 : i32
        %38:2 = scf.while (%arg5 = %36, %arg6 = %33) : (i64, i32) -> (i64, i32) {
          %39 = arith.index_cast %arg5 : i64 to index
          %40 = "enzymexla.pointer2memref"(%35) : (!llvm.ptr) -> memref<?xi32>
          %41 = memref.load %40[%39] {alignment = 4 : i64, ordering = 0 : i64} : memref<?xi32>
          %42 = arith.addi %arg6, %41 : i32
          %43 = "enzymexla.pointer2memref"(%8) : (!llvm.ptr) -> memref<1xi32>
          memref.store %42, %43[%c0] : memref<1xi32>
          %44 = arith.addi %arg5, %c1_i64 : i64
          %45 = arith.trunci %44 : i64 to i32
          %46 = arith.cmpi ne, %37, %45 : i32
          scf.condition(%46) %44, %42 : i64, i32
        } do {
        ^bb0(%arg5: i64, %arg6: i32):
          scf.yield %arg5, %arg6 : i64, i32
        }
      }
      llvm.call @__kmpc_for_static_fini(%3, %20) : (!llvm.ptr, i32) -> ()
      %28 = "enzymexla.pointer2memref"(%9) : (!llvm.ptr) -> memref<1x!llvm.ptr>
      memref.store %8, %28[%c0] : memref<1x!llvm.ptr>
      %29 = llvm.call @__kmpc_reduce_nowait(%2, %20, %c1_i32, %c8_i64, %9, %1, %0) {convergent, no_unwind} : (!llvm.ptr, i32, i32, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
      %30 = arith.index_castui %29 : i32 to index
      %31 = arith.cmpi eq, %30, %c1 : index
      scf.if %31 {
        %32 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xi32>
        %33 = memref.load %32[%c0] : memref<?xi32>
        %34 = "enzymexla.pointer2memref"(%8) : (!llvm.ptr) -> memref<1xi32>
        %35 = memref.load %34[%c0] : memref<1xi32>
        %36 = arith.addi %35, %33 : i32
        %37 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xi32>
        memref.store %36, %37[%c0] : memref<?xi32>
        llvm.call @__kmpc_end_reduce_nowait(%2, %20, %0) {convergent, no_unwind} : (!llvm.ptr, i32, !llvm.ptr) -> ()
      } else {
        %32 = arith.cmpi eq, %30, %c2 : index
        scf.if %32 {
          %33 = "enzymexla.pointer2memref"(%8) : (!llvm.ptr) -> memref<1xi32>
          %34 = memref.load %33[%c0] : memref<1xi32>
          %35 = llvm.atomicrmw add %arg3, %34 monotonic {alignment = 4 : i64} : !llvm.ptr, i32
        }
      }
    }
    llvm.return
  }
  llvm.func @__kmpc_for_static_init_4(!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_for_static_fini(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func internal @reduction.omp_outlined.omp.reduction.reduction_func(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0 = arith.constant 0 : index
    %0 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %1 = memref.load %0[%c0] : memref<?x!llvm.ptr>
    %2 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %3 = memref.load %2[%c0] : memref<?x!llvm.ptr>
    %4 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<?xi32>
    %5 = memref.load %4[%c0] : memref<?xi32>
    %6 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi32>
    %7 = memref.load %6[%c0] : memref<?xi32>
    %8 = arith.addi %7, %5 : i32
    %9 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<?xi32>
    memref.store %8, %9[%c0] : memref<?xi32>
    llvm.return
  }
  llvm.func @__kmpc_reduce_nowait(!llvm.ptr, i32, i32, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
  llvm.func @__kmpc_end_reduce_nowait(!llvm.ptr, i32, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @reduction(
// CHECK: omp.parallel reduction(byref @omp_red_add_i32 %{{.*}} -> %{{.*}} : !llvm.ptr) {
// CHECK:    omp.wsloop nowait schedule(static = %{{.*}} : i32) {
// CHECK:      omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
// CHECK:        omp.yield
// CHECK:  omp.terminator
