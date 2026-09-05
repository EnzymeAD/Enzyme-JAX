// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `#pragma omp parallel sections`: clang lowers sections onto the static
// worksharing-loop runtime, so this raises to an omp.wsloop over the section index.
module {
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_0(";unknown;unknown;0;0;;\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_1() {addr_space = 0 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %c22_i32 = arith.constant 22 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1026_i32 = arith.constant 1026 : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
    %2 = llvm.insertvalue %c0_i32, %1[0] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %3 = llvm.insertvalue %c1026_i32, %2[1] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %4 = llvm.insertvalue %c0_i32, %3[2] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %5 = llvm.insertvalue %c22_i32, %4[3] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %6 = llvm.insertvalue %0, %5[4] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    llvm.return %6 : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
  }
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_2() {addr_space = 0 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> {
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
  llvm.func @sections(%arg0: !llvm.ptr, %arg1: i32) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_2 : !llvm.ptr
    %c2_i32 = arith.constant 2 : i32
    %1 = llvm.mlir.addressof @sections.omp_outlined : !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1x!llvm.ptr>
    memref.store %arg0, %4[%c0] : memref<1x!llvm.ptr>
    %5 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<1xi32>
    memref.store %arg1, %5[%c0] : memref<1xi32>
    llvm.call @__kmpc_fork_call(%0, %c2_i32, %1, %2, %3) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @sections.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %c34_i32 = arith.constant 34 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<1xi32>
    memref.store %c0_i32, %5[%c0] : memref<1xi32>
    %6 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi32>
    memref.store %c1_i32, %6[%c0] : memref<1xi32>
    %7 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<1xi32>
    memref.store %c1_i32, %7[%c0] : memref<1xi32>
    %8 = "enzymexla.pointer2memref"(%4) : (!llvm.ptr) -> memref<1xi32>
    memref.store %c0_i32, %8[%c0] : memref<1xi32>
    %9 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
    %10 = memref.load %9[%c0] : memref<?xi32>
    llvm.call @__kmpc_for_static_init_4(%0, %10, %c34_i32, %4, %1, %2, %3, %c1_i32, %c1_i32) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) -> ()
    %11 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi32>
    %12 = memref.load %11[%c0] : memref<1xi32>
    %13 = arith.minsi %12, %c1_i32 : i32
    %14 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi32>
    memref.store %13, %14[%c0] : memref<1xi32>
    %15 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<1xi32>
    %16 = memref.load %15[%c0] : memref<1xi32>
    %17 = arith.cmpi sle, %16, %13 : i32
    scf.if %17 {
      %18 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1xi32>
      %19 = memref.load %18[%c0] : memref<1xi32>
      %20 = arith.maxsi %19, %16 : i32
      %21 = arith.addi %20, %c1_i32 : i32
      scf.for %arg4 = %16 to %21 step %c1_i32  : i32 {
        %22 = arith.cmpi ult, %arg4, %c2_i32 : i32
        scf.if %22 {
          %23 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
          %24 = memref.load %23[%c0] : memref<?x!llvm.ptr>
          %25 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xi32>
          %26 = memref.load %25[%c0] : memref<?xi32>
          llvm.call @body(%24, %26) : (!llvm.ptr, i32) -> ()
        }
      }
    }
    llvm.call @__kmpc_for_static_fini(%0, %10) : (!llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @__kmpc_for_static_init_4(!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @body(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_for_static_fini(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @sections(
// CHECK: omp.parallel {
// CHECK:  omp.wsloop nowait schedule(static = %{{.*}} : i32) {
// CHECK:    omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
// CHECK:        llvm.call @body(%{{.*}}, %{{.*}}) : (!llvm.ptr, i32) -> ()
// CHECK:      omp.yield
// CHECK:  omp.terminator
