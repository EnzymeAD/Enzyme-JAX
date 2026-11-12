// RUN: enzymexlamlir-opt --llvm-to-omp %s | FileCheck %s --implicit-check-not="llvm.call @__kmpc"

// `#pragma omp barrier` becomes omp.barrier.
module {
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_0(";unknown;unknown;0;0;;\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global private unnamed_addr constant @mlir.llvm.nameless_global_1() {addr_space = 0 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> {
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_0 : !llvm.ptr
    %c22_i32 = arith.constant 22 : i32
    %c0_i32 = arith.constant 0 : i32
    %c34_i32 = arith.constant 34 : i32
    %1 = llvm.mlir.undef : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)>
    %2 = llvm.insertvalue %c0_i32, %1[0] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
    %3 = llvm.insertvalue %c34_i32, %2[1] : !llvm.struct<"struct.ident_t.1", (i32, i32, i32, i32, ptr)> 
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
  llvm.func @barrier(%arg0: !llvm.ptr, %arg1: i32) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_2 : !llvm.ptr
    %c2_i32 = arith.constant 2 : i32
    %1 = llvm.mlir.addressof @barrier.omp_outlined : !llvm.ptr
    %2 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %c1_i32 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<1x!llvm.ptr>
    memref.store %arg0, %4[%c0] : memref<1x!llvm.ptr>
    %5 = "enzymexla.pointer2memref"(%3) : (!llvm.ptr) -> memref<1xi32>
    memref.store %arg1, %5[%c0] : memref<1xi32>
    llvm.call @__kmpc_fork_call(%0, %c2_i32, %1, %2, %3) vararg(!llvm.func<void (ptr, i32, ptr, ...)>) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
  llvm.func internal @barrier.omp_outlined(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {dso_local, sym_visibility = "private"} {
    %c0 = arith.constant 0 : index
    %0 = llvm.mlir.addressof @mlir.llvm.nameless_global_1 : !llvm.ptr
    %1 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %2 = memref.load %1[%c0] : memref<?x!llvm.ptr>
    %3 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xi32>
    %4 = memref.load %3[%c0] : memref<?xi32>
    llvm.call tail @body(%2, %4) : (!llvm.ptr, i32) -> ()
    %5 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xi32>
    %6 = memref.load %5[%c0] : memref<?xi32>
    llvm.call tail @__kmpc_barrier(%0, %6) {convergent, no_unwind} : (!llvm.ptr, i32) -> ()
    %7 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %8 = memref.load %7[%c0] : memref<?x!llvm.ptr>
    %9 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr) -> memref<?xi32>
    %10 = memref.load %9[%c0] : memref<?xi32>
    llvm.call tail @body(%8, %10) : (!llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @body(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_barrier(!llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @__kmpc_fork_call(!llvm.ptr, i32, !llvm.ptr, ...) attributes {sym_visibility = "private"}
}

// CHECK-LABEL: llvm.func @barrier(
// CHECK: omp.parallel {
// CHECK:  llvm.call tail @body(%{{.*}}, %{{.*}}) : (!llvm.ptr, i32) -> ()
// CHECK:  omp.barrier
// CHECK:  llvm.call tail @body(%{{.*}}, %{{.*}}) : (!llvm.ptr, i32) -> ()
// CHECK:  omp.terminator
