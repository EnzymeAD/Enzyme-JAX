// RUN: enzymexlamlir-opt %s -lift-tessera-annotations | FileCheck %s

module {
  llvm.mlir.global internal @__tessera_optimize_rule_0(0 : i32) {addr_space = 0 : i32, alignment = 4 : i64, dso_local} : i32
  llvm.mlir.global private unnamed_addr constant @".str"("tessera_optimize=eigen.inv(eigen.inv(x)) -> x\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("<invalid loc>\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("tessera_op=eigen.inv\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @".str.3"("/home/jessicacotturone21/Reactant/enzyme/pragma_test.c\00") {addr_space = 0 : i32, dso_local, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(4 : i32) : i32
    %2 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %3 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %4 = llvm.mlir.addressof @inverse : !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %8 = llvm.insertvalue %2, %7[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %9 = llvm.insertvalue %1, %8[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %10 = llvm.insertvalue %0, %9[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %13 = llvm.mlir.addressof @".str" : !llvm.ptr
    %14 = llvm.mlir.addressof @__tessera_optimize_rule_0 : !llvm.ptr
    %15 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %17 = llvm.insertvalue %13, %16[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %18 = llvm.insertvalue %12, %17[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %19 = llvm.insertvalue %11, %18[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %20 = llvm.insertvalue %0, %19[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %21 = llvm.mlir.undef : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %23 = llvm.insertvalue %10, %22[1] : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
    llvm.return %23 : !llvm.array<2 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }

  // CHECK: llvm.func @inverse
  // CHECK-SAME: tessera_op = "eigen.inv"
  llvm.func @inverse(%arg0 : f32) -> f32 {
    llvm.return %arg0 : f32
  }

  // CHECK-LABEL: llvm.func @main
  llvm.func @main(%x : f32) -> f32 {
    %0 = llvm.call @inverse(%x) : (f32) -> f32
    %1 = llvm.call @inverse(%0) : (f32) -> f32
    llvm.return %1 : f32
  }
}

// CHECK: tessera.optimizations
// CHECK-NEXT: tessera.optimization "eigen.inv(eigen.inv(x)) -> x"
