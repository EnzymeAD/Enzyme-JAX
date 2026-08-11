// RUN: enzymexlamlir-opt --polygeist-mem2reg %s | FileCheck %s

llvm.func @use(!llvm.array<1 x i32>)

llvm.func @wrap(%v : i32) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %slot = llvm.alloca %c1 x !llvm.array<1 x i32> {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %v, %slot : i32, !llvm.ptr
  %agg = llvm.load %slot : !llvm.ptr -> !llvm.array<1 x i32>
  llvm.call @use(%agg) : (!llvm.array<1 x i32>) -> ()
  llvm.return
}

// CHECK-LABEL:   llvm.func @wrap(
// CHECK-SAME:      %[[v:.*]]: i32) {
// CHECK-NOT:       llvm.load
// CHECK:           %[[agg:.*]] = llvm.insertvalue %[[v]], %{{.*}}[0] : !llvm.array<1 x i32>
// CHECK:           llvm.call @use(%[[agg]])
