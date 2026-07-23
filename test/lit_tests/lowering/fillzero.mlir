// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm=backend=cuda | FileCheck %s

func.func @fillzero(%x: memref<5x6xf32>) {
  enzyme.fill_zero %x : memref<5x6xf32>
  return
}

// CHECK-LABEL:   llvm.func @fillzero(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr) {
// CHECK:           %[[ZERO:.*]] = llvm.mlir.constant(0 : i8) : i8
// CHECK:           %[[SIZE:.*]] = llvm.mlir.constant(120 : i64) : i64
// CHECK:           "llvm.intr.memset"(%[[ARG0]], %[[ZERO]], %[[SIZE]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK:           llvm.return
// CHECK:         }
