// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm=backend=cuda | FileCheck %s

func.func @atomic_rmw(%idx: index, %x: f32, %mem: memref<?xf32>) {
  enzyme.atomic_rmw addf %x, %mem[%idx] monotonic {alignment = 4 : i64} : (f32, memref<?xf32>) -> f32
  return
}

// CHECK-LABEL:   llvm.func @atomic_rmw(
// CHECK-SAME:      %[[ARG0:.*]]: i64,
// CHECK-SAME:      %[[ARG1:.*]]: f32,
// CHECK-SAME:      %[[ARG2:.*]]: !llvm.ptr) {
// CHECK:           %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[ARG2]]{{\[}}%[[ARG0]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           %[[ATOMICRMW_0:.*]] = llvm.atomicrmw fadd %[[GETELEMENTPTR_0]], %[[ARG1]] monotonic {alignment = 4 : i64} : !llvm.ptr, f32
// CHECK:           llvm.return
// CHECK:         }
