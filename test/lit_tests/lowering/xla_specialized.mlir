// RUN: enzymexlamlir-opt %s --split-input-file --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s

// The trailing num_specialized inputs are scalars the runtime jits in as
// compile-time constants: they go in their own i64 array, and the call
// carries their count so the executable cache keys on their values.
module {
  func.func @use(%n: i64) {
    %memref = gpu.alloc () : memref<16xf64, 1>
    enzymexla.xla_wrapper @raised (%memref, %n) {num_specialized = 1 : i64} : (memref<16xf64, 1>, i64) -> ()
    return
  }
  func.func private @raised(%arg0: memref<16xf64, 1>, %n: tensor<i64>) {
    return
  }
}

// CHECK-LABEL: @use
// CHECK-DAG: %[[ARGS:.+]] = llvm.alloca %{{.+}} x !llvm.array<1 x i64>
// CHECK-DAG: %[[CONSTS:.+]] = llvm.alloca %{{.+}} x !llvm.array<1 x i64>
// CHECK: llvm.call @reactantXLAExec(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr) -> ()

// -----

// Without specialized scalars the same entry point takes a zero count and a
// null constant pointer.
module {
  func.func @plain() {
    %memref = gpu.alloc () : memref<16xf64, 1>
    enzymexla.xla_wrapper @raised2 (%memref) : (memref<16xf64, 1>) -> ()
    return
  }
  func.func private @raised2(%arg0: memref<16xf64, 1>) {
    return
  }
}

// CHECK-LABEL: @plain
// CHECK-NOT: llvm.call @reactantXLAExec(
// CHECK-DAG: %[[NULL:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG: %[[ZERO:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: llvm.call @reactantXLAExec(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[ZERO]], %[[NULL]])
