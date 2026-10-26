// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s

// Three launch sites, two structurally identical kernels under different
// names: the embedded module is printed under a fixed function name and
// keyed by content, so exactly one copy is emitted and every site loads it.
module {
  llvm.func @caller() {
    %memref = gpu.alloc () : memref<4xf64, 1>
    %memref_1 = gpu.alloc () : memref<4xf64, 1>
    enzymexla.xla_wrapper @k1 (%memref, %memref_1) : (memref<4xf64, 1>, memref<4xf64, 1>) -> ()
    enzymexla.xla_wrapper @k1 (%memref_1, %memref) : (memref<4xf64, 1>, memref<4xf64, 1>) -> ()
    enzymexla.xla_wrapper @k2 (%memref, %memref_1) : (memref<4xf64, 1>, memref<4xf64, 1>) -> ()
    llvm.return
  }
  func.func private @k1(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf64>
    return %arg0, %0 : tensor<4xf64>, tensor<4xf64>
  }
  func.func private @k2(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf64>
    return %arg0, %0 : tensor<4xf64>, tensor<4xf64>
  }
}

// Exactly one embedded module, printed under the fixed name; its single
// materialized address serves all three launch sites.
// CHECK-COUNT-1: llvm.mlir.global internal constant @[[MOD:xlamod\$[0-9a-f]+]]("func.func private @reactant_kernel
// CHECK-NOT: llvm.mlir.global internal constant @xlamod
// CHECK: llvm.func @caller()
// CHECK: llvm.mlir.addressof @[[MOD]] : !llvm.ptr
// CHECK-NOT: llvm.mlir.addressof @xlamod
// CHECK-COUNT-3: llvm.call @reactantXLAExec(
// CHECK-NOT: llvm.call @reactantXLAExec(
