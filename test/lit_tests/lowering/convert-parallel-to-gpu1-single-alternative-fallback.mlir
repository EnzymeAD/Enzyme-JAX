// RUN: enzymexlamlir-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(convert-parallel-to-gpu1{emit_single_alternative=true})" | FileCheck %s

// Four dimensions with no exact shape and no recorded block shape: putting
// every dimension in the grid does not split, so a block size is chosen
// instead of dropping the kernel. Without a launch bound the architecture
// maximum is tried first: the constant dimensions become the block, filled up
// to 1024 threads from a grid dimension.

module {
  func.func @no_shape(%n: index, %m: index, %out: memref<?xf64>, %v: f64) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %w = "enzymexla.gpu_wrapper"(%n, %m, %c1, %c1, %c1, %c1) ({
      scf.parallel (%i, %j, %k, %l) = (%c0, %c0, %c0, %c0) to (%n, %m, %c8, %c16) step (%c1, %c1, %c1, %c1) {
        memref.store %v, %out[%i] : memref<?xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @no_shape(
// CHECK: enzymexla.alternatives
// CHECK: gpu.launch
// CHECK: alternatives.descs = ["block_size=1024,"]

// -----

// The source launch bound caps the chosen block size: the 8x16 block does not
// fit under 64 threads, so only the 8-wide dimension joins the block, filled
// up to 64 threads from a grid dimension.

module {
  func.func @bounded(%n: index, %m: index, %out: memref<?xf64>, %v: f64) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %w = "enzymexla.gpu_wrapper"(%n, %m, %c1, %c1, %c1, %c1) ({
      scf.parallel (%i, %j, %k, %l) = (%c0, %c0, %c0, %c0) to (%n, %m, %c16, %c8) step (%c1, %c1, %c1, %c1) {
        memref.store %v, %out[%i] : memref<?xf64>
        scf.reduce
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = [["nvvm.maxntid", "64"]]} : (index, index, index, index, index, index) -> index
    return
  }
}

// CHECK-LABEL: func.func @bounded(
// CHECK: enzymexla.alternatives
// CHECK: gpu.launch
// CHECK: alternatives.descs = ["block_size=64,"]
