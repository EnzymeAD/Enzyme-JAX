// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=32 enzymexlamlir-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

func.func @degenerate_leading_dim(%gridx: index, %gridy: index, %out: memref<?xf64, 1>, %v: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  "enzymexla.gpu_wrapper"(%gridx, %gridy, %c1, %c32, %c1, %c1) ({
    %fused = arith.muli %gridx, %c32 : index
    scf.parallel (%i, %j) = (%c0, %c0) to (%gridy, %fused) step (%c1, %c1) {
      memref.store %v, %out[%j] : memref<?xf64, 1>
      scf.reduce
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL: func.func @degenerate_leading_dim(
// CHECK-SAME: %{{[^ :]+}}: index, %[[GRIDY:[^ :]+]]: index
// CHECK: %[[BLOCKS:.+]] = arith.addi
// CHECK: gpu.launch blocks({{.*}}) in (%{{[^ ]+}} = %[[BLOCKS]], %{{[^ ]+}} = %[[GRIDY]], %{{[^ ]+}} = %c1)
// CHECK: gpu.block_id x

// -----

// A bound narrowed by a min against the original grid-y dimension is still
// safe outside x. The derived bound is a different SSA value, so this also
// checks that launch-bound matching is not limited to value identity.

func.func @min_narrowed_y_bound(%gridx: index, %gridy: index, %n: index, %out: memref<?xf64, 1>, %v: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %safe_y = arith.minsi %n, %gridy : index
  "enzymexla.gpu_wrapper"(%gridx, %gridy, %c1, %c32, %c1, %c1) ({
    %fused = arith.muli %gridx, %c32 : index
    scf.parallel (%i, %j) = (%c0, %c0) to (%safe_y, %fused) step (%c1, %c1) {
      %idx = arith.addi %i, %j : index
      memref.store %v, %out[%idx] : memref<?xf64, 1>
      scf.reduce
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL: func.func @min_narrowed_y_bound(
// CHECK-SAME: %{{[^ :]+}}: index, %[[GRIDY:[^ :]+]]: index, %[[N:[^ :]+]]: index
// CHECK: %[[SAFE_Y:.+]] = arith.minsi %[[N]], %[[GRIDY]]
// CHECK: %[[BLOCKS:.+]] = arith.addi
// CHECK: gpu.launch blocks({{.*}}) in (%{{[^ ]+}} = %[[BLOCKS]], %{{[^ ]+}} = %[[SAFE_Y]], %{{[^ ]+}} = %c1)
// CHECK-DAG: gpu.block_id y
// CHECK-DAG: gpu.block_id x

// -----

// The constant half of the same test, both ways at once: 1000 fits and sinks
// to y, and 2^32 + 1000 does not and takes x. Read as an int rather than an
// APInt the wide one looks like 1000, and the two come out the other way up.

func.func @constant_bounds(%out: memref<?xf64, 1>, %v: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c1000 = arith.constant 1000 : index
  %big = arith.constant 4294968296 : index
  "enzymexla.gpu_wrapper"(%c1000, %c1, %c1, %c32, %c1, %c1) ({
    scf.parallel (%i, %j, %k) = (%c0, %c0, %c0) to (%c1000, %big, %c32) step (%c1, %c1, %c1) {
      memref.store %v, %out[%j] : memref<?xf64, 1>
      scf.reduce
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL: func.func @constant_bounds(
// CHECK-DAG: %[[C1000:.+]] = arith.constant 1000 : index
// CHECK-DAG: %[[BIG:.+]] = arith.constant 4294968296 : index
// CHECK: gpu.launch blocks({{.*}}) in (%{{[^ ]+}} = %[[BIG]], %{{[^ ]+}} = %[[C1000]], %{{[^ ]+}} = %c1)

// -----

// Only one dimension can be given x, and which of two unknown ones carries the
// work is not something the loop order says. Preserve their order; their
// runtime values determine whether the launch is valid.

func.func @two_unbounded(%n: index, %m: index, %out: memref<?xf64, 1>, %v: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  "enzymexla.gpu_wrapper"(%n, %c1, %c1, %c32, %c1, %c1) ({
    scf.parallel (%i, %j, %k) = (%c0, %c0, %c0) to (%n, %m, %c32) step (%c1, %c1, %c1) {
      memref.store %v, %out[%i] : memref<?xf64, 1>
      scf.reduce
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK-LABEL: func.func @two_unbounded(
// CHECK-SAME: %[[N:[^ :]+]]: index, %[[M:[^ :]+]]: index
// CHECK: gpu.launch blocks({{.*}}) in (%{{[^ ]+}} = %[[N]], %{{[^ ]+}} = %[[M]], %{{[^ ]+}} = %c1)
