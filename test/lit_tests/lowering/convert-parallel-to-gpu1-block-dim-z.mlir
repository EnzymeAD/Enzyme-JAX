// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=512 enzymexlamlir-opt %s --split-input-file --pass-pipeline="builtin.module(canonicalize-parallel,convert-parallel-to-gpu1)" | FileCheck %s --check-prefix=BS512
// RUN: env POLYGEIST_GPU_KERNEL_BLOCK_SIZE=1024 enzymexlamlir-opt %s --split-input-file --pass-pipeline="builtin.module(canonicalize-parallel,convert-parallel-to-gpu1)" | FileCheck %s --check-prefix=BS1024

// blockDim.x and blockDim.y hold 1024 threads, blockDim.z stops at 64, and
// dimensions become x, y and z by position. createSplitOp builds the block by
// inserting at the front, so the dimension the launch had on blockDim.x is
// pushed toward z by every dimension taken in ahead of it.

// <<<dim3(2,2), 128>>>: at a requested 512 the whole grid is taken into the
// block, 2*2*128 filling it exactly, and 128 lands on z where it cannot
// launch. At 1024 the same thing happens with budget to spare.

func.func @absorbed_into_block(%out: memref<?xf64, 1>, %v: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : index
  "enzymexla.gpu_wrapper"(%c2, %c2, %c1, %c128, %c1, %c1) ({
    scf.parallel (%bx, %by) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      scf.parallel (%tx) = (%c0) to (%c128) step (%c1) {
        %i = arith.addi %bx, %by : index
        %j = arith.addi %i, %tx : index
        memref.store %v, %out[%j] : memref<?xf64, 1>
        scf.reduce
      }
      scf.reduce
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// BS512-LABEL:  func.func @absorbed_into_block(
// BS512:        gpu.launch blocks({{.*}}) in ({{.*}}) threads({{.*}}) in (%{{[^ ]+}} = %c128, %{{[^ ]+}} = %c2, %{{[^ ]+}} = %c2)
// BS512:        gpu.thread_id x
// BS1024-LABEL: func.func @absorbed_into_block(
// BS1024:       gpu.launch blocks({{.*}}) in ({{.*}}) threads({{.*}}) in (%{{[^ ]+}} = %c128, %{{[^ ]+}} = %c2, %{{[^ ]+}} = %c2)

// -----

// The other way a dimension reaches z: the split prepends the dimension it
// creates, shifting everything already in the block one place along. Only at
// 1024 is there room for the split to fire.

func.func @split_prepended(%out: memref<?xf64, 1>, %v: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  "enzymexla.gpu_wrapper"(%c4, %c4, %c1, %c128, %c1, %c1) ({
    scf.parallel (%bx, %by) = (%c0, %c0) to (%c4, %c4) step (%c1, %c1) {
      scf.parallel (%tx) = (%c0) to (%c128) step (%c1) {
        %i = arith.addi %bx, %by : index
        %j = arith.addi %i, %tx : index
        memref.store %v, %out[%j] : memref<?xf64, 1>
        scf.reduce
      }
      scf.reduce
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// BS512-LABEL:  func.func @split_prepended(
// BS512:        gpu.launch blocks({{.*}}) in ({{.*}}) threads({{.*}}) in (%{{[^ ]+}} = %c4, %{{[^ ]+}} = %c128, %{{[^ ]+}} = %c1)
// BS1024-LABEL: func.func @split_prepended(
// BS1024:       gpu.launch blocks({{.*}}) in ({{.*}}) threads({{.*}}) in (%{{[^ ]+}} = %c128, %{{[^ ]+}} = %c4, %{{[^ ]+}} = %c2)
// BS1024:       gpu.thread_id x

// -----

// A block that fits is left exactly where it was, displaced blockDim.x and
// all: 8 threads on z is legal, so nothing here is this pass's business.

func.func @already_fits(%out: memref<?xf64, 1>, %v: f64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  "enzymexla.gpu_wrapper"(%c2, %c2, %c1, %c8, %c1, %c1) ({
    scf.parallel (%bx, %by) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      scf.parallel (%tx) = (%c0) to (%c8) step (%c1) {
        %i = arith.addi %bx, %by : index
        %j = arith.addi %i, %tx : index
        memref.store %v, %out[%j] : memref<?xf64, 1>
        scf.reduce
      }
      scf.reduce
    }
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// BS512-LABEL:  func.func @already_fits(
// BS512:        gpu.launch blocks({{.*}}) in ({{.*}}) threads({{.*}}) in (%{{[^ ]+}} = %c2, %{{[^ ]+}} = %c2, %{{[^ ]+}} = %c8)
// BS1024-LABEL: func.func @already_fits(
// BS1024:       gpu.launch blocks({{.*}}) in ({{.*}}) threads({{.*}}) in (%{{[^ ]+}} = %c2, %{{[^ ]+}} = %c2, %{{[^ ]+}} = %c8)
