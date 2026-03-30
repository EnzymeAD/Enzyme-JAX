// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(parallel-lower{wrapParallelOps=true})" | FileCheck %s --check-prefix=LOWER
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(parallel-lower{wrapParallelOps=true},canonicalize)" | FileCheck %s --check-prefix=CANON
//
// Test that parallel-lower inserts a noop in the grid parallel body to prevent
// mlir general canonicalize pass MergeNestedParallelLoops 
// from merging the 3D grid + 3D block scf.parallel ops into a single 6D scf.parallel.

module {
  llvm.func @test_noop_preserves_grid_block(
      %arg0: !llvm.ptr {llvm.noundef},
      %arg1: !llvm.ptr {llvm.noundef},
      %n: i32) {
    %ni = arith.index_cast %n : i32 to index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %ni, %gy = %c1, %gz = %c1)
              threads(%tx, %ty, %tz) in (%bdx = %c32, %bdy = %c1, %bdz = %c1) {
      %0 = llvm.load %arg0 : !llvm.ptr -> f64
      llvm.store %0, %arg1 : f64, !llvm.ptr
      gpu.terminator
    }
    llvm.return
  }
}

// LOWER: enzymexla.gpu_wrapper
// LOWER:   scf.parallel
// LOWER:     "enzymexla.noop"
// LOWER:     scf.parallel
// LOWER:       scf.reduce
// LOWER:     scf.reduce

// After canonicalize, 3D+3D must NOT be merged into 6D
// CANON: enzymexla.gpu_wrapper
// CANON:   scf.parallel
// CANON:     "enzymexla.noop"
// CANON:     scf.parallel
// CANON:       scf.reduce
// CANON:     scf.reduce
