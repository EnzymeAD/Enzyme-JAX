// RUN: enzymexlamlir-opt --affine-cfg="enable_split_on_affine_if_constants=true" --split-input-file %s | FileCheck %s

// An affine.if over a dimension yielding constants is a select no affine
// expression writes (over a dimension of unknown extent; over a constant
// range it is written as an indicator instead): a loop bounded by its
// result splits on the conditional, and each copy raises with the constant
// in place.
func.func @split_for(%d: i32, %out: memref<?xi32>, %v: i32, %n: index) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cm1_i32 = arith.constant -1 : i32
  affine.parallel (%t, %c) = (0, 0) to (8, symbol(%n)) {
    %e = affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%c) -> i32 {
      affine.yield %cm1_i32 : i32
    } else {
      affine.yield %c0_i32 : i32
    }
    %ub = arith.addi %d, %e : i32
    scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
      %ii = arith.index_cast %i : i32 to index
      memref.store %v, %out[%ii] : memref<?xi32>
    }
  }
  return
}

// CHECK:  func.func @split_for(%arg0: i32, %arg1: memref<?xi32>, %arg2: i32, %arg3: index) {
// CHECK-NEXT:   %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:   %0 = arith.addi %arg0, %c-1_i32 : i32
// CHECK-NEXT:   %1 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:   %2 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   affine.parallel (%arg4, %arg5) = (0, 0) to (8, symbol(%arg3)) {
// CHECK-NEXT:     affine.if #set(%arg5) {
// CHECK-NEXT:       affine.parallel (%arg6) = (0) to (symbol(%1)) {
// CHECK-NEXT:         affine.store %arg2, %arg1[%arg6] : memref<?xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:     } else {
// CHECK-NEXT:       affine.parallel (%arg6) = (0) to (symbol(%2)) {
// CHECK-NEXT:         affine.store %arg2, %arg1[%arg6] : memref<?xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// A guard whose condition derives from the conditional splits the same way.
func.func @split_if(%d: i32, %out: memref<?xi32>, %v: i32, %n: index) {
  %c0_i32 = arith.constant 0 : i32
  %cm1_i32 = arith.constant -1 : i32
  affine.parallel (%t, %c) = (0, 0) to (8, symbol(%n)) {
    %e = affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%c) -> i32 {
      affine.yield %cm1_i32 : i32
    } else {
      affine.yield %c0_i32 : i32
    }
    %m = arith.addi %d, %e : i32
    %ti = arith.index_cast %t : index to i32
    %q = arith.cmpi slt, %ti, %m : i32
    scf.if %q {
      memref.store %v, %out[%t] : memref<?xi32>
    }
  }
  return
}

// -----

// A conditional over symbols is a symbol itself and needs no split.
func.func @symbol_keep(%d: i32, %k: index, %out: memref<?xi32>, %v: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cm1_i32 = arith.constant -1 : i32
  affine.parallel (%t) = (0) to (8) {
    %e = affine.if affine_set<()[s0] : (s0 - 1 >= 0)>()[%k] -> i32 {
      affine.yield %cm1_i32 : i32
    } else {
      affine.yield %c0_i32 : i32
    }
    %ub = arith.addi %d, %e : i32
    scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
      %ii = arith.index_cast %i : i32 to index
      memref.store %v, %out[%ii] : memref<?xi32>
    }
  }
  return
}

// CHECK:  func.func @symbol_keep(%arg0: i32, %arg1: index, %arg2: memref<?xi32>, %arg3: i32) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:   %0 = affine.if #set()[%arg1] -> i32 {
// CHECK-NEXT:     affine.yield %c-1_i32 : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     affine.yield %c0_i32 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   %1 = arith.addi %arg0, %0 : i32
// CHECK-NEXT:   %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:   affine.parallel (%arg4, %arg5) = (0, 0) to (8, symbol(%2)) {
// CHECK-NEXT:     affine.store %arg3, %arg2[%arg5] : memref<?xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// A guard yielding a pointer stays: only scalar results select through an
// affine.if.
func.func @pointer_keep(%d: i32, %p: !llvm.ptr, %q: !llvm.ptr, %slot: !llvm.ptr) {
  %c0_i32 = arith.constant 0 : i32
  %cm1_i32 = arith.constant -1 : i32
  affine.parallel (%t, %c) = (0, 0) to (8, 2) {
    %e = affine.if affine_set<(d0) : (d0 - 1 >= 0)>(%c) -> i32 {
      affine.yield %cm1_i32 : i32
    } else {
      affine.yield %c0_i32 : i32
    }
    %n = arith.addi %d, %e : i32
    %ti = arith.index_cast %t : index to i32
    %cmp = arith.cmpi slt, %ti, %n : i32
    %s = scf.if %cmp -> !llvm.ptr {
      scf.yield %p : !llvm.ptr
    } else {
      scf.yield %q : !llvm.ptr
    }
    llvm.store %s, %slot : !llvm.ptr, !llvm.ptr
  }
  return
}

// CHECK:  func.func @pointer_keep(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) {
// CHECK-NEXT:   %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   affine.parallel (%arg4, %arg5) = (0, 0) to (8, 2) {
// CHECK-NEXT:     %1 = affine.if #set(%arg4, %arg5)[%0] -> !llvm.ptr {
// CHECK-NEXT:       affine.yield %arg1 : !llvm.ptr
// CHECK-NEXT:     } else {
// CHECK-NEXT:       affine.yield %arg2 : !llvm.ptr
// CHECK-NEXT:     }
// CHECK-NEXT:     llvm.store %1, %arg3 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
