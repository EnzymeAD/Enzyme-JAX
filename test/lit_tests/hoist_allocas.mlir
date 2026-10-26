// RUN: enzymexlamlir-opt %s -hoist-allocas -split-input-file | FileCheck %s

// An scf.if frees nothing, so what is allocated inside it lives as long as the
// function does and belongs where the function frees it.
func.func @out_of_a_conditional(%c: i1, %v: f32) -> f32 {
  %z = arith.constant 0.0 : f32
  %r = scf.if %c -> f32 {
    %i = arith.constant 0 : index
    %m = memref.alloca() : memref<4xf32>
    memref.store %v, %m[%i] : memref<4xf32>
    %l = memref.load %m[%i] : memref<4xf32>
    scf.yield %l : f32
  } else {
    scf.yield %z : f32
  }
  return %r : f32
}

// CHECK-LABEL: func.func @out_of_a_conditional(
// CHECK: memref.alloca
// CHECK: scf.if

// -----

// The same for an allocation named in the LLVM dialect.
func.func @llvm_out_of_a_conditional(%c: i1, %v: f32) -> f32 {
  %z = arith.constant 0.0 : f32
  %one = llvm.mlir.constant(1 : i32) : i32
  %r = scf.if %c -> f32 {
    %p = llvm.alloca %one x !llvm.array<4 x f32> : (i32) -> !llvm.ptr
    llvm.store %v, %p : f32, !llvm.ptr
    %l = llvm.load %p : !llvm.ptr -> f32
    scf.yield %l : f32
  } else {
    scf.yield %z : f32
  }
  return %r : f32
}

// CHECK-LABEL: func.func @llvm_out_of_a_conditional(
// CHECK: llvm.alloca
// CHECK: scf.if

// -----

// An scf.for frees what its body allocated on every iteration, so an allocation
// in it stays in it -- at the start of the body, which is where that freeing
// begins again.
func.func @stays_within_a_loop(%n: index, %v: f32, %c: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    scf.if %c {
      %m = memref.alloca() : memref<4xf32>
      memref.store %v, %m[%c0] : memref<4xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @stays_within_a_loop(
// CHECK: scf.for
// CHECK: memref.alloca
// CHECK: scf.if

// -----

// An allocation whose size is not fixed would take that much stack even where
// it was not reached, so it stays where it is.
func.func @keeps_a_dynamic_size(%c: i1, %v: f32, %n: i32) -> f32 {
  %z = arith.constant 0.0 : f32
  %r = scf.if %c -> f32 {
    %p = llvm.alloca %n x f32 : (i32) -> !llvm.ptr
    llvm.store %v, %p : f32, !llvm.ptr
    %l = llvm.load %p : !llvm.ptr -> f32
    scf.yield %l : f32
  } else {
    scf.yield %z : f32
  }
  return %r : f32
}

// CHECK-LABEL: func.func @keeps_a_dynamic_size(
// CHECK: scf.if
// CHECK: llvm.alloca
