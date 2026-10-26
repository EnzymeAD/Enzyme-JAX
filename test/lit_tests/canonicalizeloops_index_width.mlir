// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// These rewrites read an op on integers one of whose operands came from an
// index and say the op again in index. The width they check against is the
// width of the integer the index was cast to. Where the op is itself in index
// the cast runs the other way, and index has no width to be asked for -- asking
// read it as a float and took the crash that follows.

func.func @addi_index(%n: i64) -> index {
  %c8 = arith.constant 8 : i64
  %c4 = arith.constant 4 : index
  %r = arith.remui %n, %c8 : i64
  %x = arith.index_castui %r : i64 to index
  %a = arith.addi %x, %c4 : index
  return %a : index
}

// CHECK-LABEL: func.func @addi_index
// CHECK:         %[[X:.+]] = arith.index_castui
// CHECK-NEXT:    arith.addi %[[X]]

// -----

func.func @subi_index(%n: i64) -> index {
  %c8 = arith.constant 8 : i64
  %c0 = arith.constant 0 : index
  %r = arith.remui %n, %c8 : i64
  %x = arith.index_castui %r : i64 to index
  %a = arith.subi %c0, %x : index
  return %a : index
}

// CHECK-LABEL: func.func @subi_index
// CHECK:         arith.subi

// -----

func.func @muli_index(%n: i64) -> index {
  %c8 = arith.constant 8 : i64
  %c4 = arith.constant 4 : index
  %r = arith.remui %n, %c8 : i64
  %x = arith.index_castui %r : i64 to index
  %a = arith.muli %x, %c4 : index
  return %a : index
}

// CHECK-LABEL: func.func @muli_index
// CHECK:         arith.muli

// -----

func.func @shli_index(%n: i64) -> index {
  %c8 = arith.constant 8 : i64
  %c4 = arith.constant 4 : index
  %r = arith.remui %n, %c8 : i64
  %x = arith.index_castui %r : i64 to index
  %a = arith.shli %x, %c4 : index
  return %a : index
}

// CHECK-LABEL: func.func @shli_index
// CHECK:         arith.shli

// -----

func.func @shrui_index(%n: i64) -> index {
  %c8 = arith.constant 8 : i64
  %c4 = arith.constant 4 : index
  %r = arith.remui %n, %c8 : i64
  %x = arith.index_castui %r : i64 to index
  %a = arith.shrui %x, %c4 : index
  return %a : index
}

// CHECK-LABEL: func.func @shrui_index
// CHECK:         arith.shrui

// -----

func.func @divui_index(%n: i64) -> index {
  %c8 = arith.constant 8 : i64
  %c4 = arith.constant 4 : index
  %r = arith.remui %n, %c8 : i64
  %x = arith.index_castui %r : i64 to index
  %a = arith.divui %x, %c4 : index
  return %a : index
}

// CHECK-LABEL: func.func @divui_index
// CHECK:         arith.divui

// -----

// An op on integers over an index that fits still moves into index.

func.func @addi_integer(%m: memref<?xi64>) {
  affine.parallel (%i) = (0) to (16) {
    %c4 = arith.constant 4 : i64
    %x = arith.index_castui %i : index to i64
    %a = arith.addi %x, %c4 : i64
    memref.store %a, %m[%i] : memref<?xi64>
  }
  return
}

// CHECK-LABEL: func.func @addi_integer
// CHECK:         %[[C4:.+]] = arith.constant 4 : index
// CHECK:         affine.parallel (%[[I:.+]]) = (0) to (16)
// CHECK-NEXT:      %[[A:.+]] = arith.addi %[[I]], %[[C4]] : index
// CHECK-NEXT:      %[[C:.+]] = arith.index_castui %[[A]] : index to i64
// CHECK-NEXT:      affine.store %[[C]]
