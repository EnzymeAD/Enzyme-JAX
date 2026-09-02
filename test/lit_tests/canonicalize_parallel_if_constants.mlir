// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel{parallel=false})" | FileCheck %s

// An op over what a branch chose between constants rides into the arms and
// folds there, so what the branch chose reaches its user as the branch's own
// result: a byte offset cast and scaled arrives as the element slot itself.

#set = affine_set<()[s0] : (s0 >= 1)>
module {

  func.func @cast_and_divide(%s: index) -> index {
    %c4 = arith.constant 4 : index
    %c152 = arith.constant 152 : i64
    %c164 = arith.constant 164 : i64
    %off = affine.if #set()[%s] -> i64 { affine.yield %c164 : i64 } else { affine.yield %c152 : i64 }
    %i = arith.index_cast %off : i64 to index
    %j = arith.divsi %i, %c4 : index
    return %j : index
  }

  func.func @divide_unsigned(%s: index) -> i64 {
    %c8 = arith.constant 8 : i64
    %c48 = arith.constant 48 : i64
    %c64 = arith.constant 64 : i64
    %off = affine.if #set()[%s] -> i64 { affine.yield %c64 : i64 } else { affine.yield %c48 : i64 }
    %j = arith.divui %off, %c8 : i64
    return %j : i64
  }

  // the branch is the divisor
  func.func @divide_by_branched(%s: index) -> i64 {
    %c2 = arith.constant 2 : i64
    %c4 = arith.constant 4 : i64
    %c96 = arith.constant 96 : i64
    %d = affine.if #set()[%s] -> i64 { affine.yield %c4 : i64 } else { affine.yield %c2 : i64 }
    %j = arith.divsi %c96, %d : i64
    return %j : i64
  }

  // an scf branch over constants is a select before it is anything else, and
  // the cast sinks into that
  func.func @scf_branch(%c: i1) -> index {
    %c152 = arith.constant 152 : i64
    %c164 = arith.constant 164 : i64
    %off = scf.if %c -> i64 { scf.yield %c164 : i64 } else { scf.yield %c152 : i64 }
    %i = arith.index_cast %off : i64 to index
    return %i : index
  }

  // a second user of the branch's result: sinking would ask the branch twice
  func.func @two_uses(%s: index) -> (index, i64) {
    %c152 = arith.constant 152 : i64
    %c164 = arith.constant 164 : i64
    %off = affine.if #set()[%s] -> i64 { affine.yield %c164 : i64 } else { affine.yield %c152 : i64 }
    %i = arith.index_cast %off : i64 to index
    return %i, %off : index, i64
  }

  // an arm that does not choose a constant has nothing to fold against
  func.func @dynamic_arm(%s: index, %d: i64) -> index {
    %c164 = arith.constant 164 : i64
    %off = affine.if #set()[%s] -> i64 { affine.yield %c164 : i64 } else { affine.yield %d : i64 }
    %i = arith.index_cast %off : i64 to index
    return %i : index
  }
}

// CHECK:  func.func @cast_and_divide(%[[v1:.+]]: index) -> index {
// CHECK-NEXT:  %[[v2:.+]] = arith.constant 38 : index
// CHECK-NEXT:  %[[v3:.+]] = arith.constant 41 : index
// CHECK-NEXT:  %[[v4:.+]] = affine.if #set()[%[[v1]]] -> index {
// CHECK-NEXT:  affine.yield %[[v3]] : index
// CHECK-NEXT:  } else {
// CHECK-NEXT:  affine.yield %[[v2]] : index
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[v4]] : index
// CHECK-NEXT:  }

// CHECK:  func.func @divide_unsigned(%[[v1:.+]]: index) -> i64 {
// CHECK-NEXT:  %[[v2:.+]] = arith.constant 6 : i64
// CHECK-NEXT:  %[[v3:.+]] = arith.constant 8 : i64
// CHECK-NEXT:  %[[v4:.+]] = affine.if #set()[%[[v1]]] -> i64 {
// CHECK-NEXT:  affine.yield %[[v3]] : i64
// CHECK-NEXT:  } else {
// CHECK-NEXT:  affine.yield %[[v2]] : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[v4]] : i64
// CHECK-NEXT:  }

// CHECK:  func.func @divide_by_branched(%[[v1:.+]]: index) -> i64 {
// CHECK-NEXT:  %[[v2:.+]] = arith.constant 48 : i64
// CHECK-NEXT:  %[[v3:.+]] = arith.constant 24 : i64
// CHECK-NEXT:  %[[v4:.+]] = affine.if #set()[%[[v1]]] -> i64 {
// CHECK-NEXT:  affine.yield %[[v3]] : i64
// CHECK-NEXT:  } else {
// CHECK-NEXT:  affine.yield %[[v2]] : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[v4]] : i64
// CHECK-NEXT:  }

// CHECK:  func.func @scf_branch(%[[v1:.+]]: i1) -> index {
// CHECK-NEXT:  %[[v2:.+]] = arith.constant 152 : index
// CHECK-NEXT:  %[[v3:.+]] = arith.constant 164 : index
// CHECK-NEXT:  %[[v4:.+]] = arith.select %[[v1]], %[[v3]], %[[v2]] : index
// CHECK-NEXT:  return %[[v4]] : index
// CHECK-NEXT:  }

// CHECK:  func.func @two_uses(%[[v1:.+]]: index) -> (index, i64) {
// CHECK-NEXT:  %[[v2:.+]] = arith.constant 152 : i64
// CHECK-NEXT:  %[[v3:.+]] = arith.constant 164 : i64
// CHECK-NEXT:  %[[v4:.+]] = affine.if #set()[%[[v1]]] -> i64 {
// CHECK-NEXT:  affine.yield %[[v3]] : i64
// CHECK-NEXT:  } else {
// CHECK-NEXT:  affine.yield %[[v2]] : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[v5:.+]] = arith.index_cast %[[v4]] : i64 to index
// CHECK-NEXT:  return %[[v5]], %[[v4]] : index, i64
// CHECK-NEXT:  }

// CHECK:  func.func @dynamic_arm(%[[v1:.+]]: index, %[[v2:.+]]: i64) -> index {
// CHECK-NEXT:  %[[v3:.+]] = arith.constant 164 : i64
// CHECK-NEXT:  %[[v4:.+]] = affine.if #set()[%[[v1]]] -> i64 {
// CHECK-NEXT:  affine.yield %[[v3]] : i64
// CHECK-NEXT:  } else {
// CHECK-NEXT:  affine.yield %[[v2]] : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[v5:.+]] = arith.index_cast %[[v4]] : i64 to index
// CHECK-NEXT:  return %[[v5]] : index
// CHECK-NEXT:  }
