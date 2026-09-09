// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel{parallel=false})" | FileCheck %s

// An op over what a branch chose between constants rides into the arms and
// folds there, so what the branch chose reaches its user as the branch's own
// result: a byte offset cast and scaled arrives as the element slot itself.

#set = affine_set<()[s0] : (s0 >= 1)>
#set0 = affine_set<(d0) : (d0 == 0)>
#set1 = affine_set<(d0) : (d0 - 1 == 0)>
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

  // a second user of the branch's result keeps it: the branch grows a result
  func.func @two_uses(%s: index) -> (index, i64) {
    %c152 = arith.constant 152 : i64
    %c164 = arith.constant 164 : i64
    %off = affine.if #set()[%s] -> i64 { affine.yield %c164 : i64 } else { affine.yield %c152 : i64 }
    %i = arith.index_cast %off : i64 to index
    return %i, %off : index, i64
  }

  // an arm that chooses between constants itself: the cast rides into the
  // outer branch, and from its arm into the inner one, and folds at every
  // leaf; a three-way choice by a direction is two nested affine branches
  func.func @nested_choice(%d: index) -> index {
    %c4 = arith.constant 4 : index
    %c152 = arith.constant 152 : i64
    %c164 = arith.constant 164 : i64
    %inner = affine.if #set1(%d) -> i64 { affine.yield %c164 : i64 } else { affine.yield %c152 : i64 }
    %off = affine.if #set0(%d) -> i64 { affine.yield %c164 : i64 } else { affine.yield %inner : i64 }
    %i = arith.index_cast %off : i64 to index
    %j = arith.divsi %i, %c4 : index
    return %j : index
  }

  // a select whose arm is a branch between constants is neither a tree of
  // branches nor one of selects: the cast stays
  func.func @select_over_branch(%c: i1, %s: index) -> index {
    %c152 = arith.constant 152 : i64
    %c164 = arith.constant 164 : i64
    %c96 = arith.constant 96 : i64
    %inner = affine.if #set()[%s] -> i64 { affine.yield %c164 : i64 } else { affine.yield %c152 : i64 }
    %off = arith.select %c, %c96, %inner : i64
    %i = arith.index_cast %off : i64 to index
    return %i : index
  }

  // an arm that does not choose a constant has nothing to fold against
  func.func @dynamic_arm(%s: index, %d: i64) -> index {
    %c164 = arith.constant 164 : i64
    %off = affine.if #set()[%s] -> i64 { affine.yield %c164 : i64 } else { affine.yield %d : i64 }
    %i = arith.index_cast %off : i64 to index
    return %i : index
  }
}

// CHECK:  #set = affine_set<()[s0] : (s0 - 1 >= 0)>
// CHECK-NEXT:#set1 = affine_set<()[s0] : (s0 - 1 == 0)>
// CHECK-NEXT:#set2 = affine_set<()[s0] : (s0 == 0)>
// CHECK-NEXT:module {
// CHECK-NEXT:  func.func @cast_and_divide(%[[a1:.+]]: index) -> index {
// CHECK-NEXT:    %[[a2:.+]] = arith.constant 38 : index
// CHECK-NEXT:    %[[a3:.+]] = arith.constant 41 : index
// CHECK-NEXT:    %[[a4:.+]] = affine.if #set()[%[[a1]]] -> index {
// CHECK-NEXT:      affine.yield %[[a3]] : index
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a2]] : index
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a4]] : index
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @divide_unsigned(%[[a1]]: index) -> i64 {
// CHECK-NEXT:    %[[a5:.+]] = arith.constant 6 : i64
// CHECK-NEXT:    %[[a6:.+]] = arith.constant 8 : i64
// CHECK-NEXT:    %[[a4]] = affine.if #set()[%[[a1]]] -> i64 {
// CHECK-NEXT:      affine.yield %[[a6]] : i64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a5]] : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a4]] : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @divide_by_branched(%[[a1]]: index) -> i64 {
// CHECK-NEXT:    %[[a7:.+]] = arith.constant 48 : i64
// CHECK-NEXT:    %[[a8:.+]] = arith.constant 24 : i64
// CHECK-NEXT:    %[[a4]] = affine.if #set()[%[[a1]]] -> i64 {
// CHECK-NEXT:      affine.yield %[[a8]] : i64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a7]] : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a4]] : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @scf_branch(%[[a1]]: i1) -> index {
// CHECK-NEXT:    %[[a9:.+]] = arith.constant 152 : index
// CHECK-NEXT:    %[[a10:.+]] = arith.constant 164 : index
// CHECK-NEXT:    %[[a4]] = arith.select %[[a1]], %[[a10]], %[[a9]] : index
// CHECK-NEXT:    return %[[a4]] : index
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @two_uses(%[[a1]]: index) -> (index, i64) {
// CHECK-NEXT:    %[[a9]] = arith.constant 152 : index
// CHECK-NEXT:    %[[a10]] = arith.constant 164 : index
// CHECK-NEXT:    %[[a11:.+]] = arith.constant 152 : i64
// CHECK-NEXT:    %[[a12:.+]] = arith.constant 164 : i64
// CHECK-NEXT:    %[[a4]]:2 = affine.if #set()[%[[a1]]] -> (i64, index) {
// CHECK-NEXT:      affine.yield %[[a12]], %[[a10]] : i64, index
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a11]], %[[a9]] : i64, index
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a4]]#1, %[[a4]]#0 : index, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @nested_choice(%[[a1]]: index) -> index {
// CHECK-NEXT:    %[[a2]] = arith.constant 38 : index
// CHECK-NEXT:    %[[a3]] = arith.constant 41 : index
// CHECK-NEXT:    %[[a4]] = affine.if #set1()[%[[a1]]] -> index {
// CHECK-NEXT:      affine.yield %[[a3]] : index
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a2]] : index
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a13:.+]] = affine.if #set2()[%[[a1]]] -> index {
// CHECK-NEXT:      affine.yield %[[a3]] : index
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a4]] : index
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a13]] : index
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @select_over_branch(%[[a1]]: i1, %[[a14:.+]]: index) -> index {
// CHECK-NEXT:    %[[a11]] = arith.constant 152 : i64
// CHECK-NEXT:    %[[a12]] = arith.constant 164 : i64
// CHECK-NEXT:    %[[a15:.+]] = arith.constant 96 : i64
// CHECK-NEXT:    %[[a4]] = affine.if #set()[%[[a14]]] -> i64 {
// CHECK-NEXT:      affine.yield %[[a12]] : i64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a11]] : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a13]] = arith.select %[[a1]], %[[a15]], %[[a4]] : i64
// CHECK-NEXT:    %[[a16:.+]] = arith.index_cast %[[a13]] : i64 to index
// CHECK-NEXT:    return %[[a16]] : index
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @dynamic_arm(%[[a1]]: index, %[[a14]]: i64) -> index {
// CHECK-NEXT:    %[[a12]] = arith.constant 164 : i64
// CHECK-NEXT:    %[[a4]] = affine.if #set()[%[[a1]]] -> i64 {
// CHECK-NEXT:      affine.yield %[[a12]] : i64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      affine.yield %[[a14]] : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a13]] = arith.index_cast %[[a4]] : i64 to index
// CHECK-NEXT:    return %[[a13]] : index
// CHECK-NEXT:  }
