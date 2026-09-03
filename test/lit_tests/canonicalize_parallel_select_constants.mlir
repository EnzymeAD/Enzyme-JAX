// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel{parallel=false})" | FileCheck %s

// An op over a select between constants is the select between the op over
// each arm, which folds there: a byte offset chosen between two captures,
// cast and scaled, arrives as a choice between the two element slots.

module {
  func.func @cast_divide_add(%c: i1) -> index {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c144 = arith.constant 144 : i64
    %c184 = arith.constant 184 : i64
    %off = arith.select %c, %c144, %c184 : i64
    %i = arith.index_cast %off : i64 to index
    %d = arith.divsi %i, %c8 : index
    %a = arith.addi %d, %c1 : index
    return %a : index
  }

  // the select feeds two chains; both sink and the select goes with them
  func.func @two_uses(%c: i1) -> (index, index) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c144 = arith.constant 144 : i64
    %c184 = arith.constant 184 : i64
    %off = arith.select %c, %c144, %c184 : i64
    %i = arith.index_cast %off : i64 to index
    %d8 = arith.divui %i, %c8 : index
    %d4 = arith.divsi %i, %c4 : index
    return %d8, %d4 : index, index
  }

  func.func @multiply(%c: i1) -> i64 {
    %c3 = arith.constant 3 : i64
    %c5 = arith.constant 5 : i64
    %c7 = arith.constant 7 : i64
    %s = arith.select %c, %c3, %c5 : i64
    %m = arith.muli %s, %c7 : i64
    return %m : i64
  }

  // an arm that is not a constant has nothing to fold against
  func.func @dynamic_arm(%c: i1, %d: i64) -> index {
    %c144 = arith.constant 144 : i64
    %off = arith.select %c, %c144, %d : i64
    %i = arith.index_cast %off : i64 to index
    return %i : index
  }

  // the other operand is not a constant either
  func.func @dynamic_operand(%c: i1, %x: i64) -> i64 {
    %c144 = arith.constant 144 : i64
    %c184 = arith.constant 184 : i64
    %off = arith.select %c, %c144, %c184 : i64
    %a = arith.addi %off, %x : i64
    return %a : i64
  }
}

// CHECK:  func.func @cast_divide_add(%[[c:.+]]: i1) -> index {
// CHECK-DAG:  %[[v19:.+]] = arith.constant 19 : index
// CHECK-DAG:  %[[v24:.+]] = arith.constant 24 : index
// CHECK:  %[[r:.+]] = arith.select %[[c]], %[[v19]], %[[v24]] : index
// CHECK-NEXT:  return %[[r]] : index

// CHECK:  func.func @two_uses(%[[c:.+]]: i1) -> (index, index) {
// CHECK-DAG:  %[[v18:.+]] = arith.constant 18 : index
// CHECK-DAG:  %[[v23:.+]] = arith.constant 23 : index
// CHECK-DAG:  %[[v36:.+]] = arith.constant 36 : index
// CHECK-DAG:  %[[v46:.+]] = arith.constant 46 : index
// CHECK-DAG:  %[[d8:.+]] = arith.select %[[c]], %[[v18]], %[[v23]] : index
// CHECK-DAG:  %[[d4:.+]] = arith.select %[[c]], %[[v36]], %[[v46]] : index
// CHECK:  return %[[d8]], %[[d4]] : index, index

// CHECK:  func.func @multiply(%[[c:.+]]: i1) -> i64 {
// CHECK-DAG:  %[[v21:.+]] = arith.constant 21 : i64
// CHECK-DAG:  %[[v35:.+]] = arith.constant 35 : i64
// CHECK:  %[[r:.+]] = arith.select %[[c]], %[[v21]], %[[v35]] : i64
// CHECK-NEXT:  return %[[r]] : i64

// CHECK:  func.func @dynamic_arm(%[[c:.+]]: i1, %[[d:.+]]: i64) -> index {
// CHECK:  %[[s:.+]] = arith.select %[[c]], %{{.*}}, %[[d]] : i64
// CHECK-NEXT:  %[[i:.+]] = arith.index_cast %[[s]] : i64 to index
// CHECK-NEXT:  return %[[i]] : index

// CHECK:  func.func @dynamic_operand(%[[c:.+]]: i1, %[[x:.+]]: i64) -> i64 {
// CHECK:  %[[s:.+]] = arith.select %[[c]], %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:  %[[a:.+]] = arith.addi %[[s]], %[[x]] : i64
// CHECK-NEXT:  return %[[a]] : i64
