// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

func.func @sitofp_invariant_muli(%arg0: i32, %arg1: i32) -> f64 {
  %cst = arith.constant 0.0 : f64
  %res = affine.for %iv = 0 to 10 iter_args(%acc = %cst) -> (f64) {
    %mul = arith.muli %arg0, %arg1 : i32
    %fp = arith.sitofp %mul : i32 to f64
    %add = arith.addf %acc, %fp : f64
    affine.yield %add : f64
  }
  return %res : f64
}

// CHECK-LABEL:   func.func @sitofp_invariant_muli(
// CHECK-SAME:                     %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> f64 {
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[MUL:.*]] = arith.muli %[[ARG0]], %[[ARG1]] : i32
// CHECK:           %[[PAR:.*]] = affine.parallel (%[[ARG2:.*]]) = (0) to (10) reduce ("addf") -> (f64) {
// CHECK:             %[[FP:.*]] = arith.sitofp %[[MUL]] : i32 to f64
// CHECK:             affine.yield %[[FP]] : f64
// CHECK:           }
// CHECK:           %[[ADD:.*]] = arith.addf %[[PAR]], %[[CST]] : f64
// CHECK:           return %[[ADD]] : f64
