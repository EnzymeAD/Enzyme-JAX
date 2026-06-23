// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

func.func @forbound_reassoc(%arg0: i32, %init: f64) -> f64 {
  %cst = arith.constant 1.000000e+00 : f64
  %s = arith.index_cast %arg0 : i32 to index
  %r = affine.for %i = 0 to %s iter_args(%acc = %init) -> (f64) {
    %ri = affine.for %j = 0 to affine_map<(d0)[s0] -> (-d0 + s0 - 1)>(%i)[%s]
            iter_args(%a = %acc) -> (f64) {
      %sum = arith.addf %a, %cst : f64
      affine.yield %sum : f64
    }
    affine.yield %ri : f64
  }
  return %r : f64
}

// CHECK-LABEL:   func.func @forbound_reassoc(
// CHECK-SAME:                                %[[ARG0:.*]]: i32, %[[ARG1:.*]]: f64) -> f64 {
// CHECK:           %[[CST:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[ARG0]] : i32 to index
// CHECK:           %[[OUT:.*]] = affine.parallel (%[[IV0:.*]]) = (0) to (symbol(%[[IDX]])) reduce ("addf") -> (f64) {
// CHECK:             %[[IN:.*]] = affine.parallel (%[[IV1:.*]]) = (0) to (-%[[IV0]] + symbol(%[[IDX]]) - 1) reduce ("addf") -> (f64) {
// CHECK:               affine.yield %[[CST]] : f64
// CHECK:             }
// CHECK:             affine.yield %[[IN]] : f64
// CHECK:           }
// CHECK:           %[[ADD:.*]] = arith.addf %[[ARG1]], %[[OUT]] : f64
// CHECK:           return %[[ADD]] : f64
