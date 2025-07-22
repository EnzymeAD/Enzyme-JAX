// RUN: enzymexlamlir-opt --allow-unregistered-dialect --canonicalize-scf-for --split-input-file %s | FileCheck %s

func.func @foo(%arg0: memref<1x104x194xf64, 1>, %arg1: memref<35xf64, 1>, %arg2: memref<34xf64, 1>, %arg: i64) {
  %c21_i64 = arith.constant 21 : i64
  %0 = ub.poison : i64
  %1 = ub.poison : f64
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c1_i64 = arith.constant 1 : i64
  %3 = affine.load %arg1[7] : memref<35xf64, 1>
  %4:3 = scf.for %arg5 = %c1_i64 to %c21_i64 step %c1_i64 iter_args(%arg6 = %3, %arg7 = %1, %arg8 = %arg) -> (f64, f64, i64)  : i64 {
    "test.use"(%arg8) : (i64) -> ()
    %5 = arith.index_cast %arg5 : i64 to index
    %6 = arith.addi %5, %c7 : index
    %12 = memref.load %arg1[%6] : memref<35xf64, 1>
    %13 = arith.addi %arg5, %c1_i64 : i64
    scf.yield %12, %12, %13 : f64, f64, i64
  }
  "test.use"(%4#1, %4#2) : (f64, i64) -> ()
  
  return
}

// CHECK-LABEL: func @foo
// CHECK-SAME:  ({{.*}}, %[[INIT:.+]]: i64)
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG:     %[[CM1:.+]] = arith.constant -1 : i64
// CHECK-DAG:     %[[C21:.+]] = arith.constant 21 : i64
// CHECK:         %[[FOR:.+]] = scf.for %[[I:.+]] = %[[C1]] to %[[C21]] step %[[C1]] iter_args(%{{.*}} = %{{.*}}) -> (f64)  : i64 {
// CHECK:           %[[COND:.+]] = arith.cmpi eq, %[[I]], %[[C1]] : i64
// CHECK:           %[[VAL:.+]] = arith.addi %[[I]], %[[CM1]] : i64
// CHECK:           %[[SEL:.+]] = arith.select %[[COND]], %[[INIT]], %[[VAL]] : i64
// CHECK:           "test.use"(%[[SEL]])
// CHECK:           %[[LOAD:.+]] = memref.load
// CHECK:           scf.yield %[[LOAD]]
// CHECK:         "test.use"(%[[FOR]], %[[C21]])
