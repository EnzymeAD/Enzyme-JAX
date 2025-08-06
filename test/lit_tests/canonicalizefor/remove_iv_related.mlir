// RUN: enzymexlamlir-opt --allow-unregistered-dialect --canonicalize-scf-for --split-input-file %s | FileCheck %s

func.func @original_repro(%arg0: memref<1x104x194xf64, 1>, %arg1: memref<35xf64, 1>, %arg2: memref<34xf64, 1>, %arg: i64) {
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

// CHECK-LABEL: func @original_repro
// CHECK-SAME:  ({{.*}}, %[[INIT:.+]]: i64)
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG:     %[[CM1:.+]] = arith.constant -1 : i64
// CHECK-DAG:     %[[C21:.+]] = arith.constant 21 : i64
// CHECK:         %[[FOR:.+]] = scf.for %[[I:.+]] = %[[C1]] to %[[C21]] step %[[C1]] iter_args(%{{.*}} = %{{.*}}) -> (f64) : i64 {
// CHECK:           %[[COND:.+]] = arith.cmpi eq, %[[I]], %[[C1]] : i64
// CHECK:           %[[VAL:.+]] = arith.addi %[[I]], %[[CM1]] : i64
// CHECK:           %[[SEL:.+]] = arith.select %[[COND]], %[[INIT]], %[[VAL]] : i64
// CHECK:           "test.use"(%[[SEL]])
// CHECK:           %[[LOAD:.+]] = memref.load
// CHECK:           scf.yield %[[LOAD]]
// CHECK:         "test.use"(%[[FOR]], %[[C21]])

// ----

func.func @iv_related_is_first_iter_arg(%arg0: memref<10xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init = arith.constant 0 : index
  %res:2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%iv = %init, %load = %init) -> (index, index) {
    %next_iv = arith.addi %i, %c1 : index
    "test.use"(%next_iv, %iv) : (index, index) -> ()
    %next_load = memref.load %arg0[%i] : memref<10xindex>
    scf.yield %next_iv, %next_load : index, index
  }
  "test.use"(%res#0, %res#1) : (index, index) -> ()
  return
}

// CHECK-LABEL: func @iv_related_is_first_iter_arg
// CHECK-SAME:  (%[[ARG:.+]]: memref<10xindex>)
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         %[[FOR:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C10]] step %[[C1]] iter_args(%{{.*}} = %[[C0]]) -> (index) {
// CHECK:           %[[NEXT_I:.+]] = arith.addi %[[I]], %[[C1]] : index
// CHECK:           %[[COND:.+]] = arith.cmpi eq, %[[I]], %[[C0]] : index
// CHECK:           %[[VAL:.+]] = arith.subi %[[I]], %[[C1]] : index
// CHECK:           %[[SEL:.+]] = arith.select %[[COND]], %[[C0]], %[[VAL]] : index
// CHECK:           "test.use"(%[[NEXT_I]], %[[SEL]])
// CHECK:           %[[LOAD:.+]] = memref.load %[[ARG]][%[[I]]]
// CHECK:           scf.yield %[[LOAD]]
// CHECK:         "test.use"(%[[C10]], %[[FOR]])

// ----

func.func @non_unit_step(%arg0: memref<10xindex>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  %init = arith.constant 0 : index
  %res:2 = scf.for %i = %c0 to %c10 step %c2 iter_args(%iv = %init, %load = %init) -> (index, index) {
    %next_iv = arith.addi %i, %c2 : index
    "test.use"(%next_iv, %iv) : (index, index) -> ()
    %next_load = memref.load %arg0[%i] : memref<10xindex>
    scf.yield %next_iv, %next_load : index, index
  }
  "test.use"(%res#0, %res#1) : (index, index) -> ()
  return
}

// CHECK-LABEL: func @non_unit_step
// CHECK-SAME:  (%[[ARG:.+]]: memref<10xindex>)
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         %[[FOR:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C10]] step %[[C2]] iter_args(%{{.*}} = %[[C0]]) -> (index) {
// CHECK:           %[[NEXT_I:.+]] = arith.addi %[[I]], %[[C2]] : index
// CHECK:           %[[COND:.+]] = arith.cmpi eq, %[[I]], %[[C0]] : index
// CHECK:           %[[VAL:.+]] = arith.subi %[[I]], %[[C2]] : index
// CHECK:           %[[SEL:.+]] = arith.select %[[COND]], %[[C0]], %[[VAL]] : index
// CHECK:           "test.use"(%[[NEXT_I]], %[[SEL]])
// CHECK:           %[[LOAD:.+]] = memref.load %[[ARG]][%[[I]]]
// CHECK:           scf.yield %[[LOAD]]
// CHECK:         "test.use"(%[[C10]], %[[FOR]])

// ----

func.func @non_unit_non_divisible_step(%arg0: memref<10xindex>) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index
  %init = arith.constant 0 : index
  %res:2 = scf.for %i = %c0 to %c10 step %c3 iter_args(%iv = %init, %load = %init) -> (index, index) {
    %next_iv = arith.addi %i, %c3 : index
    "test.use"(%next_iv, %iv) : (index, index) -> ()
    %next_load = memref.load %arg0[%i] : memref<10xindex>
    scf.yield %next_iv, %next_load : index, index
  }
  "test.use"(%res#0, %res#1) : (index, index) -> ()
  return
}

// CHECK-LABEL: func @non_unit_non_divisible_step
// CHECK-SAME:  (%[[ARG:.+]]: memref<10xindex>)
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK-DAG:     %[[C12:.+]] = arith.constant 12 : index
// CHECK:         %[[FOR:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C10]] step %[[C3]] iter_args(%{{.*}} = %[[C0]]) -> (index) {
// CHECK:           %[[NEXT_I:.+]] = arith.addi %[[I]], %[[C3]] : index
// CHECK:           %[[COND:.+]] = arith.cmpi eq, %[[I]], %[[C0]] : index
// CHECK:           %[[VAL:.+]] = arith.subi %[[I]], %[[C3]] : index
// CHECK:           %[[SEL:.+]] = arith.select %[[COND]], %[[C0]], %[[VAL]] : index
// CHECK:           "test.use"(%[[NEXT_I]], %[[SEL]])
// CHECK:           %[[LOAD:.+]] = memref.load %[[ARG]][%[[I]]]
// CHECK:           scf.yield %[[LOAD]]
// CHECK:         "test.use"(%[[C12]], %[[FOR]])

// ----

func.func @ub_lb_equal(%arg0: memref<10xindex>) {
  %c3 = arith.constant 3 : index
  %c10 = arith.constant 10 : index
  %init = arith.constant 0 : index
  %res:2 = scf.for %i = %c10 to %c10 step %c3 iter_args(%iv = %init, %load = %init) -> (index, index) {
    %next_iv = arith.addi %i, %c3 : index
    "test.use"(%next_iv, %iv) : (index, index) -> ()
    %next_load = memref.load %arg0[%i] : memref<10xindex>
    scf.yield %next_iv, %next_load : index, index
  }
  "test.use"(%res#0, %res#1) : (index, index) -> ()
  return
}

// CHECK-LABEL: func @ub_lb_equal
// CHECK-SAME:  (%[[ARG:.+]]: memref<10xindex>)
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         %[[FOR:.+]] = scf.for %[[I:.+]] = %[[C10]] to %[[C10]] step %[[C3]] iter_args(%{{.*}} = %[[C0]]) -> (index) {
// CHECK:           %[[NEXT_I:.+]] = arith.addi %[[I]], %[[C3]] : index
// CHECK:           %[[COND:.+]] = arith.cmpi eq, %[[I]], %[[C10]] : index
// CHECK:           %[[VAL:.+]] = arith.subi %[[I]], %[[C3]] : index
// CHECK:           %[[SEL:.+]] = arith.select %[[COND]], %[[C0]], %[[VAL]] : index
// CHECK:           "test.use"(%[[NEXT_I]], %[[SEL]])
// CHECK:           %[[LOAD:.+]] = memref.load %[[ARG]][%[[I]]]
// CHECK:           scf.yield %[[LOAD]]
// CHECK:         "test.use"(%[[C0]], %[[FOR]])

// ----

func.func @non_constant_step(%arg0: memref<10xindex>, %step: index) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %init = arith.constant 0 : index
  %res:2 = scf.for %i = %c0 to %c10 step %step iter_args(%iv = %init, %load = %init) -> (index, index) {
    %next_iv = arith.addi %i, %step : index
    "test.use"(%iv) : (index) -> ()
    %next_load = memref.load %arg0[%i] : memref<10xindex>
    scf.yield %next_iv, %next_load : index, index
  }
  "test.use"(%res#0, %res#1) : (index, index) -> ()
  return
}

// CHECK-LABEL: func @non_constant_step
// CHECK-SAME:  (%[[ARG0:.+]]: memref<10xindex>, %[[ARG1:.+]]: index)
// CHECK-DAG:     %[[C9:.+]] = arith.constant 9 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         %[[FOR:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C10]] step %[[ARG1]] iter_args(%{{.*}} = %[[C0]]) -> (index) {
// CHECK:           %[[COND:.+]] = arith.cmpi eq, %[[I]], %[[C0]] : index
// CHECK:           %[[VAL:.+]] = arith.subi %[[I]], %[[ARG1]] : index
// CHECK:           %[[SEL:.+]] = arith.select %[[COND]], %[[C0]], %[[VAL]] : index
// CHECK:           "test.use"(%[[SEL]])
// CHECK:           %[[LOAD:.+]] = memref.load %[[ARG]][%[[I]]]
// CHECK:           scf.yield %[[LOAD]]
// CHECK:         %[[DIV:.+]] = arith.divui %[[C9]], %[[ARG1]] : index
// CHECK:         %[[MUL:.+]] = arith.muli %[[ARG1]], %[[DIV]] : index
// CHECK:         %[[LAST_VAL:.+]] = arith.addi %[[ARG1]], %[[MUL]] : index
// CHECK:         "test.use"(%[[LAST_VAL]], %[[FOR]])

// ----

func.func @single_iter_arg() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init = arith.constant 0 : index
  %res = scf.for %i = %c0 to %c10 step %c1 iter_args(%iv = %init) -> index {
    %next_iv = arith.addi %i, %c1 : index
    "test.use"(%iv) : (index) -> ()
    scf.yield %next_iv : index
  }
  "test.use"(%res) : (index) -> ()
  return
}

// CHECK-LABEL: func @single_iter_arg
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         scf.for %[[I:.+]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:           %[[COND:.+]] = arith.cmpi eq, %[[I]], %[[C0]] : index
// CHECK:           %[[VAL:.+]] = arith.subi %[[I]], %[[C1]] : index
// CHECK:           %[[SEL:.+]] = arith.select %[[COND]], %[[C0]], %[[VAL]] : index
// CHECK:           "test.use"(%[[SEL]])
// CHECK:         "test.use"(%[[C10]])

// ----

func.func @multiple_iv_updates(%arg0: memref<10xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %init = arith.constant 0 : index
  %res:3 = scf.for %i = %c0 to %c10 step %c1 iter_args(%iv1 = %init, %iv2 = %init, %load = %init) -> (index, index, index) {
    %next_iv1 = arith.addi %i, %c1 : index
    %next_iv2 = arith.addi %i, %c1 : index
    "test.use"(%iv1, %iv2) : (index, index) -> ()
    %next_load = memref.load %arg0[%i] : memref<10xindex>
    scf.yield %next_iv1, %next_iv2, %next_load : index, index, index
  }
  "test.use"(%res#0, %res#1, %res#2) : (index, index, index) -> ()
  return
}

// CHECK-LABEL: func @multiple_iv_updates
// CHECK-SAME:  (%[[ARG:.+]]: memref<10xindex>)
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C10:.+]] = arith.constant 10 : index
// CHECK:         %[[FOR:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C10]] step %[[C1]] iter_args(%{{.*}} = %[[C0]]) -> (index) {
// CHECK:           %[[COND1:.+]] = arith.cmpi eq, %[[I]], %[[C0]] : index
// CHECK:           %[[VAL1:.+]] = arith.subi %[[I]], %[[C1]] : index
// CHECK:           %[[SEL1:.+]] = arith.select %[[COND1]], %[[C0]], %[[VAL1]] : index
// CHECK:           %[[COND2:.+]] = arith.cmpi eq, %[[I]], %[[C0]] : index
// CHECK:           %[[VAL2:.+]] = arith.subi %[[I]], %[[C1]] : index
// CHECK:           %[[SEL2:.+]] = arith.select %[[COND2]], %[[C0]], %[[VAL2]] : index
// CHECK:           "test.use"(%[[SEL1]], %[[SEL2]])
// CHECK:           %[[LOAD:.+]] = memref.load %[[ARG]][%[[I]]]
// CHECK:           scf.yield %[[LOAD]]
// CHECK:         "test.use"(%[[C10]], %[[C10]], %[[FOR]])
