// RUN: enzymexlamlir-opt %s -split-input-file --pass-pipeline="builtin.module(enzyme-lift-cf-to-scf{rewrite_index_switch=true},canonicalize)" -allow-unregistered-dialect | FileCheck %s

// Single case + default, no results
func.func @single_case_no_results(%arg0: index) {
  scf.index_switch %arg0
  case 0 {
    "test.op0"() : () -> ()
    scf.yield
  }
  default {
    "test.default"() : () -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: func @single_case_no_results
// CHECK-SAME:    %[[ARG:.*]]: index
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[CMP:.*]] = arith.cmpi eq, %[[ARG]], %[[C0]] : index
// CHECK:       scf.if %[[CMP]] {
// CHECK-NEXT:    "test.op0"()
// CHECK-NEXT:  } else {
// CHECK-NEXT:    "test.default"()
// CHECK-NEXT:  }
// CHECK:       return

// -----

// Single case + default, with results
func.func @single_case_with_results(%arg0: index) -> i32 {
  %res = scf.index_switch %arg0 -> i32
  case 0 {
    %0 = "test.op0"() : () -> i32
    scf.yield %0 : i32
  }
  default {
    %1 = "test.default"() : () -> i32
    scf.yield %1 : i32
  }
  return %res : i32
}

// CHECK-LABEL: func @single_case_with_results
// CHECK-SAME:    %[[ARG:.*]]: index
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[CMP:.*]] = arith.cmpi eq, %[[ARG]], %[[C0]] : index
// CHECK:       %[[RES:.*]] = scf.if %[[CMP]] -> (i32) {
// CHECK-NEXT:    %[[V0:.*]] = "test.op0"()
// CHECK-NEXT:    scf.yield %[[V0]] : i32
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %[[V1:.*]] = "test.default"()
// CHECK-NEXT:    scf.yield %[[V1]] : i32
// CHECK-NEXT:  }
// CHECK:       return %[[RES]]

// -----

// Two cases + default, with results
func.func @two_cases(%arg0: index) -> i32 {
  %res = scf.index_switch %arg0 -> i32
  case 0 {
    %0 = "test.op0"() : () -> i32
    scf.yield %0 : i32
  }
  case 1 {
    %1 = "test.op1"() : () -> i32
    scf.yield %1 : i32
  }
  default {
    %2 = "test.default"() : () -> i32
    scf.yield %2 : i32
  }
  return %res : i32
}

// CHECK-LABEL: func @two_cases
// CHECK-SAME:    %[[ARG:.*]]: index
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[CMP0:.*]] = arith.cmpi eq, %[[ARG]], %[[C0]] : index
// CHECK:       %[[OUTER:.*]] = scf.if %[[CMP0]] -> (i32) {
// CHECK:         "test.op0"()
// CHECK:       } else {
// CHECK:         %[[CMP1:.*]] = arith.cmpi eq, %[[ARG]], %[[C1]] : index
// CHECK:         %[[INNER:.*]] = scf.if %[[CMP1]]
// CHECK:           "test.op1"()
// CHECK:         } else {
// CHECK:           "test.default"()
// CHECK:         }
// CHECK:         scf.yield %[[INNER]]
// CHECK:       }
// CHECK:       return %[[OUTER]]

// -----

// Three cases + default, no results
func.func @three_cases_no_results(%arg0: index) {
  scf.index_switch %arg0
  case 5 {
    "test.op5"() : () -> ()
    scf.yield
  }
  case 10 {
    "test.op10"() : () -> ()
    scf.yield
  }
  case 15 {
    "test.op15"() : () -> ()
    scf.yield
  }
  default {
    "test.default"() : () -> ()
    scf.yield
  }
  return
}

// CHECK-LABEL: func @three_cases_no_results
// CHECK-SAME:    %[[ARG:.*]]: index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:   %[[C15:.*]] = arith.constant 15 : index
// CHECK:       %[[CMP0:.*]] = arith.cmpi eq, %[[ARG]], %[[C5]] : index
// CHECK:       scf.if %[[CMP0]] {
// CHECK:         "test.op5"()
// CHECK:       } else {
// CHECK:         %[[CMP1:.*]] = arith.cmpi eq, %[[ARG]], %[[C10]] : index
// CHECK:         scf.if %[[CMP1]] {
// CHECK:           "test.op10"()
// CHECK:         } else {
// CHECK:           %[[CMP2:.*]] = arith.cmpi eq, %[[ARG]], %[[C15]] : index
// CHECK:           scf.if %[[CMP2]] {
// CHECK:             "test.op15"()
// CHECK:           } else {
// CHECK:             "test.default"()
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       return

// -----

// Default only (zero cases)
func.func @default_only(%arg0: index) -> i32 {
  %res = scf.index_switch %arg0 -> i32
  default {
    %0 = "test.default"() : () -> i32
    scf.yield %0 : i32
  }
  return %res : i32
}

// CHECK-LABEL: func @default_only
// CHECK-NOT:   scf.if
// CHECK-NOT:   scf.index_switch
// CHECK:       %[[V:.*]] = "test.default"()
// CHECK:       return %[[V]]

// -----

// Multiple results
func.func @multiple_results(%arg0: index) -> (i32, f64) {
  %res:2 = scf.index_switch %arg0 -> i32, f64
  case 0 {
    %0 = "test.op0"() : () -> i32
    %1 = "test.op0f"() : () -> f64
    scf.yield %0, %1 : i32, f64
  }
  case 1 {
    %2 = "test.op1"() : () -> i32
    %3 = "test.op1f"() : () -> f64
    scf.yield %2, %3 : i32, f64
  }
  default {
    %4 = "test.default"() : () -> i32
    %5 = "test.defaultf"() : () -> f64
    scf.yield %4, %5 : i32, f64
  }
  return %res#0, %res#1 : i32, f64
}

// CHECK-LABEL: func @multiple_results
// CHECK-SAME:    %[[ARG:.*]]: index
// CHECK:       %[[OUTER:.*]]:2 = scf.if
// CHECK:         "test.op0"()
// CHECK:         "test.op0f"()
// CHECK:       } else {
// CHECK:         %[[INNER:.*]]:2 = scf.if
// CHECK:           "test.op1"()
// CHECK:           "test.op1f"()
// CHECK:         } else {
// CHECK:           "test.default"()
// CHECK:           "test.defaultf"()
// CHECK:         }
// CHECK:         scf.yield %[[INNER]]#0, %[[INNER]]#1
// CHECK:       }
// CHECK:       return %[[OUTER]]#0, %[[OUTER]]#1

// -----

// Verify that rewrite-index-switch=false preserves index_switch
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-lift-cf-to-scf{rewrite_index_switch=false},canonicalize)" -allow-unregistered-dialect | FileCheck %s --check-prefix=NOSWITCH

func.func @preserved_switch(%arg0: index) {
  scf.index_switch %arg0
  case 0 {
    "test.op0"() : () -> ()
    scf.yield
  }
  default {
    scf.yield
  }
  return
}

// NOSWITCH-LABEL: func @preserved_switch
// NOSWITCH:       scf.index_switch
