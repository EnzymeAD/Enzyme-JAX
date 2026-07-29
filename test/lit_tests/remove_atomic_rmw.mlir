// RUN: enzymexlamlir-opt --remove-atomics %s  --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               affine.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_1]]] : memref<4xf32>
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:               affine.store %[[MULF_0]], %[[ARG3]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             }
// CHECK:             %[[PARALLEL_0:.*]] = affine.parallel (%[[VAL_2:.*]]) = (0) to (4) reduce ("addf") -> (f32) {
// CHECK:               %[[LOAD_1:.*]] = affine.load %[[ALLOC_0]]{{\[}}%[[VAL_2]]] : memref<4xf32>
// CHECK:               %[[LOAD_2:.*]] = affine.load %[[ARG4]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               affine.store %[[CONSTANT_0]], %[[ARG4]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               %[[MULF_1:.*]] = arith.mulf %[[LOAD_2]], %[[ARG0]] : f32
// CHECK:               %[[MULF_2:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:               %[[LOAD_3:.*]] = affine.load %[[ARG2]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_3]], %[[MULF_1]] : f32
// CHECK:               affine.store %[[ADDF_0]], %[[ARG2]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               affine.yield %[[MULF_2]] : f32
// CHECK:             }
// CHECK:             memref.dealloc %[[ALLOC_0]] : memref<4xf32>
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%arg0: f32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      %alloc = memref.alloc() : memref<4xf32>
      affine.parallel (%arg5) = (0) to (4) {
        %2 = affine.load %arg1[%arg5] : memref<?xf32>
        affine.store %2, %alloc[%arg5] : memref<4xf32>
        %3 = arith.mulf %2, %arg0 : f32
        affine.store %3, %arg3[%arg5] : memref<?xf32>
      }
      %1 = affine.parallel (%arg5) = (0) to (4) reduce ("addf") -> (f32) {
        %2 = affine.load %alloc[%arg5] : memref<4xf32>
        %3 = affine.load %arg4[%arg5] : memref<?xf32>
        affine.store %cst, %arg4[%arg5] : memref<?xf32>
        %4 = arith.mulf %3, %arg0 : f32
        %5 = arith.mulf %3, %2 : f32
        %6 = enzyme.affine_atomic_rmw addf %4, %arg2, (#map) [%arg5] : (f32, memref<?xf32>) -> f32
        affine.yield %5 : f32
      }
      memref.dealloc %alloc : memref<4xf32>
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @scf_if_aliasing_store_removes_atomic(
// CHECK:           scf.if
// CHECK:           affine.store
// CHECK:           %[[OLD:.*]] = affine.load %[[A:.*]]{{\[}}%[[I:.*]]] : memref<?xf32>
// CHECK:           %[[NEW:.*]] = arith.addf %[[OLD]], %[[V:.*]] : f32
// CHECK:           affine.store %[[NEW]], %[[A]]{{\[}}%[[I]]] : memref<?xf32>
// CHECK-NOT:       enzyme.affine_atomic_rmw
  func.func @scf_if_aliasing_store_removes_atomic(%v: f32, %a: memref<?xf32>, %b: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %pred = arith.cmpi eq, %i, %c0 : index
        scf.if %pred {
          affine.store %v, %b[%i] : memref<?xf32>
        }
        %r = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @scf_if_noalias_store_removes_atomic(
// CHECK:           scf.if
// CHECK:           affine.store
// CHECK:           %[[OLD:.*]] = affine.load %[[A:.*]]{{\[}}%[[I:.*]]] : memref<?xf32>
// CHECK:           %[[NEW:.*]] = arith.addf %[[OLD]], %[[V:.*]] : f32
// CHECK:           affine.store %[[NEW]], %[[A]]{{\[}}%[[I]]] : memref<?xf32>
// CHECK-NOT:       enzyme.affine_atomic_rmw
  func.func @scf_if_noalias_store_removes_atomic(%v: f32,
      %a: memref<?xf32> {llvm.noalias},
      %b: memref<?xf32> {llvm.noalias}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %pred = arith.cmpi eq, %i, %c0 : index
        scf.if %pred {
          affine.store %v, %b[%i] : memref<?xf32>
        }
        %r = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (0)>
module {
// CHECK-LABEL:   func.func @two_racy_atomics_keep_atomics(
// CHECK:           enzyme.affine_atomic_rmw addf
// CHECK:           enzyme.affine_atomic_rmw addf
  func.func @two_racy_atomics_keep_atomics(%v: f32, %a: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %r0 = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
        %r1 = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
  func.func private @unknown_side_effect()

// CHECK-LABEL:   func.func @unknown_side_effect_keeps_atomic(
// CHECK:           func.call @unknown_side_effect
// CHECK:           enzyme.affine_atomic_rmw addf
  func.func @unknown_side_effect_keeps_atomic(%v: f32, %a: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        func.call @unknown_side_effect() : () -> ()
        %r = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @memref_atomic_rmw_keeps_affine_atomic(
// CHECK:           memref.atomic_rmw addf
// CHECK:           enzyme.affine_atomic_rmw addf
  func.func @memref_atomic_rmw_keeps_affine_atomic(%v: f32, %a: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %other = memref.atomic_rmw addf %v, %a[%i] : (f32, memref<?xf32>) -> f32
        %r = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @memref_atomic_rmw_noalias_removes_affine_atomic(
// CHECK:           memref.atomic_rmw addf
// CHECK:           %[[OLD:.*]] = affine.load %[[A:.*]]{{\[}}%[[I:.*]]] : memref<?xf32>
// CHECK:           %[[NEW:.*]] = arith.addf %[[OLD]], %[[V:.*]] : f32
// CHECK:           affine.store %[[NEW]], %[[A]]{{\[}}%[[I]]] : memref<?xf32>
// CHECK-NOT:       enzyme.affine_atomic_rmw
// CHECK:           return
  func.func @memref_atomic_rmw_noalias_removes_affine_atomic(%v: f32,
      %a: memref<?xf32> {llvm.noalias},
      %b: memref<?xf32> {llvm.noalias}) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %other = memref.atomic_rmw addf %v, %b[%i] : (f32, memref<?xf32>) -> f32
        %r = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @memref_generic_atomic_rmw_keeps_affine_atomic(
// CHECK:           memref.generic_atomic_rmw
// CHECK:           memref.atomic_yield
// CHECK:           enzyme.affine_atomic_rmw addf
  func.func @memref_generic_atomic_rmw_keeps_affine_atomic(%v: f32, %a: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %other = memref.generic_atomic_rmw %a[%i] : memref<?xf32> {
        ^bb0(%current: f32):
          %next = arith.addf %current, %v : f32
          memref.atomic_yield %next : f32
        }
        %r = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @memref_generic_atomic_rmw_noalias_removes_affine_atomic(
// CHECK:           memref.generic_atomic_rmw
// CHECK:           memref.atomic_yield
// CHECK:           %[[OLD:.*]] = affine.load %[[A:.*]]{{\[}}%[[I:.*]]] : memref<?xf32>
// CHECK:           %[[NEW:.*]] = arith.addf %[[OLD]], %[[V:.*]] : f32
// CHECK:           affine.store %[[NEW]], %[[A]]{{\[}}%[[I]]] : memref<?xf32>
// CHECK-NOT:       enzyme.affine_atomic_rmw
// CHECK:           return
  func.func @memref_generic_atomic_rmw_noalias_removes_affine_atomic(%v: f32,
      %a: memref<?xf32> {llvm.noalias},
      %b: memref<?xf32> {llvm.noalias}) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %other = memref.generic_atomic_rmw %b[%i] : memref<?xf32> {
        ^bb0(%current: f32):
          %next = arith.addf %current, %v : f32
          memref.atomic_yield %next : f32
        }
        %r = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               affine.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_1]]] : memref<4xf32>
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:               affine.store %[[MULF_0]], %[[ARG3]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             }
// CHECK:             %[[PARALLEL_0:.*]] = affine.parallel (%[[VAL_2:.*]]) = (0) to (4) reduce ("addf") -> (f32) {
// CHECK:               %[[LOAD_1:.*]] = affine.load %[[ALLOC_0]]{{\[}}%[[VAL_2]]] : memref<4xf32>
// CHECK:               %[[LOAD_2:.*]] = affine.load %[[ARG4]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               affine.store %[[CONSTANT_0]], %[[ARG4]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               %[[MULF_1:.*]] = arith.mulf %[[LOAD_2]], %[[ARG0]] : f32
// CHECK:               %[[MULF_2:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:               %[[LOAD_3:.*]] = affine.load %[[ARG2]][0] : memref<?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_3]], %[[MULF_1]] : f32
// CHECK:               affine.store %[[ADDF_0]], %[[ARG2]][0] : memref<?xf32>
// CHECK:               affine.yield %[[MULF_2]] : f32
// CHECK:             }
// CHECK:             memref.dealloc %[[ALLOC_0]] : memref<4xf32>
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%arg0: f32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      %alloc = memref.alloc() : memref<4xf32>
      affine.parallel (%arg5) = (0) to (4) {
        %2 = affine.load %arg1[%arg5] : memref<?xf32>
        affine.store %2, %alloc[%arg5] : memref<4xf32>
        %3 = arith.mulf %2, %arg0 : f32
        affine.store %3, %arg3[%arg5] : memref<?xf32>
      }
      %1 = affine.parallel (%arg5) = (0) to (4) reduce ("addf") -> (f32) {
        %2 = affine.load %alloc[%arg5] : memref<4xf32>
        %3 = affine.load %arg4[%arg5] : memref<?xf32>
        affine.store %cst, %arg4[%arg5] : memref<?xf32>
        %4 = arith.mulf %3, %arg0 : f32
        %5 = arith.mulf %3, %2 : f32
        %6 = enzyme.affine_atomic_rmw addf %4, %arg2, (#map) [%arg5] : (f32, memref<?xf32>) -> f32
        affine.yield %5 : f32
      }
      memref.dealloc %alloc : memref<4xf32>
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias},
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias},
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias},
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               affine.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_1]]] : memref<4xf32>
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:               affine.store %[[MULF_0]], %[[ARG3]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             }
// CHECK:             %[[PARALLEL_0:.*]] = affine.parallel (%[[VAL_2:.*]]) = (0) to (4) reduce ("addf") -> (f32) {
// CHECK:               %[[LOAD_1:.*]] = affine.load %[[ALLOC_0]]{{\[}}%[[VAL_2]]] : memref<4xf32>
// CHECK:               %[[LOAD_2:.*]] = affine.load %[[ARG4]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               affine.store %[[CONSTANT_0]], %[[ARG4]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               %[[MULF_1:.*]] = arith.mulf %[[LOAD_2]], %[[ARG0]] : f32
// CHECK:               %[[MULF_2:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:               %[[LOAD_3:.*]] = affine.load %[[ARG2]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_3]], %[[MULF_1]] : f32
// CHECK:               affine.store %[[ADDF_0]], %[[ARG2]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               affine.yield %[[MULF_2]] : f32
// CHECK:             }
// CHECK:             memref.dealloc %[[ALLOC_0]] : memref<4xf32>
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%arg0: f32, %arg1: memref<?xf32> {llvm.noalias}, %arg2: memref<?xf32> {llvm.noalias}, %arg3: memref<?xf32> {llvm.noalias}, %arg4: memref<?xf32> {llvm.noalias}) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      %alloc = memref.alloc() : memref<4xf32>
      affine.parallel (%arg5) = (0) to (4) {
        %2 = affine.load %arg1[%arg5] : memref<?xf32>
        affine.store %2, %alloc[%arg5] : memref<4xf32>
        %3 = arith.mulf %2, %arg0 : f32
        affine.store %3, %arg3[%arg5] : memref<?xf32>
      }
      %1 = affine.parallel (%arg5) = (0) to (4) reduce ("addf") -> (f32) {
        %2 = affine.load %alloc[%arg5] : memref<4xf32>
        %3 = affine.load %arg4[%arg5] : memref<?xf32>
        affine.store %cst, %arg4[%arg5] : memref<?xf32>
        %4 = arith.mulf %3, %arg0 : f32
        %5 = arith.mulf %3, %2 : f32
        %6 = enzyme.affine_atomic_rmw addf %4, %arg2, (#map) [%arg5] : (f32, memref<?xf32>) -> f32
        affine.yield %5 : f32
      }
      memref.dealloc %alloc : memref<4xf32>
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias},
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias},
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias},
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               affine.store %[[LOAD_0]], %[[ALLOC_0]]{{\[}}%[[VAL_1]]] : memref<4xf32>
// CHECK:               %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:               affine.store %[[MULF_0]], %[[ARG3]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             }
// CHECK:             %[[PARALLEL_0:.*]] = affine.parallel (%[[VAL_2:.*]]) = (0) to (4) reduce ("addf") -> (f32) {
// CHECK:               %[[LOAD_1:.*]] = affine.load %[[ALLOC_0]]{{\[}}%[[VAL_2]]] : memref<4xf32>
// CHECK:               %[[LOAD_2:.*]] = affine.load %[[ARG4]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               affine.store %[[CONSTANT_0]], %[[ARG4]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               %[[MULF_1:.*]] = arith.mulf %[[LOAD_2]], %[[ARG0]] : f32
// CHECK:               %[[MULF_2:.*]] = arith.mulf %[[LOAD_2]], %[[LOAD_1]] : f32
// CHECK:               %[[LOAD_3:.*]] = affine.load %[[ARG2]][0] : memref<?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_3]], %[[MULF_1]] : f32
// CHECK:               affine.store %[[ADDF_0]], %[[ARG2]][0] : memref<?xf32>
// CHECK:               affine.yield %[[MULF_2]] : f32
// CHECK:             }
// CHECK:             memref.dealloc %[[ALLOC_0]] : memref<4xf32>
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%arg0: f32, %arg1: memref<?xf32> {llvm.noalias}, %arg2: memref<?xf32> {llvm.noalias}, %arg3: memref<?xf32> {llvm.noalias}, %arg4: memref<?xf32> {llvm.noalias}) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      %alloc = memref.alloc() : memref<4xf32>
      affine.parallel (%arg5) = (0) to (4) {
        %2 = affine.load %arg1[%arg5] : memref<?xf32>
        affine.store %2, %alloc[%arg5] : memref<4xf32>
        %3 = arith.mulf %2, %arg0 : f32
        affine.store %3, %arg3[%arg5] : memref<?xf32>
      }
      %1 = affine.parallel (%arg5) = (0) to (4) reduce ("addf") -> (f32) {
        %2 = affine.load %alloc[%arg5] : memref<4xf32>
        %3 = affine.load %arg4[%arg5] : memref<?xf32>
        affine.store %cst, %arg4[%arg5] : memref<?xf32>
        %4 = arith.mulf %3, %arg0 : f32
        %5 = arith.mulf %3, %2 : f32
        %6 = enzyme.affine_atomic_rmw addf %4, %arg2, (#map) [%arg5] : (f32, memref<?xf32>) -> f32
        affine.yield %5 : f32
      }
      memref.dealloc %alloc : memref<4xf32>
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @enzyme_atomic_rmw_keeps_affine_atomic(
// CHECK:           enzyme.atomic_rmw addf
// CHECK:           enzyme.affine_atomic_rmw addf
  func.func @enzyme_atomic_rmw_keeps_affine_atomic(%v: f32, %a: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %other = enzyme.atomic_rmw addf %v, %a[%i] monotonic : (f32, memref<?xf32>) -> f32
        %r = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @enzyme_atomic_rmw_noalias_removes_affine_atomic(
// CHECK:           enzyme.atomic_rmw addf
// CHECK:           %[[OLD:.*]] = affine.load %[[A:.*]]{{\[}}%[[I:.*]]] : memref<?xf32>
// CHECK:           %[[NEW:.*]] = arith.addf %[[OLD]], %[[V:.*]] : f32
// CHECK:           affine.store %[[NEW]], %[[A]]{{\[}}%[[I]]] : memref<?xf32>
// CHECK-NOT:       enzyme.affine_atomic_rmw
// CHECK:           return
  func.func @enzyme_atomic_rmw_noalias_removes_affine_atomic(%v: f32,
      %a: memref<?xf32> {llvm.noalias},
      %b: memref<?xf32> {llvm.noalias}) {
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        %other = enzyme.atomic_rmw addf %v, %b[%i] monotonic : (f32, memref<?xf32>) -> f32
        %r = enzyme.affine_atomic_rmw addf %v, %a, (#map) [%i] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (1) to (4) {
// CHECK:               affine.store %[[ARG0]], %[[ARG2]]{{\[}}%[[VAL_1]] - 1] : memref<?xf32>
// CHECK:               %[[OLD:.*]] = affine.load %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               %[[NEW:.*]] = arith.addf %[[OLD]], %[[ARG0]] : f32
// CHECK:               affine.store %[[NEW]], %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK-NOT:           enzyme.affine_atomic_rmw
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arg1: memref<?xf32>, %arr: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%iv) = (1) to (4) {
        affine.store %a, %arr[%iv - 1] : memref<?xf32>
        %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%iv] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (1) to (4) {
// CHECK:               affine.store %[[ARG0]], %[[ARG2]]{{\[}}%[[VAL_1]] + 1] : memref<?xf32>
// CHECK:               %[[OLD:.*]] = affine.load %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               %[[NEW:.*]] = arith.addf %[[OLD]], %[[ARG0]] : f32
// CHECK:               affine.store %[[NEW]], %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK-NOT:           enzyme.affine_atomic_rmw
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arg1: memref<?xf32>, %arr: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%iv) = (1) to (4) {
        affine.store %a, %arr[%iv + 1] : memref<?xf32>
        %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%iv] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (1) to (4) {
// CHECK:               affine.store %[[ARG0]], %[[ARG2]]{{\[}}%[[VAL_1]] + 1] : memref<?xf32>
// CHECK:             }
// CHECK:             affine.parallel (%[[VAL_2:.*]]) = (1) to (4) {
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[ARG2]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:               affine.store %[[ADDF_0]], %[[ARG2]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arg1: memref<?xf32>, %arr: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%iv) = (1) to (4) {
        affine.store %a, %arr[%iv + 1] : memref<?xf32>
      }
      affine.parallel (%iv) = (1) to (4) {
        %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%iv] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (1) to (4) {
// CHECK:               affine.store %[[ARG0]], %[[ARG1]]{{\[}}%[[VAL_1]] + 1] : memref<?xf32>
// CHECK:             }
// CHECK:             affine.parallel (%[[VAL_2:.*]]) = (1) to (4) {
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[ARG2]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:               affine.store %[[ADDF_0]], %[[ARG2]]{{\[}}%[[VAL_2]]] : memref<?xf32>
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arr2: memref<?xf32>, %arr: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%iv) = (1) to (4) {
        affine.store %a, %arr2[%iv + 1] : memref<?xf32>
      }
      affine.parallel (%iv) = (1) to (4) {
        %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%iv] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias},
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias},
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias},
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32> {llvm.noalias}) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               affine.store %[[ARG0]], %[[ARG1]]{{\[}}%[[VAL_1]] + 4] : memref<?xf32>
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:               affine.store %[[ADDF_0]], %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arr2: memref<?xf32> {llvm.noalias}, %arr: memref<?xf32> {llvm.noalias}, %arg3: memref<?xf32> {llvm.noalias}, %arg4: memref<?xf32> {llvm.noalias}) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%iv) = (0) to (4) {
        affine.store %a, %arr2[%iv + 4] : memref<?xf32>
        %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%iv] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               affine.store %[[ARG0]], %[[ARG1]]{{\[}}%[[VAL_1]] + 4] : memref<?xf32>
// CHECK:               %[[OLD:.*]] = affine.load %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               %[[NEW:.*]] = arith.addf %[[OLD]], %[[ARG0]] : f32
// CHECK:               affine.store %[[NEW]], %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK-NOT:           enzyme.affine_atomic_rmw
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arr2: memref<?xf32>, %arr: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%iv) = (0) to (4) {
        affine.store %a, %arr2[%iv + 4] : memref<?xf32>
        %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%iv] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               affine.store %[[ARG0]], %[[ARG2]]{{\[}}%[[VAL_1]] + 4] : memref<?xf32>
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:               affine.store %[[ADDF_0]], %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arg1: memref<?xf32>, %arr: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%iv) = (0) to (4) {
        affine.store %a, %arr[%iv + 4] : memref<?xf32>
        %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%iv] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:               affine.store %[[ADDF_0]], %[[ARG2]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:               affine.store %[[ADDF_0]], %[[ARG2]]{{\[}}%[[VAL_1]] + 4] : memref<?xf32>
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arg1: memref<?xf32>, %arr: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%iv) = (0) to (4) {
        %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%iv] : (f32, memref<?xf32>) -> f32
        affine.store %6, %arr[%iv + 4] : memref<?xf32>
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?x?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               affine.parallel (%[[VAL_2:.*]]) = (0) to (4) {
// CHECK:                 affine.store %[[ARG0]], %[[ARG1]]{{\[}}%[[VAL_1]] + 4, %[[VAL_2]]] : memref<?x?xf32>
// CHECK:                 %[[LOAD_0:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]], %[[VAL_2]]] : memref<?x?xf32>
// CHECK:                 %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:                 affine.store %[[ADDF_0]], %[[ARG1]]{{\[}}%[[VAL_1]], %[[VAL_2]]] : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arr: memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        affine.parallel (%j) = (0) to (4) {
          affine.store %a, %arr[%i + 4, %j] : memref<?x?xf32>
          %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%i, %j] : (f32, memref<?x?xf32>) -> f32
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0 + 1, d1)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?x?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               affine.parallel (%[[VAL_2:.*]]) = (0) to (4) {
// CHECK:                 affine.store %[[ARG0]], %[[ARG1]]{{\[}}%[[VAL_1]] + 4, %[[VAL_2]]] : memref<?x?xf32>
// CHECK:                 %[[OLD:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]] + 1, %[[VAL_2]]] : memref<?x?xf32>
// CHECK:                 %[[NEW:.*]] = arith.addf %[[OLD]], %[[ARG0]] : f32
// CHECK:                 affine.store %[[NEW]], %[[ARG1]]{{\[}}%[[VAL_1]] + 1, %[[VAL_2]]] : memref<?x?xf32>
// CHECK-NOT:             enzyme.affine_atomic_rmw
// CHECK:               }
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arr: memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        affine.parallel (%j) = (0) to (4) {
          affine.store %a, %arr[%i + 4, %j] : memref<?x?xf32>
          %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%i, %j] : (f32, memref<?x?xf32>) -> f32
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0 + 1, d1)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?x?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.for %[[VAL_1:.*]] = 0 to 4 {
// CHECK:               affine.parallel (%[[VAL_2:.*]]) = (0) to (4) {
// CHECK:                 affine.store %[[ARG0]], %[[ARG1]]{{\[}}%[[VAL_1]] + 4, %[[VAL_2]]] : memref<?x?xf32>
// CHECK:                 %[[LOAD_0:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]] + 1, %[[VAL_2]]] : memref<?x?xf32>
// CHECK:                 %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:                 affine.store %[[ADDF_0]], %[[ARG1]]{{\[}}%[[VAL_1]] + 1, %[[VAL_2]]] : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arr: memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.for %i = 0 to 4 {
        affine.parallel (%j) = (0) to (4) {
          affine.store %a, %arr[%i + 4, %j] : memref<?x?xf32>
          %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%i, %j] : (f32, memref<?x?xf32>) -> f32
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1 + 1)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?x?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               affine.parallel (%[[VAL_2:.*]]) = (0) to (4) {
// CHECK:                 affine.store %[[ARG0]], %[[ARG1]]{{\[}}%[[VAL_1]], %[[VAL_2]] + 4] : memref<?x?xf32>
// CHECK:                 %[[OLD:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]], %[[VAL_2]] + 1] : memref<?x?xf32>
// CHECK:                 %[[NEW:.*]] = arith.addf %[[OLD]], %[[ARG0]] : f32
// CHECK:                 affine.store %[[NEW]], %[[ARG1]]{{\[}}%[[VAL_1]], %[[VAL_2]] + 1] : memref<?x?xf32>
// CHECK-NOT:             enzyme.affine_atomic_rmw
// CHECK:               }
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arr: memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        affine.parallel (%j) = (0) to (4) {
          affine.store %a, %arr[%i, %j + 4] : memref<?x?xf32>
          %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%i, %j] : (f32, memref<?x?xf32>) -> f32
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1 + 1)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?x?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]]) = (0) to (4) {
// CHECK:               affine.for %[[VAL_2:.*]] = 0 to 4 {
// CHECK:                 affine.store %[[ARG0]], %[[ARG1]]{{\[}}%[[VAL_1]], %[[VAL_2]] + 4] : memref<?x?xf32>
// CHECK:                 %[[LOAD_0:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]], %[[VAL_2]] + 1] : memref<?x?xf32>
// CHECK:                 %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:                 affine.store %[[ADDF_0]], %[[ARG1]]{{\[}}%[[VAL_1]], %[[VAL_2]] + 1] : memref<?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arr: memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i) = (0) to (4) {
        affine.for %j = 0 to 4 {
          affine.store %a, %arr[%i, %j + 4] : memref<?x?xf32>
          %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%i, %j] : (f32, memref<?x?xf32>) -> f32
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?x?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]], %[[VAL_2:.*]]) = (0, 0) to (4, 4) {
// CHECK:               affine.store %[[ARG0]], %[[ARG1]]{{\[}}%[[VAL_1]] + 4, %[[VAL_2]]] : memref<?x?xf32>
// CHECK:               %[[LOAD_0:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]], %[[VAL_2]]] : memref<?x?xf32>
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[LOAD_0]], %[[ARG0]] : f32
// CHECK:               affine.store %[[ADDF_0]], %[[ARG1]]{{\[}}%[[VAL_1]], %[[VAL_2]]] : memref<?x?xf32>
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arr: memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i, %j) = (0, 0) to (4, 4) {
        affine.store %a, %arr[%i + 4, %j] : memref<?x?xf32>
        %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%i, %j] : (f32, memref<?x?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0 + 1, d1)>
module {
// CHECK-LABEL:   func.func @affine(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<?x?xf32>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]] = "enzymexla.gpu_wrapper"(%[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]], %[[CONSTANT_1]]) ({
// CHECK:             affine.parallel (%[[VAL_1:.*]], %[[VAL_2:.*]]) = (0, 0) to (4, 4) {
// CHECK:               affine.store %[[ARG0]], %[[ARG1]]{{\[}}%[[VAL_1]] + 4, %[[VAL_2]]] : memref<?x?xf32>
// CHECK:               %[[OLD:.*]] = affine.load %[[ARG1]]{{\[}}%[[VAL_1]] + 1, %[[VAL_2]]] : memref<?x?xf32>
// CHECK:               %[[NEW:.*]] = arith.addf %[[OLD]], %[[ARG0]] : f32
// CHECK:               affine.store %[[NEW]], %[[ARG1]]{{\[}}%[[VAL_1]] + 1, %[[VAL_2]]] : memref<?x?xf32>
// CHECK-NOT:           enzyme.affine_atomic_rmw
// CHECK:             }
// CHECK:             "enzymexla.polygeist_yield"() : () -> ()
// CHECK:           }) : (index, index, index, index, index, index) -> index
// CHECK:           return
// CHECK:         }
  func.func @affine(%a: f32, %arr: memref<?x?xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c1, %c1, %c1) ({
      affine.parallel (%i, %j) = (0, 0) to (4, 4) {
        affine.store %a, %arr[%i + 4, %j] : memref<?x?xf32>
        %6 = enzyme.affine_atomic_rmw addf %a, %arr, (#map) [%i, %j] : (f32, memref<?x?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    return
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0 + d1 * 5 + 14)>
module {
// CHECK-LABEL:   func.func @par(
// CHECK:           affine.parallel (%[[I:.*]], %[[J:.*]]) = (0, 0) to (4, 5) {
// CHECK:             %[[PRED:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK:             %[[SEL:.*]] = scf.if %[[PRED]] -> (f32) {
// CHECK:             } else {
// CHECK:               affine.load %[[ARR:.*]]{{\[}}%[[I]] + %[[J]] * 5 + 16] : memref<?xf32>
// CHECK:             }
// CHECK:             %[[OLD:.*]] = affine.load %[[ARR]]{{\[}}%[[I]] + %[[J]] * 5 + 14] : memref<?xf32>
// CHECK:             %[[NEW:.*]] = arith.addf %[[OLD]], %[[SEL]] : f32
// CHECK:             affine.store %[[NEW]], %[[ARR]]{{\[}}%[[I]] + %[[J]] * 5 + 14] : memref<?xf32>
// CHECK-NOT:         enzyme.affine_atomic_rmw
  func.func @par(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
    %1 = "enzymexla.gpu_wrapper"(%c4, %c5) ({
      affine.parallel (%arg2, %arg3) = (0, 0) to (4, 5) {
        %2 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xi8>
        %3 = affine.load %2[%arg2] : memref<?xi8>
        %4 = arith.extui %3 : i8 to i32
        %5 = arith.andi %4, %c1_i32 : i32
        %6 = arith.cmpi eq, %5, %c0_i32 : i32
        %7 = scf.if %6 -> (f32) {
          scf.yield %cst : f32
        } else {
          %9 = affine.load %0[%arg2 + %arg3 * 5 + 16] : memref<?xf32>
          scf.yield %9 : f32
        }
        %8 = enzyme.affine_atomic_rmw addf %7, %0, (#map) [%arg2, %arg3] : (f32, memref<?xf32>) -> f32
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index) -> index
    return
  }
}
