// RUN: enzymexlamlir-opt --remove-atomics %s  --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
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
// CHECK:               %[[AFFINE_ATOMIC_RMW_0:.*]] = enzyme.affine_atomic_rmw addf %[[MULF_1]], %[[ARG2]], (#[[$ATTR_0]]) {{\[}}%[[VAL_2]]] : (f32, memref<?xf32>) -> f32
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

#map = affine_map<(d0) -> (0)>
module {
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (0)>
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
// CHECK:               %[[AFFINE_ATOMIC_RMW_0:.*]] = enzyme.affine_atomic_rmw addf %[[MULF_1]], %[[ARG2]], (#[[$ATTR_1]]) {{\[}}%[[VAL_2]]] : (f32, memref<?xf32>) -> f32
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
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (0)>
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
// CHECK:               %[[AFFINE_ATOMIC_RMW_0:.*]] = enzyme.affine_atomic_rmw addf %[[MULF_1]], %[[ARG2]], (#[[$ATTR_2]]) {{\[}}%[[VAL_2]]] : (f32, memref<?xf32>) -> f32
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
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0) -> (d0)>
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
// CHECK:               %[[AFFINE_ATOMIC_RMW_0:.*]] = enzyme.affine_atomic_rmw addf %[[ARG0]], %[[ARG2]], (#[[$ATTR_3]]) {{\[}}%[[VAL_1]]] : (f32, memref<?xf32>) -> f32
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
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0) -> (d0)>
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
// CHECK:               %[[AFFINE_ATOMIC_RMW_0:.*]] = enzyme.affine_atomic_rmw addf %[[ARG0]], %[[ARG2]], (#[[$ATTR_4]]) {{\[}}%[[VAL_1]]] : (f32, memref<?xf32>) -> f32
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
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0) -> (d0)>
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
// CHECK:               %[[AFFINE_ATOMIC_RMW_0:.*]] = enzyme.affine_atomic_rmw addf %[[ARG0]], %[[ARG2]], (#[[$ATTR_5]]) {{\[}}%[[VAL_1]]] : (f32, memref<?xf32>) -> f32
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
