// RUN: enzymexlamlir-opt %s --split-input-file --pass-pipeline="builtin.module(raise-affine-to-stablehlo{err_if_not_fully_raised=false},enzyme-hlo-opt{max_constant_expansion=0})" | FileCheck %s

#set = affine_set<(d0) : (-d0 + 89 >= 0)>
#set2 = affine_set<(d0) : (-d0 + 70 >= 0)>
  func.func private @if_with_load(%m1: memref<194xf64, 1>, %m2: memref<194xf64, 1>, %m3: memref<194xf64, 1>) {
    affine.parallel (%arg1) = (0) to (191) {
      affine.if #set2(%arg1) {
        %ld = affine.load %m1[%arg1 + 2] : memref<194xf64, 1>
        affine.store %ld, %m3[%arg1] : memref<194xf64, 1>
      } else {
        %ld = affine.load %m2[%arg1 + 3] : memref<194xf64, 1>
        affine.store %ld, %m3[%arg1] : memref<194xf64, 1>
      }
    }
    return
  }
  // -----
#set = affine_set<(d0) : (-d0 + 89 >= 0)>
#set2 = affine_set<(d0) : (-d0 + 70 >= 0)>
  func.func private @if_yield_with_load(%m1: memref<194xf64, 1>, %m2: memref<194xf64, 1>, %m3: memref<194xf64, 1>) {
    affine.parallel (%arg1) = (0) to (191) {
      %1 = affine.if #set2(%arg1) -> f64 {
        %ld = affine.load %m1[%arg1 + 2] : memref<194xf64, 1>
        affine.yield %ld : f64
      } else {
        %ld = affine.load %m2[%arg1 + 3] : memref<194xf64, 1>
        affine.yield %ld : f64
      }
      affine.store %1, %m3[%arg1] : memref<194xf64, 1>
    }
    return
  }
  // -----
#set = affine_set<(d0) : (-d0 + 89 >= 0)>
#set2 = affine_set<(d0) : (-d0 + 70 >= 0)>
  func.func private @with_load_out_of_bounds(%m1: memref<194xf64, 1>, %m2: memref<194xf64, 1>, %m3: memref<194xf64, 1>) {
    affine.parallel (%arg1) = (0) to (193) {
      %1 = affine.if #set2(%arg1) -> f64 {
        %ld = affine.load %m1[%arg1 + 2] : memref<194xf64, 1>
        affine.yield %ld : f64
      } else {
        %ld = affine.load %m2[%arg1 + 3] : memref<194xf64, 1>
        affine.yield %ld : f64
      }
      affine.store %1, %m3[%arg1] : memref<194xf64, 1>
    }
    return
  }
  // -----
#set = affine_set<(d0) : (-d0 + 89 >= 0)>
#set2 = affine_set<(d0) : (-d0 + 70 >= 0)>
  func.func private @if_with_multidimload(%m1: memref<100x194xf64, 1>, %m2: memref<100x194xf64, 1>, %m3: memref<100x194xf64, 1>) {
    affine.parallel (%a1, %arg1) = (2, 0) to (100, 192) {
      %1 = affine.if #set2(%arg1) -> f64 {
        %ld = affine.load %m1[%a1 - 2, %arg1 + 2] : memref<100x194xf64, 1>
        affine.yield %ld : f64
      } else {
        %ld = affine.load %m2[%a1 - 2, %arg1 + 2] : memref<100x194xf64, 1>
        affine.yield %ld : f64
      }
      affine.store %1, %m3[%a1, %arg1] : memref<100x194xf64, 1>
    }
    return
  }
  // -----
#set = affine_set<(d0) : (-d0 + 89 >= 0)>
#set2 = affine_set<(d0) : (-d0 + 70 >= 0)>
  func.func private @if_with_multidimload_out_of_bounds(%m1: memref<100x194xf64, 1>, %m2: memref<100x194xf64, 1>, %m3: memref<100x194xf64, 1>) {
    affine.parallel (%a1, %arg1) = (1, 0) to (100, 192) {
      %1 = affine.if #set2(%arg1) -> f64 {
        %ld = affine.load %m1[%a1 - 2, %arg1 + 2] : memref<100x194xf64, 1>
        affine.yield %ld : f64
      } else {
        %ld = affine.load %m2[%a1 - 2, %arg1 + 2] : memref<100x194xf64, 1>
        affine.yield %ld : f64
      }
      affine.store %1, %m3[%a1, %arg1] : memref<100x194xf64, 1>
    }
    return
  }



// CHECK-LABEL:   func.func private @if_with_load(
// CHECK-SAME:                                    %[[VAL_0:.*]]: memref<194xf64, 1>, %[[VAL_1:.*]]: memref<194xf64, 1>, %[[VAL_2:.*]]: memref<194xf64, 1>) {
// CHECK:           affine.parallel (%arg3) = (0) to (191) {
// CHECK:           affine.if #set(%arg3) {
// CHECK:           %[[VAL_3:.*]] = affine.load %[[VAL_0]][%arg3 + 2] : memref<194xf64, 1>
// CHECK:           affine.store %[[VAL_3]], %[[VAL_2]][%arg3] : memref<194xf64, 1>
// CHECK:           } else {
// CHECK:           %[[VAL_3]] = affine.load %[[VAL_1]][%arg3 + 3] : memref<194xf64, 1>
// CHECK:           affine.store %[[VAL_3]], %[[VAL_2]][%arg3] : memref<194xf64, 1>
// CHECK:           }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @if_yield_with_load(
// CHECK-SAME:                                          %[[VAL_0:.*]]: memref<194xf64, 1>, %[[VAL_1:.*]]: memref<194xf64, 1>, %[[VAL_2:.*]]: memref<194xf64, 1>) {
// CHECK:           affine.parallel (%arg3) = (0) to (191) {
// CHECK:           %[[VAL_3:.*]] = affine.if #set(%arg3) -> f64 {
// CHECK:           %[[VAL_4:.*]] = affine.load %[[VAL_0]][%arg3 + 2] : memref<194xf64, 1>
// CHECK:           affine.yield %[[VAL_4]] : f64
// CHECK:           } else {
// CHECK:           %[[VAL_4]] = affine.load %[[VAL_1]][%arg3 + 3] : memref<194xf64, 1>
// CHECK:           affine.yield %[[VAL_4]] : f64
// CHECK:           }
// CHECK:           affine.store %[[VAL_3]], %[[VAL_2]][%arg3] : memref<194xf64, 1>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @with_load_out_of_bounds(
// CHECK-SAME:                                               %[[VAL_0:.*]]: memref<194xf64, 1>, %[[VAL_1:.*]]: memref<194xf64, 1>, %[[VAL_2:.*]]: memref<194xf64, 1>) {
// CHECK:           affine.parallel (%arg3) = (0) to (193) {
// CHECK:           %[[VAL_3:.*]] = affine.if #set(%arg3) -> f64 {
// CHECK:           %[[VAL_4:.*]] = affine.load %[[VAL_0]][%arg3 + 2] : memref<194xf64, 1>
// CHECK:           affine.yield %[[VAL_4]] : f64
// CHECK:           } else {
// CHECK:           %[[VAL_4]] = affine.load %[[VAL_1]][%arg3 + 3] : memref<194xf64, 1>
// CHECK:           affine.yield %[[VAL_4]] : f64
// CHECK:           }
// CHECK:           affine.store %[[VAL_3]], %[[VAL_2]][%arg3] : memref<194xf64, 1>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @if_with_multidimload(
// CHECK-SAME:                                            %[[VAL_0:.*]]: memref<100x194xf64, 1>, %[[VAL_1:.*]]: memref<100x194xf64, 1>, %[[VAL_2:.*]]: memref<100x194xf64, 1>) {
// CHECK:           affine.parallel (%arg3, %[[VAL_3:.*]] = (2, 0) to (100, 192) {
// CHECK:           %[[VAL_4:.*]] = affine.if #set(%[[VAL_3]]) -> f64 {
// CHECK:           %[[VAL_5:.*]] = affine.load %[[VAL_0]][%arg3 - 2, %[[VAL_3]] + 2] : memref<100x194xf64, 1>
// CHECK:           affine.yield %[[VAL_5]] : f64
// CHECK:           } else {
// CHECK:           %[[VAL_5]] = affine.load %[[VAL_1]][%arg3 - 2, %[[VAL_3]] + 2] : memref<100x194xf64, 1>
// CHECK:           affine.yield %[[VAL_5]] : f64
// CHECK:           }
// CHECK:           affine.store %[[VAL_4]], %[[VAL_2]][%arg3, %[[VAL_3]]] : memref<100x194xf64, 1>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @if_with_multidimload_out_of_bounds(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: memref<100x194xf64, 1>, %[[VAL_1:.*]]: memref<100x194xf64, 1>, %[[VAL_2:.*]]: memref<100x194xf64, 1>) {
// CHECK:           affine.parallel (%arg3, %[[VAL_3:.*]] = (1, 0) to (100, 192) {
// CHECK:           %[[VAL_4:.*]] = affine.if #set(%[[VAL_3]]) -> f64 {
// CHECK:           %[[VAL_5:.*]] = affine.load %[[VAL_0]][%arg3 - 2, %[[VAL_3]] + 2] : memref<100x194xf64, 1>
// CHECK:           affine.yield %[[VAL_5]] : f64
// CHECK:           } else {
// CHECK:           %[[VAL_5]] = affine.load %[[VAL_1]][%arg3 - 2, %[[VAL_3]] + 2] : memref<100x194xf64, 1>
// CHECK:           affine.yield %[[VAL_5]] : f64
// CHECK:           }
// CHECK:           affine.store %[[VAL_4]], %[[VAL_2]][%arg3, %[[VAL_3]]] : memref<100x194xf64, 1>
// CHECK:           }
// CHECK:           return
// CHECK:         }
