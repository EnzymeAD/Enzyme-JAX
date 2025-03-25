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
// CHECK:           affine

// CHECK-LABEL:   func.func private @if_yield_with_load_raised(
// CHECK:           stablehlo

// CHECK-LABEL:   func.func private @with_load_out_of_bounds(
// CHECK:           affine

// CHECK-LABEL:   func.func private @if_with_multidimload_raised(
// CHECK:           stablehlo

// CHECK-LABEL:   func.func private @if_with_multidimload_out_of_bounds
// CHECK:           affine
