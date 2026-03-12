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



// CHECK-LABEL:   func.func private @if_with_load_raised(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<194xf64>, %[[VAL_1:.*]]: tensor<194xf64>, %[[VAL_2:.*]]: tensor<194xf64>) -> (tensor<194xf64>, tensor<194xf64>, tensor<194xf64>) {
// CHECK:           %[[C70:.*]] = stablehlo.constant {{.*}} dense<70> : tensor<191xi64>
// CHECK:           %[[C0:.*]] = stablehlo.constant dense<0> : tensor<191xi64>
// CHECK:           %[[IOTA:.*]] = stablehlo.iota dim = 0 : tensor<191xi64>
// CHECK:           %[[SUB:.*]] = stablehlo.subtract %[[C70]], %[[IOTA]] {{.*}} : tensor<191xi64>
// CHECK:           %[[CMP:.*]] = stablehlo.compare  GE, %[[SUB]], %[[C0]] : (tensor<191xi64>, tensor<191xi64>) -> tensor<191xi1>
// CHECK:           %[[SL0:.*]] = stablehlo.slice %[[VAL_0]] [2:193] : (tensor<194xf64>) -> tensor<191xf64>
// CHECK:           %[[SL2:.*]] = stablehlo.slice %[[VAL_2]] [0:191] : (tensor<194xf64>) -> tensor<191xf64>
// CHECK:           %[[SEL0:.*]] = stablehlo.select %[[CMP]], %[[SL0]], %[[SL2]] : tensor<191xi1>, tensor<191xf64>
// CHECK:           %[[SL1:.*]] = stablehlo.slice %[[VAL_1]] [3:194] : (tensor<194xf64>) -> tensor<191xf64>
// CHECK:           %[[SEL1:.*]] = stablehlo.select %[[CMP]], %[[SEL0]], %[[SL1]] : tensor<191xi1>, tensor<191xf64>
// CHECK:           %[[TAIL:.*]] = stablehlo.slice %[[VAL_2]] [191:194] : (tensor<194xf64>) -> tensor<3xf64>
// CHECK:           %[[CAT:.*]] = stablehlo.concatenate %[[SEL1]], %[[TAIL]], dim = 0 : (tensor<191xf64>, tensor<3xf64>) -> tensor<194xf64>
// CHECK:           return %[[VAL_0]], %[[VAL_1]], %[[CAT]] : tensor<194xf64>, tensor<194xf64>, tensor<194xf64>
// CHECK:         }

// CHECK-LABEL:   func.func private @if_yield_with_load_raised(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: tensor<194xf64>, %[[VAL_1:.*]]: tensor<194xf64>, %[[VAL_2:.*]]: tensor<194xf64>) -> (tensor<194xf64>, tensor<194xf64>, tensor<194xf64>) {
// CHECK:           %[[C70:.*]] = stablehlo.constant {{.*}} dense<70> : tensor<191xi64>
// CHECK:           %[[C0:.*]] = stablehlo.constant dense<0> : tensor<191xi64>
// CHECK:           %[[IOTA:.*]] = stablehlo.iota dim = 0 : tensor<191xi64>
// CHECK:           %[[SUB:.*]] = stablehlo.subtract %[[C70]], %[[IOTA]] {{.*}} : tensor<191xi64>
// CHECK:           %[[CMP:.*]] = stablehlo.compare  GE, %[[SUB]], %[[C0]] : (tensor<191xi64>, tensor<191xi64>) -> tensor<191xi1>
// CHECK:           %[[SL0:.*]] = stablehlo.slice %[[VAL_0]] [2:193] : (tensor<194xf64>) -> tensor<191xf64>
// CHECK:           %[[SL1:.*]] = stablehlo.slice %[[VAL_1]] [3:194] : (tensor<194xf64>) -> tensor<191xf64>
// CHECK:           %[[SEL:.*]] = stablehlo.select %[[CMP]], %[[SL0]], %[[SL1]] : tensor<191xi1>, tensor<191xf64>
// CHECK:           %[[TAIL:.*]] = stablehlo.slice %[[VAL_2]] [191:194] : (tensor<194xf64>) -> tensor<3xf64>
// CHECK:           %[[CAT:.*]] = stablehlo.concatenate %[[SEL]], %[[TAIL]], dim = 0 : (tensor<191xf64>, tensor<3xf64>) -> tensor<194xf64>
// CHECK:           return %[[VAL_0]], %[[VAL_1]], %[[CAT]] : tensor<194xf64>, tensor<194xf64>, tensor<194xf64>
// CHECK:         }

// CHECK-LABEL:   func.func private @with_load_out_of_bounds_raised(
// CHECK-SAME:                                                       %[[VAL_0:.*]]: tensor<194xf64>, %[[VAL_1:.*]]: tensor<194xf64>, %[[VAL_2:.*]]: tensor<194xf64>) -> (tensor<194xf64>, tensor<194xf64>, tensor<194xf64>) {
// CHECK:           %[[C70:.*]] = stablehlo.constant {{.*}} dense<70> : tensor<193xi64>
// CHECK:           %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:           %[[C0:.*]] = stablehlo.constant dense<0> : tensor<193xi64>
// CHECK:           %[[IOTA:.*]] = stablehlo.iota dim = 0 : tensor<193xi64>
// CHECK:           %[[SUB:.*]] = stablehlo.subtract %[[C70]], %[[IOTA]] {{.*}} : tensor<193xi64>
// CHECK:           %[[CMP:.*]] = stablehlo.compare  GE, %[[SUB]], %[[C0]] : (tensor<193xi64>, tensor<193xi64>) -> tensor<193xi1>
// CHECK:           %[[SL0:.*]] = stablehlo.slice %[[VAL_0]] [2:194] : (tensor<194xf64>) -> tensor<192xf64>
// CHECK:           %[[PAD0:.*]] = stablehlo.pad %[[SL0]], %[[CST]], low = [0], high = [1], interior = [0] : (tensor<192xf64>, tensor<f64>) -> tensor<193xf64>
// CHECK:           %[[SL1:.*]] = stablehlo.slice %[[VAL_1]] [3:194] : (tensor<194xf64>) -> tensor<191xf64>
// CHECK:           %[[PAD1:.*]] = stablehlo.pad %[[SL1]], %[[CST]], low = [0], high = [2], interior = [0] : (tensor<191xf64>, tensor<f64>) -> tensor<193xf64>
// CHECK:           %[[SEL:.*]] = stablehlo.select %[[CMP]], %[[PAD0]], %[[PAD1]] : tensor<193xi1>, tensor<193xf64>
// CHECK:           %[[TAIL:.*]] = stablehlo.slice %[[VAL_2]] [193:194] : (tensor<194xf64>) -> tensor<1xf64>
// CHECK:           %[[CAT:.*]] = stablehlo.concatenate %[[SEL]], %[[TAIL]], dim = 0 : (tensor<193xf64>, tensor<1xf64>) -> tensor<194xf64>
// CHECK:           return %[[VAL_0]], %[[VAL_1]], %[[CAT]] : tensor<194xf64>, tensor<194xf64>, tensor<194xf64>
// CHECK:         }

// CHECK-LABEL:   func.func private @if_with_multidimload_raised(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: tensor<100x194xf64>, %[[VAL_1:.*]]: tensor<100x194xf64>, %[[VAL_2:.*]]: tensor<100x194xf64>) -> (tensor<100x194xf64>, tensor<100x194xf64>, tensor<100x194xf64>) {
// CHECK:           %[[C70:.*]] = stablehlo.constant {{.*}} dense<70> : tensor<192xi64>
// CHECK:           %[[C0I:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[C2I:.*]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK:           %[[C0:.*]] = stablehlo.constant dense<0> : tensor<192xi64>
// CHECK:           %[[IOTA:.*]] = stablehlo.iota dim = 0 : tensor<192xi64>
// CHECK:           %[[SUB:.*]] = stablehlo.subtract %[[C70]], %[[IOTA]] {{.*}} : tensor<192xi64>
// CHECK:           %[[CMP:.*]] = stablehlo.compare  GE, %[[SUB]], %[[C0]] : (tensor<192xi64>, tensor<192xi64>) -> tensor<192xi1>
// CHECK:           %[[SL0:.*]] = stablehlo.slice %[[VAL_0]] [0:98, 2:194] : (tensor<100x194xf64>) -> tensor<98x192xf64>
// CHECK:           %[[SL1:.*]] = stablehlo.slice %[[VAL_1]] [0:98, 2:194] : (tensor<100x194xf64>) -> tensor<98x192xf64>
// CHECK:           %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CMP]], dims = [0] : (tensor<192xi1>) -> tensor<192x98xi1>
// CHECK:           %[[TR0:.*]] = stablehlo.transpose %[[SL0]], dims = [1, 0] : (tensor<98x192xf64>) -> tensor<192x98xf64>
// CHECK:           %[[TR1:.*]] = stablehlo.transpose %[[SL1]], dims = [1, 0] : (tensor<98x192xf64>) -> tensor<192x98xf64>
// CHECK:           %[[SEL:.*]] = stablehlo.select %[[BCAST]], %[[TR0]], %[[TR1]] : tensor<192x98xi1>, tensor<192x98xf64>
// CHECK:           %[[TR2:.*]] = stablehlo.transpose %[[SEL]], dims = [1, 0] : (tensor<192x98xf64>) -> tensor<98x192xf64>
// CHECK:           %[[UPD:.*]] = stablehlo.dynamic_update_slice %[[VAL_2]], %[[TR2]], %[[C2I]], %[[C0I]] : (tensor<100x194xf64>, tensor<98x192xf64>, tensor<i64>, tensor<i64>) -> tensor<100x194xf64>
// CHECK:           return %[[VAL_0]], %[[VAL_1]], %[[UPD]] : tensor<100x194xf64>, tensor<100x194xf64>, tensor<100x194xf64>
// CHECK:         }

// CHECK-LABEL:   func.func private @if_with_multidimload_out_of_bounds_raised(
// CHECK-SAME:                                                                   %[[VAL_0:.*]]: tensor<100x194xf64>, %[[VAL_1:.*]]: tensor<100x194xf64>, %[[VAL_2:.*]]: tensor<100x194xf64>) -> (tensor<100x194xf64>, tensor<100x194xf64>, tensor<100x194xf64>) {
// CHECK:           %[[C70:.*]] = stablehlo.constant {{.*}} dense<70> : tensor<192xi64>
// CHECK:           %[[C0I:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[C1I:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:           %[[C0:.*]] = stablehlo.constant dense<0> : tensor<192xi64>
// CHECK:           %[[IOTA:.*]] = stablehlo.iota dim = 0 : tensor<192xi64>
// CHECK:           %[[SUB:.*]] = stablehlo.subtract %[[C70]], %[[IOTA]] {{.*}} : tensor<192xi64>
// CHECK:           %[[CMP:.*]] = stablehlo.compare  GE, %[[SUB]], %[[C0]] : (tensor<192xi64>, tensor<192xi64>) -> tensor<192xi1>
// CHECK:           %[[SL0:.*]] = stablehlo.slice %[[VAL_0]] [0:98, 2:194] : (tensor<100x194xf64>) -> tensor<98x192xf64>
// CHECK:           %[[SL1:.*]] = stablehlo.slice %[[VAL_1]] [0:98, 2:194] : (tensor<100x194xf64>) -> tensor<98x192xf64>
// CHECK:           %[[TR0:.*]] = stablehlo.transpose %[[SL0]], dims = [1, 0] : (tensor<98x192xf64>) -> tensor<192x98xf64>
// CHECK:           %[[TR1:.*]] = stablehlo.transpose %[[SL1]], dims = [1, 0] : (tensor<98x192xf64>) -> tensor<192x98xf64>
// CHECK:           %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[CMP]], dims = [0] : (tensor<192xi1>) -> tensor<192x98xi1>
// CHECK:           %[[SEL:.*]] = stablehlo.select %[[BCAST]], %[[TR0]], %[[TR1]] : tensor<192x98xi1>, tensor<192x98xf64>
// CHECK:           %[[TR2:.*]] = stablehlo.transpose %[[SEL]], dims = [1, 0] : (tensor<192x98xf64>) -> tensor<98x192xf64>
// CHECK:           %[[PAD:.*]] = stablehlo.pad %[[TR2]], %[[CST]], low = [1, 0], high = [0, 0], interior = [0, 0] : (tensor<98x192xf64>, tensor<f64>) -> tensor<99x192xf64>
// CHECK:           %[[UPD:.*]] = stablehlo.dynamic_update_slice %[[VAL_2]], %[[PAD]], %[[C1I]], %[[C0I]] : (tensor<100x194xf64>, tensor<99x192xf64>, tensor<i64>, tensor<i64>) -> tensor<100x194xf64>
// CHECK:           return %[[VAL_0]], %[[VAL_1]], %[[UPD]] : tensor<100x194xf64>, tensor<100x194xf64>, tensor<100x194xf64>
// CHECK:         }
