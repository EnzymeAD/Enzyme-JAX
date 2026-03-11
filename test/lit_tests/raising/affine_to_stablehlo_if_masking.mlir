// RUN: enzymexlamlir-opt %s -raise-affine-to-stablehlo | FileCheck %s

func.func @test_if_masking(%arg0: memref<10xf32>, %arg1: memref<10xi1>) {
  affine.for %i = 0 to 10 {
    %cond = affine.load %arg1[%i] : memref<10xi1>
    scf.if %cond {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.addf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    }
  }
  return
}

// CHECK-LABEL:   func.func private @test_if_masking_raised(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<10xf32>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.dynamic_slice %[[VAL_1]], %[[VAL_2]], sizes = [10] : (tensor<10xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK:           %[[VAL_4:.*]] = stablehlo.reshape %[[VAL_3]] : (tensor<10xi1>) -> tensor<10xi1>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[VAL_5]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.reshape %[[VAL_6]] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_8:.*]] = arith.addf %[[VAL_7]], %[[VAL_7]] : tensor<10xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_10:.*]] = stablehlo.broadcast_in_dim %[[VAL_8]], dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.dynamic_slice %[[VAL_0]], %[[VAL_9]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.select %[[VAL_4]], %[[VAL_10]], %[[VAL_11]] : tensor<10xi1>, tensor<10xf32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %[[VAL_12]], %[[VAL_9]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.select %[[VAL_4]], %[[VAL_13]], %[[VAL_0]] : tensor<10xi1>, tensor<10xf32>
// CHECK:           return %[[VAL_14]], %[[VAL_1]] : tensor<10xf32>, tensor<10xi1>
// CHECK:         }

func.func @test_nested_if_masking(%arg0: memref<10xf32>, %arg1: memref<10xi1>, %arg2: memref<10xi1>) {
  affine.for %i = 0 to 10 {
    %cond1 = affine.load %arg1[%i] : memref<10xi1>
    scf.if %cond1 {
      %cond2 = affine.load %arg2[%i] : memref<10xi1>
      scf.if %cond2 {
        %v = affine.load %arg0[%i] : memref<10xf32>
        %v2 = arith.addf %v, %v : f32
        affine.store %v2, %arg0[%i] : memref<10xf32>
      }
    }
  }
  return
}

func.func @test_if_else_masking(%arg0: memref<10xf32>, %arg1: memref<10xi1>) {
  affine.for %i = 0 to 10 {
    %cond = affine.load %arg1[%i] : memref<10xi1>
    scf.if %cond {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.addf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    } else {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.mulf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    }
  }
  return
}

func.func @test_affine_if_masking(%arg0: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    affine.if affine_set<(d0) : (d0 - 5 >= 0)>(%i) {
      %v = affine.load %arg0[%i] : memref<10xf32>
      %v2 = arith.addf %v, %v : f32
      affine.store %v2, %arg0[%i] : memref<10xf32>
    }
  }
  return
}

// CHECK-LABEL:   func.func private @test_affine_if_masking_raised(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<10xf32>) -> tensor<10xf32> {
// CHECK:           %0 = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.add %0, %[[VAL_1]] : tensor<10xi64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.multiply %[[VAL_2]], %[[VAL_3]] : tensor<10xi64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<-5> : tensor<i64>
// CHECK:           %3 = stablehlo.broadcast_in_dim %[[VAL_5]], dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.add %[[VAL_4]], %3 : tensor<10xi64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[VAL_8:.*]] = stablehlo.compare  GE, %[[VAL_6]], %[[VAL_7]] : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %6 = stablehlo.dynamic_slice %[[VAL_0]], %[[VAL_9]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.reshape %6 : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_11:.*]] = arith.addf %[[VAL_10]], %[[VAL_10]] : tensor<10xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %9 = stablehlo.broadcast_in_dim %[[VAL_11]], dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %9, %[[VAL_12]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.select %[[VAL_8]], %[[VAL_13]], %[[VAL_0]] : tensor<10xi1>, tensor<10xf32>
// CHECK:           return %[[VAL_14]] : tensor<10xf32>
// CHECK:         }

// CHECK-LABEL:   func.func private @test_if_else_masking_raised(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<10xf32>, %[[VAL_1:.*]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<10> : tensor<i64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           %0:3 = stablehlo.while(%iterArg = %[[VAL_2]], %iterArg_2 = %[[VAL_0]], %iterArg_3 = %[[VAL_1]]) : tensor<i64>, tensor<10xf32>, tensor<10xi1>
// CHECK:           cond {
// CHECK:           %[[VAL_5:.*]] = stablehlo.compare  LT, %iterArg, %[[VAL_3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK:           stablehlo.return %[[VAL_5]] : tensor<i1>
// CHECK:           } do {
// CHECK:           %[[VAL_5]] = stablehlo.dynamic_slice %iterArg_3, %iterArg, sizes = [1] : (tensor<10xi1>, tensor<i64>) -> tensor<1xi1>
// CHECK:           %[[VAL_6:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:           %3 = stablehlo.dynamic_slice %iterArg_2, %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_8:.*]] = arith.addf %[[VAL_7]], %[[VAL_7]] : tensor<f32>
// CHECK:           %6 = stablehlo.broadcast_in_dim %[[VAL_8]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.dynamic_update_slice %iterArg_2, %6, %iterArg : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %8 = stablehlo.dynamic_slice %iterArg_2, %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_11:.*]] = arith.mulf %[[VAL_10]], %[[VAL_10]] : tensor<f32>
// CHECK:           %11 = stablehlo.broadcast_in_dim %[[VAL_11]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.dynamic_update_slice %iterArg_2, %11, %iterArg : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %13 = stablehlo.broadcast_in_dim %[[VAL_6]], dims = [] : (tensor<i1>) -> tensor<10xi1>
// CHECK:           %[[VAL_13:.*]] = stablehlo.select %13, %[[VAL_9]], %[[VAL_12]] : tensor<10xi1>, tensor<10xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.add %iterArg, %[[VAL_4]] : tensor<i64>
// CHECK:           stablehlo.return %[[VAL_14]], %[[VAL_13]], %iterArg_3 : tensor<i64>, tensor<10xf32>, tensor<10xi1>
// CHECK:           }
// CHECK:           return %0#1, %0#2 : tensor<10xf32>, tensor<10xi1>
// CHECK:         }

// CHECK-LABEL:   func.func private @test_nested_if_masking_raised(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<10xf32>, %[[VAL_1:.*]]: tensor<10xi1>, %[[VAL_2:.*]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>, tensor<10xi1>) {
// CHECK:           %0 = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.add %0, %[[VAL_3]] : tensor<10xi64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.multiply %[[VAL_4]], %[[VAL_5]] : tensor<10xi64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %3 = stablehlo.dynamic_slice %[[VAL_1]], %[[VAL_7]], sizes = [10] : (tensor<10xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK:           %[[VAL_8:.*]] = stablehlo.reshape %3 : (tensor<10xi1>) -> tensor<10xi1>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %5 = stablehlo.dynamic_slice %[[VAL_2]], %[[VAL_9]], sizes = [10] : (tensor<10xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK:           %[[VAL_10:.*]] = stablehlo.reshape %5 : (tensor<10xi1>) -> tensor<10xi1>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %7 = stablehlo.dynamic_slice %[[VAL_0]], %[[VAL_11]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.reshape %7 : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_12]], %[[VAL_12]] : tensor<10xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %10 = stablehlo.broadcast_in_dim %[[VAL_13]], dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %10, %[[VAL_14]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.select %[[VAL_10]], %[[VAL_15]], %[[VAL_0]] : tensor<10xi1>, tensor<10xf32>
// CHECK:           %[[VAL_17:.*]] = stablehlo.select %[[VAL_8]], %[[VAL_16]], %[[VAL_0]] : tensor<10xi1>, tensor<10xf32>
// CHECK:           return %[[VAL_17]], %[[VAL_1]], %[[VAL_2]] : tensor<10xf32>, tensor<10xi1>, tensor<10xi1>
// CHECK:         }


// CHECK-LABEL:   func.func private @test_affine_if_masking_raised(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<10xf32>) -> tensor<10xf32> {
// CHECK:           %0 = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.add %0, %[[VAL_1]] : tensor<10xi64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.multiply %[[VAL_2]], %[[VAL_3]] : tensor<10xi64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<-5> : tensor<i64>
// CHECK:           %3 = stablehlo.broadcast_in_dim %[[VAL_5]], dims = [] : (tensor<i64>) -> tensor<10xi64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.add %[[VAL_4]], %3 : tensor<10xi64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[VAL_8:.*]] = stablehlo.compare  GE, %[[VAL_6]], %[[VAL_7]] : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %6 = stablehlo.dynamic_slice %[[VAL_0]], %[[VAL_9]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.reshape %6 : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_11:.*]] = arith.addf %[[VAL_10]], %[[VAL_10]] : tensor<10xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %9 = stablehlo.broadcast_in_dim %[[VAL_11]], dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %9, %[[VAL_12]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<10> : tensor<i64>
// CHECK:           %[[VAL_16:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           %11:2 = stablehlo.while(%iterArg = %[[VAL_14]], %iterArg_8 = %[[VAL_0]]) : tensor<i64>, tensor<10xf32>
// CHECK:           cond {
// CHECK:           %[[VAL_17:.*]] = stablehlo.compare  LT, %iterArg, %[[VAL_15]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK:           stablehlo.return %[[VAL_17]] : tensor<i1>
// CHECK:           } do {
// CHECK:           %[[VAL_18:.*]] = stablehlo.constant dense<-5> : tensor<i64>
// CHECK:           %[[VAL_17]] = stablehlo.add %iterArg, %[[VAL_18]] : tensor<i64>
// CHECK:           %[[VAL_19:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_20:.*]] = stablehlo.compare  GE, %[[VAL_17]], %[[VAL_19]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK:           %14 = stablehlo.dynamic_slice %iterArg_8, %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.reshape %14 : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_22:.*]] = arith.addf %[[VAL_21]], %[[VAL_21]] : tensor<f32>
// CHECK:           %17 = stablehlo.broadcast_in_dim %[[VAL_22]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_23:.*]] = stablehlo.dynamic_update_slice %iterArg_8, %17, %iterArg : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %19 = stablehlo.broadcast_in_dim %[[VAL_20]], dims = [] : (tensor<i1>) -> tensor<10xi1>
// CHECK:           %[[VAL_24:.*]] = stablehlo.select %19, %[[VAL_23]], %iterArg_8 : tensor<10xi1>, tensor<10xf32>
// CHECK:           %[[VAL_25:.*]] = stablehlo.add %iterArg, %[[VAL_16]] : tensor<i64>
// CHECK:           stablehlo.return %[[VAL_25]], %[[VAL_24]] : tensor<i64>, tensor<10xf32>
// CHECK:           }
// CHECK:           return %11#1 : tensor<10xf32>
// CHECK:         }

// CHECK-LABEL:   func.func private @test_if_else_masking_raised(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: tensor<10xf32>, %[[VAL_1:.*]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>) {
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<10> : tensor<i64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           %0:3 = stablehlo.while(%iterArg = %[[VAL_2]], %iterArg_2 = %[[VAL_0]], %iterArg_3 = %[[VAL_1]]) : tensor<i64>, tensor<10xf32>, tensor<10xi1>
// CHECK:           cond {
// CHECK:           %[[VAL_5:.*]] = stablehlo.compare  LT, %iterArg, %[[VAL_3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK:           stablehlo.return %[[VAL_5]] : tensor<i1>
// CHECK:           } do {
// CHECK:           %[[VAL_5]] = stablehlo.dynamic_slice %iterArg_3, %iterArg, sizes = [1] : (tensor<10xi1>, tensor<i64>) -> tensor<1xi1>
// CHECK:           %[[VAL_6:.*]] = stablehlo.reshape %[[VAL_5]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:           %3 = stablehlo.dynamic_slice %iterArg_2, %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.reshape %3 : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_8:.*]] = arith.addf %[[VAL_7]], %[[VAL_7]] : tensor<f32>
// CHECK:           %6 = stablehlo.broadcast_in_dim %[[VAL_8]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.dynamic_update_slice %iterArg_2, %6, %iterArg : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %8 = stablehlo.dynamic_slice %iterArg_2, %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[VAL_10:.*]] = stablehlo.reshape %8 : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_11:.*]] = arith.mulf %[[VAL_10]], %[[VAL_10]] : tensor<f32>
// CHECK:           %11 = stablehlo.broadcast_in_dim %[[VAL_11]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.dynamic_update_slice %iterArg_2, %11, %iterArg : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %13 = stablehlo.broadcast_in_dim %[[VAL_6]], dims = [] : (tensor<i1>) -> tensor<10xi1>
// CHECK:           %[[VAL_13:.*]] = stablehlo.select %13, %[[VAL_9]], %[[VAL_12]] : tensor<10xi1>, tensor<10xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.add %iterArg, %[[VAL_4]] : tensor<i64>
// CHECK:           stablehlo.return %[[VAL_14]], %[[VAL_13]], %iterArg_3 : tensor<i64>, tensor<10xf32>, tensor<10xi1>
// CHECK:           }
// CHECK:           return %0#1, %0#2 : tensor<10xf32>, tensor<10xi1>
// CHECK:         }

// CHECK-LABEL:   func.func private @test_nested_if_masking_raised(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<10xf32>, %[[VAL_1:.*]]: tensor<10xi1>, %[[VAL_2:.*]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>, tensor<10xi1>) {
// CHECK:           %0 = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.add %0, %[[VAL_3]] : tensor<10xi64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.multiply %[[VAL_4]], %[[VAL_5]] : tensor<10xi64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %3 = stablehlo.dynamic_slice %[[VAL_1]], %[[VAL_7]], sizes = [10] : (tensor<10xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK:           %[[VAL_8:.*]] = stablehlo.reshape %3 : (tensor<10xi1>) -> tensor<10xi1>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %5 = stablehlo.dynamic_slice %[[VAL_2]], %[[VAL_9]], sizes = [10] : (tensor<10xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK:           %[[VAL_10:.*]] = stablehlo.reshape %5 : (tensor<10xi1>) -> tensor<10xi1>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %7 = stablehlo.dynamic_slice %[[VAL_0]], %[[VAL_11]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.reshape %7 : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_12]], %[[VAL_12]] : tensor<10xf32>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %10 = stablehlo.broadcast_in_dim %[[VAL_13]], dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_15:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %10, %[[VAL_14]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_17:.*]] = stablehlo.constant dense<10> : tensor<i64>
// CHECK:           %[[VAL_18:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           %12:4 = stablehlo.while(%iterArg = %[[VAL_16]], %iterArg_8 = %[[VAL_0]], %iterArg_9 = %[[VAL_1]], %iterArg_10 = %[[VAL_2]]) : tensor<i64>, tensor<10xf32>, tensor<10xi1>, tensor<10xi1>
// CHECK:           cond {
// CHECK:           %[[VAL_19:.*]] = stablehlo.compare  LT, %iterArg, %[[VAL_17]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK:           stablehlo.return %[[VAL_19]] : tensor<i1>
// CHECK:           } do {
// CHECK:           %[[VAL_19]] = stablehlo.dynamic_slice %iterArg_9, %iterArg, sizes = [1] : (tensor<10xi1>, tensor<i64>) -> tensor<1xi1>
// CHECK:           %[[VAL_20:.*]] = stablehlo.reshape %[[VAL_19]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:           %15 = stablehlo.dynamic_slice %iterArg_10, %iterArg, sizes = [1] : (tensor<10xi1>, tensor<i64>) -> tensor<1xi1>
// CHECK:           %[[VAL_21:.*]] = stablehlo.reshape %15 : (tensor<1xi1>) -> tensor<i1>
// CHECK:           %17 = stablehlo.dynamic_slice %iterArg_8, %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.reshape %17 : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_23:.*]] = arith.addf %[[VAL_22]], %[[VAL_22]] : tensor<f32>
// CHECK:           %20 = stablehlo.broadcast_in_dim %[[VAL_23]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_24:.*]] = stablehlo.dynamic_update_slice %iterArg_8, %20, %iterArg : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %22 = stablehlo.broadcast_in_dim %[[VAL_21]], dims = [] : (tensor<i1>) -> tensor<10xi1>
// CHECK:           %[[VAL_25:.*]] = stablehlo.select %22, %[[VAL_24]], %iterArg_8 : tensor<10xi1>, tensor<10xf32>
// CHECK:           %24 = stablehlo.broadcast_in_dim %[[VAL_20]], dims = [] : (tensor<i1>) -> tensor<10xi1>
// CHECK:           %[[VAL_26:.*]] = stablehlo.select %24, %[[VAL_25]], %iterArg_8 : tensor<10xi1>, tensor<10xf32>
// CHECK:           %[[VAL_27:.*]] = stablehlo.add %iterArg, %[[VAL_18]] : tensor<i64>
// CHECK:           stablehlo.return %[[VAL_27]], %[[VAL_26]], %iterArg_9, %iterArg_10 : tensor<i64>, tensor<10xf32>, tensor<10xi1>, tensor<10xi1>
// CHECK:           }
// CHECK:           return %12#1, %12#2, %12#3 : tensor<10xf32>, tensor<10xi1>, tensor<10xi1>
// CHECK:         }

// CHECK-LABEL:   func.func private @test_if_masking_raised(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<10xf32>, %[[VAL_1:.*]]: tensor<10xi1>) -> (tensor<10xf32>, tensor<10xi1>) {
// CHECK:           %0 = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.add %0, %[[VAL_2]] : tensor<10xi64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.multiply %[[VAL_3]], %[[VAL_4]] : tensor<10xi64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %3 = stablehlo.dynamic_slice %[[VAL_1]], %[[VAL_6]], sizes = [10] : (tensor<10xi1>, tensor<i64>) -> tensor<10xi1>
// CHECK:           %[[VAL_7:.*]] = stablehlo.reshape %3 : (tensor<10xi1>) -> tensor<10xi1>
// CHECK:           %[[VAL_8:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %5 = stablehlo.dynamic_slice %[[VAL_0]], %[[VAL_8]], sizes = [10] : (tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.reshape %5 : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_10:.*]] = arith.addf %[[VAL_9]], %[[VAL_9]] : tensor<10xf32>
// CHECK:           %[[VAL_11:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %8 = stablehlo.broadcast_in_dim %[[VAL_10]], dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.dynamic_update_slice %[[VAL_0]], %8, %[[VAL_11]] : (tensor<10xf32>, tensor<10xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %[[VAL_13:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK:           %[[VAL_14:.*]] = stablehlo.constant dense<10> : tensor<i64>
// CHECK:           %[[VAL_15:.*]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK:           %10:3 = stablehlo.while(%iterArg = %[[VAL_13]], %iterArg_7 = %[[VAL_0]], %iterArg_8 = %[[VAL_1]]) : tensor<i64>, tensor<10xf32>, tensor<10xi1>
// CHECK:           cond {
// CHECK:           %[[VAL_16:.*]] = stablehlo.compare  LT, %iterArg, %[[VAL_14]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK:           stablehlo.return %[[VAL_16]] : tensor<i1>
// CHECK:           } do {
// CHECK:           %[[VAL_16]] = stablehlo.dynamic_slice %iterArg_8, %iterArg, sizes = [1] : (tensor<10xi1>, tensor<i64>) -> tensor<1xi1>
// CHECK:           %[[VAL_17:.*]] = stablehlo.reshape %[[VAL_16]] : (tensor<1xi1>) -> tensor<i1>
// CHECK:           %13 = stablehlo.dynamic_slice %iterArg_7, %iterArg, sizes = [1] : (tensor<10xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK:           %[[VAL_18:.*]] = stablehlo.reshape %13 : (tensor<1xf32>) -> tensor<f32>
// CHECK:           %[[VAL_19:.*]] = arith.addf %[[VAL_18]], %[[VAL_18]] : tensor<f32>
// CHECK:           %16 = stablehlo.broadcast_in_dim %[[VAL_19]], dims = [] : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_20:.*]] = stablehlo.dynamic_update_slice %iterArg_7, %16, %iterArg : (tensor<10xf32>, tensor<1xf32>, tensor<i64>) -> tensor<10xf32>
// CHECK:           %18 = stablehlo.broadcast_in_dim %[[VAL_17]], dims = [] : (tensor<i1>) -> tensor<10xi1>
// CHECK:           %[[VAL_21:.*]] = stablehlo.select %18, %[[VAL_20]], %iterArg_7 : tensor<10xi1>, tensor<10xf32>
// CHECK:           %[[VAL_22:.*]] = stablehlo.add %iterArg, %[[VAL_15]] : tensor<i64>
// CHECK:           stablehlo.return %[[VAL_22]], %[[VAL_21]], %iterArg_8 : tensor<i64>, tensor<10xf32>, tensor<10xi1>
// CHECK:           }
// CHECK:           return %10#1, %10#2 : tensor<10xf32>, tensor<10xi1>
// CHECK:         }
