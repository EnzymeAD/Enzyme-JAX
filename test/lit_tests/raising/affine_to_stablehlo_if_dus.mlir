// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file --canonicalize | FileCheck %s

func.func @band(%a: memref<128x8xf32, 1>, %b: memref<128x8xf32, 1>, %out: memref<128x8xf32, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (128, 8) {
    %x = affine.load %a[%i, %j] : memref<128x8xf32, 1>
    %y = affine.load %b[%i, %j] : memref<128x8xf32, 1>
    %r = affine.if affine_set<(d0) : (-d0 + 125 >= 0, d0 - 1 >= 0)>(%i) -> f32 {
      affine.yield %x : f32
    } else {
      affine.yield %y : f32
    }
    affine.store %r, %out[%i, %j] : memref<128x8xf32, 1>
  }
  return
}

// CHECK:  func.func private @band_raised(%arg0: tensor<128x8xf32>, %arg1: tensor<128x8xf32>, %arg2: tensor<128x8xf32>) -> (tensor<128x8xf32>, tensor<128x8xf32>, tensor<128x8xf32>) {
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[v1:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.reshape %arg0 : (tensor<128x8xf32>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v3:.+]] = stablehlo.reshape %arg1 : (tensor<128x8xf32>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v4:.+]] = stablehlo.slice %[[v2]] [1:126, 0:8] : (tensor<128x8xf32>) -> tensor<125x8xf32>
// CHECK-NEXT:    %[[v5:.+]] = stablehlo.dynamic_update_slice %[[v3]], %[[v4]], %[[v1]], %[[v0]] : (tensor<128x8xf32>, tensor<125x8xf32>, tensor<i64>, tensor<i64>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v6:.+]] = stablehlo.broadcast_in_dim %[[v5]], dims = [0, 1] : (tensor<128x8xf32>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v7:.+]] = stablehlo.dynamic_update_slice %arg2, %[[v6]], %[[v0]], %[[v0]] : (tensor<128x8xf32>, tensor<128x8xf32>, tensor<i64>, tensor<i64>) -> tensor<128x8xf32>
// CHECK-NEXT:    return %arg0, %arg1, %[[v7]] : tensor<128x8xf32>, tensor<128x8xf32>, tensor<128x8xf32>
// CHECK-NEXT:  }

// -----

func.func @eqset(%a: memref<128x8xf32, 1>, %b: memref<128x8xf32, 1>, %out: memref<128x8xf32, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (128, 8) {
    %x = affine.load %a[%i, %j] : memref<128x8xf32, 1>
    %y = affine.load %b[%i, %j] : memref<128x8xf32, 1>
    %r = affine.if affine_set<(d0) : (d0 - 5 == 0)>(%i) -> f32 {
      affine.yield %x : f32
    } else {
      affine.yield %y : f32
    }
    affine.store %r, %out[%i, %j] : memref<128x8xf32, 1>
  }
  return
}

// CHECK:  func.func private @eqset_raised(%arg0: tensor<128x8xf32>, %arg1: tensor<128x8xf32>, %arg2: tensor<128x8xf32>) -> (tensor<128x8xf32>, tensor<128x8xf32>, tensor<128x8xf32>) {
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[v1:.+]] = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.reshape %arg0 : (tensor<128x8xf32>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v3:.+]] = stablehlo.reshape %arg1 : (tensor<128x8xf32>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v4:.+]] = stablehlo.slice %[[v2]] [5:6, 0:8] : (tensor<128x8xf32>) -> tensor<1x8xf32>
// CHECK-NEXT:    %[[v5:.+]] = stablehlo.dynamic_update_slice %[[v3]], %[[v4]], %[[v1]], %[[v0]] : (tensor<128x8xf32>, tensor<1x8xf32>, tensor<i64>, tensor<i64>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v6:.+]] = stablehlo.broadcast_in_dim %[[v5]], dims = [0, 1] : (tensor<128x8xf32>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v7:.+]] = stablehlo.dynamic_update_slice %arg2, %[[v6]], %[[v0]], %[[v0]] : (tensor<128x8xf32>, tensor<128x8xf32>, tensor<i64>, tensor<i64>) -> tensor<128x8xf32>
// CHECK-NEXT:    return %arg0, %arg1, %[[v7]] : tensor<128x8xf32>, tensor<128x8xf32>, tensor<128x8xf32>
// CHECK-NEXT:  }

// -----

// The branch values do not carry the constrained axis, so the box would not
// be representable as a slice: fall back to a select.
func.func @noaxis(%a: memref<128x8xf32, 1>, %b: memref<128x8xf32, 1>, %out: memref<128x8xf32, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (128, 8) {
    %x = affine.load %a[0, %j] : memref<128x8xf32, 1>
    %y = affine.load %b[0, %j] : memref<128x8xf32, 1>
    %r = affine.if affine_set<(d0) : (-d0 + 125 >= 0, d0 - 1 >= 0)>(%i) -> f32 {
      affine.yield %x : f32
    } else {
      affine.yield %y : f32
    }
    affine.store %r, %out[%i, %j] : memref<128x8xf32, 1>
  }
  return
}

// CHECK:  func.func private @noaxis_raised(%arg0: tensor<128x8xf32>, %arg1: tensor<128x8xf32>, %arg2: tensor<128x8xf32>) -> (tensor<128x8xf32>, tensor<128x8xf32>, tensor<128x8xf32>) {
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[v1:.+]] = stablehlo.constant dense<125> : tensor<i64>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %[[v3:.+]] = stablehlo.constant dense<1> : tensor<128xi64>
// CHECK-NEXT:    %[[v4:.+]] = stablehlo.constant dense<0> : tensor<128xi64>
// CHECK-NEXT:    %[[v5:.+]] = stablehlo.iota dim = 0 : tensor<128xi64>
// CHECK-NEXT:    %[[v6:.+]] = stablehlo.add %[[v5]], %[[v4]] : tensor<128xi64>
// CHECK-NEXT:    %[[v7:.+]] = stablehlo.multiply %[[v6]], %[[v3]] : tensor<128xi64>
// CHECK-NEXT:    %[[v8:.+]] = stablehlo.slice %arg0 [0:1, 0:8] : (tensor<128x8xf32>) -> tensor<1x8xf32>
// CHECK-NEXT:    %[[v9:.+]] = stablehlo.reshape %[[v8]] : (tensor<1x8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %[[v10:.+]] = stablehlo.slice %arg1 [0:1, 0:8] : (tensor<128x8xf32>) -> tensor<1x8xf32>
// CHECK-NEXT:    %[[v11:.+]] = stablehlo.reshape %[[v10]] : (tensor<1x8xf32>) -> tensor<8xf32>
// CHECK-NEXT:    %[[v12:.+]] = stablehlo.broadcast_in_dim %[[v2]], dims = [] : (tensor<i64>) -> tensor<128xi64>
// CHECK-NEXT:    %[[v13:.+]] = stablehlo.multiply %[[v7]], %[[v12]] : tensor<128xi64>
// CHECK-NEXT:    %[[v14:.+]] = stablehlo.broadcast_in_dim %[[v1]], dims = [] : (tensor<i64>) -> tensor<128xi64>
// CHECK-NEXT:    %[[v15:.+]] = stablehlo.add %[[v13]], %[[v14]] : tensor<128xi64>
// CHECK-NEXT:    %[[v16:.+]] = stablehlo.compare GE, %[[v15]], %[[v4]] : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
// CHECK-NEXT:    %[[v17:.+]] = stablehlo.broadcast_in_dim %[[v2]], dims = [] : (tensor<i64>) -> tensor<128xi64>
// CHECK-NEXT:    %[[v18:.+]] = stablehlo.add %[[v7]], %[[v17]] : tensor<128xi64>
// CHECK-NEXT:    %[[v19:.+]] = stablehlo.compare GE, %[[v18]], %[[v4]] : (tensor<128xi64>, tensor<128xi64>) -> tensor<128xi1>
// CHECK-NEXT:    %[[v20:.+]] = stablehlo.and %[[v16]], %[[v19]] : tensor<128xi1>
// CHECK-NEXT:    %[[v21:.+]] = stablehlo.broadcast_in_dim %[[v20]], dims = [0] : (tensor<128xi1>) -> tensor<128x8xi1>
// CHECK-NEXT:    %[[v22:.+]] = stablehlo.broadcast_in_dim %[[v9]], dims = [1] : (tensor<8xf32>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v23:.+]] = stablehlo.broadcast_in_dim %[[v11]], dims = [1] : (tensor<8xf32>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v24:.+]] = stablehlo.select %[[v21]], %[[v22]], %[[v23]] : tensor<128x8xi1>, tensor<128x8xf32>
// CHECK-NEXT:    %[[v25:.+]] = stablehlo.broadcast_in_dim %[[v24]], dims = [0, 1] : (tensor<128x8xf32>) -> tensor<128x8xf32>
// CHECK-NEXT:    %[[v26:.+]] = stablehlo.dynamic_update_slice %arg2, %[[v25]], %[[v0]], %[[v0]] : (tensor<128x8xf32>, tensor<128x8xf32>, tensor<i64>, tensor<i64>) -> tensor<128x8xf32>
// CHECK-NEXT:    return %arg0, %arg1, %[[v26]] : tensor<128x8xf32>, tensor<128x8xf32>, tensor<128x8xf32>
// CHECK-NEXT:  }

// -----

func.func @lb10(%a: memref<100xf32, 1>, %b: memref<100xf32, 1>, %o: memref<100xf32, 1>) {
  affine.parallel (%i) = (10) to (100) {
    %x = affine.load %a[%i] : memref<100xf32, 1>
    %y = affine.load %b[%i] : memref<100xf32, 1>
    %r = affine.if affine_set<(d0) : (d0 - 20 >= 0, -d0 + 89 >= 0)>(%i) -> f32 {
      affine.yield %x : f32
    } else {
      affine.yield %y : f32
    }
    affine.store %r, %o[%i] : memref<100xf32, 1>
  }
  return
}

// CHECK:  func.func private @lb10_raised(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>, %arg2: tensor<100xf32>) -> (tensor<100xf32>, tensor<100xf32>, tensor<100xf32>) {
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.constant dense<10> : tensor<i64>
// CHECK-NEXT:    %[[v1:.+]] = stablehlo.slice %arg0 [10:100] : (tensor<100xf32>) -> tensor<90xf32>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.reshape %[[v1]] : (tensor<90xf32>) -> tensor<90xf32>
// CHECK-NEXT:    %[[v3:.+]] = stablehlo.slice %arg1 [10:100] : (tensor<100xf32>) -> tensor<90xf32>
// CHECK-NEXT:    %[[v4:.+]] = stablehlo.reshape %[[v3]] : (tensor<90xf32>) -> tensor<90xf32>
// CHECK-NEXT:    %[[v5:.+]] = stablehlo.slice %[[v2]] [10:80] : (tensor<90xf32>) -> tensor<70xf32>
// CHECK-NEXT:    %[[v6:.+]] = stablehlo.dynamic_update_slice %[[v4]], %[[v5]], %[[v0]] : (tensor<90xf32>, tensor<70xf32>, tensor<i64>) -> tensor<90xf32>
// CHECK-NEXT:    %[[v7:.+]] = stablehlo.broadcast_in_dim %[[v6]], dims = [0] : (tensor<90xf32>) -> tensor<90xf32>
// CHECK-NEXT:    %[[v8:.+]] = stablehlo.dynamic_update_slice %arg2, %[[v7]], %[[v0]] : (tensor<100xf32>, tensor<90xf32>, tensor<i64>) -> tensor<100xf32>
// CHECK-NEXT:    return %arg0, %arg1, %[[v8]] : tensor<100xf32>, tensor<100xf32>, tensor<100xf32>
// CHECK-NEXT:  }

// -----

func.func @step2(%a: memref<100xf32, 1>, %b: memref<100xf32, 1>, %o: memref<100xf32, 1>) {
  affine.parallel (%i) = (0) to (100) step (2) {
    %x = affine.load %a[%i] : memref<100xf32, 1>
    %y = affine.load %b[%i] : memref<100xf32, 1>
    %r = affine.if affine_set<(d0) : (d0 - 21 >= 0, -d0 + 60 >= 0)>(%i) -> f32 {
      affine.yield %x : f32
    } else {
      affine.yield %y : f32
    }
    affine.store %r, %o[%i] : memref<100xf32, 1>
  }
  return
}

// CHECK:  func.func private @step2_raised(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>, %arg2: tensor<100xf32>) -> (tensor<100xf32>, tensor<100xf32>, tensor<100xf32>) {
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.constant dense<11> : tensor<i64>
// CHECK-NEXT:    %[[v1:.+]] = stablehlo.constant dense<2> : tensor<50xi64>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.constant dense<0> : tensor<50xi64>
// CHECK-NEXT:    %[[v3:.+]] = stablehlo.iota dim = 0 : tensor<50xi64>
// CHECK-NEXT:    %[[v4:.+]] = stablehlo.add %[[v3]], %[[v2]] : tensor<50xi64>
// CHECK-NEXT:    %[[v5:.+]] = stablehlo.multiply %[[v4]], %[[v1]] : tensor<50xi64>
// CHECK-NEXT:    %[[v6:.+]] = stablehlo.slice %arg0 [0:100:2] : (tensor<100xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v7:.+]] = stablehlo.reshape %[[v6]] : (tensor<50xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v8:.+]] = stablehlo.slice %arg1 [0:100:2] : (tensor<100xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v9:.+]] = stablehlo.reshape %[[v8]] : (tensor<50xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v10:.+]] = stablehlo.slice %[[v7]] [11:31] : (tensor<50xf32>) -> tensor<20xf32>
// CHECK-NEXT:    %[[v11:.+]] = stablehlo.dynamic_update_slice %[[v9]], %[[v10]], %[[v0]] : (tensor<50xf32>, tensor<20xf32>, tensor<i64>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v12:.+]] = stablehlo.reshape %[[v5]] : (tensor<50xi64>) -> tensor<50x1xi64>
// CHECK-NEXT:    %[[v13:.+]] = stablehlo.broadcast_in_dim %[[v11]], dims = [0] : (tensor<50xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v14:.+]] = "stablehlo.scatter"(%arg2, %[[v12]], %[[v13]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:      stablehlo.return %arg4 : tensor<f32>
// CHECK-NEXT:    }) : (tensor<100xf32>, tensor<50x1xi64>, tensor<50xf32>) -> tensor<100xf32>
// CHECK-NEXT:    return %arg0, %arg1, %[[v14]] : tensor<100xf32>, tensor<100xf32>, tensor<100xf32>
// CHECK-NEXT:  }

// -----

// 21 is not reachable with a step of 2, so the equality has no solution in
// the iteration space and the branch stays a select.
func.func @eqodd(%a: memref<100xf32, 1>, %b: memref<100xf32, 1>, %o: memref<100xf32, 1>) {
  affine.parallel (%i) = (0) to (100) step (2) {
    %x = affine.load %a[%i] : memref<100xf32, 1>
    %y = affine.load %b[%i] : memref<100xf32, 1>
    %r = affine.if affine_set<(d0) : (d0 - 21 == 0)>(%i) -> f32 {
      affine.yield %x : f32
    } else {
      affine.yield %y : f32
    }
    affine.store %r, %o[%i] : memref<100xf32, 1>
  }
  return
}

// CHECK:  func.func private @eqodd_raised(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>, %arg2: tensor<100xf32>) -> (tensor<100xf32>, tensor<100xf32>, tensor<100xf32>) {
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.constant dense<-21> : tensor<i64>
// CHECK-NEXT:    %[[v1:.+]] = stablehlo.constant dense<2> : tensor<50xi64>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.constant dense<0> : tensor<50xi64>
// CHECK-NEXT:    %[[v3:.+]] = stablehlo.iota dim = 0 : tensor<50xi64>
// CHECK-NEXT:    %[[v4:.+]] = stablehlo.add %[[v3]], %[[v2]] : tensor<50xi64>
// CHECK-NEXT:    %[[v5:.+]] = stablehlo.multiply %[[v4]], %[[v1]] : tensor<50xi64>
// CHECK-NEXT:    %[[v6:.+]] = stablehlo.slice %arg0 [0:100:2] : (tensor<100xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v7:.+]] = stablehlo.reshape %[[v6]] : (tensor<50xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v8:.+]] = stablehlo.slice %arg1 [0:100:2] : (tensor<100xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v9:.+]] = stablehlo.reshape %[[v8]] : (tensor<50xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v10:.+]] = stablehlo.broadcast_in_dim %[[v0]], dims = [] : (tensor<i64>) -> tensor<50xi64>
// CHECK-NEXT:    %[[v11:.+]] = stablehlo.add %[[v5]], %[[v10]] : tensor<50xi64>
// CHECK-NEXT:    %[[v12:.+]] = stablehlo.compare EQ, %[[v11]], %[[v2]] : (tensor<50xi64>, tensor<50xi64>) -> tensor<50xi1>
// CHECK-NEXT:    %[[v13:.+]] = stablehlo.select %[[v12]], %[[v7]], %[[v9]] : tensor<50xi1>, tensor<50xf32>
// CHECK-NEXT:    %[[v14:.+]] = stablehlo.reshape %[[v5]] : (tensor<50xi64>) -> tensor<50x1xi64>
// CHECK-NEXT:    %[[v15:.+]] = stablehlo.broadcast_in_dim %[[v13]], dims = [0] : (tensor<50xf32>) -> tensor<50xf32>
// CHECK-NEXT:    %[[v16:.+]] = "stablehlo.scatter"(%arg2, %[[v14]], %[[v15]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:      stablehlo.return %arg4 : tensor<f32>
// CHECK-NEXT:    }) : (tensor<100xf32>, tensor<50x1xi64>, tensor<50xf32>) -> tensor<100xf32>
// CHECK-NEXT:    return %arg0, %arg1, %[[v16]] : tensor<100xf32>, tensor<100xf32>, tensor<100xf32>
// CHECK-NEXT:  }
