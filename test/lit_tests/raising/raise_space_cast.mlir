// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A memory_space_cast view of scratch would split one buffer into two SSA
// roots; accesses do not care about the address space, so they retarget to
// the source and the cast drops, letting the scratch raise as usual.
func.func @spacecast(%out: memref<10xf64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  %view = memref.memory_space_cast %tmp : memref<10xf64> to memref<10xf64, 3>
  affine.parallel (%i) = (0) to (10) {
    %c = arith.constant 2.0 : f64
    affine.store %c, %view[%i] : memref<10xf64, 3>
    %v = affine.load %view[9 - %i] : memref<10xf64, 3>
    affine.store %v, %out[%i] : memref<10xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @spacecast_raised(
// CHECK-NOT: memory_space_cast
// CHECK-DAG: stablehlo.constant dense<2.000000e+00> : tensor<f64>
// CHECK-DAG: stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK: stablehlo.reverse

// -----

// An atomic accumulation into the scratch is an access too: it retargets
// to the source, and raises as a scatter-add.
func.func @spacecast_atomic(%in: memref<4xf64, 1>, %out: memref<2xf64, 1>) {
  %tmp = memref.alloca() : memref<2xf64>
  %view = memref.memory_space_cast %tmp : memref<2xf64> to memref<2xf64, 3>
  %z = arith.constant 0.0 : f64
  affine.store %z, %view[0] : memref<2xf64, 3>
  affine.store %z, %view[1] : memref<2xf64, 3>
  affine.parallel (%i) = (0) to (4) {
    %v = affine.load %in[%i] : memref<4xf64, 1>
    %old = enzyme.affine_atomic_rmw addf %v, %view, (affine_map<(d0) -> (d0 floordiv 2)>)[%i] monotonic : (f64, memref<2xf64, 3>) -> f64
  }
  %a = affine.load %view[0] : memref<2xf64, 3>
  %b = affine.load %view[1] : memref<2xf64, 3>
  affine.store %a, %out[0] : memref<2xf64, 1>
  affine.store %b, %out[1] : memref<2xf64, 1>
  return
}

// CHECK-LABEL: func.func private @spacecast_atomic_raised(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf64>, %[[V1:[^ ,:)]+]]: tensor<2xf64>) -> (tensor<4xf64>, tensor<2xf64>) {
// CHECK-NEXT: %[[V2:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V2]], dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V3]], %[[V10]], %[[V9]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V2]], dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V11]], %[[V18]], %[[V17]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT: %[[V22:[^ ,:)]+]] = stablehlo.add %[[V20]], %[[V21]] : tensor<4xi64>
// CHECK-NEXT: %[[V23:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT: %[[V24:[^ ,:)]+]] = stablehlo.multiply %[[V22]], %[[V23]] : tensor<4xi64>
// CHECK-NEXT: %[[V25:[^ ,:)]+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT: %[[V26:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V25]], dims = [] : (tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT: %[[V27:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT: %[[V28:[^ ,:)]+]] = stablehlo.compare LT, %[[V24]], %[[V27]] : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
// CHECK-NEXT: %[[V29:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT: %[[V30:[^ ,:)]+]] = stablehlo.negate %[[V24]] : tensor<4xi64>
// CHECK-NEXT: %[[V31:[^ ,:)]+]] = stablehlo.subtract %[[V30]], %[[V29]] : tensor<4xi64>
// CHECK-NEXT: %[[V32:[^ ,:)]+]] = stablehlo.select %[[V28]], %[[V31]], %[[V24]] : tensor<4xi1>, tensor<4xi64>
// CHECK-NEXT: %[[V33:[^ ,:)]+]] = stablehlo.divide %[[V32]], %[[V26]] : tensor<4xi64>
// CHECK-NEXT: %[[V34:[^ ,:)]+]] = stablehlo.negate %[[V33]] : tensor<4xi64>
// CHECK-NEXT: %[[V35:[^ ,:)]+]] = stablehlo.subtract %[[V34]], %[[V29]] : tensor<4xi64>
// CHECK-NEXT: %[[V36:[^ ,:)]+]] = stablehlo.select %[[V28]], %[[V35]], %[[V33]] : tensor<4xi1>, tensor<4xi64>
// CHECK-NEXT: %[[V37:[^ ,:)]+]] = stablehlo.reshape %[[V36]] : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT: %[[V38:[^ ,:)]+]] = "stablehlo.scatter"(%[[V19]], %[[V37]], %[[V0]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V39:[^ ,:)]+]]: tensor<f64>, %[[V40:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: %[[V41:[^ ,:)]+]] = stablehlo.add %[[V39]], %[[V40]] : tensor<f64>
// CHECK-NEXT: stablehlo.return %[[V41]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<2xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V42:[^ ,:)]+]] = stablehlo.slice %[[V38]] [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V43:[^ ,:)]+]] = stablehlo.reshape %[[V42]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[V44:[^ ,:)]+]] = stablehlo.slice %[[V38]] [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V45:[^ ,:)]+]] = stablehlo.reshape %[[V44]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[V46:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V47:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V48:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V49:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V50:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V51:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V52:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V43]], dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V53:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V1]], %[[V52]], %[[V51]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V54:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V55:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V56:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V57:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V58:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V59:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V60:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V45]], dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V61:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V53]], %[[V60]], %[[V59]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: return %[[V0]], %[[V61]] : tensor<4xf64>, tensor<2xf64>
// CHECK-NEXT: }

// -----

// So is one at plain indices.
func.func @spacecast_atomic_plain(%in: memref<4xf64, 1>, %idx: memref<4xi32, 1>, %out: memref<2xf64, 1>) {
  %tmp = memref.alloca() : memref<2xf64>
  %view = memref.memory_space_cast %tmp : memref<2xf64> to memref<2xf64, 3>
  %z = arith.constant 0.0 : f64
  affine.store %z, %view[0] : memref<2xf64, 3>
  affine.store %z, %view[1] : memref<2xf64, 3>
  affine.parallel (%i) = (0) to (4) {
    %v = affine.load %in[%i] : memref<4xf64, 1>
    %j = affine.load %idx[%i] : memref<4xi32, 1>
    %jj = arith.index_cast %j : i32 to index
    %old = enzyme.atomic_rmw addf %v, %view[%jj] monotonic : (f64, memref<2xf64, 3>) -> f64
  }
  %a = affine.load %view[0] : memref<2xf64, 3>
  %b = affine.load %view[1] : memref<2xf64, 3>
  affine.store %a, %out[0] : memref<2xf64, 1>
  affine.store %b, %out[1] : memref<2xf64, 1>
  return
}

// CHECK-LABEL: func.func private @spacecast_atomic_plain_raised(
// CHECK-SAME: %[[V0:[^ ,:)]+]]: tensor<4xf64>, %[[V1:[^ ,:)]+]]: tensor<4xi32>, %[[V2:[^ ,:)]+]]: tensor<2xf64>) -> (tensor<4xf64>, tensor<4xi32>, tensor<2xf64>) {
// CHECK-NEXT: %[[V3:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[V4:[^ ,:)]+]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf64>
// CHECK-NEXT: %[[V5:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V6:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V7:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V8:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V9:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V10:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V11:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V3]], dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V12:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V4]], %[[V11]], %[[V10]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V13:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V14:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V15:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V16:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V17:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V18:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V19:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V3]], dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V20:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V12]], %[[V19]], %[[V18]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V21:[^ ,:)]+]] = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT: %[[V22:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT: %[[V23:[^ ,:)]+]] = stablehlo.add %[[V21]], %[[V22]] : tensor<4xi64>
// CHECK-NEXT: %[[V24:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT: %[[V25:[^ ,:)]+]] = stablehlo.multiply %[[V23]], %[[V24]] : tensor<4xi64>
// CHECK-NEXT: %[[V26:[^ ,:)]+]] = stablehlo.convert %[[V1]] : (tensor<4xi32>) -> tensor<4xi64>
// CHECK-NEXT: %[[V27:[^ ,:)]+]] = stablehlo.reshape %[[V26]] : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT: %[[V28:[^ ,:)]+]] = "stablehlo.scatter"(%[[V20]], %[[V27]], %[[V0]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT: ^bb0(%[[V29:[^ ,:)]+]]: tensor<f64>, %[[V30:[^ ,:)]+]]: tensor<f64>):
// CHECK-NEXT: %[[V31:[^ ,:)]+]] = stablehlo.add %[[V29]], %[[V30]] : tensor<f64>
// CHECK-NEXT: stablehlo.return %[[V31]] : tensor<f64>
// CHECK-NEXT: }) : (tensor<2xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V32:[^ ,:)]+]] = stablehlo.slice %[[V28]] [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V33:[^ ,:)]+]] = stablehlo.reshape %[[V32]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[V34:[^ ,:)]+]] = stablehlo.slice %[[V28]] [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V35:[^ ,:)]+]] = stablehlo.reshape %[[V34]] : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT: %[[V36:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V37:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V38:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V39:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V40:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V41:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[V42:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V33]], dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V43:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V2]], %[[V42]], %[[V41]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: %[[V44:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V45:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V46:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V47:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V48:[^ ,:)]+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT: %[[V49:[^ ,:)]+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT: %[[V50:[^ ,:)]+]] = stablehlo.broadcast_in_dim %[[V35]], dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT: %[[V51:[^ ,:)]+]] = stablehlo.dynamic_update_slice %[[V43]], %[[V50]], %[[V49]] : (tensor<2xf64>, tensor<1xf64>, tensor<i64>) -> tensor<2xf64>
// CHECK-NEXT: return %[[V0]], %[[V1]], %[[V51]] : tensor<4xf64>, tensor<4xi32>, tensor<2xf64>
// CHECK-NEXT: }
