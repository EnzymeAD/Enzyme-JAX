// RUN: enzymexlamlir-opt --raise-affine-to-stablehlo --split-input-file %s | FileCheck %s

// A rotated do-while raises by peeling one before-region execution and
// carrying (condition, args, buffers) through a stablehlo.while whose body
// runs the do region then the before region again.
func.func @dowhile_loop(%out: memref<100xf64, 1>, %nb: memref<i32, 1>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  affine.parallel (%t) = (0) to (100) {
    %n = affine.load %nb[] : memref<i32, 1>
    %r = scf.while (%i = %c0) : (i32) -> i32 {
      %ip = arith.addi %i, %c1 : i32
      %cond = arith.cmpi slt, %ip, %n : i32
      scf.condition(%cond) %ip : i32
    } do {
    ^bb0(%i2: i32):
      scf.yield %i2 : i32
    }
    %f = arith.sitofp %r : i32 to f64
    affine.store %f, %out[%t] : memref<100xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @dowhile_loop_raised(
// CHECK: %[[I0:.+]] = arith.addi %{{.+}}, %{{.+}} : tensor<i32>
// CHECK: %[[C0:.+]] = arith.cmpi slt, %[[I0]], %{{.+}} : tensor<i32>
// CHECK: %[[W:.+]]:4 = stablehlo.while(%[[C:[a-zA-Z0-9_]+]] = %[[C0]], %[[I:.+]] = %[[I0]], %{{.+}}) : tensor<i1>, tensor<i32>, tensor<100xf64>, tensor<i32>
// CHECK: cond {
// CHECK: stablehlo.return %[[C]] : tensor<i1>
// CHECK: } do {
// CHECK: %[[I1:.+]] = arith.addi %[[I]], %{{.+}} : tensor<i32>
// CHECK: %[[C1:.+]] = arith.cmpi slt, %[[I1]], %{{.+}} : tensor<i32>
// CHECK: stablehlo.return %[[C1]], %[[I1]], %{{.+}}, %{{.+}} : tensor<i1>, tensor<i32>, tensor<100xf64>, tensor<i32>
// CHECK: arith.sitofp %[[W]]#1

// -----

// The do region's stores land in carried buffers; the before region's
// re-evaluation reads the carried state.
func.func @dowhile_store(%out: memref<100xf64, 1>, %acc: memref<f64, 1>, %nb: memref<i32, 1>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 1.0 : f64
  affine.parallel (%t) = (0) to (100) {
    %n = affine.load %nb[] : memref<i32, 1>
    %r = scf.while (%i = %c0) : (i32) -> i32 {
      %ip = arith.addi %i, %c1 : i32
      %a = affine.load %acc[] : memref<f64, 1>
      %a1 = arith.addf %a, %cst : f64
      affine.store %a1, %acc[] : memref<f64, 1>
      %cond = arith.cmpi slt, %ip, %n : i32
      scf.condition(%cond) %ip : i32
    } do {
    ^bb0(%i2: i32):
      scf.yield %i2 : i32
    }
    %f = arith.sitofp %r : i32 to f64
    affine.store %f, %out[%t] : memref<100xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @dowhile_store_raised(
// CHECK: %[[A0:.+]] = stablehlo.dynamic_update_slice %arg1, %{{.+}} : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK: stablehlo.while(%[[C:[a-zA-Z0-9_]+]] = %{{[^,]+}}, %[[I:.+]] = %{{[^,]+}}, %{{.+}} = %arg0, %[[A:.+]] = %[[A0]], %{{.+}}) : tensor<i1>, tensor<i32>, tensor<100xf64>, tensor<f64>, tensor<i32>
// CHECK: cond {
// CHECK: stablehlo.return %[[C]] : tensor<i1>
// CHECK: } do {
// CHECK: %[[I1:.+]] = arith.addi %[[I]], %{{.+}} : tensor<i32>
// CHECK: %[[S:.+]] = arith.addf
// CHECK: %[[B:.+]] = stablehlo.broadcast_in_dim %[[S]]
// CHECK: %[[A1:.+]] = stablehlo.dynamic_update_slice %[[A]], %[[B]]
// CHECK: %[[C1:.+]] = arith.cmpi slt, %[[I1]], %{{.+}} : tensor<i32>
// CHECK: stablehlo.return %[[C1]], %[[I1]], %{{.+}}, %[[A1]], %{{.+}} : tensor<i1>, tensor<i32>, tensor<100xf64>, tensor<f64>, tensor<i32>
