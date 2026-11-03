// RUN: enzymexlamlir-opt %s --split-input-file --raise-affine-to-stablehlo --canonicalize --cse | FileCheck %s

// trmm: B[i][j] = alpha * (B[i][j] + sum_{k > i} A[k][i] * B[k][j]). The i
// loop is not parallel (iteration i reads B[k][j], which iteration k writes),
// but every iteration reads before any writes, so it runs in lock step with
// the j axis and the k loop nested in it: one while over k, the body over
// (i, j).
#set = affine_set<(d0) : (-d0 + 2 >= 0)>
func.func private @trmm(%alpha: memref<f64>, %A: memref<4x4xf64>, %B: memref<4x5xf64>) {
  %cst = arith.constant 0.000000e+00 : f64
  %0 = affine.load %alpha[] : memref<f64>
  affine.for %i = 0 to 4 {
    affine.parallel (%j) = (0) to (5) {
      %1 = affine.if #set(%i) -> f64 {
        %6 = affine.load %B[%i, %j] : memref<4x5xf64>
        affine.yield %6 : f64
      } else {
        affine.yield %cst : f64
      }
      %2 = affine.parallel (%k) = (%i + 1) to (4) reduce ("addf") -> (f64) {
        %6 = affine.load %A[%k, %i] : memref<4x4xf64>
        %7 = affine.load %B[%k, %j] : memref<4x5xf64>
        %8 = arith.mulf %6, %7 : f64
        affine.yield %8 : f64
      }
      %3 = arith.addf %1, %2 : f64
      affine.if #set(%i) {
        affine.store %3, %B[%i, %j] : memref<4x5xf64>
      }
      %4 = affine.load %B[%i, %j] : memref<4x5xf64>
      %5 = arith.mulf %0, %4 : f64
      affine.store %5, %B[%i, %j] : memref<4x5xf64>
    }
  }
  return
}

// CHECK-LABEL:  func.func private @trmm_raised(%arg0: tensor<f64>, %arg1: tensor<4x4xf64>, %arg2: tensor<4x5xf64>) -> (tensor<f64>, tensor<4x4xf64>, tensor<4x5xf64>) {
// CHECK:          %[[W:.+]]:5 = stablehlo.while(%iterArg = %c{{.*}}, %iterArg_{{[0-9]+}} = %{{.*}}, %iterArg_{{[0-9]+}} = %arg0, %iterArg_{{[0-9]+}} = %arg1, %iterArg_{{[0-9]+}} = %arg2) : tensor<i64>, tensor<4x5xf64>, tensor<f64>, tensor<4x4xf64>, tensor<4x5xf64>
// CHECK:          "stablehlo.gather"(%iterArg_{{[0-9]+}}, %{{.*}}) {{.*}} : (tensor<4x4xf64>, tensor<4x2xi64>) -> tensor<4xf64>
// CHECK:          "stablehlo.gather"(%iterArg_{{[0-9]+}}, %{{.*}}) {{.*}} : (tensor<4x5xf64>, tensor<4x5x2xi64>) -> tensor<4x5xf64>
// CHECK:          arith.mulf %{{.*}}, %{{.*}} : tensor<4x5xf64>
// CHECK:          arith.addf %iterArg_{{[0-9]+}}, %{{.*}} : tensor<4x5xf64>
// CHECK-NOT:      stablehlo.while
// CHECK:          %[[S:.+]] = arith.addf %{{.*}}, %[[W]]#1 : tensor<4x5xf64>
// CHECK-NEXT:     %[[G:.+]] = stablehlo.select %{{.*}}, %[[S]], %[[W]]#4 : tensor<4x5xi1>, tensor<4x5xf64>
// CHECK-NEXT:     %[[B:.+]] = stablehlo.dynamic_update_slice %[[W]]#4, %[[G]], %c{{.*}}, %c{{.*}} : (tensor<4x5xf64>, tensor<4x5xf64>, tensor<i64>, tensor<i64>) -> tensor<4x5xf64>
// CHECK-NEXT:     %[[A:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f64>) -> tensor<4x5xf64>
// CHECK-NEXT:     %[[M:.+]] = arith.mulf %[[A]], %[[B]] : tensor<4x5xf64>
// CHECK-NEXT:     %[[R:.+]] = stablehlo.dynamic_update_slice %[[B]], %[[M]], %c{{.*}}, %c{{.*}} : (tensor<4x5xf64>, tensor<4x5xf64>, tensor<i64>, tensor<i64>) -> tensor<4x5xf64>
// CHECK-NEXT:     return %[[W]]#2, %[[W]]#3, %[[R]] : tensor<f64>, tensor<4x4xf64>, tensor<4x5xf64>
// -----

// X[i] = 1; Y[i] = X[i + 1]: iteration i reads X[i + 1] before iteration
// i + 1 writes it. In lock step the writes would come first (a WAR the
// source of which is after its destination in program order), so the loop
// does not run in lock step.
func.func private @war(%X: memref<4xf64>, %Y: memref<3xf64>) {
  %one = arith.constant 1.0 : f64
  affine.for %i = 0 to 3 {
    affine.store %one, %X[%i] : memref<4xf64>
    %v = affine.load %X[%i + 1] : memref<4xf64>
    affine.store %v, %Y[%i] : memref<3xf64>
  }
  return
}

// CHECK-LABEL:  func.func private @war_raised(%arg0: tensor<4xf64>, %arg1: tensor<3xf64>) -> (tensor<4xf64>, tensor<3xf64>) {
// CHECK:          stablehlo.while
// CHECK:          %[[ONE:.+]] = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:     %[[X:.+]] = stablehlo.dynamic_update_slice %iterArg_{{[0-9]+}}, %[[ONE]], %iterArg : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
// CHECK-NEXT:     %[[N:.+]] = stablehlo.add %iterArg, %c{{.*}} : tensor<i64>
// CHECK-NEXT:     stablehlo.dynamic_slice %[[X]], %[[N]], sizes = [1] : (tensor<4xf64>, tensor<i64>) -> tensor<1xf64>
// -----

// A copy nested two loops deep over distinct buffers runs in lock step over
// all three.
func.func private @copy(%in: memref<2x3x4xf64>, %out: memref<2x3x4xf64>) {
  affine.for %i = 0 to 2 {
    affine.for %j = 0 to 3 {
      affine.for %k = 0 to 4 {
        %v = affine.load %in[%i, %j, %k] : memref<2x3x4xf64>
        affine.store %v, %out[%i, %j, %k] : memref<2x3x4xf64>
      }
    }
  }
  return
}

// CHECK-LABEL:  func.func private @copy_raised(%arg0: tensor<2x3x4xf64>, %arg1: tensor<2x3x4xf64>) -> (tensor<2x3x4xf64>, tensor<2x3x4xf64>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %[[O:.+]] = stablehlo.dynamic_update_slice %arg1, %arg0, %c, %c, %c : (tensor<2x3x4xf64>, tensor<2x3x4xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x3x4xf64>
// CHECK-NEXT:     return %arg0, %[[O]] : tensor<2x3x4xf64>, tensor<2x3x4xf64>
// -----

// X[i][k] written, then X[i - 1][k + 1] read: in program order the write
// comes first, but should the k loop iterate with i as an axis, iteration k
// reads what iteration k + 1 writes. The i loop does not run in lock step.
func.func private @skew(%X: memref<4x5xf64>, %Y: memref<4x5xf64>) {
  %one = arith.constant 1.0 : f64
  affine.for %i = 1 to 4 {
    affine.for %k = 0 to 4 {
      affine.store %one, %X[%i, %k] : memref<4x5xf64>
      %v = affine.load %X[%i - 1, %k + 1] : memref<4x5xf64>
      affine.store %v, %Y[%i, %k] : memref<4x5xf64>
    }
  }
  return
}

// CHECK-LABEL:  func.func private @skew_raised(%arg0: tensor<4x5xf64>, %arg1: tensor<4x5xf64>) -> (tensor<4x5xf64>, tensor<4x5xf64>) {
// CHECK:          stablehlo.while
// CHECK:          %[[ONE:.+]] = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<1x4xf64>
// CHECK-NEXT:     %[[X:.+]] = stablehlo.dynamic_update_slice %iterArg_{{[0-9]+}}, %[[ONE]], %iterArg, %c{{.*}} : (tensor<4x5xf64>, tensor<1x4xf64>, tensor<i64>, tensor<i64>) -> tensor<4x5xf64>
// CHECK-NEXT:     %[[N:.+]] = stablehlo.add %iterArg, %c{{.*}} : tensor<i64>
// CHECK-NEXT:     stablehlo.dynamic_slice %[[X]], %[[N]], %c{{.*}}, sizes = [1, 4] : (tensor<4x5xf64>, tensor<i64>, tensor<i64>) -> tensor<1x4xf64>
