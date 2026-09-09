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

// CHECK:    func.func private @dowhile_loop_raised(%[[a1:.+]]: tensor<100xf64>, %[[a2:.+]]: tensor<i32>) -> (tensor<100xf64>, tensor<i32>) {
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.iota dim = 0 : tensor<100xi64>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.constant dense<0> : tensor<100xi64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.add %[[a5]], %[[a6]] : tensor<100xi64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.constant dense<1> : tensor<100xi64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.multiply %[[a7]], %[[a8]] : tensor<100xi64>
// CHECK-NEXT:    %[[a10:.+]] = arith.addi %[[a3]], %[[a4]] : tensor<i32>
// CHECK-NEXT:    %[[a11:.+]] = arith.cmpi slt, %[[a10]], %[[a2]] : tensor<i32>
// CHECK-NEXT:    %[[a12:.+]]:4 = stablehlo.while(%[[a13:.+]] = %[[a11]], %[[a14:.+]] = %[[a10]], %[[a15:.+]] = %[[a1]], %[[a16:.+]] = %[[a2]]) : tensor<i1>, tensor<i32>, tensor<100xf64>, tensor<i32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      stablehlo.return %[[a13]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[a17:.+]] = arith.addi %[[a14]], %[[a4]] : tensor<i32>
// CHECK-NEXT:      %[[a18:.+]] = arith.cmpi slt, %[[a17]], %[[a2]] : tensor<i32>
// CHECK-NEXT:      stablehlo.return %[[a18]], %[[a17]], %[[a15]], %[[a16]] : tensor<i1>, tensor<i32>, tensor<100xf64>, tensor<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a19:.+]] = arith.sitofp %[[a12]]#1 : tensor<i32> to tensor<f64>
// CHECK-NEXT:    %[[a20:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a21:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a22:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a23:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a24:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a25:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a26:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<100xf64>
// CHECK-NEXT:    %[[a27:.+]] = stablehlo.dynamic_update_slice %[[a12]]#2, %[[a26]], %[[a25]] : (tensor<100xf64>, tensor<100xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK-NEXT:    return %[[a27]], %[[a12]]#3 : tensor<100xf64>, tensor<i32>
// CHECK-NEXT:  }

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

// CHECK:    func.func private @dowhile_store_raised(%[[a1:.+]]: tensor<100xf64>, %[[a2:.+]]: tensor<f64>, %[[a3:.+]]: tensor<i32>) -> (tensor<100xf64>, tensor<f64>, tensor<i32>) {
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.iota dim = 0 : tensor<100xi64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.constant dense<0> : tensor<100xi64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.add %[[a7]], %[[a8]] : tensor<100xi64>
// CHECK-NEXT:    %[[a10:.+]] = stablehlo.constant dense<1> : tensor<100xi64>
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.multiply %[[a9]], %[[a10]] : tensor<100xi64>
// CHECK-NEXT:    %[[a12:.+]] = arith.addi %[[a4]], %[[a5]] : tensor<i32>
// CHECK-NEXT:    %[[a13:.+]] = arith.addf %[[a2]], %[[a6]] : tensor<f64>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.dynamic_update_slice %[[a2]], %[[a13]] : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:    %[[a15:.+]] = arith.cmpi slt, %[[a12]], %[[a3]] : tensor<i32>
// CHECK-NEXT:    %[[a16:.+]]:5 = stablehlo.while(%[[a17:.+]] = %[[a15]], %[[a18:.+]] = %[[a12]], %[[a19:.+]] = %[[a1]], %[[a20:.+]] = %[[a14]], %[[a21:.+]] = %[[a3]]) : tensor<i1>, tensor<i32>, tensor<100xf64>, tensor<f64>, tensor<i32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      stablehlo.return %[[a17]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[a22:.+]] = arith.addi %[[a18]], %[[a5]] : tensor<i32>
// CHECK-NEXT:      %[[a23:.+]] = arith.addf %[[a20]], %[[a6]] : tensor<f64>
// CHECK-NEXT:      %[[a24:.+]] = stablehlo.dynamic_update_slice %[[a20]], %[[a23]] : (tensor<f64>, tensor<f64>) -> tensor<f64>
// CHECK-NEXT:      %[[a25:.+]] = arith.cmpi slt, %[[a22]], %[[a3]] : tensor<i32>
// CHECK-NEXT:      stablehlo.return %[[a25]], %[[a22]], %[[a19]], %[[a24]], %[[a21]] : tensor<i1>, tensor<i32>, tensor<100xf64>, tensor<f64>, tensor<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a26:.+]] = arith.sitofp %[[a16]]#1 : tensor<i32> to tensor<f64>
// CHECK-NEXT:    %[[a27:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a28:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a29:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a30:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a31:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a32:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a33:.+]] = stablehlo.broadcast_in_dim %[[a26]], dims = [] : (tensor<f64>) -> tensor<100xf64>
// CHECK-NEXT:    %[[a34:.+]] = stablehlo.dynamic_update_slice %[[a16]]#2, %[[a33]], %[[a32]] : (tensor<100xf64>, tensor<100xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK-NEXT:    return %[[a34]], %[[a16]]#3, %[[a16]]#4 : tensor<100xf64>, tensor<f64>, tensor<i32>
// CHECK-NEXT:  }
