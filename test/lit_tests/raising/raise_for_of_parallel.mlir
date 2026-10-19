// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A sequential loop whose trip count is only known at runtime iterates as a
// stablehlo.while; the parallel body raises batched inside it and the buffers
// it writes are carried through the loop.
func.func @timeloop(%out: memref<100xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.for %t = 0 to %ni {
    affine.parallel (%i) = (0) to (100) {
      %v = affine.load %out[%i] : memref<100xf64, 1>
      %c = arith.constant 1.0 : f64
      %s = arith.addf %v, %c : f64
      affine.store %s, %out[%i] : memref<100xf64, 1>
    }
  }
  return
}

// CHECK:    func.func private @timeloop_raised(%[[a1:.+]]: tensor<100xf64>, %[[a2:.+]]: tensor<i64>) -> (tensor<100xf64>, tensor<i64>) {
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[a6:.+]]:3 = stablehlo.while(%[[a7:.+]] = %[[a4]], %[[a8:.+]] = %[[a1]], %[[a9:.+]] = %[[a2]]) : tensor<i64>, tensor<100xf64>, tensor<i64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %[[a10:.+]] = stablehlo.compare LT, %[[a7]], %[[a2]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[a10]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[a10]] = stablehlo.iota dim = 0 : tensor<100xi64>
// CHECK-NEXT:      %[[a11:.+]] = stablehlo.constant dense<0> : tensor<100xi64>
// CHECK-NEXT:      %[[a12:.+]] = stablehlo.add %[[a10]], %[[a11]] : tensor<100xi64>
// CHECK-NEXT:      %[[a13:.+]] = stablehlo.constant dense<1> : tensor<100xi64>
// CHECK-NEXT:      %[[a14:.+]] = stablehlo.multiply %[[a12]], %[[a13]] : tensor<100xi64>
// CHECK-NEXT:      %[[a15:.+]] = stablehlo.broadcast_in_dim %[[a3]], dims = [] : (tensor<f64>) -> tensor<100xf64>
// CHECK-NEXT:      %[[a16:.+]] = arith.addf %[[a8]], %[[a15]] : tensor<100xf64>
// CHECK-NEXT:      %[[a17:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a18:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a19:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a20:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a21:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a22:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %[[a23:.+]] = stablehlo.dynamic_update_slice %[[a8]], %[[a16]], %[[a22]] : (tensor<100xf64>, tensor<100xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK-NEXT:      %[[a24:.+]] = stablehlo.add %[[a7]], %[[a5]] : tensor<i64>
// CHECK-NEXT:      stablehlo.return %[[a24]], %[[a23]], %[[a9]] : tensor<i64>, tensor<100xf64>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a6]]#1, %[[a6]]#2 : tensor<100xf64>, tensor<i64>
// CHECK-NEXT:  }

// -----

// Scratch allocated outside the loop is carried through the while like any
// other buffer, starting from a zero splat.
func.func @scratchloop(%out: memref<10xf64, 1>, %nbuf: memref<i64, 1>) {
  %tmp = memref.alloca() : memref<10xf64>
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.for %t = 0 to %ni {
    affine.parallel (%i) = (0) to (10) {
      %v = affine.load %out[%i] : memref<10xf64, 1>
      affine.store %v, %tmp[%i] : memref<10xf64>
    }
    affine.parallel (%i) = (0) to (10) {
      %v = affine.load %tmp[9 - %i] : memref<10xf64>
      affine.store %v, %out[%i] : memref<10xf64, 1>
    }
  }
  return
}

// CHECK:    func.func private @scratchloop_raised(%[[a1:.+]]: tensor<10xf64>, %[[a2:.+]]: tensor<i64>) -> (tensor<10xf64>, tensor<i64>) {
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<10xf64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[a6:.+]]:4 = stablehlo.while(%[[a7:.+]] = %[[a4]], %[[a8:.+]] = %[[a1]], %[[a9:.+]] = %[[a2]], %[[a10:.+]] = %[[a3]]) : tensor<i64>, tensor<10xf64>, tensor<i64>, tensor<10xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %[[a11:.+]] = stablehlo.compare LT, %[[a7]], %[[a2]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[a11]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[a11]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT:      %[[a12:.+]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK-NEXT:      %[[a13:.+]] = stablehlo.add %[[a11]], %[[a12]] : tensor<10xi64>
// CHECK-NEXT:      %[[a14:.+]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK-NEXT:      %[[a15:.+]] = stablehlo.multiply %[[a13]], %[[a14]] : tensor<10xi64>
// CHECK-NEXT:      %[[a16:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a17:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a18:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a19:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a20:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a21:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %[[a22:.+]] = stablehlo.dynamic_update_slice %[[a10]], %[[a8]], %[[a21]] : (tensor<10xf64>, tensor<10xf64>, tensor<i64>) -> tensor<10xf64>
// CHECK-NEXT:      %[[a23:.+]] = stablehlo.iota dim = 0 : tensor<10xi64>
// CHECK-NEXT:      %[[a24:.+]] = stablehlo.constant dense<0> : tensor<10xi64>
// CHECK-NEXT:      %[[a25:.+]] = stablehlo.add %[[a23]], %[[a24]] : tensor<10xi64>
// CHECK-NEXT:      %[[a26:.+]] = stablehlo.constant dense<1> : tensor<10xi64>
// CHECK-NEXT:      %[[a27:.+]] = stablehlo.multiply %[[a25]], %[[a26]] : tensor<10xi64>
// CHECK-NEXT:      %[[a28:.+]] = stablehlo.reverse %[[a22]], dims = [0] : tensor<10xf64>
// CHECK-NEXT:      %[[a29:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a30:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a31:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a32:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a33:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a34:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %[[a35:.+]] = stablehlo.dynamic_update_slice %[[a8]], %[[a28]], %[[a34]] : (tensor<10xf64>, tensor<10xf64>, tensor<i64>) -> tensor<10xf64>
// CHECK-NEXT:      %[[a36:.+]] = stablehlo.add %[[a7]], %[[a5]] : tensor<i64>
// CHECK-NEXT:      stablehlo.return %[[a36]], %[[a35]], %[[a9]], %[[a22]] : tensor<i64>, tensor<10xf64>, tensor<i64>, tensor<10xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a6]]#1, %[[a6]]#2 : tensor<10xf64>, tensor<i64>
// CHECK-NEXT:  }
