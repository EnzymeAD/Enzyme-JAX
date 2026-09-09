// RUN: enzymexlamlir-opt %s -polygeist-mem2reg --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A grid-stride reduction: an scf.for with uniform runtime bounds raises as
// a stablehlo.while whose counter is a rank-0 scalar of the loop's integer
// type, carrying the per-lane accumulator as a batched tensor.
func.func @scfred(%out: memref<100xf64, 1>, %in: memref<?xf64, 1>, %nb: memref<i32, 1>) {
  %z = arith.constant 0.0 : f64
  %c100 = arith.constant 100 : i32
  affine.parallel (%t) = (0) to (100) {
    %n = affine.load %nb[] : memref<i32, 1>
    %sum = scf.for %j = %c100 to %n step %c100 iter_args(%acc = %z) -> (f64) : i32 {
      %ti = arith.index_castui %t : index to i32
      %idx32 = arith.addi %j, %ti : i32
      %idx = arith.index_cast %idx32 : i32 to index
      %v = memref.load %in[%idx] : memref<?xf64, 1>
      %a = arith.addf %acc, %v : f64
      scf.yield %a : f64
    }
    affine.store %sum, %out[%t] : memref<100xf64, 1>
  }
  return
}

// CHECK:    func.func private @scfred_raised(%[[a1:.+]]: tensor<100xf64>, %[[a2:.+]]: tensor<?xf64>, %[[a3:.+]]: tensor<i32>) -> (tensor<100xf64>, tensor<?xf64>, tensor<i32>) {
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.constant dense<100> : tensor<i32>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.iota dim = 0 : tensor<100xi64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.constant dense<0> : tensor<100xi64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.add %[[a6]], %[[a7]] : tensor<100xi64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.constant dense<1> : tensor<100xi64>
// CHECK-NEXT:    %[[a10:.+]] = stablehlo.multiply %[[a8]], %[[a9]] : tensor<100xi64>
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.broadcast_in_dim %[[a4]], dims = [] : (tensor<f64>) -> tensor<100xf64>
// CHECK-NEXT:    %[[a12:.+]]:5 = stablehlo.while(%[[a13:.+]] = %[[a5]], %[[a14:.+]] = %[[a11]], %[[a15:.+]] = %[[a1]], %[[a16:.+]] = %[[a2]], %[[a17:.+]] = %[[a3]]) : tensor<i32>, tensor<100xf64>, tensor<100xf64>, tensor<?xf64>, tensor<i32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %[[a18:.+]] = stablehlo.compare LT, %[[a13]], %[[a3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[a18]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[a18]] = stablehlo.convert %[[a10]] : (tensor<100xi64>) -> tensor<100xi32>
// CHECK-NEXT:      %[[a19:.+]] = stablehlo.broadcast_in_dim %[[a13]], dims = [] : (tensor<i32>) -> tensor<100xi32>
// CHECK-NEXT:      %[[a20:.+]] = arith.addi %[[a19]], %[[a18]] : tensor<100xi32>
// CHECK-NEXT:      %[[a21:.+]] = stablehlo.convert %[[a20]] : (tensor<100xi32>) -> tensor<100xi64>
// CHECK-NEXT:      %[[a22:.+]] = stablehlo.reshape %[[a21]] : (tensor<100xi64>) -> tensor<100x1xi64>
// CHECK-NEXT:      %[[a23:.+]] = "stablehlo.gather"(%[[a16]], %[[a22]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<?xf64>, tensor<100x1xi64>) -> tensor<100xf64>
// CHECK-NEXT:      %[[a24:.+]] = arith.addf %[[a14]], %[[a23]] : tensor<100xf64>
// CHECK-NEXT:      %[[a25:.+]] = stablehlo.add %[[a13]], %[[a5]] : tensor<i32>
// CHECK-NEXT:      stablehlo.return %[[a25]], %[[a24]], %[[a15]], %[[a16]], %[[a17]] : tensor<i32>, tensor<100xf64>, tensor<100xf64>, tensor<?xf64>, tensor<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a26:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a27:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a28:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a29:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a30:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a31:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a32:.+]] = stablehlo.dynamic_update_slice %[[a12]]#2, %[[a12]]#1, %[[a31]] : (tensor<100xf64>, tensor<100xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK-NEXT:    return %[[a32]], %[[a12]]#3, %[[a12]]#4 : tensor<100xf64>, tensor<?xf64>, tensor<i32>
// CHECK-NEXT:  }

// -----

// Ping-pong buffers, the kernel handed both: branches yielding one of two
// buffers are split into a branch per access by polygeist-mem2reg (#3116,
// #3133) when the access sits inside a GPU wrapper, so loads select between
// the carried tensors and stores mask into each, with no frozen alias.
#even = affine_set<(d0) : (d0 mod 2 == 0)>
func.func @pingpong(%a: memref<100xf64, 1>, %b: memref<100xf64, 1>, %ni: index) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  "enzymexla.gpu_wrapper"(%ni, %c1, %c1, %c100, %c1, %c1) ({
    affine.for %s = 0 to %ni {
      affine.parallel (%i) = (0) to (100) {
        %src = affine.if #even(%s) -> memref<100xf64, 1> {
          affine.yield %a : memref<100xf64, 1>
        } else {
          affine.yield %b : memref<100xf64, 1>
        }
        %dst = affine.if #even(%s) -> memref<100xf64, 1> {
          affine.yield %b : memref<100xf64, 1>
        } else {
          affine.yield %a : memref<100xf64, 1>
        }
        %v = affine.load %src[%i] : memref<100xf64, 1>
        %w = arith.mulf %v, %v : f64
        affine.store %w, %dst[%i] : memref<100xf64, 1>
      }
    } {enzymexla.parallel}
    "enzymexla.polygeist_yield"() : () -> ()
  }) : (index, index, index, index, index, index) -> index
  return
}

// CHECK:    func.func @pingpong(%[[a1:.+]]: memref<100xf64, 1>, %[[a2:.+]]: memref<100xf64, 1>, %[[a3:.+]]: index) {
// CHECK-NEXT:    %[[a4:.+]] = memref.alloca() : memref<i64>
// CHECK-NEXT:    %[[a5:.+]] = arith.constant 1 : index
// CHECK-NEXT:    %[[a6:.+]] = arith.constant 100 : index
// CHECK-NEXT:    %[[a7:.+]] = gpu.alloc  () : memref<i64, 1>
// CHECK-NEXT:    %[[a8:.+]] = arith.index_cast %[[a3]] : index to i64
// CHECK-NEXT:    affine.store %[[a8]], %[[a4]][] : memref<i64>
// CHECK-NEXT:    %[[a9:.+]] = arith.constant 8 : index
// CHECK-NEXT:    enzymexla.memcpy  %[[a7]], %[[a4]], %[[a9]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%[[a1]], %[[a2]], %[[a7]]) : (memref<100xf64, 1>, memref<100xf64, 1>, memref<i64, 1>) -> ()
// CHECK-NEXT:    %[[a10:.+]] = arith.constant 0 : index
// CHECK-NEXT:    gpu.dealloc  %[[a7]] : memref<i64, 1>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @rxla$raised_0(%[[a1]]: tensor<100xf64>, %[[a2]]: tensor<100xf64>, %[[a3]]: tensor<i64>) -> (tensor<100xf64>, tensor<100xf64>, tensor<i64>) {
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[a8]]:3 = stablehlo.while(%[[a13:.+]] = %[[a11]], %[[a14:.+]] = %[[a1]], %[[a15:.+]] = %[[a2]]) : tensor<i64>, tensor<100xf64>, tensor<100xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %[[a16:.+]] = stablehlo.compare LT, %[[a13]], %[[a3]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[a16]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[a16]] = stablehlo.iota dim = 0 : tensor<100xi64>
// CHECK-NEXT:      %[[a17:.+]] = stablehlo.constant dense<0> : tensor<100xi64>
// CHECK-NEXT:      %[[a18:.+]] = stablehlo.add %[[a16]], %[[a17]] : tensor<100xi64>
// CHECK-NEXT:      %[[a19:.+]] = stablehlo.constant dense<1> : tensor<100xi64>
// CHECK-NEXT:      %[[a20:.+]] = stablehlo.multiply %[[a18]], %[[a19]] : tensor<100xi64>
// CHECK-NEXT:      %[[a21:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:      %[[a22:.+]] = stablehlo.remainder %[[a13]], %[[a21]] : tensor<i64>
// CHECK-NEXT:      %[[a23:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %[[a24:.+]] = stablehlo.compare LT, %[[a22]], %[[a23]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %[[a25:.+]] = stablehlo.add %[[a22]], %[[a21]] : tensor<i64>
// CHECK-NEXT:      %[[a26:.+]] = stablehlo.select %[[a24]], %[[a25]], %[[a22]] : tensor<i1>, tensor<i64>
// CHECK-NEXT:      %[[a27:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %[[a28:.+]] = stablehlo.compare EQ, %[[a26]], %[[a27]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %[[a29:.+]] = stablehlo.not %[[a28]] : tensor<i1>
// CHECK-NEXT:      %[[a30:.+]] = stablehlo.broadcast_in_dim %[[a28]], dims = [] : (tensor<i1>) -> tensor<100xi1>
// CHECK-NEXT:      %[[a31:.+]] = stablehlo.select %[[a30]], %[[a14]], %[[a15]] : tensor<100xi1>, tensor<100xf64>
// CHECK-NEXT:      %[[a32:.+]] = arith.mulf %[[a31]], %[[a31]] : tensor<100xf64>
// CHECK-NEXT:      %[[a33:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:      %[[a34:.+]] = stablehlo.remainder %[[a13]], %[[a33]] : tensor<i64>
// CHECK-NEXT:      %[[a35:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %[[a36:.+]] = stablehlo.compare LT, %[[a34]], %[[a35]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %[[a37:.+]] = stablehlo.add %[[a34]], %[[a33]] : tensor<i64>
// CHECK-NEXT:      %[[a38:.+]] = stablehlo.select %[[a36]], %[[a37]], %[[a34]] : tensor<i1>, tensor<i64>
// CHECK-NEXT:      %[[a39:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %[[a40:.+]] = stablehlo.compare EQ, %[[a38]], %[[a39]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %[[a41:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a42:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a43:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a44:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a45:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a46:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %[[a47:.+]] = stablehlo.broadcast_in_dim %[[a40]], dims = [] : (tensor<i1>) -> tensor<100xi1>
// CHECK-NEXT:      %[[a48:.+]] = stablehlo.select %[[a47]], %[[a32]], %[[a15]] : tensor<100xi1>, tensor<100xf64>
// CHECK-NEXT:      %[[a49:.+]] = stablehlo.dynamic_update_slice %[[a15]], %[[a48]], %[[a46]] : (tensor<100xf64>, tensor<100xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK-NEXT:      %[[a50:.+]] = stablehlo.not %[[a40]] : tensor<i1>
// CHECK-NEXT:      %[[a51:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a52:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a53:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a54:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a55:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %[[a56:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %[[a57:.+]] = stablehlo.broadcast_in_dim %[[a50]], dims = [] : (tensor<i1>) -> tensor<100xi1>
// CHECK-NEXT:      %[[a58:.+]] = stablehlo.select %[[a57]], %[[a32]], %[[a14]] : tensor<100xi1>, tensor<100xf64>
// CHECK-NEXT:      %[[a59:.+]] = stablehlo.dynamic_update_slice %[[a14]], %[[a58]], %[[a56]] : (tensor<100xf64>, tensor<100xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK-NEXT:      %[[a60:.+]] = stablehlo.add %[[a13]], %[[a12]] : tensor<i64>
// CHECK-NEXT:      stablehlo.return %[[a60]], %[[a59]], %[[a49]] : tensor<i64>, tensor<100xf64>, tensor<100xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[a8]]#1, %[[a8]]#2, %[[a3]] : tensor<100xf64>, tensor<100xf64>, tensor<i64>
// CHECK-NEXT:  }

// -----

// A remainder loop whose bounds vary per lane iterates a scalar counter to
// the maximum lane trip count, masking finished lanes: stores mask, and iter
// args keep their value once a lane finishes.
func.func @lanefor(%out: memref<8xf64, 1>, %in: memref<?xf64, 1>) {
  %z = arith.constant 0.0 : f64
  affine.parallel (%t) = (0) to (8) {
    %s = affine.for %j = affine_map<(d0) -> (d0)>(%t) to 8 iter_args(%acc = %z) -> (f64) {
      %v = affine.load %in[%j] : memref<?xf64, 1>
      %a = arith.addf %acc, %v : f64
      affine.yield %a : f64
    }
    affine.store %s, %out[%t] : memref<8xf64, 1>
  }
  return
}

// CHECK:    func.func private @lanefor_raised(%[[a1:.+]]: tensor<8xf64>, %[[a2:.+]]: tensor<?xf64>) -> (tensor<8xf64>, tensor<?xf64>) {
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.iota dim = 0 : tensor<8xi64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.constant dense<0> : tensor<8xi64>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.add %[[a4]], %[[a5]] : tensor<8xi64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.constant dense<1> : tensor<8xi64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.multiply %[[a6]], %[[a7]] : tensor<8xi64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %[[a10:.+]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[a13:.+]] = stablehlo.broadcast_in_dim %[[a11]], dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.subtract %[[a13]], %[[a8]] : tensor<8xi64>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.constant dense<0> : tensor<8xi64>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.maximum %[[a15]], %[[a16]] : tensor<8xi64>
// CHECK-NEXT:    %[[a18:.+]] = stablehlo.constant dense<1> : tensor<8xi64>
// CHECK-NEXT:    %[[a19:.+]] = stablehlo.subtract %[[a14]], %[[a18]] : tensor<8xi64>
// CHECK-NEXT:    %[[a20:.+]] = stablehlo.add %[[a17]], %[[a19]] : tensor<8xi64>
// CHECK-NEXT:    %[[a21:.+]] = stablehlo.divide %[[a20]], %[[a14]] : tensor<8xi64>
// CHECK-NEXT:    %[[a22:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a23:.+]] = stablehlo.reduce(%[[a21]] init: %[[a22]]) applies stablehlo.maximum across dimensions = [0] : (tensor<8xi64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:    %[[a24:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a25:.+]] = stablehlo.broadcast_in_dim %[[a3]], dims = [] : (tensor<f64>) -> tensor<8xf64>
// CHECK-NEXT:    %[[a26:.+]]:4 = stablehlo.while(%[[a27:.+]] = %[[a24]], %[[a28:.+]] = %[[a25]], %[[a29:.+]] = %[[a1]], %[[a30:.+]] = %[[a2]]) : tensor<i64>, tensor<8xf64>, tensor<8xf64>, tensor<?xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %[[a31:.+]] = stablehlo.compare LT, %[[a27]], %[[a23]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %[[a31]] : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %[[a31]] = stablehlo.broadcast_in_dim %[[a27]], dims = [] : (tensor<i64>) -> tensor<8xi64>
// CHECK-NEXT:      %[[a32:.+]] = stablehlo.multiply %[[a31]], %[[a14]] : tensor<8xi64>
// CHECK-NEXT:      %[[a33:.+]] = stablehlo.add %[[a8]], %[[a32]] : tensor<8xi64>
// CHECK-NEXT:      %[[a34:.+]] = stablehlo.compare LT, %[[a33]], %[[a13]] : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
// CHECK-NEXT:      %[[a35:.+]] = stablehlo.reshape %[[a33]] : (tensor<8xi64>) -> tensor<8x1xi64>
// CHECK-NEXT:      %[[a36:.+]] = "stablehlo.gather"(%[[a30]], %[[a35]]) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<?xf64>, tensor<8x1xi64>) -> tensor<8xf64>
// CHECK-NEXT:      %[[a37:.+]] = arith.addf %[[a28]], %[[a36]] : tensor<8xf64>
// CHECK-NEXT:      %[[a38:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:      %[[a39:.+]] = stablehlo.add %[[a27]], %[[a38]] : tensor<i64>
// CHECK-NEXT:      %[[a40:.+]] = stablehlo.select %[[a34]], %[[a37]], %[[a28]] : tensor<8xi1>, tensor<8xf64>
// CHECK-NEXT:      stablehlo.return %[[a39]], %[[a40]], %[[a29]], %[[a30]] : tensor<i64>, tensor<8xf64>, tensor<8xf64>, tensor<?xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a41:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a42:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a43:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a44:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a45:.+]] = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %[[a46:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a47:.+]] = stablehlo.dynamic_update_slice %[[a26]]#2, %[[a26]]#1, %[[a46]] : (tensor<8xf64>, tensor<8xf64>, tensor<i64>) -> tensor<8xf64>
// CHECK-NEXT:    return %[[a47]], %[[a26]]#3 : tensor<8xf64>, tensor<?xf64>
// CHECK-NEXT:  }

// -----

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
