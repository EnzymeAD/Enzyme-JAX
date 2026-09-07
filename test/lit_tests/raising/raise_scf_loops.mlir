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

// CHECK-LABEL: func.func private @scfred_raised(
// CHECK: %[[W:[0-9]+]]:5 = stablehlo.while(%[[IV:[a-zA-Z0-9_]+]] = %{{[^,]+}}, %[[ACC:[a-zA-Z0-9_]+]] = %{{[^,]+}}, %{{.+}}) : tensor<i32>, tensor<100xf64>, tensor<100xf64>, tensor<?xf64>, tensor<i32>
// CHECK: cond {
// CHECK: stablehlo.compare LT, %[[IV]], %{{.+}} : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK: } do {
// CHECK: "stablehlo.gather"
// CHECK: arith.addf %[[ACC]], %{{.+}} : tensor<100xf64>
// CHECK: stablehlo.add %[[IV]], %{{.+}} : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice %[[W]]#2, %{{.+}} : (tensor<100xf64>, tensor<100xf64>, tensor<i64>) -> tensor<100xf64>

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

// CHECK:    func.func @pingpong(%arg0: memref<100xf64, 1>, %arg1: memref<100xf64, 1>, %arg2: index) {
// CHECK-NEXT:    %alloca = memref.alloca() : memref<i64>
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c100 = arith.constant 100 : index
// CHECK-NEXT:    %memref = gpu.alloc  () : memref<i64, 1>
// CHECK-NEXT:    %0 = arith.index_cast %arg2 : index to i64
// CHECK-NEXT:    affine.store %0, %alloca[] : memref<i64>
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    enzymexla.memcpy  %memref, %alloca, %c8 : memref<i64, 1>, memref<i64>
// CHECK-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%arg0, %arg1, %memref) : (memref<100xf64, 1>, memref<100xf64, 1>, memref<i64, 1>) -> ()
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    gpu.dealloc  %memref : memref<i64, 1>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func private @rxla$raised_0(%arg0: tensor<100xf64>, %arg1: tensor<100xf64>, %arg2: tensor<i64>) -> (tensor<100xf64>, tensor<100xf64>, tensor<i64>) {
// CHECK-NEXT:    %0 = stablehlo.reshape %arg2 : (tensor<i64>) -> tensor<i64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %1:3 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0, %iterArg_2 = %arg1) : tensor<i64>, tensor<100xf64>, tensor<100xf64> attributes {enzymexla.parallel}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %2 = stablehlo.compare LT, %iterArg, %0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %2 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %2 = stablehlo.iota dim = 0 : tensor<100xi64>
// CHECK-NEXT:      %c_3 = stablehlo.constant dense<0> : tensor<100xi64>
// CHECK-NEXT:      %3 = stablehlo.add %2, %c_3 : tensor<100xi64>
// CHECK-NEXT:      %c_4 = stablehlo.constant dense<1> : tensor<100xi64>
// CHECK-NEXT:      %4 = stablehlo.multiply %3, %c_4 : tensor<100xi64>
// CHECK-NEXT:      %c_5 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:      %5 = stablehlo.remainder %iterArg, %c_5 : tensor<i64>
// CHECK-NEXT:      %c_6 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %6 = stablehlo.compare LT, %5, %c_6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %7 = stablehlo.add %5, %c_5 : tensor<i64>
// CHECK-NEXT:      %8 = stablehlo.select %6, %7, %5 : tensor<i1>, tensor<i64>
// CHECK-NEXT:      %c_7 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.compare EQ, %8, %c_7 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %10 = stablehlo.reshape %iterArg_1 : (tensor<100xf64>) -> tensor<100xf64>
// CHECK-NEXT:      %11 = stablehlo.not %9 : tensor<i1>
// CHECK-NEXT:      %12 = stablehlo.reshape %iterArg_2 : (tensor<100xf64>) -> tensor<100xf64>
// CHECK-NEXT:      %13 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<i1>) -> tensor<100xi1>
// CHECK-NEXT:      %14 = stablehlo.select %13, %10, %12 : tensor<100xi1>, tensor<100xf64>
// CHECK-NEXT:      %15 = arith.mulf %14, %14 : tensor<100xf64>
// CHECK-NEXT:      %c_8 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:      %16 = stablehlo.remainder %iterArg, %c_8 : tensor<i64>
// CHECK-NEXT:      %c_9 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %17 = stablehlo.compare LT, %16, %c_9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %18 = stablehlo.add %16, %c_8 : tensor<i64>
// CHECK-NEXT:      %19 = stablehlo.select %17, %18, %16 : tensor<i1>, tensor<i64>
// CHECK-NEXT:      %c_10 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %20 = stablehlo.compare EQ, %19, %c_10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_12 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_16 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %21 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<100xf64>) -> tensor<100xf64>
// CHECK-NEXT:      %22 = stablehlo.dynamic_slice %iterArg_2, %c_16, sizes = [100] : (tensor<100xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK-NEXT:      %23 = stablehlo.reshape %21 : (tensor<100xf64>) -> tensor<100xf64>
// CHECK-NEXT:      %24 = stablehlo.reshape %22 : (tensor<100xf64>) -> tensor<100xf64>
// CHECK-NEXT:      %25 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<i1>) -> tensor<100xi1>
// CHECK-NEXT:      %26 = stablehlo.select %25, %23, %24 : tensor<100xi1>, tensor<100xf64>
// CHECK-NEXT:      %27 = stablehlo.broadcast_in_dim %26, dims = [0] : (tensor<100xf64>) -> tensor<100xf64>
// CHECK-NEXT:      %28 = stablehlo.dynamic_update_slice %iterArg_2, %27, %c_16 : (tensor<100xf64>, tensor<100xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK-NEXT:      %29 = stablehlo.not %20 : tensor<i1>
// CHECK-NEXT:      %c_17 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_18 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_19 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_20 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_21 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_22 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %30 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<100xf64>) -> tensor<100xf64>
// CHECK-NEXT:      %31 = stablehlo.dynamic_slice %iterArg_1, %c_22, sizes = [100] : (tensor<100xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK-NEXT:      %32 = stablehlo.reshape %30 : (tensor<100xf64>) -> tensor<100xf64>
// CHECK-NEXT:      %33 = stablehlo.reshape %31 : (tensor<100xf64>) -> tensor<100xf64>
// CHECK-NEXT:      %34 = stablehlo.broadcast_in_dim %29, dims = [] : (tensor<i1>) -> tensor<100xi1>
// CHECK-NEXT:      %35 = stablehlo.select %34, %32, %33 : tensor<100xi1>, tensor<100xf64>
// CHECK-NEXT:      %36 = stablehlo.broadcast_in_dim %35, dims = [0] : (tensor<100xf64>) -> tensor<100xf64>
// CHECK-NEXT:      %37 = stablehlo.dynamic_update_slice %iterArg_1, %36, %c_22 : (tensor<100xf64>, tensor<100xf64>, tensor<i64>) -> tensor<100xf64>
// CHECK-NEXT:      %38 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %38, %37, %28 : tensor<i64>, tensor<100xf64>, tensor<100xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1#1, %1#2, %arg2 : tensor<100xf64>, tensor<100xf64>, tensor<i64>
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

// CHECK-LABEL: func.func private @lanefor_raised(
// CHECK: stablehlo.reduce{{.*}}applies stablehlo.maximum
// CHECK: stablehlo.while
// CHECK: } do {
// CHECK: %[[ACTIVE:.+]] = stablehlo.compare LT, %{{.+}}, %{{.+}} : (tensor<8xi64>, tensor<8xi64>) -> tensor<8xi1>
// CHECK: stablehlo.select %[[ACTIVE]], %{{.+}}, %{{.+}} : tensor<8xi1>, tensor<8xf64>

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

// CHECK-LABEL: func.func private @dowhile_loop_raised(
// CHECK: stablehlo.while(%[[C:[a-zA-Z0-9_]+]] = %{{[^,]+}}, %{{.+}}) : tensor<i1>, tensor<i32>, tensor<100xf64>, tensor<i32>
// CHECK: cond {
// CHECK: stablehlo.return %[[C]] : tensor<i1>
