// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --enzyme-hlo-opt --split-input-file | FileCheck %s --check-prefix=OPT

// A tid==0 broadcast store: the mask covers the thread axis but the store
// does not index it, and the stored value is invariant along it, so the mask
// or-reduces over that axis.
func.func @lane0(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>) {
  affine.parallel (%t, %j) = (0, 0) to (32, 16) {
    affine.if affine_set<(d0) : (d0 == 0)>(%t) {
      %v = affine.load %in[%j] : memref<16xf64, 1>
      affine.store %v, %out[%j] : memref<16xf64, 1>
    }
  }
  return
}

// CHECK:    func.func private @lane0_raised(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<32xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<32xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<32xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<32xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_1 : tensor<16xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_2 : tensor<16xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:    %6 = stablehlo.compare EQ, %2, %c_3 : (tensor<32xi64>, tensor<32xi64>) -> tensor<32xi1>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %7 = stablehlo.reduce(%6 init: %c_10) applies stablehlo.or across dimensions = [0] : (tensor<32xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i1>) -> tensor<16xi1>
// CHECK-NEXT:    %9 = stablehlo.select %8, %arg1, %arg0 : tensor<16xi1>, tensor<16xf64>
// CHECK-NEXT:    %10 = stablehlo.dynamic_update_slice %arg0, %9, %c_9 : (tensor<16xf64>, tensor<16xf64>, tensor<i64>) -> tensor<16xf64>
// CHECK-NEXT:    return %10, %arg1 : tensor<16xf64>, tensor<16xf64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @lane0_raised(%arg0: tensor<16xf64>, %arg1: tensor<16xf64>) -> (tensor<16xf64>, tensor<16xf64>) {
// OPT-NEXT:    return %arg1, %arg1 : tensor<16xf64>, tensor<16xf64>
// OPT-NEXT:  }

// -----
// The same lane-0 broadcast store at scale, 4096 elements by 256 lanes:
// the lane mask or-reduced over 256 lanes is a pad of an all-true row too
// large to constant-fold (reduce_or_and_pad folds it once merged).
func.func @lane0_large(%out: memref<4096xf64, 1>, %in: memref<4096xf64, 1>) {
  affine.parallel (%t, %e) = (0, 0) to (256, 4096) {
    affine.if affine_set<(d0) : (d0 == 0)>(%t) {
      %v = affine.load %in[%e] : memref<4096xf64, 1>
      affine.store %v, %out[%e] : memref<4096xf64, 1>
    }
  }
  return
}

// CHECK:    func.func private @lane0_large_raised(%arg0: tensor<4096xf64>, %arg1: tensor<4096xf64>) -> (tensor<4096xf64>, tensor<4096xf64>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<256xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<256xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<256xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<256xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<256xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<4096xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<4096xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_1 : tensor<4096xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<4096xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_2 : tensor<4096xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<256xi64>
// CHECK-NEXT:    %6 = stablehlo.compare EQ, %2, %c_3 : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %7 = stablehlo.reduce(%6 init: %c_10) applies stablehlo.or across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %7, dims = [] : (tensor<i1>) -> tensor<4096xi1>
// CHECK-NEXT:    %9 = stablehlo.select %8, %arg1, %arg0 : tensor<4096xi1>, tensor<4096xf64>
// CHECK-NEXT:    %10 = stablehlo.dynamic_update_slice %arg0, %9, %c_9 : (tensor<4096xf64>, tensor<4096xf64>, tensor<i64>) -> tensor<4096xf64>
// CHECK-NEXT:    return %10, %arg1 : tensor<4096xf64>, tensor<4096xf64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @lane0_large_raised(%arg0: tensor<4096xf64>, %arg1: tensor<4096xf64>) -> (tensor<4096xf64>, tensor<4096xf64>) {
// OPT-NEXT:    return %arg1, %arg1 : tensor<4096xf64>, tensor<4096xf64>
// OPT-NEXT:  }

// -----
// Two-dimensional buffers with one row selected by the guard: only row 5 is
// read and written, chosen by an affine.if on the row axis, which the store
// does not index. The mask over the row axis is one-hot at 5.
func.func @row_small(%out: memref<16x32xf64, 1>, %in: memref<16x32xf64, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (16, 32) {
    affine.if affine_set<(d0) : (d0 - 5 == 0)>(%i) {
      %v = affine.load %in[5, %j] : memref<16x32xf64, 1>
      affine.store %v, %out[5, %j] : memref<16x32xf64, 1>
    }
  }
  return
}

// CHECK:    func.func private @row_small_raised(%arg0: tensor<16x32xf64>, %arg1: tensor<16x32xf64>) -> (tensor<16x32xf64>, tensor<16x32xf64>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<16xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<16xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<32xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_1 : tensor<32xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<32xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_2 : tensor<32xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<-5> : tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %7 = stablehlo.add %2, %6 : tensor<16xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %8 = stablehlo.compare EQ, %7, %c_4 : (tensor<16xi64>, tensor<16xi64>) -> tensor<16xi1>
// CHECK-NEXT:    %9 = stablehlo.slice %arg1 [5:6, 0:32] : (tensor<16x32xf64>) -> tensor<1x32xf64>
// CHECK-NEXT:    %10 = stablehlo.reshape %9 : (tensor<1x32xf64>) -> tensor<32xf64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<32xf64>) -> tensor<1x32xf64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %12 = stablehlo.reduce(%8 init: %c_17) applies stablehlo.or across dimensions = [0] : (tensor<16xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    %13 = stablehlo.slice %arg0 [5:6, 0:32] : (tensor<16x32xf64>) -> tensor<1x32xf64>
// CHECK-NEXT:    %14 = stablehlo.reshape %11 : (tensor<1x32xf64>) -> tensor<32xf64>
// CHECK-NEXT:    %15 = stablehlo.reshape %13 : (tensor<1x32xf64>) -> tensor<32xf64>
// CHECK-NEXT:    %16 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i1>) -> tensor<32xi1>
// CHECK-NEXT:    %17 = stablehlo.select %16, %14, %15 : tensor<32xi1>, tensor<32xf64>
// CHECK-NEXT:    %18 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<32xf64>) -> tensor<1x32xf64>
// CHECK-NEXT:    %19 = stablehlo.dynamic_update_slice %arg0, %18, %c_10, %c_16 : (tensor<16x32xf64>, tensor<1x32xf64>, tensor<i64>, tensor<i64>) -> tensor<16x32xf64>
// CHECK-NEXT:    return %19, %arg1 : tensor<16x32xf64>, tensor<16x32xf64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @row_small_raised(%arg0: tensor<16x32xf64>, %arg1: tensor<16x32xf64>) -> (tensor<16x32xf64>, tensor<16x32xf64>) {
// OPT-NEXT:    %0 = stablehlo.slice %arg1 [5:6, 0:32] : (tensor<16x32xf64>) -> tensor<1x32xf64>
// OPT-NEXT:    %1 = stablehlo.slice %arg0 [0:5, 0:32] : (tensor<16x32xf64>) -> tensor<5x32xf64>
// OPT-NEXT:    %2 = stablehlo.slice %arg0 [6:16, 0:32] : (tensor<16x32xf64>) -> tensor<10x32xf64>
// OPT-NEXT:    %3 = stablehlo.concatenate %1, %0, %2, dim = 0 : (tensor<5x32xf64>, tensor<1x32xf64>, tensor<10x32xf64>) -> tensor<16x32xf64>
// OPT-NEXT:    return %3, %arg1 : tensor<16x32xf64>, tensor<16x32xf64>
// OPT-NEXT:  }

// -----
// The same row selection at scale, 256 rows by 4096 columns: the row mask
// is a pad too large to constant-fold and still folds away.
func.func @row_large(%out: memref<256x4096xf64, 1>, %in: memref<256x4096xf64, 1>) {
  affine.parallel (%i, %j) = (0, 0) to (256, 4096) {
    affine.if affine_set<(d0) : (d0 - 5 == 0)>(%i) {
      %v = affine.load %in[5, %j] : memref<256x4096xf64, 1>
      affine.store %v, %out[5, %j] : memref<256x4096xf64, 1>
    }
  }
  return
}

// CHECK:    func.func private @row_large_raised(%arg0: tensor<256x4096xf64>, %arg1: tensor<256x4096xf64>) -> (tensor<256x4096xf64>, tensor<256x4096xf64>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<256xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<256xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<256xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<256xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<256xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<4096xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<4096xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_1 : tensor<4096xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<4096xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_2 : tensor<4096xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<-5> : tensor<i64>
// CHECK-NEXT:    %6 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i64>) -> tensor<256xi64>
// CHECK-NEXT:    %7 = stablehlo.add %2, %6 : tensor<256xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<256xi64>
// CHECK-NEXT:    %8 = stablehlo.compare EQ, %7, %c_4 : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
// CHECK-NEXT:    %9 = stablehlo.slice %arg1 [5:6, 0:4096] : (tensor<256x4096xf64>) -> tensor<1x4096xf64>
// CHECK-NEXT:    %10 = stablehlo.reshape %9 : (tensor<1x4096xf64>) -> tensor<4096xf64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<5> : tensor<i64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<4096xf64>) -> tensor<1x4096xf64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %12 = stablehlo.reduce(%8 init: %c_17) applies stablehlo.or across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    %13 = stablehlo.slice %arg0 [5:6, 0:4096] : (tensor<256x4096xf64>) -> tensor<1x4096xf64>
// CHECK-NEXT:    %14 = stablehlo.reshape %11 : (tensor<1x4096xf64>) -> tensor<4096xf64>
// CHECK-NEXT:    %15 = stablehlo.reshape %13 : (tensor<1x4096xf64>) -> tensor<4096xf64>
// CHECK-NEXT:    %16 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i1>) -> tensor<4096xi1>
// CHECK-NEXT:    %17 = stablehlo.select %16, %14, %15 : tensor<4096xi1>, tensor<4096xf64>
// CHECK-NEXT:    %18 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<4096xf64>) -> tensor<1x4096xf64>
// CHECK-NEXT:    %19 = stablehlo.dynamic_update_slice %arg0, %18, %c_10, %c_16 : (tensor<256x4096xf64>, tensor<1x4096xf64>, tensor<i64>, tensor<i64>) -> tensor<256x4096xf64>
// CHECK-NEXT:    return %19, %arg1 : tensor<256x4096xf64>, tensor<256x4096xf64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @row_large_raised(%arg0: tensor<256x4096xf64>, %arg1: tensor<256x4096xf64>) -> (tensor<256x4096xf64>, tensor<256x4096xf64>) {
// OPT-NEXT:    %0 = stablehlo.slice %arg1 [5:6, 0:4096] : (tensor<256x4096xf64>) -> tensor<1x4096xf64>
// OPT-NEXT:    %1 = stablehlo.slice %arg0 [0:5, 0:4096] : (tensor<256x4096xf64>) -> tensor<5x4096xf64>
// OPT-NEXT:    %2 = stablehlo.slice %arg0 [6:256, 0:4096] : (tensor<256x4096xf64>) -> tensor<250x4096xf64>
// OPT-NEXT:    %3 = stablehlo.concatenate %1, %0, %2, dim = 0 : (tensor<5x4096xf64>, tensor<1x4096xf64>, tensor<250x4096xf64>) -> tensor<256x4096xf64>
// OPT-NEXT:    return %3, %arg1 : tensor<256x4096xf64>, tensor<256x4096xf64>
// OPT-NEXT:  }

// -----
// The block-wide pattern: scratch filled by every lane, one lane broadcasting
// a scalar, a barrier, then every lane combining both. The barrier is a
// no-op in raised form and the broadcast mask or-reduces over the thread
// axis it does not index.
func.func @bcast(%out: memref<?xf64, 1>, %in: memref<?xf64, 1>, %nbuf: memref<i64, 1>) {
  %n = affine.load %nbuf[] : memref<i64, 1>
  %ni = arith.index_cast %n : i64 to index
  affine.for %e = 0 to %ni {
    %scr = memref.alloca() : memref<32xf64>
    %b = memref.alloca() : memref<1xf64>
    affine.parallel (%t) = (0) to (32) {
      %v = affine.load %in[%e * 32 + %t] : memref<?xf64, 1>
      affine.store %v, %scr[%t] : memref<32xf64>
      affine.if affine_set<(d0) : (d0 == 0)>(%t) {
        %s = affine.load %in[%e] : memref<?xf64, 1>
        affine.store %s, %b[0] : memref<1xf64>
      }
      "enzymexla.barrier"(%t) : (index) -> ()
      %x = affine.load %scr[%t] : memref<32xf64>
      %y = affine.load %b[0] : memref<1xf64>
      %m = arith.mulf %x, %y : f64
      affine.store %m, %out[%e * 32 + %t] : memref<?xf64, 1>
    }
  } {enzymexla.parallel}
  return
}

// CHECK:    func.func private @bcast_raised(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<i64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<i64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %0:4 = stablehlo.while(%iterArg = %c, %iterArg_1 = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %arg2) : tensor<i64>, tensor<?xf64>, tensor<?xf64>, tensor<i64> attributes {enzymexla.parallel}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %1 = stablehlo.compare LT, %iterArg, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %cst = stablehlo.constant dense<0.000000e+00> : tensor<32xf64>
// CHECK-NEXT:      %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<1xf64>
// CHECK-NEXT:      %1 = stablehlo.iota dim = 0 : tensor<32xi64>
// CHECK-NEXT:      %c_5 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:      %2 = stablehlo.add %1, %c_5 : tensor<32xi64>
// CHECK-NEXT:      %c_6 = stablehlo.constant dense<1> : tensor<32xi64>
// CHECK-NEXT:      %3 = stablehlo.multiply %2, %c_6 : tensor<32xi64>
// CHECK-NEXT:      %c_7 = stablehlo.constant dense<32> : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.multiply %iterArg, %c_7 : tensor<i64>
// CHECK-NEXT:      %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:      %6 = stablehlo.add %5, %3 : tensor<32xi64>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<32xi64>) -> tensor<32x1xi64>
// CHECK-NEXT:      %8 = "stablehlo.gather"(%iterArg_2, %7) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<?xf64>, tensor<32x1xi64>) -> tensor<32xf64>
// CHECK-NEXT:      %c_8 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_9 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_10 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_11 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_12 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_13 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.dynamic_update_slice %cst, %8, %c_13 : (tensor<32xf64>, tensor<32xf64>, tensor<i64>) -> tensor<32xf64>
// CHECK-NEXT:      %c_14 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:      %10 = stablehlo.compare EQ, %3, %c_14 : (tensor<32xi64>, tensor<32xi64>) -> tensor<32xi1>
// CHECK-NEXT:      %11 = stablehlo.dynamic_slice %iterArg_2, %iterArg, sizes = [1] : (tensor<?xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:      %12 = stablehlo.reshape %11 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:      %c_15 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_16 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_17 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_18 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_19 = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:      %c_20 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:      %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:      %c_21 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:      %14 = stablehlo.reduce(%10 init: %c_21) applies stablehlo.or across dimensions = [0] : (tensor<32xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:      %15 = stablehlo.reshape %13 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:      %16 = stablehlo.reshape %cst_4 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:      %17 = stablehlo.select %14, %15, %16 : tensor<i1>, tensor<f64>
// CHECK-NEXT:      %18 = stablehlo.broadcast_in_dim %17, dims = [] : (tensor<f64>) -> tensor<1xf64>
// CHECK-NEXT:      %19 = stablehlo.dynamic_update_slice %cst_4, %18, %c_20 : (tensor<1xf64>, tensor<1xf64>, tensor<i64>) -> tensor<1xf64>
// CHECK-NEXT:      %20 = stablehlo.reshape %19 : (tensor<1xf64>) -> tensor<f64>
// CHECK-NEXT:      %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f64>) -> tensor<32xf64>
// CHECK-NEXT:      %22 = arith.mulf %9, %21 : tensor<32xf64>
// CHECK-NEXT:      %c_22 = stablehlo.constant dense<32> : tensor<i64>
// CHECK-NEXT:      %23 = stablehlo.multiply %iterArg, %c_22 : tensor<i64>
// CHECK-NEXT:      %24 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<i64>) -> tensor<32xi64>
// CHECK-NEXT:      %25 = stablehlo.add %24, %3 : tensor<32xi64>
// CHECK-NEXT:      %26 = stablehlo.reshape %25 : (tensor<32xi64>) -> tensor<32x1xi64>
// CHECK-NEXT:      %27 = "stablehlo.scatter"(%iterArg_1, %26, %22) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// CHECK-NEXT:      ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// CHECK-NEXT:        stablehlo.return %arg4 : tensor<f64>
// CHECK-NEXT:      }) : (tensor<?xf64>, tensor<32x1xi64>, tensor<32xf64>) -> tensor<?xf64>
// CHECK-NEXT:      %28 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %28, %27, %iterArg_2, %iterArg_3 : tensor<i64>, tensor<?xf64>, tensor<?xf64>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#1, %0#2, %0#3 : tensor<?xf64>, tensor<?xf64>, tensor<i64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @bcast_raised(%arg0: tensor<?xf64>, %arg1: tensor<?xf64>, %arg2: tensor<i64>) -> (tensor<?xf64>, tensor<?xf64>, tensor<i64>) {
// OPT-NEXT:    %c = stablehlo.constant dense<{{\[\[}}0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31]]> : tensor<32x1xi64>
// OPT-NEXT:    %c_0 = stablehlo.constant dense<32> : tensor<i64>
// OPT-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// OPT-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i64>
// OPT-NEXT:    %0:4 = stablehlo.while(%iterArg = %c_1, %iterArg_3 = %arg0, %iterArg_4 = %arg1, %iterArg_5 = %arg2) : tensor<i64>, tensor<?xf64>, tensor<?xf64>, tensor<i64> attributes {enzymexla.parallel}
// OPT-NEXT:    cond {
// OPT-NEXT:      %1 = stablehlo.compare LT, %iterArg, %arg2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// OPT-NEXT:      stablehlo.return %1 : tensor<i1>
// OPT-NEXT:    } do {
// OPT-NEXT:      %1 = stablehlo.multiply %iterArg, %c_0 : tensor<i64>
// OPT-NEXT:      %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<i64>) -> tensor<32x1xi64>
// OPT-NEXT:      %3 = stablehlo.add %2, %c : tensor<32x1xi64>
// OPT-NEXT:      %4 = "stablehlo.gather"(%iterArg_4, %3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<?xf64>, tensor<32x1xi64>) -> tensor<32xf64>
// OPT-NEXT:      %5 = stablehlo.dynamic_slice %iterArg_4, %iterArg, sizes = [1] : (tensor<?xf64>, tensor<i64>) -> tensor<1xf64>
// OPT-NEXT:      %6 = stablehlo.broadcast_in_dim %5, dims = [0] : (tensor<1xf64>) -> tensor<32xf64>
// OPT-NEXT:      %7 = arith.mulf %4, %6 : tensor<32xf64>
// OPT-NEXT:      %8 = "stablehlo.scatter"(%iterArg_3, %3, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// OPT-NEXT:      ^bb0(%arg3: tensor<f64>, %arg4: tensor<f64>):
// OPT-NEXT:        stablehlo.return %arg4 : tensor<f64>
// OPT-NEXT:      }) : (tensor<?xf64>, tensor<32x1xi64>, tensor<32xf64>) -> tensor<?xf64>
// OPT-NEXT:      %9 = stablehlo.add %iterArg, %c_2 : tensor<i64>
// OPT-NEXT:      stablehlo.return %9, %8, %iterArg_4, %iterArg_5 : tensor<i64>, tensor<?xf64>, tensor<?xf64>, tensor<i64>
// OPT-NEXT:    }
// OPT-NEXT:    return %0#1, %0#2, %0#3 : tensor<?xf64>, tensor<?xf64>, tensor<i64>
// OPT-NEXT:  }

// -----
// The non-affine store path: a mask axis the scatter grid does not carry
// (the tid==0 guard) or-reduces instead of producing a rank-mismatched
// broadcast.
func.func @bcast_scatter(%out: memref<100xf64, 1>, %in: memref<100xf64, 1>) {
  affine.parallel (%e, %t) = (0, 0) to (4, 32) {
    %v = affine.load %in[%e * 25] : memref<100xf64, 1>
    affine.if affine_set<(d0) : (d0 == 0)>(%t) {
      %ei = arith.index_castui %e : index to i64
      %e2 = arith.muli %ei, %ei : i64
      %idx = arith.index_cast %e2 : i64 to index
      memref.store %v, %out[%idx] : memref<100xf64, 1>
    }
  }
  return
}

// CHECK:    func.func private @bcast_scatter_raised(%arg0: tensor<100xf64>, %arg1: tensor<100xf64>) -> (tensor<100xf64>, tensor<100xf64>) {
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<4xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<4xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<4xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<32xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_1 : tensor<32xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<32xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_2 : tensor<32xi64>
// CHECK-NEXT:    %6 = stablehlo.slice %arg1 [0:100:25] : (tensor<100xf64>) -> tensor<4xf64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<32xi64>
// CHECK-NEXT:    %7 = stablehlo.compare EQ, %5, %c_3 : (tensor<32xi64>, tensor<32xi64>) -> tensor<32xi1>
// CHECK-NEXT:    %8 = arith.muli %2, %2 : tensor<4xi64>
// CHECK-NEXT:    %9 = stablehlo.reshape %8 : (tensor<4xi64>) -> tensor<4x1xi64>
// CHECK-NEXT:    %10 = "stablehlo.gather"(%arg0, %9) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<100xf64>, tensor<4x1xi64>) -> tensor<4xf64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %11 = stablehlo.reduce(%7 init: %c_4) applies stablehlo.or across dimensions = [0] : (tensor<32xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<i1>) -> tensor<4xi1>
// CHECK-NEXT:    %13 = stablehlo.select %12, %6, %10 : tensor<4xi1>, tensor<4xf64>
// CHECK-NEXT:    %14 = "stablehlo.scatter"(%arg0, %9, %13) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<100xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<100xf64>
// CHECK-NEXT:    return %14, %arg1 : tensor<100xf64>, tensor<100xf64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @bcast_scatter_raised(%arg0: tensor<100xf64>, %arg1: tensor<100xf64>) -> (tensor<100xf64>, tensor<100xf64>) {
// OPT-NEXT:    %c = stablehlo.constant dense<{{\[\[}}0], [1], [4], [9]]> : tensor<4x1xi64>
// OPT-NEXT:    %0 = stablehlo.slice %arg1 [0:100:25] : (tensor<100xf64>) -> tensor<4xf64>
// OPT-NEXT:    %1 = "stablehlo.scatter"(%arg0, %c, %0) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// OPT-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// OPT-NEXT:      stablehlo.return %arg3 : tensor<f64>
// OPT-NEXT:    }) : (tensor<100xf64>, tensor<4x1xi64>, tensor<4xf64>) -> tensor<100xf64>
// OPT-NEXT:    return %1, %arg1 : tensor<100xf64>, tensor<100xf64>
// OPT-NEXT:  }

// -----
// The scatter path at scale, 4096 elements by 256 lanes: the lane mask the
// scatter grid does not carry is a pad too large to constant-fold.
func.func @bcast_scatter_large(%out: memref<8192xf64, 1>, %in: memref<4096xf64, 1>) {
  affine.parallel (%e, %t) = (0, 0) to (4096, 256) {
    %v = affine.load %in[%e] : memref<4096xf64, 1>
    affine.if affine_set<(d0) : (d0 == 0)>(%t) {
      %ei = arith.index_castui %e : index to i64
      %e2 = arith.muli %ei, %ei : i64
      %c4095 = arith.constant 4095 : i64
      %masked = arith.andi %e2, %c4095 : i64
      %idx = arith.index_cast %masked : i64 to index
      memref.store %v, %out[%idx] : memref<8192xf64, 1>
    }
  }
  return
}

// CHECK:    func.func private @bcast_scatter_large_raised(%arg0: tensor<8192xf64>, %arg1: tensor<4096xf64>) -> (tensor<8192xf64>, tensor<4096xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<4095> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4096xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<4096xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_0 : tensor<4096xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<4096xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_1 : tensor<4096xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<256xi64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<256xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_2 : tensor<256xi64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<256xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_3 : tensor<256xi64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<256xi64>
// CHECK-NEXT:    %6 = stablehlo.compare EQ, %5, %c_4 : (tensor<256xi64>, tensor<256xi64>) -> tensor<256xi1>
// CHECK-NEXT:    %7 = arith.muli %2, %2 : tensor<4096xi64>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<4096xi64>
// CHECK-NEXT:    %9 = arith.andi %7, %8 : tensor<4096xi64>
// CHECK-NEXT:    %10 = stablehlo.reshape %9 : (tensor<4096xi64>) -> tensor<4096x1xi64>
// CHECK-NEXT:    %11 = "stablehlo.gather"(%arg0, %10) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<8192xf64>, tensor<4096x1xi64>) -> tensor<4096xf64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %12 = stablehlo.reduce(%6 init: %c_5) applies stablehlo.or across dimensions = [0] : (tensor<256xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<i1>) -> tensor<4096xi1>
// CHECK-NEXT:    %14 = stablehlo.select %13, %arg1, %11 : tensor<4096xi1>, tensor<4096xf64>
// CHECK-NEXT:    %15 = "stablehlo.scatter"(%arg0, %10, %14) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// CHECK-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// CHECK-NEXT:      stablehlo.return %arg3 : tensor<f64>
// CHECK-NEXT:    }) : (tensor<8192xf64>, tensor<4096x1xi64>, tensor<4096xf64>) -> tensor<8192xf64>
// CHECK-NEXT:    return %15, %arg1 : tensor<8192xf64>, tensor<4096xf64>
// CHECK-NEXT:  }

// OPT:  module {
// OPT-NEXT:  func.func private @bcast_scatter_large_raised(%arg0: tensor<8192xf64>, %arg1: tensor<4096xf64>) -> (tensor<8192xf64>, tensor<4096xf64>) {
// OPT-NEXT:    %c = stablehlo.constant dense<4095> : tensor<4096xi64>
// OPT-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<4096xi64>
// OPT-NEXT:    %1 = arith.muli %0, %0 : tensor<4096xi64>
// OPT-NEXT:    %2 = arith.andi %1, %c : tensor<4096xi64>
// OPT-NEXT:    %3 = stablehlo.reshape %2 : (tensor<4096xi64>) -> tensor<4096x1xi64>
// OPT-NEXT:    %4 = "stablehlo.scatter"(%arg0, %3, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// OPT-NEXT:    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
// OPT-NEXT:      stablehlo.return %arg3 : tensor<f64>
// OPT-NEXT:    }) : (tensor<8192xf64>, tensor<4096x1xi64>, tensor<4096xf64>) -> tensor<8192xf64>
// OPT-NEXT:    return %4, %arg1 : tensor<8192xf64>, tensor<4096xf64>
// OPT-NEXT:  }
