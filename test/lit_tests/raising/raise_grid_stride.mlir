// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo=err_if_not_fully_raised=false > %t
// RUN: FileCheck %s --check-prefix=CONST < %t
// RUN: FileCheck %s --check-prefix=SYM < %t
// RUN: FileCheck %s --check-prefix=STEP < %t
// RUN: FileCheck %s --check-prefix=FOREACH < %t
// RUN: FileCheck %s --check-prefix=NESTED < %t
// RUN: FileCheck %s --check-prefix=CARRIED < %t
// RUN: FileCheck %s --check-prefix=LANE < %t
// RUN: FileCheck %s --check-prefix=GUARD < %t
// RUN: FileCheck %s --check-prefix=SCATTER < %t
// RUN: FileCheck %s --check-prefix=FOREACH2D < %t

// A grid-stride loop (thread t handles t, t + s, t + 2s, ...) inside a
// parallel of extent s covers exactly one iteration per lane index; it is
// coalesced into a single parallel over the lane index behind t == 0, which
// raises as whole-tensor ops instead of a stablehlo.while over the strides.

#ub = affine_map<(d0) -> ((6 - d0 + 3) floordiv 4)>
func.func private @const_stride(%arg0: memref<6xf64, 1>, %arg1: memref<6xf64, 1>) {
  affine.parallel (%t) = (0) to (4) {
    affine.for %k = 0 to #ub(%t) {
      %0 = affine.load %arg0[%t + %k * 4] : memref<6xf64, 1>
      %1 = arith.addf %0, %0 : f64
      affine.store %1, %arg1[%t + %k * 4] : memref<6xf64, 1>
    }
  }
  return
}

// CONST-LABEL: func.func private @const_stride_raised(
// CONST-NOT: stablehlo.while
// CONST: arith.addf %{{.*}}, %{{.*}} : tensor<6xf64>
// CONST-NOT: stablehlo.while
// CONST: stablehlo.dynamic_update_slice
// CONST-NOT: stablehlo.while

#ubs = affine_map<(d0)[s0] -> ((64 - d0 + s0 - 1) floordiv s0)>
func.func private @sym_stride(%arg0: memref<64xf64, 1>, %arg1: memref<64xf64, 1>, %sbuf: memref<i32, 1>) {
  %c8_i32 = arith.constant 8 : i32
  %n = affine.load %sbuf[] : memref<i32, 1>
  %b = arith.minsi %n, %c8_i32 : i32
  %s = arith.index_cast %b : i32 to index
  affine.parallel (%t) = (0) to (symbol(%s)) {
    affine.for %k = 0 to #ubs(%t)[%s] {
      %0 = affine.load %arg0[%t + %k * symbol(%s)] : memref<64xf64, 1>
      %1 = arith.addf %0, %0 : f64
      affine.store %1, %arg1[%t + %k * symbol(%s)] : memref<64xf64, 1>
    }
  }
  return
}

// SYM-LABEL: func.func private @sym_stride_raised(
// SYM-NOT: stablehlo.while
// SYM: arith.addf %{{.*}}, %{{.*}} : tensor<64xf64>
// SYM-NOT: stablehlo.while
// SYM: stablehlo.dynamic_update_slice
// SYM-NOT: stablehlo.while

// The unnormalized form `for j = t to N step s` with a shifted start.
func.func private @step_form(%arg0: memref<8xf64, 1>, %arg1: memref<8xf64, 1>) {
  affine.parallel (%t) = (0) to (4) {
    affine.for %j = affine_map<(d0) -> (d0 + 2)>(%t) to 8 step 4 {
      %0 = affine.load %arg0[%j] : memref<8xf64, 1>
      %1 = arith.mulf %0, %0 : f64
      affine.store %1, %arg1[%j - 2] : memref<8xf64, 1>
    }
  }
  return
}

// STEP-LABEL: func.func private @step_form_raised(
// STEP-NOT: stablehlo.while
// STEP: stablehlo.slice %{{.*}} [2:8] : (tensor<8xf64>) -> tensor<6xf64>
// STEP: arith.mulf %{{.*}}, %{{.*}} : tensor<6xf64>
// STEP-NOT: stablehlo.while
// STEP: stablehlo.dynamic_update_slice
// STEP-NOT: stablehlo.while

// MFEM_FOREACH_THREAD: a 2-D thread block where the y-axis strides over rows
// into shared scratch, guarded on the other axis being zero, followed by a
// barrier and a per-thread read of the scratch.
#set = affine_set<(d0, d1) : (d0 == 0, -d1 + 5 >= 0)>
#uby = affine_map<(d0) -> ((6 - d0 + 3) floordiv 4)>
func.func private @foreach_thread(%arg0: memref<6xf64, 1>, %arg1: memref<6xf64, 1>) {
  %c1 = arith.constant 1 : index
  affine.parallel (%tz, %ty) = (0, 0) to (2, 4) {
    %scr = memref.alloca() : memref<6xf64>
    affine.if #set(%tz, %ty) {
      affine.for %k = 0 to #uby(%ty) {
        %0 = affine.load %arg0[%ty + %k * 4] : memref<6xf64, 1>
        %1 = arith.addf %0, %0 : f64
        affine.store %1, %scr[%ty + %k * 4] : memref<6xf64>
      }
    }
    "enzymexla.barrier"(%ty, %tz, %c1) : (index, index, index) -> ()
    affine.if affine_set<(d0) : (d0 == 0)>(%tz) {
      %2 = affine.load %scr[%ty] : memref<6xf64>
      %3 = affine.load %scr[%ty + 1] : memref<6xf64>
      %4 = arith.addf %2, %3 : f64
      affine.store %4, %arg1[%ty] : memref<6xf64, 1>
    }
  }
  return
}

// FOREACH-LABEL: func.func private @foreach_thread_raised(
// FOREACH-NOT: stablehlo.while
// FOREACH: arith.addf %{{.*}}, %{{.*}} : tensor<6xf64>
// FOREACH-NOT: stablehlo.while
// FOREACH: stablehlo.dynamic_update_slice
// FOREACH-NOT: stablehlo.while

// Both axes stride: the loops coalesce inner-first into one 2-D parallel.
#ub2 = affine_map<(d0) -> ((6 - d0 + 3) floordiv 4)>
func.func private @nested(%arg0: memref<6x6xf64, 1>, %arg1: memref<6x6xf64, 1>) {
  affine.parallel (%tx, %ty) = (0, 0) to (4, 4) {
    affine.for %ky = 0 to #ub2(%ty) {
      affine.for %kx = 0 to #ub2(%tx) {
        %0 = affine.load %arg0[%ty + %ky * 4, %tx + %kx * 4] : memref<6x6xf64, 1>
        %1 = arith.addf %0, %0 : f64
        affine.store %1, %arg1[%ty + %ky * 4, %tx + %kx * 4] : memref<6x6xf64, 1>
      }
    }
  }
  return
}

// NESTED-LABEL: func.func private @nested_raised(
// NESTED-NOT: stablehlo.while
// NESTED: arith.addf %{{.*}}, %{{.*}} : tensor<6x6xf64>
// NESTED-NOT: stablehlo.while
// NESTED: stablehlo.dynamic_update_slice
// NESTED-NOT: stablehlo.while

// A loop-carried dependence (every stride accumulates into the lane's own
// slot) is not parallel over the lane index and stays a loop.
func.func private @carried(%arg0: memref<8xf64, 1>, %arg1: memref<4xf64, 1>) {
  affine.parallel (%t) = (0) to (4) {
    affine.for %j = affine_map<(d0) -> (d0)>(%t) to 8 step 4 {
      %0 = affine.load %arg0[%j] : memref<8xf64, 1>
      %1 = affine.load %arg1[%t] : memref<4xf64, 1>
      %2 = arith.addf %0, %1 : f64
      affine.store %2, %arg1[%t] : memref<4xf64, 1>
    }
  }
  return
}

// CARRIED-LABEL: func.func private @carried(
// CARRIED: affine.parallel
// CARRIED-NEXT: affine.for

// A per-lane value loaded outside the loop cannot be expressed in terms of
// the lane index, so the loop stays.
func.func private @lane_value(%arg0: memref<8xf64, 1>, %arg1: memref<8xf64, 1>, %arg2: memref<4xf64, 1>) {
  affine.parallel (%t) = (0) to (4) {
    %v = affine.load %arg2[%t] : memref<4xf64, 1>
    affine.for %j = affine_map<(d0) -> (d0)>(%t) to 8 step 4 {
      %0 = affine.load %arg0[%j] : memref<8xf64, 1>
      %1 = arith.addf %0, %v : f64
      affine.store %1, %arg1[%j] : memref<8xf64, 1>
    }
  }
  return
}

// LANE-LABEL: func.func private @lane_value(
// LANE: affine.parallel
// LANE-NEXT: affine.load
// LANE-NEXT: affine.for

// The coalesced form's store sits behind a guard on an axis it never indexes
// by; the write lands wherever any lane is admitted.
func.func private @lane_guard(%arg0: memref<8xf64, 1>, %arg1: memref<8xf64, 1>) {
  affine.parallel (%t) = (0) to (4) {
    affine.if affine_set<(d0) : (d0 == 0)>(%t) {
      affine.parallel (%i) = (0) to (8) {
        %0 = affine.load %arg0[%i] : memref<8xf64, 1>
        affine.store %0, %arg1[%i] : memref<8xf64, 1>
      }
    }
  }
  return
}

// GUARD-LABEL: func.func private @lane_guard_raised(
// GUARD: %[[T:.+]] = stablehlo.iota dim = 0 : tensor<4xi64>
// GUARD: %[[M:.+]] = stablehlo.compare EQ, %{{.*}}, %{{.*}} : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
// GUARD: %[[F:.+]] = stablehlo.constant dense<false> : tensor<i1>
// GUARD: %[[ANY:.+]] = stablehlo.reduce(%[[M]] init: %[[F]]) applies stablehlo.or across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
// GUARD: %[[MB:.+]] = stablehlo.broadcast_in_dim %[[ANY]], dims = [] : (tensor<i1>) -> tensor<8xi1>
// GUARD: stablehlo.select %[[MB]], %{{.*}}, %{{.*}} : tensor<8xi1>, tensor<8xf64>

// The same any-lane guard over a store that raises as a scatter (linearized
// index over two axes).
func.func private @lane_guard_scatter(%arg0: memref<15xf64, 1>, %arg1: memref<15xf64, 1>) {
  affine.parallel (%t) = (0) to (4) {
    affine.if affine_set<(d0) : (d0 == 0)>(%t) {
      affine.parallel (%i, %j) = (0, 0) to (5, 3) {
        %0 = affine.load %arg0[%j * 5 + %i] : memref<15xf64, 1>
        affine.store %0, %arg1[%i * 3 + %j] : memref<15xf64, 1>
      }
    }
  }
  return
}

// SCATTER-LABEL: func.func private @lane_guard_scatter_raised(
// SCATTER: %[[M:.+]] = stablehlo.compare EQ, %{{.*}}, %{{.*}} : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi1>
// SCATTER: %[[ANY:.+]] = stablehlo.reduce(%[[M]] init: %{{.*}}) applies stablehlo.or across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
// SCATTER: %[[MB:.+]] = stablehlo.broadcast_in_dim %[[ANY]], dims = [] : (tensor<i1>) -> tensor<5x3xi1>
// SCATTER: stablehlo.select %[[MB]], %{{.*}}, %{{.*}} : tensor<5x3xi1>, tensor<5x3xf64>
// SCATTER: "stablehlo.scatter"

// The MFEM_FOREACH_THREAD(dy,y,D1D-1) MFEM_FOREACH_THREAD(qx,x,Q1D) transpose
// load of a basis matrix into shared memory: both axes stride, the outer one
// behind a guard on the z-axis, and the scratch is indexed linearly.
#set_y = affine_set<(d0, d1) : (d0 == 0, -d1 + 2 >= 0)>
#set_x = affine_set<(d0) : (-d0 + 4 >= 0)>
#uby2 = affine_map<(d0) -> ((3 - d0 + 4) floordiv 5)>
#ubx2 = affine_map<(d0) -> ((5 - d0 + 4) floordiv 5)>
func.func private @foreach_thread_2d(%B: memref<20xf64, 1>, %out: memref<15xf64, 1>) {
  affine.parallel (%tx, %ty, %tz) = (0, 0, 0) to (5, 5, 2) {
    %sB = memref.alloca() : memref<30xf64>
    affine.if #set_y(%tz, %ty) {
      affine.for %ky = 0 to #uby2(%ty) {
        affine.if #set_x(%tx) {
          affine.for %kx = 0 to #ubx2(%tx) {
            %b = affine.load %B[(%ty + %ky * 5) * 5 + %tx + %kx * 5] : memref<20xf64, 1>
            affine.store %b, %sB[(%tx + %kx * 5) * 3 + %ty + %ky * 5] : memref<30xf64>
          }
        }
      }
    }
    "enzymexla.barrier"(%tx, %ty, %tz) : (index, index, index) -> ()
    affine.if #set_y(%tz, %ty) {
      %v = affine.load %sB[%tx * 3 + %ty] : memref<30xf64>
      affine.store %v, %out[%tx * 3 + %ty] : memref<15xf64, 1>
    }
  }
  return
}

// FOREACH2D-LABEL: func.func private @foreach_thread_2d_raised(
// FOREACH2D-NOT: stablehlo.while
// FOREACH2D: stablehlo.gather{{.*}} -> tensor<3x5xf64>
// FOREACH2D-NOT: stablehlo.while
// FOREACH2D: stablehlo.scatter
// FOREACH2D-NOT: stablehlo.while
