// RUN: enzymexlamlir-opt %s --affine-cfg --affine-parallelize=parallel-reductions=1 --affine-cfg | FileCheck %s

#map = affine_map<(d0) -> (d0 + 1)>

// CHECK-LABEL: func.func @kernel_trmm(
// CHECK-NOT: affine.if
// CHECK: affine.for %[[I:.*]] = 0 to 1000 {
// CHECK-NEXT: affine.parallel (%[[J:.*]]) = (0) to (1200) {
// CHECK-NEXT: %[[INIT:.*]] = affine.load %arg4[%[[I]], %[[J]]] : memref<1000x1200xf64>
// CHECK-NEXT: %[[RED:.*]] = affine.parallel (%[[K:.*]]) = (%[[I]] + 1) to (1000) reduce ("addf") -> (f64) {
// CHECK: %[[SUM:.*]] = arith.addf %[[INIT]], %[[RED]] : f64
// CHECK-NEXT: affine.store %[[SUM]], %arg4[%[[I]], %[[J]]] : memref<1000x1200xf64>
func.func @kernel_trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<1000x1000xf64>, %arg4: memref<1000x1200xf64>) {
  affine.for %arg5 = 0 to 1000 {
    affine.for %arg6 = 0 to 1200 {
      affine.for %arg7 = #map(%arg5) to 1000 {
        %2 = affine.load %arg3[%arg7, %arg5] : memref<1000x1000xf64>
        %3 = affine.load %arg4[%arg7, %arg6] : memref<1000x1200xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = affine.load %arg4[%arg5, %arg6] : memref<1000x1200xf64>
        %6 = arith.addf %5, %4 : f64
        affine.store %6, %arg4[%arg5, %arg6] : memref<1000x1200xf64>
      }
      %0 = affine.load %arg4[%arg5, %arg6] : memref<1000x1200xf64>
      %1 = arith.mulf %arg2, %0 : f64
      affine.store %1, %arg4[%arg5, %arg6] : memref<1000x1200xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @inbounds_without_trailing_access(
// CHECK-NOT: affine.if
// CHECK: affine.for %[[I2:.*]] = 0 to 1000 {
// CHECK-NEXT: affine.parallel (%[[J2:.*]]) = (0) to (1200) {
// CHECK-NEXT: %[[INIT2:.*]] = affine.load %arg1[%[[I2]], %[[J2]]] : memref<1000x1200xf64>
// CHECK-NEXT: %[[RED2:.*]] = affine.parallel (%[[K2:.*]]) = (%[[I2]] + 1) to (1000) reduce ("addf") -> (f64) {
// CHECK: %[[SUM2:.*]] = arith.addf %[[INIT2]], %[[RED2]] : f64
// CHECK-NEXT: affine.store %[[SUM2]], %arg1[%[[I2]], %[[J2]]] : memref<1000x1200xf64>
func.func @inbounds_without_trailing_access(%arg3: memref<1000x1000xf64>, %arg4: memref<1000x1200xf64>) {
  affine.for %arg5 = 0 to 1000 {
    affine.for %arg6 = 0 to 1200 {
      affine.for %arg7 = #map(%arg5) to 1000 {
        %2 = affine.load %arg3[%arg7, %arg5] : memref<1000x1000xf64>
        %3 = affine.load %arg4[%arg7, %arg6] : memref<1000x1200xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = affine.load %arg4[%arg5, %arg6] : memref<1000x1200xf64>
        %6 = arith.addf %5, %4 : f64
        affine.store %6, %arg4[%arg5, %arg6] : memref<1000x1200xf64>
      }
    }
  }
  return
}

// When %arg5 == 999, %arg5 + 1 == 1000 is out of bounds for dim 0 of memref<1000x1200xf64>,
// so the load and store must remain guarded by affine.if.
// CHECK-LABEL: func.func @oob_on_empty_trip_kept_guarded(
// CHECK: affine.for %[[I3:.*]] = 0 to 1000 {
// CHECK-NEXT: affine.parallel (%[[J3:.*]]) = (0) to (1200) {
// CHECK-NEXT: %[[G_INIT:.*]] = affine.if {{.*}}(%[[I3]]) -> f64 {
// CHECK: %[[RED3:.*]] = affine.parallel (%[[K3:.*]]) = (%[[I3]] + 1) to (1000) reduce ("addf") -> (f64) {
// CHECK: %[[SUM3:.*]] = arith.addf %[[G_INIT]], %[[RED3]] : f64
// CHECK-NEXT: affine.if {{.*}}(%[[I3]]) {
// CHECK-NEXT: affine.store %[[SUM3]], %arg1[%[[I3]] + 1, %[[J3]]] : memref<1000x1200xf64>
func.func @oob_on_empty_trip_kept_guarded(%arg3: memref<1000x1000xf64>, %arg4: memref<1000x1200xf64>) {
  affine.for %arg5 = 0 to 1000 {
    affine.for %arg6 = 0 to 1200 {
      affine.for %arg7 = #map(%arg5) to 1000 {
        %2 = affine.load %arg3[%arg7, %arg5] : memref<1000x1000xf64>
        %3 = affine.load %arg4[%arg7, %arg6] : memref<1000x1200xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = affine.load %arg4[%arg5 + 1, %arg6] : memref<1000x1200xf64>
        %6 = arith.addf %5, %4 : f64
        affine.store %6, %arg4[%arg5 + 1, %arg6] : memref<1000x1200xf64>
      }
    }
  }
  return
}
