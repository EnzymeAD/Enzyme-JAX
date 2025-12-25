// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

module {
  func.func private @use(%arg0: f32)
  func.func @affine2(%arg0: f32, %arg1: memref<?xf32>) {
    affine.parallel (%arg2) = (0) to (4) {
      %0 = memref.atomic_rmw addf %arg0, %arg1[%arg2] : (f32, memref<?xf32>) -> f32
      func.call @use(%0) : (f32) -> ()
    }
    return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func private @use(f32)

// CHECK-LABEL:   func.func @affine2(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: memref<?xf32>) {
// CHECK:           affine.parallel (%[[VAL_0:.*]]) = (0) to (4) {
// CHECK:             %[[AFFINE_ATOMIC_RMW_0:.*]] = enzyme.affine_atomic_rmw addf %[[ARG0]], %[[ARG1]], (#[[$ATTR_0]]) {{\[}}%[[VAL_0]]] : (f32, memref<?xf32>) -> f32
// CHECK:             func.call @use(%[[AFFINE_ATOMIC_RMW_0]]) : (f32) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

