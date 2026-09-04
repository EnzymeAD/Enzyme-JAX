// RUN: enzymexlamlir-opt --affine-cfg --split-input-file %s | FileCheck %s

// A bitwise-or is under a power of two exactly when every operand is; the
// thread id the axis bounds put under 4 drops out, leaving the other operand's
// own comparison.
func.func @drop(%b: i32, %out: memref<?xi1>) {
  affine.parallel (%t) = (0) to (4) {
    %a = arith.index_castui %t : index to i32
    %c4 = arith.constant 4 : i32
    %ab = arith.ori %a, %b : i32
    %r = arith.cmpi ult, %ab, %c4 : i32
    affine.store %r, %out[%t] : memref<?xi1>
  }
  return
}

// CHECK-LABEL: func.func @drop(
// CHECK-SAME:    %[[B:.+]]: i32
// CHECK-NOT:     arith.ori
// CHECK:         %[[R:.+]] = arith.cmpi ult, %[[B]], %{{.+}} : i32
// CHECK-NEXT:    affine.store %[[R]],

// -----

// An operand the bounds put at or above the power of two settles `ult` false.
func.func @never_under(%b: i32, %out: memref<?xi1>) {
  affine.parallel (%t) = (4) to (8) {
    %a = arith.index_castui %t : index to i32
    %c4 = arith.constant 4 : i32
    %ab = arith.ori %a, %b : i32
    %r = arith.cmpi ult, %ab, %c4 : i32
    affine.store %r, %out[%t] : memref<?xi1>
  }
  return
}

// CHECK-LABEL: func.func @never_under(
// CHECK-NOT:     arith.cmpi
// CHECK:         %[[F:.+]] = arith.constant false
// CHECK:         affine.store %[[F]],

// -----

// And reaches a power of two exactly when some operand does: an operand at or
// above it settles `uge` true, one under it drops out.
func.func @atleast(%b: i32, %out: memref<?xi1>, %out2: memref<?xi1>) {
  affine.parallel (%t) = (4) to (8) {
    %a = arith.index_castui %t : index to i32
    %c4 = arith.constant 4 : i32
    %ab = arith.ori %a, %b : i32
    %r = arith.cmpi uge, %ab, %c4 : i32
    affine.store %r, %out[%t] : memref<?xi1>
  }
  affine.parallel (%t) = (0) to (4) {
    %a = arith.index_castui %t : index to i32
    %c4 = arith.constant 4 : i32
    %ab = arith.ori %a, %b : i32
    %r = arith.cmpi uge, %ab, %c4 : i32
    affine.store %r, %out2[%t] : memref<?xi1>
  }
  return
}

// CHECK-LABEL: func.func @atleast(
// CHECK-SAME:    %[[B:.+]]: i32
// CHECK-NOT:     arith.ori
// CHECK:         %[[T:.+]] = arith.constant true
// CHECK:         affine.store %[[T]],
// CHECK:         %[[R:.+]] = arith.cmpi uge, %[[B]], %{{.+}} : i32
// CHECK-NEXT:    affine.store %[[R]],

// -----

// Every operand of an or chain decided drops it entirely.
func.func @all(%out: memref<?xi1>) {
  affine.parallel (%t1, %t2, %t3) = (0, 0, 0) to (4, 2, 1) {
    %a = arith.index_castui %t1 : index to i32
    %b = arith.index_castui %t2 : index to i32
    %c = arith.index_castui %t3 : index to i32
    %c4 = arith.constant 4 : i32
    %ab = arith.ori %a, %b : i32
    %abc = arith.ori %ab, %c : i32
    %r = arith.cmpi ult, %abc, %c4 : i32
    affine.store %r, %out[%t1 * 2 + %t2 + %t3] : memref<?xi1>
  }
  return
}

// CHECK-LABEL: func.func @all(
// CHECK-NOT:     arith.cmpi
// CHECK:         %[[T:.+]] = arith.constant true
// CHECK:         affine.store %[[T]],

// -----

// Nothing folds when no operand is decided, when the bound is not a power of
// two, or when the comparison is signed.
func.func @keep(%a: i32, %b: i32, %out: memref<?xi1>) {
  affine.parallel (%t) = (0) to (4) {
    %ti = arith.index_castui %t : index to i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %ab = arith.ori %a, %b : i32
    %r = arith.cmpi ult, %ab, %c4 : i32
    %tb = arith.ori %ti, %b : i32
    %s = arith.cmpi ult, %tb, %c3 : i32
    %u = arith.cmpi slt, %tb, %c4 : i32
    affine.store %r, %out[%t] : memref<?xi1>
    affine.store %s, %out[%t] : memref<?xi1>
    affine.store %u, %out[%t] : memref<?xi1>
  }
  return
}

// CHECK-LABEL: func.func @keep(
// CHECK:         %[[AB:.+]] = arith.ori %arg0, %arg1 : i32
// CHECK:         arith.cmpi ult, %[[AB]], %{{.+}} : i32
// CHECK:         %[[TB:.+]] = arith.ori %{{.+}}, %arg1 : i32
// CHECK:         arith.cmpi ult, %[[TB]], %{{.+}} : i32
// CHECK:         arith.cmpi slt, %[[TB]], %{{.+}} : i32
