// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// The du[3] scratch shape: clang zeroes a double as an i64 0 store and copies
// one as an i64 load, so the alloca is viewed as both f64 and i64 memrefs.
// The integer views retype to the alloca's element type with bitcasts at each
// access, which fold against the constant and the paired store, leaving one
// consistently typed buffer.
module {
  func.func @punned(%out: memref<?xf64>, %v: f64) {
    %c1 = arith.constant 1 : i32
    %z = arith.constant 0 : i64
    %du = llvm.alloca %c1 x !llvm.array<3 x f64> : (i32) -> !llvm.ptr
    %fview = "enzymexla.pointer2memref"(%du) : (!llvm.ptr) -> memref<?xf64>
    %iview = "enzymexla.pointer2memref"(%du) : (!llvm.ptr) -> memref<?xi64>
    affine.store %v, %fview[0] : memref<?xf64>
    affine.store %z, %iview[2] : memref<?xi64>
    %bits = affine.load %iview[0] : memref<?xi64>
    %oview = "enzymexla.pointer2memref"(%du) : (!llvm.ptr) -> memref<?xi64>
    affine.store %bits, %oview[1] : memref<?xi64>
    %r = affine.load %fview[1] : memref<?xf64>
    affine.store %r, %out[0] : memref<?xf64>
    %r2 = affine.load %fview[2] : memref<?xf64>
    affine.store %r2, %out[1] : memref<?xf64>
    return
  }
}

// CHECK-LABEL: func.func @punned(
// CHECK-SAME: %[[OUT:.+]]: memref<?xf64>, %[[V:.+]]: f64
// CHECK-NEXT: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT: %[[DU:.+]] = memref.alloca() : memref<3xf64>
// CHECK-NEXT: affine.store %[[V]], %[[DU]][0] : memref<3xf64>
// CHECK-NEXT: affine.store %[[ZERO]], %[[DU]][2] : memref<3xf64>
// CHECK-NEXT: %[[L0:.+]] = affine.load %[[DU]][0] : memref<3xf64>
// CHECK-NEXT: %[[B0:.+]] = arith.bitcast %[[L0]] : f64 to i64
// CHECK-NEXT: %[[B1:.+]] = arith.bitcast %[[B0]] : i64 to f64
// CHECK-NEXT: affine.store %[[B1]], %[[DU]][1] : memref<3xf64>
// CHECK-NEXT: %[[L1:.+]] = affine.load %[[DU]][1] : memref<3xf64>
// CHECK-NEXT: affine.store %[[L1]], %[[OUT]][0] : memref<?xf64>
// CHECK-NEXT: %[[L2:.+]] = affine.load %[[DU]][2] : memref<3xf64>
// CHECK-NEXT: affine.store %[[L2]], %[[OUT]][1] : memref<?xf64>
// CHECK-NEXT: return
// CHECK-NEXT: }

// The same shape through non-affine accesses at a dynamic position.
func.func @punned_memref(%out: memref<?xf64>, %v: f64, %i: index) {
  %c1 = arith.constant 1 : i32
  %z = arith.constant 0 : i64
  %du = llvm.alloca %c1 x !llvm.array<3 x f64> : (i32) -> !llvm.ptr
  %fview = "enzymexla.pointer2memref"(%du) : (!llvm.ptr) -> memref<?xf64>
  %iview = "enzymexla.pointer2memref"(%du) : (!llvm.ptr) -> memref<?xi64>
  affine.store %v, %fview[0] : memref<?xf64>
  memref.store %z, %iview[%i] : memref<?xi64>
  %b = memref.load %iview[%i] : memref<?xi64>
  %f = arith.bitcast %b : i64 to f64
  affine.store %f, %out[0] : memref<?xf64>
  return
}

// CHECK-LABEL: func.func @punned_memref(
// CHECK-SAME: %[[MOUT:.+]]: memref<?xf64>, %[[MV:.+]]: f64, %[[MI:.+]]: index
// CHECK-NEXT: %[[MZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT: %[[MDU:.+]] = memref.alloca() : memref<3xf64>
// CHECK-NEXT: affine.store %[[MV]], %[[MDU]][0] : memref<3xf64>
// CHECK-NEXT: memref.store %[[MZERO]], %[[MDU]][%[[MI]]] : memref<3xf64>
// CHECK-NEXT: %[[ML:.+]] = memref.load %[[MDU]][%[[MI]]] : memref<3xf64>
// CHECK-NEXT: %[[MB0:.+]] = arith.bitcast %[[ML]] : f64 to i64
// CHECK-NEXT: %[[MB1:.+]] = arith.bitcast %[[MB0]] : i64 to f64
// CHECK-NEXT: affine.store %[[MB1]], %[[MOUT]][0] : memref<?xf64>
// CHECK-NEXT: return
// CHECK-NEXT: }
