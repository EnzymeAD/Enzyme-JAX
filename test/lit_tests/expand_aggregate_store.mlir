// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// A lambda capture copied whole into a stack slot after an insertvalue
// update: the aggregate store expands into per-field stores, the
// extractvalues fold against the insertvalue chain, the piece stores
// forward to their loads, and the dead slot disappears.
module {
  func.func @capture(%cap: !llvm.struct<(f64, i32, f64)>, %n: i32, %out: memref<?xf64>) {
    %c1 = arith.constant 1 : i32
    %upd = llvm.insertvalue %n, %cap[1] : !llvm.struct<(f64, i32, f64)>
    %slot = llvm.alloca %c1 x !llvm.struct<(f64, i32, f64)> : (i32) -> !llvm.ptr
    %view = "enzymexla.pointer2memref"(%slot) : (!llvm.ptr) -> memref<?x!llvm.struct<(f64, i32, f64)>>
    affine.store %upd, %view[0] : memref<?x!llvm.struct<(f64, i32, f64)>>
    %fview = "enzymexla.pointer2memref"(%slot) : (!llvm.ptr) -> memref<?xf64>
    %f0 = affine.load %fview[0] : memref<?xf64>
    %f2 = affine.load %fview[2] : memref<?xf64>
    %sum = arith.addf %f0, %f2 : f64
    affine.store %sum, %out[0] : memref<?xf64>
    return
  }
}

// CHECK-LABEL: func.func @capture(
// CHECK-SAME: %[[CAP:.+]]: !llvm.struct<(f64, i32, f64)>, %[[N:.+]]: i32, %[[OUT:.+]]: memref<?xf64>
// CHECK-NEXT: %[[F0:.+]] = llvm.extractvalue %[[CAP]][0] : !llvm.struct<(f64, i32, f64)>
// CHECK-NEXT: %[[F2:.+]] = llvm.extractvalue %[[CAP]][2] : !llvm.struct<(f64, i32, f64)>
// CHECK-NEXT: %[[SUM:.+]] = arith.addf %[[F0]], %[[F2]] : f64
// CHECK-NEXT: affine.store %[[SUM]], %[[OUT]][0] : memref<?xf64>
// CHECK-NEXT: return
// CHECK-NEXT: }
