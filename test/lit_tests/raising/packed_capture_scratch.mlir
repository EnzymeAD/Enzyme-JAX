// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// A not-quite-inlined device lambda packs its captures into a stack struct
// and reads them back through typed views: pointer slots forward to the
// stored pointer, and a shape array read at a runtime index resolves as a
// select over the constant-offset stores.

// CHECK-LABEL: @packed_raised
// CHECK-NOT: llvm.alloca

module {
  func.func private @packed(%out: memref<16xf64, 1>, %in: memref<16xf64, 1>, %nbuf: memref<1xi32, 1>) {
    %c1 = arith.constant 1 : i32
    %n = affine.load %nbuf[0] : memref<1xi32, 1>
    %st = llvm.alloca %c1 x !llvm.struct<packed (ptr<1>, i32, i32)> : (i32) -> !llvm.ptr
    %pv = "enzymexla.pointer2memref"(%st) : (!llvm.ptr) -> memref<?x!llvm.ptr<1>>
    %iv = "enzymexla.pointer2memref"(%st) : (!llvm.ptr) -> memref<?xi32>
    %p = "enzymexla.memref2pointer"(%in) : (memref<16xf64, 1>) -> !llvm.ptr<1>
    affine.store %p, %pv[0] : memref<?x!llvm.ptr<1>>
    affine.store %n, %iv[2] : memref<?xi32>
    affine.store %c1, %iv[3] : memref<?xi32>
    affine.parallel (%t) = (0) to (16) {
      %pl = affine.load %pv[0] : memref<?x!llvm.ptr<1>>
      %view = "enzymexla.pointer2memref"(%pl) : (!llvm.ptr<1>) -> memref<?xf64, 1>
      %d = affine.load %iv[%t mod 2 + 2] : memref<?xi32>
      %di = arith.index_cast %d : i32 to index
      %vidx = arith.addi %t, %di : index
      %v = memref.load %view[%vidx] : memref<?xf64, 1>
      affine.store %v, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}

// -----

// A member chosen by side arrives as a select of byte offsets into the
// struct; the slot it names is constant per arm, so the read forwards to a
// select of the two stored values rather than a scan over every slot.

// CHECK-LABEL: @sided_raised
// CHECK-NOT: llvm.alloca
// CHECK: %[[side:.+]] = arith.cmpi ne
// CHECK-NOT: arith.cmpi
// CHECK: arith.select %[[side]], %{{.+}}, %{{.+}} : tensor<i1>, tensor<i32>
// CHECK-NOT: arith.cmpi

module {
  func.func private @sided(%out: memref<16xi32, 1>, %nbuf: memref<2xi32, 1>, %fbuf: memref<1xi32, 1>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i64
    %c8 = arith.constant 8 : i64
    %c12 = arith.constant 12 : i64
    %f = affine.load %fbuf[0] : memref<1xi32, 1>
    %n0 = affine.load %nbuf[0] : memref<2xi32, 1>
    %n1 = affine.load %nbuf[1] : memref<2xi32, 1>
    %st = llvm.alloca %c1 x !llvm.struct<packed (i32, i32, i32, i32)> : (i32) -> !llvm.ptr
    %iv = "enzymexla.pointer2memref"(%st) : (!llvm.ptr) -> memref<?xi32>
    affine.store %f, %iv[0] : memref<?xi32>
    affine.store %n0, %iv[2] : memref<?xi32>
    affine.store %n1, %iv[3] : memref<?xi32>
    affine.parallel (%t) = (0) to (16) {
      %fl = affine.load %iv[0] : memref<?xi32>
      %side = arith.cmpi ne, %fl, %c0 : i32
      %off = arith.select %side, %c12, %c8 : i64
      %slot = arith.divsi %off, %c4 : i64
      %si = arith.index_cast %slot : i64 to index
      %n = memref.load %iv[%si] : memref<?xi32>
      affine.store %n, %out[%t] : memref<16xi32, 1>
    }
    return
  }
}
