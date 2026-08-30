// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

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
