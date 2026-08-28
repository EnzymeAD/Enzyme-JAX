// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

// An empty optional buffer arrives as a null base pointer: accesses through
// it sit on paths that can only fault, so loads read as zero, stores vanish,
// and the null never becomes a kernel argument. The null may also hide
// behind offset arithmetic inside a select.

// CHECK-LABEL: @nullgep_raised
// CHECK-NOT: llvm.mlir.zero

module {
  func.func private @nullbuf(%out: memref<16xf64, 1>, %a: memref<16xf64, 1>) {
    %null = llvm.mlir.zero : !llvm.ptr<1>
    %nv = "enzymexla.pointer2memref"(%null) : (!llvm.ptr<1>) -> memref<?xf64, 1>
    affine.parallel (%t) = (0) to (16) {
      %v = affine.load %a[%t] : memref<16xf64, 1>
      %z = affine.load %nv[%t] : memref<?xf64, 1>
      %s = arith.addf %v, %z : f64
      affine.store %s, %out[%t] : memref<16xf64, 1>
    }
    return
  }

  // CHECK-LABEL: @nullbuf_raised
  // CHECK-NOT: llvm.mlir.zero
  func.func private @nullgep(%out: memref<16xf64, 1>, %a: memref<64xf64, 1>, %nbuf: memref<1xi32, 1>) {
    %n = affine.load %nbuf[0] : memref<1xi32, 1>
    %cond = arith.cmpi sgt, %n, %n : i32
    %null = llvm.mlir.zero : !llvm.ptr<1>
    %p = "enzymexla.memref2pointer"(%a) : (memref<64xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%t) = (0) to (16) {
      %i = arith.index_castui %t : index to i64
      %ga = llvm.getelementptr %p[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %gn = llvm.getelementptr %null[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %sel = arith.select %cond, %gn, %ga : !llvm.ptr<1>
      %view = "enzymexla.pointer2memref"(%sel) : (!llvm.ptr<1>) -> memref<?xf64, 1>
      %v = affine.load %view[0] : memref<?xf64, 1>
      affine.store %v, %out[%t] : memref<16xf64, 1>
    }
    return
  }
}
