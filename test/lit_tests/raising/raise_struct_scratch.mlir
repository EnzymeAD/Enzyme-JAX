// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// Struct-element scratch splits into one primitive scratch per field: the
// i32 views address the fields at even/odd offsets, and the whole-pair i64
// load reads both fields of pair zero at once.
func.func @intpair(%out: memref<256xi32, 1>, %in: memref<256xi32, 1>) {
  %scr = memref.alloca() {alignment = 4 : i64} : memref<256x!llvm.struct<"struct.mfem::DevicePair", (i32, i32)>>
  %ptr = "enzymexla.memref2pointer"(%scr) : (memref<256x!llvm.struct<"struct.mfem::DevicePair", (i32, i32)>>) -> !llvm.ptr<3>
  %c2 = arith.constant 2 : index
  affine.parallel (%t) = (0) to (256) {
    %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr<3>) -> memref<?xi32, 3>
    %v = affine.load %in[%t] : memref<256xi32, 1>
    affine.store %v, %view[%t * 2] : memref<?xi32, 3>
    %ti = arith.index_castui %t : index to i32
    affine.store %ti, %view[%t * 2 + 1] : memref<?xi32, 3>
    "enzymexla.barrier"(%t) : (index) -> ()
    %wide = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr<3>) -> memref<?xi64, 3>
    %pair0 = affine.load %wide[0] : memref<?xi64, 3>
    %lo = arith.trunci %pair0 : i64 to i32
    %t2 = arith.muli %t, %c2 : index
    %e = memref.load %view[%t2] : memref<?xi32, 3>
    %r = arith.addi %lo, %e : i32
    affine.store %r, %out[%t] : memref<256xi32, 1>
  }
  return
}

// CHECK-LABEL: func.func private @intpair_raised(
// CHECK-NOT: llvm.struct
// CHECK: stablehlo.slice

// -----

// Mixed-type pairs keep each field in its own scratch, and the whole-pair
// memcpy between struct-strided geps becomes per-field moves.
func.func @mixedpair(%out: memref<256xf64, 1>, %in: memref<256xf64, 1>) {
  %scr = memref.alloca() {alignment = 8 : i64} : memref<256x!llvm.struct<"struct.mfem::DevicePair.0", (f64, i32)>>
  %ptr = "enzymexla.memref2pointer"(%scr) : (memref<256x!llvm.struct<"struct.mfem::DevicePair.0", (f64, i32)>>) -> !llvm.ptr<3>
  %gp = llvm.addrspacecast %ptr : !llvm.ptr<3> to !llvm.ptr
  %c16_i64 = arith.constant 16 : i64
  %c1_i64 = arith.constant 1 : i64
  %c255_i64 = arith.constant 255 : i64
  affine.parallel (%t) = (0) to (256) {
    %fview = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr<3>) -> memref<?xf64, 3>
    %iview = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr<3>) -> memref<?xi32, 3>
    %v = affine.load %in[%t] : memref<256xf64, 1>
    affine.store %v, %fview[%t * 2] : memref<?xf64, 3>
    %ti = arith.index_castui %t : index to i32
    affine.store %ti, %iview[%t * 4 + 2] : memref<?xi32, 3>
    "enzymexla.barrier"(%t) : (index) -> ()
    %tl = arith.index_castui %t : index to i64
    %tn = arith.addi %tl, %c1_i64 : i64
    %ts = arith.andi %tn, %c255_i64 : i64
    %dst = llvm.getelementptr inbounds %gp[%tl] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16 x i8>
    %src = llvm.getelementptr inbounds %gp[%ts] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16 x i8>
    "llvm.intr.memcpy"(%dst, %src, %c16_i64) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    "enzymexla.barrier"(%t) : (index) -> ()
    %r = affine.load %fview[%t * 2] : memref<?xf64, 3>
    affine.store %r, %out[%t] : memref<256xf64, 1>
  }
  return
}

// CHECK-LABEL: func.func private @mixedpair_raised(
// CHECK-NOT: llvm.struct
// CHECK: stablehlo.gather
// CHECK: stablehlo.scatter
