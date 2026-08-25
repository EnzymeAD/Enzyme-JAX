// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --split-input-file | FileCheck %s

// A small constant-length memcpy/memset of homogeneous-struct data (MFEM's
// DevicePair reductions writing work[i] = buffer[0]) expands into typed
// load/store pairs the affine access raising handles.
func.func @structcopy(%dst: !llvm.ptr, %idx: i64) {
  %buf = memref.alloca() : memref<256x!llvm.struct<"struct.mfem::DevicePair", (f64, f64)>>
  %src = "enzymexla.memref2pointer"(%buf) : (memref<256x!llvm.struct<"struct.mfem::DevicePair", (f64, f64)>>) -> !llvm.ptr
  %c0 = llvm.mlir.constant(0 : i8) : i8
  %c16 = llvm.mlir.constant(16 : i64) : i64
  "llvm.intr.memset"(%src, %c0, %c16) <{arg_attrs = [{llvm.align = 8 : i64}, {}, {}], isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %gep = llvm.getelementptr inbounds %dst[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16 x i8>
  "llvm.intr.memcpy"(%gep, %src, %c16) <{arg_attrs = [{llvm.align = 8 : i64}, {llvm.align = 8 : i64}, {}], isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  return
}

// CHECK-LABEL: func.func @structcopy(
// CHECK-SAME: %[[DST:.+]]: !llvm.ptr, %[[IDX:.+]]: i64
// CHECK-NEXT: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT: %[[BUF:.+]] = memref.alloca() : memref<256x!llvm.struct<"struct.mfem::DevicePair", (f64, f64)>>
// CHECK-NEXT: %[[SRC:.+]] = "enzymexla.memref2pointer"(%[[BUF]]) : (memref<256x!llvm.struct<"struct.mfem::DevicePair", (f64, f64)>>) -> !llvm.ptr
// CHECK-NEXT: llvm.store %[[ZERO]], %[[SRC]] {alignment = 8 : i64} : f64, !llvm.ptr
// CHECK-NEXT: %[[SF1:.+]] = llvm.getelementptr %[[SRC]][1] : (!llvm.ptr) -> !llvm.ptr, f64
// CHECK-NEXT: llvm.store %[[ZERO]], %[[SF1]] {alignment = 8 : i64} : f64, !llvm.ptr
// CHECK-NEXT: %[[DGEP:.+]] = llvm.getelementptr inbounds %[[DST]][%[[IDX]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<16 x i8>
// CHECK-NEXT: %[[V0:.+]] = llvm.load %[[SRC]] {alignment = 8 : i64} : !llvm.ptr -> f64
// CHECK-NEXT: llvm.store %[[V0]], %[[DGEP]] {alignment = 8 : i64} : f64, !llvm.ptr
// CHECK-NEXT: %[[SF1B:.+]] = llvm.getelementptr %[[SRC]][1] : (!llvm.ptr) -> !llvm.ptr, f64
// CHECK-NEXT: %[[DF1:.+]] = llvm.getelementptr %[[DGEP]][1] : (!llvm.ptr) -> !llvm.ptr, f64
// CHECK-NEXT: %[[V1:.+]] = llvm.load %[[SF1B]] {alignment = 8 : i64} : !llvm.ptr -> f64
// CHECK-NEXT: llvm.store %[[V1]], %[[DF1]] {alignment = 8 : i64} : f64, !llvm.ptr
// CHECK-NEXT: return

// -----

// The zero initializer of a small stack array (real_t du[3] = {0,0,0} in
// MFEM's quadrature interpolator fallbacks) is a memset whose root is the
// llvm.alloca itself; it expands the same way, letting the alloca convert.
func.func @arrayinit() -> f64 {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c0_i8 = llvm.mlir.constant(0 : i8) : i8
  %c24 = llvm.mlir.constant(24 : i64) : i64
  %du = llvm.alloca %c1 x !llvm.array<3 x f64> {alignment = 16 : i64} : (i32) -> !llvm.ptr
  "llvm.intr.memset"(%du, %c0_i8, %c24) <{arg_attrs = [{llvm.align = 16 : i64}, {}, {}], isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  %v = llvm.load %du : !llvm.ptr -> f64
  return %v : f64
}

// CHECK-LABEL: func.func @arrayinit(
// CHECK-NOT: llvm.intr.memset
