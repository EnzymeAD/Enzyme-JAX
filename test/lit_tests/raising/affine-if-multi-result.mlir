// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access,canonicalize,affine-cfg)" | FileCheck %s

// When a value used in an affine map is a non-first result of a pure
// multi-result affine.if, symbol legalization hoists the if and must hand
// back the result the value came from. It handed back result zero for
// every result, so the second access below read with the first access's
// offset: in MFEM's PAHdivMassSetup3D (vector-coefficient branch, the
// (i,j,k)=(2,2,2) term) that meant C(1) was read where C(2) was needed and
// every 3D Hdiv PA operator with a vector coefficient was wrong.

#set = affine_set<()[s0] : (s0 - 3 == 0)>
module {
  func.func @f(%cd: index, %m: !llvm.ptr, %out: memref<?xf64>) {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    affine.parallel (%q) = (0) to (8) {
      %of:2 = affine.if #set()[%cd] -> (i32, i32) {
        affine.yield %c1_i32, %c2_i32 : i32, i32
      } else {
        affine.yield %c0_i32, %c0_i32 : i32, i32
      }
      %qi = arith.index_cast %q : index to i32
      %e = arith.muli %qi, %c8_i32 : i32
      %e64 = arith.extsi %e : i32 to i64
      %base = llvm.getelementptr inbounds %m[%e64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
      affine.for %r = 0 to 4 {
        %ri = arith.index_cast %r : index to i32
        %i1 = arith.addi %ri, %of#0 : i32
        %i1e = arith.extsi %i1 : i32 to i64
        %p1 = llvm.getelementptr inbounds %base[%i1e] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
        %m1 = "enzymexla.pointer2memref"(%p1) : (!llvm.ptr) -> memref<?xf64>
        %c0 = arith.constant 0 : index
        %v1 = memref.load %m1[%c0] : memref<?xf64>
        %i2 = arith.addi %ri, %of#1 : i32
        %i2e = arith.extsi %i2 : i32 to i64
        %p2 = llvm.getelementptr inbounds %base[%i2e] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
        %m2 = "enzymexla.pointer2memref"(%p2) : (!llvm.ptr) -> memref<?xf64>
        %v2 = memref.load %m2[%c0] : memref<?xf64>
        %s = arith.addf %v1, %v2 : f64
        affine.store %s, %out[%r + %q * 4] : memref<?xf64>
      }
    }
    return
  }
}

// CHECK: %[[IF:.+]]:2 = affine.if
// CHECK: %[[S0:.+]] = arith.index_cast %[[IF]]#0 : i32 to index
// CHECK: %[[S1:.+]] = arith.index_cast %[[IF]]#1 : i32 to index
// CHECK: affine.load %{{.+}}[%{{.+}} + symbol(%[[S0]]) + %{{.+}} * 8]
// CHECK: affine.load %{{.+}}[%{{.+}} + symbol(%[[S1]]) + %{{.+}} * 8]
