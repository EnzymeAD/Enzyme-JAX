// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(canonicalize-parallel)" --split-input-file | FileCheck %s

// A gep chosen by a branch runs once, over a branch of what the arms
// disagree on: one operand (the base), or the leading constant index a gep
// keeps in an attribute rather than in its operands. Chains then collapse
// to a single branch over a scalar offset.

#set = affine_set<()[s0] : (s0 >= 1)>
module {
  // arms yield geps that differ only in the base
  func.func private @diff_base(%b1: !llvm.ptr<3>, %b2: !llvm.ptr<3>, %i: i64, %s: index) -> !llvm.ptr<3> {
    %r = affine.if #set()[%s] -> !llvm.ptr<3> {
      %g = llvm.getelementptr inbounds|nuw %b2[%i] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
      affine.yield %g : !llvm.ptr<3>
    } else {
      %g = llvm.getelementptr inbounds|nuw %b1[%i] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
      affine.yield %g : !llvm.ptr<3>
    }
    return %r : !llvm.ptr<3>
  }
  // arms yield geps off one base that differ only in a constant index
  func.func private @diff_const(%b: !llvm.ptr<3>, %s: index) -> !llvm.ptr<3> {
    %r = affine.if #set()[%s] -> !llvm.ptr<3> {
      %g = llvm.getelementptr inbounds|nuw %b[576] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
      affine.yield %g : !llvm.ptr<3>
    } else {
      %g = llvm.getelementptr inbounds|nuw %b[288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
      affine.yield %g : !llvm.ptr<3>
    }
    return %r : !llvm.ptr<3>
  }
  // the chained case: gep-of-gep, both levels multiplexed
  func.func private @chained(%b: !llvm.ptr<3>, %i: i64, %s: index) -> memref<?xf64> {
    %p1 = llvm.getelementptr inbounds|nuw %b[288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %p2 = llvm.getelementptr inbounds|nuw %b[576] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %r = affine.if #set()[%s] -> memref<?xf64> {
      %g = llvm.getelementptr inbounds|nuw %p2[%i] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
      %v = "enzymexla.pointer2memref"(%g) : (!llvm.ptr<3>) -> memref<?xf64>
      affine.yield %v : memref<?xf64>
    } else {
      %g = llvm.getelementptr inbounds|nuw %p1[%i] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
      %v = "enzymexla.pointer2memref"(%g) : (!llvm.ptr<3>) -> memref<?xf64>
      affine.yield %v : memref<?xf64>
    }
    return %r : memref<?xf64>
  }
}

// CHECK: func.func private @diff_base(%arg0: !llvm.ptr<3>, %arg1: !llvm.ptr<3>, %arg2: i64, %arg3: index) -> !llvm.ptr<3> {
// CHECK-NEXT: %0 = affine.if #set()[%arg3] -> !llvm.ptr<3> {
// CHECK-NEXT: affine.yield %arg1 : !llvm.ptr<3>
// CHECK-NEXT: } else {
// CHECK-NEXT: affine.yield %arg0 : !llvm.ptr<3>
// CHECK-NEXT: }
// CHECK-NEXT: %1 = llvm.getelementptr inbounds|nuw %0[%arg2] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
// CHECK-NEXT: return %1 : !llvm.ptr<3>
// CHECK-NEXT: }

// CHECK: func.func private @diff_const(%arg0: !llvm.ptr<3>, %arg1: index) -> !llvm.ptr<3> {
// CHECK-NEXT: %c288_i64 = arith.constant 288 : i64
// CHECK-NEXT: %c576_i64 = arith.constant 576 : i64
// CHECK-NEXT: %0 = affine.if #set()[%arg1] -> i64 {
// CHECK-NEXT: affine.yield %c576_i64 : i64
// CHECK-NEXT: } else {
// CHECK-NEXT: affine.yield %c288_i64 : i64
// CHECK-NEXT: }
// CHECK-NEXT: %1 = llvm.getelementptr inbounds|nuw %arg0[%0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
// CHECK-NEXT: return %1 : !llvm.ptr<3>
// CHECK-NEXT: }

// CHECK: func.func private @chained(%arg0: !llvm.ptr<3>, %arg1: i64, %arg2: index) -> memref<?xf64> {
// CHECK-NEXT: %c288_i64 = arith.constant 288 : i64
// CHECK-NEXT: %c576_i64 = arith.constant 576 : i64
// CHECK-NEXT: %0 = affine.if #set()[%arg2] -> i64 {
// CHECK-NEXT: affine.yield %c576_i64 : i64
// CHECK-NEXT: } else {
// CHECK-NEXT: affine.yield %c288_i64 : i64
// CHECK-NEXT: }
// CHECK-NEXT: %1 = llvm.getelementptr inbounds|nuw %arg0[%0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
// CHECK-NEXT: %2 = llvm.getelementptr inbounds|nuw %1[%arg1] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
// CHECK-NEXT: %3 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr<3>) -> memref<?xf64>
// CHECK-NEXT: return %3 : memref<?xf64>
// CHECK-NEXT: }

// -----



// A select is an if whose arms only yield: the same two matches apply.

module {
  // select of geps differing only in the base
  func.func private @sel_base(%b1: !llvm.ptr<3>, %b2: !llvm.ptr<3>, %i: i64, %c: i1) -> !llvm.ptr<3> {
    %g1 = llvm.getelementptr inbounds|nuw %b1[%i] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
    %g2 = llvm.getelementptr inbounds|nuw %b2[%i] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
    %r = arith.select %c, %g1, %g2 : !llvm.ptr<3>
    return %r : !llvm.ptr<3>
  }
  // select of geps off one base differing in a constant index
  func.func private @sel_const(%b: !llvm.ptr<3>, %c: i1) -> !llvm.ptr<3> {
    %g1 = llvm.getelementptr inbounds|nuw %b[288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %g2 = llvm.getelementptr inbounds|nuw %b[576] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %r = arith.select %c, %g1, %g2 : !llvm.ptr<3>
    return %r : !llvm.ptr<3>
  }
  // chained through a select
  func.func private @sel_chain(%b: !llvm.ptr<3>, %i: i64, %c: i1) -> memref<?xf64> {
    %p1 = llvm.getelementptr inbounds|nuw %b[288] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %p2 = llvm.getelementptr inbounds|nuw %b[576] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %g1 = llvm.getelementptr inbounds|nuw %p1[%i] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
    %g2 = llvm.getelementptr inbounds|nuw %p2[%i] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
    %v1 = "enzymexla.pointer2memref"(%g1) : (!llvm.ptr<3>) -> memref<?xf64>
    %v2 = "enzymexla.pointer2memref"(%g2) : (!llvm.ptr<3>) -> memref<?xf64>
    %r = arith.select %c, %v1, %v2 : memref<?xf64>
    return %r : memref<?xf64>
  }
}

// CHECK: func.func private @sel_base(%arg0: !llvm.ptr<3>, %arg1: !llvm.ptr<3>, %arg2: i64, %arg3: i1) -> !llvm.ptr<3> {
// CHECK-NEXT: %0 = arith.select %arg3, %arg0, %arg1 : !llvm.ptr<3>
// CHECK-NEXT: %1 = llvm.getelementptr inbounds|nuw %0[%arg2] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
// CHECK-NEXT: return %1 : !llvm.ptr<3>
// CHECK-NEXT: }

// CHECK: func.func private @sel_const(%arg0: !llvm.ptr<3>, %arg1: i1) -> !llvm.ptr<3> {
// CHECK-NEXT: %c288_i64 = arith.constant 288 : i64
// CHECK-NEXT: %c576_i64 = arith.constant 576 : i64
// CHECK-NEXT: %0 = arith.select %arg1, %c288_i64, %c576_i64 : i64
// CHECK-NEXT: %1 = llvm.getelementptr inbounds|nuw %arg0[%0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
// CHECK-NEXT: return %1 : !llvm.ptr<3>
// CHECK-NEXT: }

// CHECK: func.func private @sel_chain(%arg0: !llvm.ptr<3>, %arg1: i64, %arg2: i1) -> memref<?xf64> {
// CHECK-NEXT: %c288_i64 = arith.constant 288 : i64
// CHECK-NEXT: %c576_i64 = arith.constant 576 : i64
// CHECK-NEXT: %0 = arith.select %arg2, %c288_i64, %c576_i64 : i64
// CHECK-NEXT: %1 = llvm.getelementptr inbounds|nuw %arg0[%0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i8
// CHECK-NEXT: %2 = llvm.getelementptr inbounds|nuw %1[%arg1] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<8 x i8>
// CHECK-NEXT: %3 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr<3>) -> memref<?xf64>
// CHECK-NEXT: return %3 : memref<?xf64>
// CHECK-NEXT: }
