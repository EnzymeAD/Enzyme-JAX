// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

module {
  func.func @zerofill(%i: i64) {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %a = llvm.alloca %c1 x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
    "llvm.intr.memset"(%a, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    %g = llvm.getelementptr inbounds %a[%i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<24 x i8>
    "llvm.intr.memset"(%g, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    return
  }
  func.func @ptrs() {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %a = llvm.alloca %c1 x !llvm.array<8 x ptr> : (i32) -> !llvm.ptr
    "llvm.intr.memset"(%a, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    return
  }
  func.func @structs() {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %a = llvm.alloca %c1 x !llvm.array<2 x struct<(f64, i64)>> : (i32) -> !llvm.ptr
    "llvm.intr.memset"(%a, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    return
  }
  func.func @ragged() {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i8
    %c12 = arith.constant 12 : i64
    %a = llvm.alloca %c1 x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
    "llvm.intr.memset"(%a, %c0, %c12) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    return
  }
}

// CHECK:  func.func @zerofill(%[[m1:.+]]: i64) {
// CHECK-NEXT:  %[[m2:.+]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:  %[[m3:.+]] = arith.constant 1 : i32
// CHECK-NEXT:  %[[m4:.+]] = llvm.alloca %[[m3]] x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
// CHECK-NEXT:  %[[m5:.+]] = "enzymexla.pointer2memref"(%[[m4]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  affine.for %arg1 = 0 to 3 {
// CHECK-NEXT:  affine.store %[[m2]], %[[m5]][%arg1] : memref<?xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[m6:.+]] = llvm.getelementptr inbounds %[[m4]][%[[m1]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<24 x i8>
// CHECK-NEXT:  %[[m7:.+]] = "enzymexla.pointer2memref"(%[[m6]]) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:  affine.for %arg1 = 0 to 3 {
// CHECK-NEXT:  affine.store %[[m2]], %[[m7]][%arg1] : memref<?xf64>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @ptrs() {
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @structs() {
// CHECK-NEXT:  %[[s1:.+]] = arith.constant 1 : i32
// CHECK-NEXT:  %[[s2:.+]] = arith.constant 0 : i8
// CHECK-NEXT:  %[[s3:.+]] = arith.constant 24 : i64
// CHECK-NEXT:  %[[s4:.+]] = llvm.alloca %[[s1]] x !llvm.array<2 x struct<(f64, i64)>> : (i32) -> !llvm.ptr
// CHECK-NEXT:  "llvm.intr.memset"(%[[s4]], %[[s2]], %[[s3]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:  return
// CHECK-NEXT:  }

// CHECK:  func.func @ragged() {
// CHECK-NEXT:  %[[m1:.+]] = arith.constant 1 : i32
// CHECK-NEXT:  %[[m2:.+]] = arith.constant 0 : i8
// CHECK-NEXT:  %[[m3:.+]] = arith.constant 12 : i64
// CHECK-NEXT:  %[[m4:.+]] = llvm.alloca %[[m1]] x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
// CHECK-NEXT:  "llvm.intr.memset"(%[[m4]], %[[m2]], %[[m3]]) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
// CHECK-NEXT:  return
// CHECK-NEXT:  }
