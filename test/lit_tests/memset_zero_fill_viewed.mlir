// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

module {
  func.func @chosen(%c: i1, %i: i64) {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %a = llvm.alloca %c1 x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
    %b = llvm.alloca %c1 x !llvm.array<4 x f64> : (i32) -> !llvm.ptr
    %ga = llvm.getelementptr %a[%i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
    %gb = llvm.getelementptr %b[%i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<8 x i8>
    %s = arith.select %c, %ga, %gb : !llvm.ptr
    "llvm.intr.memset"(%s, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    return
  }
  func.func @cast(%c: i1) {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %a = llvm.alloca %c1 x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
    %b = llvm.alloca %c1 x !llvm.array<4 x f64> : (i32) -> !llvm.ptr
    %ca = llvm.addrspacecast %a : !llvm.ptr to !llvm.ptr<3>
    %cb = llvm.addrspacecast %b : !llvm.ptr to !llvm.ptr<3>
    %s = arith.select %c, %ca, %cb : !llvm.ptr<3>
    "llvm.intr.memset"(%s, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr<3>, i8, i64) -> ()
    return
  }
  func.func @mismatched(%c: i1) {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %a = llvm.alloca %c1 x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
    %b = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
    %s = arith.select %c, %a, %b : !llvm.ptr
    "llvm.intr.memset"(%s, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    return
  }
  func.func @argument(%a: !llvm.ptr, %c: i1) {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i8
    %c24 = arith.constant 24 : i64
    %b = llvm.alloca %c1 x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
    %s = arith.select %c, %a, %b : !llvm.ptr
    "llvm.intr.memset"(%s, %c0, %c24) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
    return
  }
}

// CHECK-LABEL:  func.func @chosen(
// CHECK-NOT:  llvm.intr.memset
// CHECK:  %[[sel:.+]] = arith.select %arg0, %{{.*}}, %{{.*}} : memref<?xf64>
// CHECK-NEXT:  affine.for %[[iv:.+]] = 0 to 3 {
// CHECK-NEXT:  affine.store %{{.*}}, %[[sel]][%[[iv]]] : memref<?xf64>
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @cast(
// CHECK-NOT:  llvm.intr.memset
// CHECK:  %[[sel:.+]] = arith.select %arg0, %{{.*}}, %{{.*}} : memref<?xf64, 3 : index>
// CHECK-NEXT:  affine.for %[[iv:.+]] = 0 to 3 {
// CHECK-NEXT:  affine.store %{{.*}}, %[[sel]][%[[iv]]] : memref<?xf64, 3 : index>
// CHECK-NEXT:  }

// CHECK-LABEL:  func.func @mismatched(
// CHECK:  llvm.intr.memset

// CHECK-LABEL:  func.func @argument(
// CHECK:  llvm.intr.memset
