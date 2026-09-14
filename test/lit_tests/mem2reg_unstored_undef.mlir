// RUN: enzymexlamlir-opt --polygeist-mem2reg %s | FileCheck %s

// Bytes of an allocation nothing writes hold uninitialized memory, so a read
// of them is undef: a load of a slot no store reaches, and a field extracted
// from a load of a struct whose other fields were stored. The stored fields
// forward as before, and the whole load goes with its last extract.

func.func @unstored() -> i32 {
  %alloca = memref.alloca() : memref<i32>
  %0 = memref.load %alloca[] : memref<i32>
  return %0 : i32
}

llvm.func @padding(%arg0: i32, %arg1: f64) -> !llvm.struct<(i32, f64, i8)> {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %alloca = llvm.alloca %c1 x !llvm.struct<(i32, array<4 x i8>, f64)> : (i32) -> !llvm.ptr
  llvm.store %arg0, %alloca : i32, !llvm.ptr
  %gep = llvm.getelementptr %alloca[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, f64)>
  llvm.store %arg1, %gep : f64, !llvm.ptr
  %0 = llvm.load %alloca : !llvm.ptr -> !llvm.struct<(i32, array<4 x i8>, f64)>
  %1 = llvm.extractvalue %0[0] : !llvm.struct<(i32, array<4 x i8>, f64)>
  %2 = llvm.extractvalue %0[2] : !llvm.struct<(i32, array<4 x i8>, f64)>
  %3 = llvm.extractvalue %0[1, 0] : !llvm.struct<(i32, array<4 x i8>, f64)>
  %4 = llvm.mlir.undef : !llvm.struct<(i32, f64, i8)>
  %5 = llvm.insertvalue %1, %4[0] : !llvm.struct<(i32, f64, i8)>
  %6 = llvm.insertvalue %2, %5[1] : !llvm.struct<(i32, f64, i8)>
  %7 = llvm.insertvalue %3, %6[2] : !llvm.struct<(i32, f64, i8)>
  llvm.return %7 : !llvm.struct<(i32, f64, i8)>
}

// CHECK:    func.func @unstored() -> i32 {
// CHECK-NEXT:    %0 = llvm.mlir.undef : i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @padding(%arg0: i32, %arg1: f64) -> !llvm.struct<(i32, f64, i8)> {
// CHECK-NEXT:    %0 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %1 = llvm.alloca %0 x !llvm.struct<(i32, array<4 x i8>, f64)> : (i32) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %arg0, %1 : i32, !llvm.ptr
// CHECK-NEXT:    %2 = llvm.getelementptr %1[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<4 x i8>, f64)>
// CHECK-NEXT:    llvm.store %arg1, %2 : f64, !llvm.ptr
// CHECK-NEXT:    %3 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i32, array<4 x i8>, f64)>
// CHECK-NEXT:    %4 = llvm.mlir.undef : i8
// CHECK-NEXT:    %5 = llvm.mlir.undef : !llvm.struct<(i32, f64, i8)>
// CHECK-NEXT:    %6 = llvm.insertvalue %arg0, %5[0] : !llvm.struct<(i32, f64, i8)> 
// CHECK-NEXT:    %7 = llvm.insertvalue %arg1, %6[1] : !llvm.struct<(i32, f64, i8)> 
// CHECK-NEXT:    %8 = llvm.insertvalue %4, %7[2] : !llvm.struct<(i32, f64, i8)> 
// CHECK-NEXT:    llvm.return %8 : !llvm.struct<(i32, f64, i8)>
// CHECK-NEXT:  }
