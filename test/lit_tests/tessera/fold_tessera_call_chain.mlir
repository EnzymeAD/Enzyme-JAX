// RUN: enzymexlamlir-opt %s -canonicalize | FileCheck %s

module {
  tessera.define @eigen.inv(%arg0: !llvm.ptr {llvm.sret = !llvm.struct<(array<4 x f32>)>, llvm.nocapture, llvm.writeonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.readonly}) -> () attributes {argSizes = array<i64: 16, 16>, byRefArgs = array<i1: true, true>, pure = true} {
    tessera.return
  }

  llvm.func @main(%arg0: i128) -> !llvm.struct<(array<4 x f32>)> {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(3 : i32) : i32
    %4 = llvm.mlir.poison : vector<4xf32>
    %5 = tessera.call @eigen.inv(%arg0) : (i128) -> !llvm.struct<(array<4 x f32>)>
    %6 = llvm.extractvalue %5[0, 0] : !llvm.struct<(array<4 x f32>)> 
    %7 = llvm.extractvalue %5[0, 1] : !llvm.struct<(array<4 x f32>)> 
    %8 = llvm.extractvalue %5[0, 2] : !llvm.struct<(array<4 x f32>)>
    %9 = llvm.extractvalue %5[0, 3] : !llvm.struct<(array<4 x f32>)>
    %10 = llvm.insertelement %6, %4[%0 : i32] : vector<4xf32>
    %11 = llvm.insertelement %7, %10[%1 : i32] : vector<4xf32> 
    %12 = llvm.insertelement %8, %11[%2 : i32] : vector<4xf32>
    %13 = llvm.insertelement %9, %12[%3 : i32] : vector<4xf32>
    %14 = llvm.bitcast %13 : vector<4xf32> to i128
    %15 = tessera.call @eigen.inv(%14) : (i128) -> !llvm.struct<(array<4 x f32>)>
    llvm.return %15 : !llvm.struct<(array<4 x f32>)>
  }
}

// CHECK: tessera.define @eigen.inv(%[[ARG0:.*]]: !llvm.ptr {llvm.nocapture, llvm.sret = !llvm.struct<(array<4 x f32>)>, llvm.writeonly}, %[[ARG1:.*]]: !llvm.ptr {llvm.nocapture, llvm.readonly})
// CHECK-NEXT: tessera.return

// CHECK: llvm.func @main(%[[VAL:.*]]: i128) -> !llvm.struct<(array<4 x f32>)> {
// CHECK-NEXT: %[[RES1:.*]] = tessera.call @eigen.inv(%[[VAL]]) : (i128) -> !llvm.struct<(array<4 x f32>)>
// CHECK-NEXT: %[[RES2:.*]] = tessera.call @eigen.inv(%[[RES1]]) : (!llvm.struct<(array<4 x f32>)>) -> !llvm.struct<(array<4 x f32>)>
// CHECK-NEXT: llvm.return %[[RES2]] : !llvm.struct<(array<4 x f32>)>
// CHECK-NEXT: }