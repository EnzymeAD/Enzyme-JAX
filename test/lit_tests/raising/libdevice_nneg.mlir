// RUN: enzymexlamlir-opt %s --libdevice-funcs-raise | FileCheck %s

// The nneg flag of llvm.uitofp and llvm.zext is kept on the arith op, so that
// arith-raise can turn the conversion into a plain stablehlo.convert.

// CHECK-LABEL: @nneg
// CHECK: arith.uitofp %arg0 nneg : i64 to f64
// CHECK: arith.uitofp %arg1 : i64 to f64
// CHECK: arith.extui %arg2 nneg : i32 to i64
// CHECK: arith.extui %arg3 : i32 to i64
module {
  llvm.func @nneg(%a: i64, %b: i64, %c: i32, %d: i32) -> !llvm.struct<(f64, i64)> {
    %0 = llvm.uitofp nneg %a : i64 to f64
    %1 = llvm.uitofp %b : i64 to f64
    %2 = llvm.fadd %0, %1 : f64
    %3 = llvm.zext nneg %c : i32 to i64
    %4 = llvm.zext %d : i32 to i64
    %5 = llvm.add %3, %4 : i64
    %6 = llvm.mlir.undef : !llvm.struct<(f64, i64)>
    %7 = llvm.insertvalue %2, %6[0] : !llvm.struct<(f64, i64)>
    %8 = llvm.insertvalue %5, %7[1] : !llvm.struct<(f64, i64)>
    llvm.return %8 : !llvm.struct<(f64, i64)>
  }
}
