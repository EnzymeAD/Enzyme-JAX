// RUN: enzymexlamlir-opt %s -canonicalize | FileCheck %s

module {
  llvm.func local_unnamed_addr @f(%arg0: !llvm.struct<"struct.Eigen::internal::plain_array", (array<4 x f32>)>) -> vector<4xf32> {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.extractvalue %arg0[0] : !llvm.struct<"struct.Eigen::internal::plain_array", (array<4 x f32>)> 
    %2 = llvm.extractvalue %1[0] : !llvm.array<4 x f32> 
    %3 = llvm.extractvalue %1[1] : !llvm.array<4 x f32> 
    %4 = llvm.extractvalue %1[2] : !llvm.array<4 x f32>
    %5 = llvm.extractvalue %1[3] : !llvm.array<4 x f32> 
    %6 = llvm.mlir.poison : vector<4xf32>
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.insertelement %2, %6[%7 : i32] : vector<4xf32>
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.insertelement %3, %8[%9 : i32] : vector<4xf32> 
    %11 = llvm.mlir.constant(2 : i32) : i32
    %12 = llvm.insertelement %4, %10[%11 : i32] : vector<4xf32>
    %13 = llvm.mlir.constant(3 : i32) : i32
    %14 = llvm.insertelement %5, %12[%13 : i32] : vector<4xf32>
    llvm.return %14 : vector<4xf32>
  }
}