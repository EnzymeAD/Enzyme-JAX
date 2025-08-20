// RUN: enzymexlamlir-opt --allow-unregistered-dialect --canonicalize-scf-for -split-input-file %s | FileCheck %s

module {
  func.func @if_trunc(%89 : i32, %87 : i64, %68: i32, %0 : i64, %2: !llvm.ptr, %57: f64, %69: !llvm.ptr, %70: !llvm.ptr, %arg16 : f64, %71: !llvm.ptr, %arg11: f64, %72: i32) {
    %cst = arith.constant 1.200000e+03 : f64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
          %90:5 = scf.for %arg21 = %c1_i32 to %89 step %c1_i32 iter_args(%arg22 = %87, %arg23 = %c0_i32, %arg24 = %68, %arg25 = %0, %arg26 = %2) -> (i64, i32, i32, i64, !llvm.ptr)  : i32 {
            %95 = arith.sitofp %arg24 : i32 to f64
            %96 = arith.mulf %95, %cst {fastmathFlags = #llvm.fastmath<fast>} : f64
            %97 = arith.divf %96, %57 {fastmathFlags = #llvm.fastmath<fast>} : f64
            %98 = llvm.load %69 {alignment = 1 : i64} : !llvm.ptr -> i64
            %99 = llvm.inttoptr %98 : i64 to !llvm.ptr
            %100 = llvm.getelementptr inbounds %99[%arg22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
            llvm.store %97, %100 {alignment = 8 : i64} : f64, !llvm.ptr
            %101 = llvm.load %70 {alignment = 1 : i64} : !llvm.ptr -> i64
            %102 = llvm.inttoptr %101 : i64 to !llvm.ptr
            %103 = llvm.getelementptr inbounds %102[%arg22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
            llvm.store %arg16, %103 {alignment = 8 : i64} : f64, !llvm.ptr
            %104 = llvm.load %71 {alignment = 1 : i64} : !llvm.ptr -> i64
            %105 = llvm.inttoptr %104 : i64 to !llvm.ptr
            %106 = llvm.getelementptr inbounds %105[%arg22] : (!llvm.ptr, i64) -> !llvm.ptr, f64
            llvm.store %arg11, %106 {alignment = 8 : i64} : f64, !llvm.ptr
            %107 = arith.addi %arg22, %c1_i64 : i64
            %108 = arith.addi %72, %arg23 : i32
            %109 = arith.addi %arg23, %c1_i32 : i32
            scf.yield %107, %109, %108, %107, %99 : i64, i32, i32, i64, !llvm.ptr
          }
    return
  }
}

// CHECK:  func.func @if_trunc(%arg0: i32, %arg1: i64, %arg2: i32, %arg3: i64, %arg4: !llvm.ptr, %arg5: f64, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: f64, %arg9: !llvm.ptr, %arg10: f64, %arg11: i32) {
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %cst = arith.constant 1.200000e+03 : f64
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %0 = scf.for %arg12 = %c1_i32 to %arg0 step %c1_i32 iter_args(%arg13 = %arg2) -> (i32)  : i32 {
// CHECK-NEXT:      %1 = arith.addi %arg12, %c-1_i32 : i32
// CHECK-NEXT:      %2 = arith.addi %arg12, %c-1_i32 : i32
// CHECK-NEXT:      %3 = arith.trunci %arg1 : i64 to i32
// CHECK-NEXT:      %4 = arith.addi %3, %2 : i32
// CHECK-NEXT:      %5 = arith.extsi %4 : i32 to i64
// CHECK-NEXT:      %6 = arith.sitofp %arg13 : i32 to f64
// CHECK-NEXT:      %7 = arith.mulf %6, %cst {fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:      %8 = arith.divf %7, %arg5 {fastmathFlags = #llvm.fastmath<fast>} : f64
// CHECK-NEXT:      %9 = llvm.load %arg6 {alignment = 1 : i64} : !llvm.ptr -> i64
// CHECK-NEXT:      %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
// CHECK-NEXT:      %11 = llvm.getelementptr inbounds %10[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:      llvm.store %8, %11 {alignment = 8 : i64} : f64, !llvm.ptr
// CHECK-NEXT:      %12 = llvm.load %arg7 {alignment = 1 : i64} : !llvm.ptr -> i64
// CHECK-NEXT:      %13 = llvm.inttoptr %12 : i64 to !llvm.ptr
// CHECK-NEXT:      %14 = llvm.getelementptr inbounds %13[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:      llvm.store %arg8, %14 {alignment = 8 : i64} : f64, !llvm.ptr
// CHECK-NEXT:      %15 = llvm.load %arg9 {alignment = 1 : i64} : !llvm.ptr -> i64
// CHECK-NEXT:      %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
// CHECK-NEXT:      %17 = llvm.getelementptr inbounds %16[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:      llvm.store %arg10, %17 {alignment = 8 : i64} : f64, !llvm.ptr
// CHECK-NEXT:      %18 = arith.addi %arg11, %1 : i32
// CHECK-NEXT:      scf.yield %18 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
