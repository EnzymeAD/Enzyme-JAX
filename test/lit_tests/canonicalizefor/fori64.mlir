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

// CHECK-LABEL:     func.func @if_trunc
// CHECK-SAME:            (%[[ub:.+]]: i32, %[[arg1:.+]]: i64, %[[arg2:.+]]: i32, %[[arg3:.+]]: i64,
// CHECK-DAG:             %[[cm1:.*]] = arith.constant -1 : i32
// CHECK-DAG:             %[[c1:.*]] = arith.constant 1 : i32
// CHECK:             scf.for %[[I:.*]] = %[[c1]] to %[[ub]] step %[[c1]]
// CHECK:               %[[TRUNC:.+]] = arith.trunci %[[arg1]] : i64 to i32
// CHECK:               %[[ADD:.+]] = arith.addi %[[TRUNC]], %[[VAL:.+]] : i32
// CHECK:               %[[EXT:.+]] = arith.extsi %[[ADD]] : i32 to i64
// CHECK:               llvm.getelementptr inbounds %[[ADDR:.+]][%[[EXT]]] : (!llvm.ptr, i64) -> !llvm.ptr, f64
