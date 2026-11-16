// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

// `ifelse` on a complex number in a kernel is an `llvm.select` of
// `!llvm.array<2 x f32>` (EnzymeAD/Reactant.jl#3475). Splitting the aggregate
// store per leaf leaves extractvalues of that select, which raising cannot
// handle; they must become selects of the leaves, folded through the
// insertvalue chain and the aggregate constant, as arith ops.

// CHECK-LABEL: func.func private @zero_first
// CHECK-SAME:    (%[[OUT:.+]]: memref<2x2xf32, 1>, %[[A:.+]]: memref<2x2xf32, 1>)
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         affine.parallel (%[[I:.+]]) = (0) to (2) {
// CHECK:           %[[RE:.+]] = affine.load %[[A]][%[[I]], 0]
// CHECK:           %[[IM:.+]] = affine.load %[[A]][%[[I]], 1]
// CHECK:           %[[C:.+]] = arith.cmpi ne
// CHECK-DAG:       %[[SRE:.+]] = arith.select %[[C]], %[[RE]], %[[ZERO]] : f32
// CHECK-DAG:       %[[SIM:.+]] = arith.select %[[C]], %[[IM]], %[[ZERO]] : f32
// CHECK-DAG:       affine.store %[[SRE]], %[[OUT]][%[[I]], 0]
// CHECK-DAG:       affine.store %[[SIM]], %[[OUT]][%[[I]], 1]
// CHECK-NOT:     llvm.select
// CHECK-NOT:     llvm.extractvalue
// CHECK-NOT:     llvm.mlir.constant
func.func private @zero_first(%arg0: memref<2x2xf32, 1>, %arg1: memref<2x2xf32, 1>) {
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %0 = llvm.mlir.constant(dense<0.000000e+00> : tensor<2xf32>) : !llvm.array<2 x f32>
  %1 = "enzymexla.memref2pointer"(%arg0) : (memref<2x2xf32, 1>) -> !llvm.ptr<1>
  %2 = "enzymexla.memref2pointer"(%arg1) : (memref<2x2xf32, 1>) -> !llvm.ptr<1>
  affine.parallel (%arg2) = (0) to (2) {
    %3 = arith.addi %arg2, %c1 : index
    %4 = arith.index_castui %3 : index to i64
    %5 = arith.index_castui %arg2 : index to i64
    %6 = llvm.getelementptr inbounds %2[%5] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<2 x f32>
    %7 = llvm.load %6 invariant <alignment = 4> : !llvm.ptr<1> -> !llvm.array<2 x f32>
    %8 = arith.cmpi ne, %4, %c1_i64 : i64
    %9 = llvm.select %8, %7, %0 : i1, !llvm.array<2 x f32>
    %10 = llvm.getelementptr inbounds %1[%5] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, !llvm.array<2 x f32>
    llvm.store %9, %10 <alignment = 4> : !llvm.array<2 x f32>, !llvm.ptr<1>
  }
  return
}
