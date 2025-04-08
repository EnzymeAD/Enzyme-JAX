// RUN: enzymexlamlir-opt %s  --canonicalize-scf-for --split-input-file | FileCheck %s

// Check that we correctly identify ambigous main IV

func.func @foo(%arg0: memref<1x104x194xf64, 1>, %arg1: memref<35xf64, 1>, %arg2: memref<34xf64, 1>) {
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c1_i64 = arith.constant 1 : i64
  %c20_i64 = arith.constant 20 : i64
  affine.parallel (%arg3, %arg4) = (0, 0) to (90, 180) {
    %0 = affine.load %arg0[0, %arg3 + 7, %arg4 + 7] : memref<1x104x194xf64, 1>
    %1 = affine.load %arg1[7] : memref<35xf64, 1>
    affine.store %1, %arg0[0, %arg3 + 7, %arg4 + 7] : memref<1x104x194xf64, 1>
    %2:2 = scf.while (%arg5 = %1, %arg6 = %c1_i64) : (f64, i64) -> (f64, i64) {
      %3 = arith.index_cast %arg6 : i64 to index
      %4 = arith.addi %3, %c7 : index
      %5 = memref.load %arg1[%4] : memref<35xf64, 1>
      %6 = arith.index_cast %arg6 : i64 to index
      %7 = arith.addi %6, %c6 : index
      %8 = memref.load %arg2[%7] : memref<34xf64, 1>
      %9 = arith.cmpf ole, %8, %0 : f64
      %10 = arith.select %9, %5, %arg5 : f64
      affine.store %10, %arg0[0, %arg3 + 7, %arg4 + 7] : memref<1x104x194xf64, 1>
      %11 = arith.addi %arg6, %c1_i64 : i64
      %12 = arith.cmpi ne, %arg6, %c20_i64 : i64
      scf.condition(%12) %10, %11 : f64, i64
    } do {
    ^bb0(%arg5: f64, %arg6: i64):
      scf.yield %arg5, %arg6 : f64, i64
    }
  }
  return
}

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<1x104x194xf64, 1>,
// CHECK-SAME:                   %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<35xf64, 1>,
// CHECK-SAME:                   %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<34xf64, 1>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 21 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 7 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK:           affine.parallel (%[[VAL_7:.*]], %[[VAL_8:.*]]) = (0, 0) to (90, 180) {
// CHECK:             %[[VAL_9:.*]] = affine.load %[[VAL_0]][0, %[[VAL_7]] + 7, %[[VAL_8]] + 7] : memref<1x104x194xf64, 1>
// CHECK:             %[[VAL_10:.*]] = affine.load %[[VAL_1]][7] : memref<35xf64, 1>
// CHECK:             affine.store %[[VAL_10]], %[[VAL_0]][0, %[[VAL_7]] + 7, %[[VAL_8]] + 7] : memref<1x104x194xf64, 1>
// CHECK:             %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_6]] iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (f64)  : i64 {
// CHECK:               %[[VAL_14:.*]] = arith.index_cast %[[VAL_12]] : i64 to index
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_5]] : index
// CHECK:               %[[VAL_16:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_15]]] : memref<35xf64, 1>
// CHECK:               %[[VAL_17:.*]] = arith.index_cast %[[VAL_12]] : i64 to index
// CHECK:               %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_4]] : index
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_18]]] : memref<34xf64, 1>
// CHECK:               %[[VAL_20:.*]] = arith.cmpf ole, %[[VAL_19]], %[[VAL_9]] : f64
// CHECK:               %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_16]], %[[VAL_13]] : f64
// CHECK:               affine.store %[[VAL_21]], %[[VAL_0]][0, %[[VAL_7]] + 7, %[[VAL_8]] + 7] : memref<1x104x194xf64, 1>
// CHECK:               scf.yield %[[VAL_21]] : f64
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }


// -----


module {
  func.func private @while_2_to_10(%arg0: memref<26x48x48xf64, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 479232 : i64, llvm.noalias}, %arg1: memref<10x32x32xf64, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 81920 : i64, llvm.noalias}) {
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c-2_i64 = arith.constant -2 : i64
    %c16_i16 = arith.constant 16 : i16
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c32_i64 = arith.constant 32 : i64
    %c7_i64 = arith.constant 7 : i64
    %c48_i64 = arith.constant 48 : i64
    %c18439_i64 = arith.constant 18439 : i64
    %cst = arith.constant 1.000000e-01 : f64
    %c-1_i64 = arith.constant -1 : i64
    %c2_i64 = arith.constant 2 : i64
    %cst_0 = arith.constant -1.000000e-01 : f64
    %c1024_i64 = arith.constant 1024 : i64
    %cst_1 = arith.constant 1.200000e+00 : f64
    %cst_2 = arith.constant 2.2204460492503131E-15 : f64
    %c6_i64 = arith.constant 6 : i64
    %c2304_i64 = arith.constant 2304 : i64
    %c10_i64 = arith.constant 10 : i64
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<26x48x48xf64, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.memref2pointer"(%arg1) : (memref<10x32x32xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%arg2, %arg3) = (0, 0) to (4, 256) {
      %2 = arith.addi %arg2, %c1 : index
      %3 = arith.addi %arg3, %c1 : index
      %4 = arith.index_castui %2 : index to i64
      %5 = arith.subi %4, %c1_i64 : i64
      %6 = arith.trunci %5 : i64 to i32
      %7 = arith.divui %6, %c2_i32 : i32
      %8 = arith.extui %7 : i32 to i64
      %9 = arith.muli %8, %c-2_i64 : i64
      %10 = arith.index_castui %3 : index to i64
      %11 = arith.subi %10, %c1_i64 : i64
      %12 = arith.trunci %11 : i64 to i16
      %13 = arith.divui %12, %c16_i16 : i16
      %14 = arith.extui %13 : i16 to i64
      %15 = arith.subi %c0_i64, %14 : i64
      %16 = arith.addi %14, %c1_i64 : i64
      %17 = arith.addi %15, %5 : i64
      %18 = arith.addi %17, %9 : i64
      %19 = arith.muli %18, %c16_i64 : i64
      %20 = arith.addi %10, %19 : i64
      %21 = arith.muli %8, %c16_i64 : i64
      %22 = arith.addi %16, %21 : i64
      %23 = arith.cmpi sge, %20, %c1_i64 : i64
      %24 = arith.cmpi sle, %20, %c32_i64 : i64
      %25 = arith.andi %23, %24 : i1
      scf.if %25 {
        %26 = arith.addi %22, %c7_i64 : i64
        %27 = arith.muli %26, %c48_i64 : i64
        %28 = arith.addi %20, %c18439_i64 : i64
        %29 = arith.addi %28, %27 : i64
        %30 = llvm.getelementptr inbounds %0[%29] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
        %31 = llvm.load %30 : !llvm.ptr<1> -> f64
        %32 = arith.divf %31, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
        llvm.store %32, %30 : f64, !llvm.ptr<1>
        %33 = arith.subi %22, %c1_i64 : i64
        %34 = arith.muli %33, %c32_i64 : i64
        %35 = arith.addi %34, %c-1_i64 : i64
        %36 = arith.addi %35, %20 : i64
        %37 = arith.addi %20, %c7_i64 : i64
        %38 = arith.addi %37, %27 : i64
        %39:2 = scf.while (%arg4 = %c2_i64, %arg5 = %cst) : (i64, f64) -> (i64, f64) {
          %40 = arith.divf %cst_0, %arg5 {fastmathFlags = #llvm.fastmath<none>} : f64
          %41 = arith.subi %arg4, %c1_i64 : i64
          %42 = arith.muli %41, %c1024_i64 : i64
          %43 = arith.addi %36, %42 : i64
          %44 = llvm.getelementptr inbounds %1[%43] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
          llvm.store %40, %44  : f64, !llvm.ptr<1>
          %45 = arith.mulf %40, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
          %46 = arith.addf %45, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
          %47 = math.absf %46 : f64
          %48 = arith.cmpf olt, %cst_2, %47 {fastmathFlags = #llvm.fastmath<none>} : f64
          %49 = arith.addi %arg4, %c6_i64 : i64
          %50 = arith.muli %49, %c2304_i64 : i64
          %51 = arith.addi %38, %50 : i64
          %52 = llvm.getelementptr inbounds %0[%51] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
          %53 = llvm.load %52  : !llvm.ptr<1> -> f64
          %54 = arith.addi %arg4, %c7_i64 : i64
          %55 = arith.muli %54, %c2304_i64 : i64
          %56 = arith.addi %38, %55 : i64
          %57 = llvm.getelementptr inbounds %0[%56] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
          %58 = llvm.load %57  : !llvm.ptr<1> -> f64
          %59 = arith.select %48, %53, %58 : f64
          llvm.store %59, %57  : f64, !llvm.ptr<1>
          %60 = arith.addi %arg4, %c1_i64 : i64
          %61 = arith.cmpi ne, %arg4, %c10_i64 : i64
          scf.condition(%61) %60, %46 : i64, f64
        } do {
        ^bb0(%arg4: i64, %arg5: f64):
          scf.yield %arg4, %arg5 : i64, f64
        }
      }
    }
    return
  }
}

// CHECK-LABEL:  func.func private @while_2_to_10(
// CHECK:    %[[C11:.+]] = arith.constant 11 : i64
// CHECK:    %[[C1:.+]] = arith.constant 1 : i64
// CHECK:    %[[C2:.+]] = arith.constant 2 : i64
// CHECK:    %[[LOOP:.+]] = scf.for %[[IV:.+]] = %[[C2]] to %[[C11]] step %[[C1]] iter_args(%[[CARRIED:.+]] = %[[CST:.+]]) -> (f64)  : i64 {
