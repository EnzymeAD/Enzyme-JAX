// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func private @red
// CHECK: stablehlo.while

module {
  func.func private @red(%arg0: memref<34x104x194xf64, 1>, %arg1: memref<34x104x194xf64, 1>, %arg2: memref<35x104x194xf64, 1>, %arg3: memref<34xf64, 1>, %arg4: memref<34xf64, 1>, %arg5: memref<104x194xf64, 1>, %arg6: memref<104x194xf64, 1>, %arg7: memref<104x194xf64, 1>, %arg8: memref<1x104x194xf64, 1>) {
    %c-4_i64 = arith.constant -4 : i64
    %c-6_i64 = arith.constant -6 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-5_i64 = arith.constant -5 : i64
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f64
    %cst2 = arith.constant 2.000000e+00 : f64
    %c20_i64 = arith.constant 20 : i64
    affine.parallel (%arg9, %arg10) = (0, 0) to (102, 192) {
      %0 = arith.index_castui %arg9 : index to i64
      %1 = arith.addi %0, %c-5_i64 : i64
      affine.store %cst, %arg2[7, %arg9 + 1, %arg10 + 1] : memref<35x104x194xf64, 1>
      %2 = arith.cmpi slt, %1, %c1_i64 : i64
      %3 = arith.addi %0, %c-6_i64 : i64
      %4 = arith.cmpi slt, %3, %c1_i64 : i64
      %5 = arith.addi %0, %c-4_i64 : i64
      %6 = arith.cmpi slt, %5, %c1_i64 : i64
      affine.for %arg11 = 0 to 20 {
        %72 = affine.load %arg2[%arg11 + 7, %arg9 + 1, %arg10 + 1] : memref<35x104x194xf64, 1>
        %73 = arith.subf %72, %cst2 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %73, %arg2[%arg11 + 8, %arg9 + 1, %arg10 + 1] : memref<35x104x194xf64, 1>
      }
    }
    return
  }
}

// -----

// CHECK-LABEL:   func.func private @big
// CHECK: stablehlo.while

module {
  func.func private @big(%arg0: memref<34x104x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<35xf64, 1>, %arg3: memref<34x104x194xf64, 1>, %arg4: memref<34x104x194xf64, 1>) {
    %cst = arith.constant 4.000000e+01 : f64
    %cst_0 = arith.constant 3.200000e+01 : f64
    %cst_1 = arith.constant 40.18861714285714 : f64
    %cst_2 = arith.constant 1.000000e+04 : f64
    %cst_3 = arith.constant 0.37969820454999997 : f64
    %cst_4 = arith.constant 0.018507636718000001 : f64
    %cst_5 = arith.constant -0.023342758796999999 : f64
    %cst_6 = arith.constant -1.2419983026000001 : f64
    %cst_7 = arith.constant 0.21311365518 : f64
    %cst_8 = arith.constant 2.0564311498999999 : f64
    %cst_9 = arith.constant 2.5019633244000001 : f64
    %cst_10 = arith.constant -4.9527603988999997 : f64
    %cst_11 = arith.constant 2.0660924175000002 : f64
    %cst_12 = arith.constant 0.55927935969999998 : f64
    %cst_13 = arith.constant 0.55077101278999996 : f64
    %cst_14 = arith.constant -2.4649669533999998 : f64
    %cst_15 = arith.constant 1.8795372995999999 : f64
    %cst_16 = arith.constant 3.5063081279000001 : f64
    %cst_17 = arith.constant 6.7080479603000001 : f64
    %cst_18 = arith.constant 0.65399043664000001 : f64
    %cst_19 = arith.constant 5.0042598061000003 : f64
    %cst_20 = arith.constant -4.4870114575000004 : f64
    %cst_21 = arith.constant -13.336301112999999 : f64
    %cst_22 = arith.constant 6.6051753096999999 : f64
    %cst_23 = arith.constant -30.938076334000002 : f64
    %cst_24 = arith.constant 50.774768217999998 : f64
    %cst_25 = arith.constant -42.549998213999999 : f64
    %cst_26 = arith.constant 19.681925208999999 : f64
    %cst_27 = arith.constant 0.19083568887999999 : f64
    %cst_28 = arith.constant 0.48169980162999998 : f64
    %cst_29 = arith.constant 0.54048723790999997 : f64
    %cst_30 = arith.constant 5.3563304045000004 : f64
    %cst_31 = arith.constant 11.311538583999999 : f64
    %cst_32 = arith.constant -8.3627885466999992 : f64
    %cst_33 = arith.constant 3.1742946532 : f64
    %cst_34 = arith.constant 19.717078466 : f64
    %cst_35 = arith.constant -33.449108469000002 : f64
    %cst_36 = arith.constant 21.661789529 : f64
    %cst_37 = arith.constant 5.4723692739000001 : f64
    %cst_38 = arith.constant 29.130021252999999 : f64
    %cst_39 = arith.constant -60.362551500999999 : f64
    %cst_40 = arith.constant 61.548258126999997 : f64
    %cst_41 = arith.constant -37.074170416999998 : f64
    %cst_42 = arith.constant 1.9193502195000001 : f64
    %cst_43 = arith.constant 17.681814114000002 : f64
    %cst_44 = arith.constant -56.888046320999997 : f64
    %cst_45 = arith.constant 81.770425107999997 : f64
    %cst_46 = arith.constant -65.281885265 : f64
    %cst_47 = arith.constant 26.010145068 : f64
    %cst_48 = arith.constant 60.579916611999998 : f64
    %cst_49 = arith.constant 432.27585684000002 : f64
    %cst_50 = arith.constant -1284.9161071000001 : f64
    %cst_51 = arith.constant 2037.5295546 : f64
    %cst_52 = arith.constant -1786.4682637000001 : f64
    %cst_53 = arith.constant 866.72408165000002 : f64
    %cst_54 = arith.constant 801.89615746000004 : f64
    %cst_55 = arith.constant 1.020000e+03 : f64
    %cst_56 = arith.constant 9.8066499999999994 : f64
    %cst_57 = arith.constant 5.000000e-01 : f64
    affine.parallel (%arg5, %arg6) = (0, 0) to (92, 182) {
      %0 = affine.load %arg3[26, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
      %1 = affine.load %arg4[26, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
      %2 = affine.load %arg1[26] {alignment = 16 : i64, ordering = 0 : i64 } : memref<34xf64, 1>
      %3 = arith.divf %0, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      %4 = arith.addf %1, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
      %5 = arith.divf %4, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
      %6 = math.sqrt %5 : f64
      %7 = arith.negf %2 {fastmathFlags = #llvm.fastmath<none>} : f64
      %8 = arith.divf %7, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
      %9 = arith.mulf %3, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
      %10 = arith.mulf %6, %cst_4 {fastmathFlags = #llvm.fastmath<none>} : f64
      %11 = arith.subf %9, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
      %12 = arith.addf %11, %cst_5 {fastmathFlags = #llvm.fastmath<none>} : f64
      %13 = arith.mulf %8, %12 {fastmathFlags = #llvm.fastmath<none>} : f64
      %14 = arith.mulf %3, %cst_6 {fastmathFlags = #llvm.fastmath<none>} : f64
      %15 = arith.mulf %6, %cst_7 {fastmathFlags = #llvm.fastmath<none>} : f64
      %16 = arith.subf %14, %15 {fastmathFlags = #llvm.fastmath<none>} : f64
      %17 = arith.addf %16, %cst_8 {fastmathFlags = #llvm.fastmath<none>} : f64
      %18 = arith.mulf %3, %17 {fastmathFlags = #llvm.fastmath<none>} : f64
      %19 = arith.mulf %6, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f64
      %20 = arith.addf %19, %cst_10 {fastmathFlags = #llvm.fastmath<none>} : f64
      %21 = arith.mulf %6, %20 {fastmathFlags = #llvm.fastmath<none>} : f64
      %22 = arith.addf %21, %18 {fastmathFlags = #llvm.fastmath<none>} : f64
      %23 = arith.addf %22, %cst_11 {fastmathFlags = #llvm.fastmath<none>} : f64
      %24 = arith.addf %13, %23 {fastmathFlags = #llvm.fastmath<none>} : f64
      %25 = arith.mulf %8, %24 {fastmathFlags = #llvm.fastmath<none>} : f64
      %26 = arith.mulf %3, %cst_12 {fastmathFlags = #llvm.fastmath<none>} : f64
      %27 = arith.mulf %6, %cst_13 {fastmathFlags = #llvm.fastmath<none>} : f64
      %28 = arith.subf %26, %27 {fastmathFlags = #llvm.fastmath<none>} : f64
      %29 = arith.addf %28, %cst_14 {fastmathFlags = #llvm.fastmath<none>} : f64
      %30 = arith.mulf %3, %29 {fastmathFlags = #llvm.fastmath<none>} : f64
      %31 = arith.mulf %6, %cst_15 {fastmathFlags = #llvm.fastmath<none>} : f64
      %32 = arith.subf %cst_16, %31 {fastmathFlags = #llvm.fastmath<none>} : f64
      %33 = arith.mulf %6, %32 {fastmathFlags = #llvm.fastmath<none>} : f64
      %34 = arith.addf %33, %30 {fastmathFlags = #llvm.fastmath<none>} : f64
      %35 = arith.addf %34, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f64
      %36 = arith.mulf %3, %35 {fastmathFlags = #llvm.fastmath<none>} : f64
      %37 = arith.mulf %6, %cst_18 {fastmathFlags = #llvm.fastmath<none>} : f64
      %38 = arith.subf %cst_19, %37 {fastmathFlags = #llvm.fastmath<none>} : f64
      %39 = arith.mulf %6, %38 {fastmathFlags = #llvm.fastmath<none>} : f64
      %40 = arith.addf %39, %cst_20 {fastmathFlags = #llvm.fastmath<none>} : f64
      %41 = arith.mulf %6, %40 {fastmathFlags = #llvm.fastmath<none>} : f64
      %42 = arith.addf %41, %36 {fastmathFlags = #llvm.fastmath<none>} : f64
      %43 = arith.addf %42, %cst_21 {fastmathFlags = #llvm.fastmath<none>} : f64
      %44 = arith.mulf %3, %43 {fastmathFlags = #llvm.fastmath<none>} : f64
      %45 = arith.mulf %6, %cst_22 {fastmathFlags = #llvm.fastmath<none>} : f64
      %46 = arith.addf %45, %cst_23 {fastmathFlags = #llvm.fastmath<none>} : f64
      %47 = arith.mulf %6, %46 {fastmathFlags = #llvm.fastmath<none>} : f64
      %48 = arith.addf %47, %cst_24 {fastmathFlags = #llvm.fastmath<none>} : f64
      %49 = arith.mulf %6, %48 {fastmathFlags = #llvm.fastmath<none>} : f64
      %50 = arith.addf %49, %cst_25 {fastmathFlags = #llvm.fastmath<none>} : f64
      %51 = arith.mulf %6, %50 {fastmathFlags = #llvm.fastmath<none>} : f64
      %52 = arith.addf %51, %44 {fastmathFlags = #llvm.fastmath<none>} : f64
      %53 = arith.addf %52, %cst_26 {fastmathFlags = #llvm.fastmath<none>} : f64
      %54 = arith.addf %25, %53 {fastmathFlags = #llvm.fastmath<none>} : f64
      %55 = arith.mulf %8, %54 {fastmathFlags = #llvm.fastmath<none>} : f64
      %56 = arith.mulf %3, %cst_27 {fastmathFlags = #llvm.fastmath<none>} : f64
      %57 = arith.mulf %6, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f64
      %58 = arith.subf %57, %56 {fastmathFlags = #llvm.fastmath<none>} : f64
      %59 = arith.addf %58, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f64
      %60 = arith.mulf %3, %59 {fastmathFlags = #llvm.fastmath<none>} : f64
      %61 = arith.mulf %6, %cst_30 {fastmathFlags = #llvm.fastmath<none>} : f64
      %62 = arith.subf %cst_31, %61 {fastmathFlags = #llvm.fastmath<none>} : f64
      %63 = arith.mulf %6, %62 {fastmathFlags = #llvm.fastmath<none>} : f64
      %64 = arith.addf %63, %60 {fastmathFlags = #llvm.fastmath<none>} : f64
      %65 = arith.addf %64, %cst_32 {fastmathFlags = #llvm.fastmath<none>} : f64
      %66 = arith.mulf %3, %65 {fastmathFlags = #llvm.fastmath<none>} : f64
      %67 = arith.mulf %6, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f64
      %68 = arith.subf %cst_34, %67 {fastmathFlags = #llvm.fastmath<none>} : f64
      %69 = arith.mulf %6, %68 {fastmathFlags = #llvm.fastmath<none>} : f64
      %70 = arith.addf %69, %cst_35 {fastmathFlags = #llvm.fastmath<none>} : f64
      %71 = arith.mulf %6, %70 {fastmathFlags = #llvm.fastmath<none>} : f64
      %72 = arith.addf %71, %66 {fastmathFlags = #llvm.fastmath<none>} : f64
      %73 = arith.addf %72, %cst_36 {fastmathFlags = #llvm.fastmath<none>} : f64
      %74 = arith.mulf %3, %73 {fastmathFlags = #llvm.fastmath<none>} : f64
      %75 = arith.mulf %6, %cst_37 {fastmathFlags = #llvm.fastmath<none>} : f64
      %76 = arith.subf %cst_38, %75 {fastmathFlags = #llvm.fastmath<none>} : f64
      %77 = arith.mulf %6, %76 {fastmathFlags = #llvm.fastmath<none>} : f64
      %78 = arith.addf %77, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f64
      %79 = arith.mulf %6, %78 {fastmathFlags = #llvm.fastmath<none>} : f64
      %80 = arith.addf %79, %cst_40 {fastmathFlags = #llvm.fastmath<none>} : f64
      %81 = arith.mulf %6, %80 {fastmathFlags = #llvm.fastmath<none>} : f64
      %82 = arith.addf %81, %74 {fastmathFlags = #llvm.fastmath<none>} : f64
      %83 = arith.addf %82, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f64
      %84 = arith.mulf %3, %83 {fastmathFlags = #llvm.fastmath<none>} : f64
      %85 = arith.mulf %6, %cst_42 {fastmathFlags = #llvm.fastmath<none>} : f64
      %86 = arith.subf %cst_43, %85 {fastmathFlags = #llvm.fastmath<none>} : f64
      %87 = arith.mulf %6, %86 {fastmathFlags = #llvm.fastmath<none>} : f64
      %88 = arith.addf %87, %cst_44 {fastmathFlags = #llvm.fastmath<none>} : f64
      %89 = arith.mulf %6, %88 {fastmathFlags = #llvm.fastmath<none>} : f64
      %90 = arith.addf %89, %cst_45 {fastmathFlags = #llvm.fastmath<none>} : f64
      %91 = arith.mulf %6, %90 {fastmathFlags = #llvm.fastmath<none>} : f64
      %92 = arith.addf %91, %cst_46 {fastmathFlags = #llvm.fastmath<none>} : f64
      %93 = arith.mulf %6, %92 {fastmathFlags = #llvm.fastmath<none>} : f64
      %94 = arith.addf %93, %84 {fastmathFlags = #llvm.fastmath<none>} : f64
      %95 = arith.addf %94, %cst_47 {fastmathFlags = #llvm.fastmath<none>} : f64
      %96 = arith.mulf %3, %95 {fastmathFlags = #llvm.fastmath<none>} : f64
      %97 = arith.mulf %6, %cst_48 {fastmathFlags = #llvm.fastmath<none>} : f64
      %98 = arith.subf %cst_49, %97 {fastmathFlags = #llvm.fastmath<none>} : f64
      %99 = arith.mulf %6, %98 {fastmathFlags = #llvm.fastmath<none>} : f64
      %100 = arith.addf %99, %cst_50 {fastmathFlags = #llvm.fastmath<none>} : f64
      %101 = arith.mulf %6, %100 {fastmathFlags = #llvm.fastmath<none>} : f64
      %102 = arith.addf %101, %cst_51 {fastmathFlags = #llvm.fastmath<none>} : f64
      %103 = arith.mulf %6, %102 {fastmathFlags = #llvm.fastmath<none>} : f64
      %104 = arith.addf %103, %cst_52 {fastmathFlags = #llvm.fastmath<none>} : f64
      %105 = arith.mulf %6, %104 {fastmathFlags = #llvm.fastmath<none>} : f64
      %106 = arith.addf %105, %cst_53 {fastmathFlags = #llvm.fastmath<none>} : f64
      %107 = arith.mulf %6, %106 {fastmathFlags = #llvm.fastmath<none>} : f64
      %108 = arith.addf %107, %96 {fastmathFlags = #llvm.fastmath<none>} : f64
      %109 = arith.addf %108, %cst_54 {fastmathFlags = #llvm.fastmath<none>} : f64
      %110 = arith.addf %55, %109 {fastmathFlags = #llvm.fastmath<none>} : f64
      %111 = arith.subf %110, %cst_55 {fastmathFlags = #llvm.fastmath<none>} : f64
      %112 = arith.mulf %111, %cst_56 {fastmathFlags = #llvm.fastmath<none>} : f64
      %113 = arith.divf %112, %cst_55 {fastmathFlags = #llvm.fastmath<none>} : f64
      %114 = arith.negf %113 {fastmathFlags = #llvm.fastmath<none>} : f64
      %115 = affine.load %arg3[27, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
      %116 = affine.load %arg4[27, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
      %117 = affine.load %arg2[28] {alignment = 32 : i64, ordering = 0 : i64 } : memref<35xf64, 1>
      %118 = arith.addf %2, %117 {fastmathFlags = #llvm.fastmath<none>} : f64
      %119 = arith.divf %115, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      %120 = arith.addf %116, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
      %121 = arith.divf %120, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
      %122 = math.sqrt %121 : f64
      %123 = arith.negf %118 {fastmathFlags = #llvm.fastmath<none>} : f64
      %124 = arith.divf %123, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
      %125 = arith.mulf %119, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
      %126 = arith.mulf %122, %cst_4 {fastmathFlags = #llvm.fastmath<none>} : f64
      %127 = arith.subf %125, %126 {fastmathFlags = #llvm.fastmath<none>} : f64
      %128 = arith.addf %127, %cst_5 {fastmathFlags = #llvm.fastmath<none>} : f64
      %129 = arith.mulf %124, %128 {fastmathFlags = #llvm.fastmath<none>} : f64
      %130 = arith.mulf %119, %cst_6 {fastmathFlags = #llvm.fastmath<none>} : f64
      %131 = arith.mulf %122, %cst_7 {fastmathFlags = #llvm.fastmath<none>} : f64
      %132 = arith.subf %130, %131 {fastmathFlags = #llvm.fastmath<none>} : f64
      %133 = arith.addf %132, %cst_8 {fastmathFlags = #llvm.fastmath<none>} : f64
      %134 = arith.mulf %119, %133 {fastmathFlags = #llvm.fastmath<none>} : f64
      %135 = arith.mulf %122, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f64
      %136 = arith.addf %135, %cst_10 {fastmathFlags = #llvm.fastmath<none>} : f64
      %137 = arith.mulf %122, %136 {fastmathFlags = #llvm.fastmath<none>} : f64
      %138 = arith.addf %137, %134 {fastmathFlags = #llvm.fastmath<none>} : f64
      %139 = arith.addf %138, %cst_11 {fastmathFlags = #llvm.fastmath<none>} : f64
      %140 = arith.addf %129, %139 {fastmathFlags = #llvm.fastmath<none>} : f64
      %141 = arith.mulf %124, %140 {fastmathFlags = #llvm.fastmath<none>} : f64
      %142 = arith.mulf %119, %cst_12 {fastmathFlags = #llvm.fastmath<none>} : f64
      %143 = arith.mulf %122, %cst_13 {fastmathFlags = #llvm.fastmath<none>} : f64
      %144 = arith.subf %142, %143 {fastmathFlags = #llvm.fastmath<none>} : f64
      %145 = arith.addf %144, %cst_14 {fastmathFlags = #llvm.fastmath<none>} : f64
      %146 = arith.mulf %119, %145 {fastmathFlags = #llvm.fastmath<none>} : f64
      %147 = arith.mulf %122, %cst_15 {fastmathFlags = #llvm.fastmath<none>} : f64
      %148 = arith.subf %cst_16, %147 {fastmathFlags = #llvm.fastmath<none>} : f64
      %149 = arith.mulf %122, %148 {fastmathFlags = #llvm.fastmath<none>} : f64
      %150 = arith.addf %149, %146 {fastmathFlags = #llvm.fastmath<none>} : f64
      %151 = arith.addf %150, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f64
      %152 = arith.mulf %119, %151 {fastmathFlags = #llvm.fastmath<none>} : f64
      %153 = arith.mulf %122, %cst_18 {fastmathFlags = #llvm.fastmath<none>} : f64
      %154 = arith.subf %cst_19, %153 {fastmathFlags = #llvm.fastmath<none>} : f64
      %155 = arith.mulf %122, %154 {fastmathFlags = #llvm.fastmath<none>} : f64
      %156 = arith.addf %155, %cst_20 {fastmathFlags = #llvm.fastmath<none>} : f64
      %157 = arith.mulf %122, %156 {fastmathFlags = #llvm.fastmath<none>} : f64
      %158 = arith.addf %157, %152 {fastmathFlags = #llvm.fastmath<none>} : f64
      %159 = arith.addf %158, %cst_21 {fastmathFlags = #llvm.fastmath<none>} : f64
      %160 = arith.mulf %119, %159 {fastmathFlags = #llvm.fastmath<none>} : f64
      %161 = arith.mulf %122, %cst_22 {fastmathFlags = #llvm.fastmath<none>} : f64
      %162 = arith.addf %161, %cst_23 {fastmathFlags = #llvm.fastmath<none>} : f64
      %163 = arith.mulf %122, %162 {fastmathFlags = #llvm.fastmath<none>} : f64
      %164 = arith.addf %163, %cst_24 {fastmathFlags = #llvm.fastmath<none>} : f64
      %165 = arith.mulf %122, %164 {fastmathFlags = #llvm.fastmath<none>} : f64
      %166 = arith.addf %165, %cst_25 {fastmathFlags = #llvm.fastmath<none>} : f64
      %167 = arith.mulf %122, %166 {fastmathFlags = #llvm.fastmath<none>} : f64
      %168 = arith.addf %167, %160 {fastmathFlags = #llvm.fastmath<none>} : f64
      %169 = arith.addf %168, %cst_26 {fastmathFlags = #llvm.fastmath<none>} : f64
      %170 = arith.addf %141, %169 {fastmathFlags = #llvm.fastmath<none>} : f64
      %171 = arith.mulf %124, %170 {fastmathFlags = #llvm.fastmath<none>} : f64
      %172 = arith.mulf %119, %cst_27 {fastmathFlags = #llvm.fastmath<none>} : f64
      %173 = arith.mulf %122, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f64
      %174 = arith.subf %173, %172 {fastmathFlags = #llvm.fastmath<none>} : f64
      %175 = arith.addf %174, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f64
      %176 = arith.mulf %119, %175 {fastmathFlags = #llvm.fastmath<none>} : f64
      %177 = arith.mulf %122, %cst_30 {fastmathFlags = #llvm.fastmath<none>} : f64
      %178 = arith.subf %cst_31, %177 {fastmathFlags = #llvm.fastmath<none>} : f64
      %179 = arith.mulf %122, %178 {fastmathFlags = #llvm.fastmath<none>} : f64
      %180 = arith.addf %179, %176 {fastmathFlags = #llvm.fastmath<none>} : f64
      %181 = arith.addf %180, %cst_32 {fastmathFlags = #llvm.fastmath<none>} : f64
      %182 = arith.mulf %119, %181 {fastmathFlags = #llvm.fastmath<none>} : f64
      %183 = arith.mulf %122, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f64
      %184 = arith.subf %cst_34, %183 {fastmathFlags = #llvm.fastmath<none>} : f64
      %185 = arith.mulf %122, %184 {fastmathFlags = #llvm.fastmath<none>} : f64
      %186 = arith.addf %185, %cst_35 {fastmathFlags = #llvm.fastmath<none>} : f64
      %187 = arith.mulf %122, %186 {fastmathFlags = #llvm.fastmath<none>} : f64
      %188 = arith.addf %187, %182 {fastmathFlags = #llvm.fastmath<none>} : f64
      %189 = arith.addf %188, %cst_36 {fastmathFlags = #llvm.fastmath<none>} : f64
      %190 = arith.mulf %119, %189 {fastmathFlags = #llvm.fastmath<none>} : f64
      %191 = arith.mulf %122, %cst_37 {fastmathFlags = #llvm.fastmath<none>} : f64
      %192 = arith.subf %cst_38, %191 {fastmathFlags = #llvm.fastmath<none>} : f64
      %193 = arith.mulf %122, %192 {fastmathFlags = #llvm.fastmath<none>} : f64
      %194 = arith.addf %193, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f64
      %195 = arith.mulf %122, %194 {fastmathFlags = #llvm.fastmath<none>} : f64
      %196 = arith.addf %195, %cst_40 {fastmathFlags = #llvm.fastmath<none>} : f64
      %197 = arith.mulf %122, %196 {fastmathFlags = #llvm.fastmath<none>} : f64
      %198 = arith.addf %197, %190 {fastmathFlags = #llvm.fastmath<none>} : f64
      %199 = arith.addf %198, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f64
      %200 = arith.mulf %119, %199 {fastmathFlags = #llvm.fastmath<none>} : f64
      %201 = arith.mulf %122, %cst_42 {fastmathFlags = #llvm.fastmath<none>} : f64
      %202 = arith.subf %cst_43, %201 {fastmathFlags = #llvm.fastmath<none>} : f64
      %203 = arith.mulf %122, %202 {fastmathFlags = #llvm.fastmath<none>} : f64
      %204 = arith.addf %203, %cst_44 {fastmathFlags = #llvm.fastmath<none>} : f64
      %205 = arith.mulf %122, %204 {fastmathFlags = #llvm.fastmath<none>} : f64
      %206 = arith.addf %205, %cst_45 {fastmathFlags = #llvm.fastmath<none>} : f64
      %207 = arith.mulf %122, %206 {fastmathFlags = #llvm.fastmath<none>} : f64
      %208 = arith.addf %207, %cst_46 {fastmathFlags = #llvm.fastmath<none>} : f64
      %209 = arith.mulf %122, %208 {fastmathFlags = #llvm.fastmath<none>} : f64
      %210 = arith.addf %209, %200 {fastmathFlags = #llvm.fastmath<none>} : f64
      %211 = arith.addf %210, %cst_47 {fastmathFlags = #llvm.fastmath<none>} : f64
      %212 = arith.mulf %119, %211 {fastmathFlags = #llvm.fastmath<none>} : f64
      %213 = arith.mulf %122, %cst_48 {fastmathFlags = #llvm.fastmath<none>} : f64
      %214 = arith.subf %cst_49, %213 {fastmathFlags = #llvm.fastmath<none>} : f64
      %215 = arith.mulf %122, %214 {fastmathFlags = #llvm.fastmath<none>} : f64
      %216 = arith.addf %215, %cst_50 {fastmathFlags = #llvm.fastmath<none>} : f64
      %217 = arith.mulf %122, %216 {fastmathFlags = #llvm.fastmath<none>} : f64
      %218 = arith.addf %217, %cst_51 {fastmathFlags = #llvm.fastmath<none>} : f64
      %219 = arith.mulf %122, %218 {fastmathFlags = #llvm.fastmath<none>} : f64
      %220 = arith.addf %219, %cst_52 {fastmathFlags = #llvm.fastmath<none>} : f64
      %221 = arith.mulf %122, %220 {fastmathFlags = #llvm.fastmath<none>} : f64
      %222 = arith.addf %221, %cst_53 {fastmathFlags = #llvm.fastmath<none>} : f64
      %223 = arith.mulf %122, %222 {fastmathFlags = #llvm.fastmath<none>} : f64
      %224 = arith.addf %223, %212 {fastmathFlags = #llvm.fastmath<none>} : f64
      %225 = arith.addf %224, %cst_54 {fastmathFlags = #llvm.fastmath<none>} : f64
      %226 = arith.addf %171, %225 {fastmathFlags = #llvm.fastmath<none>} : f64
      %227 = arith.subf %226, %cst_55 {fastmathFlags = #llvm.fastmath<none>} : f64
      %228 = arith.mulf %227, %cst_56 {fastmathFlags = #llvm.fastmath<none>} : f64
      %229 = arith.divf %228, %cst_55 {fastmathFlags = #llvm.fastmath<none>} : f64
      %230 = arith.negf %229 {fastmathFlags = #llvm.fastmath<none>} : f64
      %231 = arith.addf %114, %230 {fastmathFlags = #llvm.fastmath<none>} : f64
      %232 = arith.mulf %231, %cst_57 {fastmathFlags = #llvm.fastmath<none>} : f64
      %233 = arith.negf %232 {fastmathFlags = #llvm.fastmath<none>} : f64
      %234 = arith.mulf %117, %233 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %234, %arg0[26, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
      affine.for %arg7 = 0 to 19 {
        %235 = affine.load %arg0[-%arg7 + 26, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
        %236 = affine.load %arg3[-%arg7 + 25, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
        %237 = affine.load %arg4[-%arg7 + 25, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
        %238 = affine.load %arg1[-%arg7 + 25] {alignment = 8 : i64, ordering = 0 : i64 } : memref<34xf64, 1>
        %239 = arith.divf %236, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
        %240 = arith.addf %237, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
        %241 = arith.divf %240, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
        %242 = math.sqrt %241 : f64
        %243 = arith.negf %238 {fastmathFlags = #llvm.fastmath<none>} : f64
        %244 = arith.divf %243, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
        %245 = arith.mulf %239, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
        %246 = arith.mulf %242, %cst_4 {fastmathFlags = #llvm.fastmath<none>} : f64
        %247 = arith.subf %245, %246 {fastmathFlags = #llvm.fastmath<none>} : f64
        %248 = arith.addf %247, %cst_5 {fastmathFlags = #llvm.fastmath<none>} : f64
        %249 = arith.mulf %248, %244 {fastmathFlags = #llvm.fastmath<none>} : f64
        %250 = arith.mulf %239, %cst_6 {fastmathFlags = #llvm.fastmath<none>} : f64
        %251 = arith.mulf %242, %cst_7 {fastmathFlags = #llvm.fastmath<none>} : f64
        %252 = arith.subf %250, %251 {fastmathFlags = #llvm.fastmath<none>} : f64
        %253 = arith.addf %252, %cst_8 {fastmathFlags = #llvm.fastmath<none>} : f64
        %254 = arith.mulf %239, %253 {fastmathFlags = #llvm.fastmath<none>} : f64
        %255 = arith.mulf %242, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f64
        %256 = arith.addf %255, %cst_10 {fastmathFlags = #llvm.fastmath<none>} : f64
        %257 = arith.mulf %242, %256 {fastmathFlags = #llvm.fastmath<none>} : f64
        %258 = arith.addf %257, %254 {fastmathFlags = #llvm.fastmath<none>} : f64
        %259 = arith.addf %258, %cst_11 {fastmathFlags = #llvm.fastmath<none>} : f64
        %260 = arith.addf %249, %259 {fastmathFlags = #llvm.fastmath<none>} : f64
        %261 = arith.mulf %244, %260 {fastmathFlags = #llvm.fastmath<none>} : f64
        %262 = arith.mulf %239, %cst_12 {fastmathFlags = #llvm.fastmath<none>} : f64
        %263 = arith.mulf %242, %cst_13 {fastmathFlags = #llvm.fastmath<none>} : f64
        %264 = arith.subf %262, %263 {fastmathFlags = #llvm.fastmath<none>} : f64
        %265 = arith.addf %264, %cst_14 {fastmathFlags = #llvm.fastmath<none>} : f64
        %266 = arith.mulf %239, %265 {fastmathFlags = #llvm.fastmath<none>} : f64
        %267 = arith.mulf %242, %cst_15 {fastmathFlags = #llvm.fastmath<none>} : f64
        %268 = arith.subf %cst_16, %267 {fastmathFlags = #llvm.fastmath<none>} : f64
        %269 = arith.mulf %242, %268 {fastmathFlags = #llvm.fastmath<none>} : f64
        %270 = arith.addf %269, %266 {fastmathFlags = #llvm.fastmath<none>} : f64
        %271 = arith.addf %270, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f64
        %272 = arith.mulf %239, %271 {fastmathFlags = #llvm.fastmath<none>} : f64
        %273 = arith.mulf %242, %cst_18 {fastmathFlags = #llvm.fastmath<none>} : f64
        %274 = arith.subf %cst_19, %273 {fastmathFlags = #llvm.fastmath<none>} : f64
        %275 = arith.mulf %242, %274 {fastmathFlags = #llvm.fastmath<none>} : f64
        %276 = arith.addf %275, %cst_20 {fastmathFlags = #llvm.fastmath<none>} : f64
        %277 = arith.mulf %242, %276 {fastmathFlags = #llvm.fastmath<none>} : f64
        %278 = arith.addf %277, %272 {fastmathFlags = #llvm.fastmath<none>} : f64
        %279 = arith.addf %278, %cst_21 {fastmathFlags = #llvm.fastmath<none>} : f64
        %280 = arith.mulf %239, %279 {fastmathFlags = #llvm.fastmath<none>} : f64
        %281 = arith.mulf %242, %cst_22 {fastmathFlags = #llvm.fastmath<none>} : f64
        %282 = arith.addf %281, %cst_23 {fastmathFlags = #llvm.fastmath<none>} : f64
        %283 = arith.mulf %242, %282 {fastmathFlags = #llvm.fastmath<none>} : f64
        %284 = arith.addf %283, %cst_24 {fastmathFlags = #llvm.fastmath<none>} : f64
        %285 = arith.mulf %242, %284 {fastmathFlags = #llvm.fastmath<none>} : f64
        %286 = arith.addf %285, %cst_25 {fastmathFlags = #llvm.fastmath<none>} : f64
        %287 = arith.mulf %242, %286 {fastmathFlags = #llvm.fastmath<none>} : f64
        %288 = arith.addf %287, %280 {fastmathFlags = #llvm.fastmath<none>} : f64
        %289 = arith.addf %288, %cst_26 {fastmathFlags = #llvm.fastmath<none>} : f64
        %290 = arith.addf %261, %289 {fastmathFlags = #llvm.fastmath<none>} : f64
        %291 = arith.mulf %244, %290 {fastmathFlags = #llvm.fastmath<none>} : f64
        %292 = arith.mulf %239, %cst_27 {fastmathFlags = #llvm.fastmath<none>} : f64
        %293 = arith.mulf %242, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f64
        %294 = arith.subf %293, %292 {fastmathFlags = #llvm.fastmath<none>} : f64
        %295 = arith.addf %294, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f64
        %296 = arith.mulf %239, %295 {fastmathFlags = #llvm.fastmath<none>} : f64
        %297 = arith.mulf %242, %cst_30 {fastmathFlags = #llvm.fastmath<none>} : f64
        %298 = arith.subf %cst_31, %297 {fastmathFlags = #llvm.fastmath<none>} : f64
        %299 = arith.mulf %242, %298 {fastmathFlags = #llvm.fastmath<none>} : f64
        %300 = arith.addf %299, %296 {fastmathFlags = #llvm.fastmath<none>} : f64
        %301 = arith.addf %300, %cst_32 {fastmathFlags = #llvm.fastmath<none>} : f64
        %302 = arith.mulf %239, %301 {fastmathFlags = #llvm.fastmath<none>} : f64
        %303 = arith.mulf %242, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f64
        %304 = arith.subf %cst_34, %303 {fastmathFlags = #llvm.fastmath<none>} : f64
        %305 = arith.mulf %242, %304 {fastmathFlags = #llvm.fastmath<none>} : f64
        %306 = arith.addf %305, %cst_35 {fastmathFlags = #llvm.fastmath<none>} : f64
        %307 = arith.mulf %242, %306 {fastmathFlags = #llvm.fastmath<none>} : f64
        %308 = arith.addf %307, %302 {fastmathFlags = #llvm.fastmath<none>} : f64
        %309 = arith.addf %308, %cst_36 {fastmathFlags = #llvm.fastmath<none>} : f64
        %310 = arith.mulf %239, %309 {fastmathFlags = #llvm.fastmath<none>} : f64
        %311 = arith.mulf %242, %cst_37 {fastmathFlags = #llvm.fastmath<none>} : f64
        %312 = arith.subf %cst_38, %311 {fastmathFlags = #llvm.fastmath<none>} : f64
        %313 = arith.mulf %242, %312 {fastmathFlags = #llvm.fastmath<none>} : f64
        %314 = arith.addf %313, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f64
        %315 = arith.mulf %242, %314 {fastmathFlags = #llvm.fastmath<none>} : f64
        %316 = arith.addf %315, %cst_40 {fastmathFlags = #llvm.fastmath<none>} : f64
        %317 = arith.mulf %242, %316 {fastmathFlags = #llvm.fastmath<none>} : f64
        %318 = arith.addf %317, %310 {fastmathFlags = #llvm.fastmath<none>} : f64
        %319 = arith.addf %318, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f64
        %320 = arith.mulf %239, %319 {fastmathFlags = #llvm.fastmath<none>} : f64
        %321 = arith.mulf %242, %cst_42 {fastmathFlags = #llvm.fastmath<none>} : f64
        %322 = arith.subf %cst_43, %321 {fastmathFlags = #llvm.fastmath<none>} : f64
        %323 = arith.mulf %242, %322 {fastmathFlags = #llvm.fastmath<none>} : f64
        %324 = arith.addf %323, %cst_44 {fastmathFlags = #llvm.fastmath<none>} : f64
        %325 = arith.mulf %242, %324 {fastmathFlags = #llvm.fastmath<none>} : f64
        %326 = arith.addf %325, %cst_45 {fastmathFlags = #llvm.fastmath<none>} : f64
        %327 = arith.mulf %242, %326 {fastmathFlags = #llvm.fastmath<none>} : f64
        %328 = arith.addf %327, %cst_46 {fastmathFlags = #llvm.fastmath<none>} : f64
        %329 = arith.mulf %242, %328 {fastmathFlags = #llvm.fastmath<none>} : f64
        %330 = arith.addf %329, %320 {fastmathFlags = #llvm.fastmath<none>} : f64
        %331 = arith.addf %330, %cst_47 {fastmathFlags = #llvm.fastmath<none>} : f64
        %332 = arith.mulf %239, %331 {fastmathFlags = #llvm.fastmath<none>} : f64
        %333 = arith.mulf %242, %cst_48 {fastmathFlags = #llvm.fastmath<none>} : f64
        %334 = arith.subf %cst_49, %333 {fastmathFlags = #llvm.fastmath<none>} : f64
        %335 = arith.mulf %242, %334 {fastmathFlags = #llvm.fastmath<none>} : f64
        %336 = arith.addf %335, %cst_50 {fastmathFlags = #llvm.fastmath<none>} : f64
        %337 = arith.mulf %242, %336 {fastmathFlags = #llvm.fastmath<none>} : f64
        %338 = arith.addf %337, %cst_51 {fastmathFlags = #llvm.fastmath<none>} : f64
        %339 = arith.mulf %242, %338 {fastmathFlags = #llvm.fastmath<none>} : f64
        %340 = arith.addf %339, %cst_52 {fastmathFlags = #llvm.fastmath<none>} : f64
        %341 = arith.mulf %242, %340 {fastmathFlags = #llvm.fastmath<none>} : f64
        %342 = arith.addf %341, %cst_53 {fastmathFlags = #llvm.fastmath<none>} : f64
        %343 = arith.mulf %242, %342 {fastmathFlags = #llvm.fastmath<none>} : f64
        %344 = arith.addf %343, %332 {fastmathFlags = #llvm.fastmath<none>} : f64
        %345 = arith.addf %344, %cst_54 {fastmathFlags = #llvm.fastmath<none>} : f64
        %346 = arith.addf %291, %345 {fastmathFlags = #llvm.fastmath<none>} : f64
        %347 = arith.subf %346, %cst_55 {fastmathFlags = #llvm.fastmath<none>} : f64
        %348 = arith.mulf %347, %cst_56 {fastmathFlags = #llvm.fastmath<none>} : f64
        %349 = arith.divf %348, %cst_55 {fastmathFlags = #llvm.fastmath<none>} : f64
        %350 = arith.negf %349 {fastmathFlags = #llvm.fastmath<none>} : f64
        %351 = affine.load %arg3[-%arg7 + 26, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
        %352 = affine.load %arg4[-%arg7 + 26, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
        %353 = affine.load %arg1[-%arg7 + 26] {alignment = 8 : i64, ordering = 0 : i64 } : memref<34xf64, 1>
        %354 = arith.divf %351, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
        %355 = arith.addf %352, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
        %356 = arith.divf %355, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
        %357 = math.sqrt %356 : f64
        %358 = arith.negf %353 {fastmathFlags = #llvm.fastmath<none>} : f64
        %359 = arith.divf %358, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
        %360 = arith.mulf %354, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
        %361 = arith.mulf %357, %cst_4 {fastmathFlags = #llvm.fastmath<none>} : f64
        %362 = arith.subf %360, %361 {fastmathFlags = #llvm.fastmath<none>} : f64
        %363 = arith.addf %362, %cst_5 {fastmathFlags = #llvm.fastmath<none>} : f64
        %364 = arith.mulf %363, %359 {fastmathFlags = #llvm.fastmath<none>} : f64
        %365 = arith.mulf %354, %cst_6 {fastmathFlags = #llvm.fastmath<none>} : f64
        %366 = arith.mulf %357, %cst_7 {fastmathFlags = #llvm.fastmath<none>} : f64
        %367 = arith.subf %365, %366 {fastmathFlags = #llvm.fastmath<none>} : f64
        %368 = arith.addf %367, %cst_8 {fastmathFlags = #llvm.fastmath<none>} : f64
        %369 = arith.mulf %354, %368 {fastmathFlags = #llvm.fastmath<none>} : f64
        %370 = arith.mulf %357, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f64
        %371 = arith.addf %370, %cst_10 {fastmathFlags = #llvm.fastmath<none>} : f64
        %372 = arith.mulf %357, %371 {fastmathFlags = #llvm.fastmath<none>} : f64
        %373 = arith.addf %372, %369 {fastmathFlags = #llvm.fastmath<none>} : f64
        %374 = arith.addf %373, %cst_11 {fastmathFlags = #llvm.fastmath<none>} : f64
        %375 = arith.addf %364, %374 {fastmathFlags = #llvm.fastmath<none>} : f64
        %376 = arith.mulf %359, %375 {fastmathFlags = #llvm.fastmath<none>} : f64
        %377 = arith.mulf %354, %cst_12 {fastmathFlags = #llvm.fastmath<none>} : f64
        %378 = arith.mulf %357, %cst_13 {fastmathFlags = #llvm.fastmath<none>} : f64
        %379 = arith.subf %377, %378 {fastmathFlags = #llvm.fastmath<none>} : f64
        %380 = arith.addf %379, %cst_14 {fastmathFlags = #llvm.fastmath<none>} : f64
        %381 = arith.mulf %354, %380 {fastmathFlags = #llvm.fastmath<none>} : f64
        %382 = arith.mulf %357, %cst_15 {fastmathFlags = #llvm.fastmath<none>} : f64
        %383 = arith.subf %cst_16, %382 {fastmathFlags = #llvm.fastmath<none>} : f64
        %384 = arith.mulf %357, %383 {fastmathFlags = #llvm.fastmath<none>} : f64
        %385 = arith.addf %384, %381 {fastmathFlags = #llvm.fastmath<none>} : f64
        %386 = arith.addf %385, %cst_17 {fastmathFlags = #llvm.fastmath<none>} : f64
        %387 = arith.mulf %354, %386 {fastmathFlags = #llvm.fastmath<none>} : f64
        %388 = arith.mulf %357, %cst_18 {fastmathFlags = #llvm.fastmath<none>} : f64
        %389 = arith.subf %cst_19, %388 {fastmathFlags = #llvm.fastmath<none>} : f64
        %390 = arith.mulf %357, %389 {fastmathFlags = #llvm.fastmath<none>} : f64
        %391 = arith.addf %390, %cst_20 {fastmathFlags = #llvm.fastmath<none>} : f64
        %392 = arith.mulf %357, %391 {fastmathFlags = #llvm.fastmath<none>} : f64
        %393 = arith.addf %392, %387 {fastmathFlags = #llvm.fastmath<none>} : f64
        %394 = arith.addf %393, %cst_21 {fastmathFlags = #llvm.fastmath<none>} : f64
        %395 = arith.mulf %354, %394 {fastmathFlags = #llvm.fastmath<none>} : f64
        %396 = arith.mulf %357, %cst_22 {fastmathFlags = #llvm.fastmath<none>} : f64
        %397 = arith.addf %396, %cst_23 {fastmathFlags = #llvm.fastmath<none>} : f64
        %398 = arith.mulf %357, %397 {fastmathFlags = #llvm.fastmath<none>} : f64
        %399 = arith.addf %398, %cst_24 {fastmathFlags = #llvm.fastmath<none>} : f64
        %400 = arith.mulf %357, %399 {fastmathFlags = #llvm.fastmath<none>} : f64
        %401 = arith.addf %400, %cst_25 {fastmathFlags = #llvm.fastmath<none>} : f64
        %402 = arith.mulf %357, %401 {fastmathFlags = #llvm.fastmath<none>} : f64
        %403 = arith.addf %402, %395 {fastmathFlags = #llvm.fastmath<none>} : f64
        %404 = arith.addf %403, %cst_26 {fastmathFlags = #llvm.fastmath<none>} : f64
        %405 = arith.addf %376, %404 {fastmathFlags = #llvm.fastmath<none>} : f64
        %406 = arith.mulf %359, %405 {fastmathFlags = #llvm.fastmath<none>} : f64
        %407 = arith.mulf %354, %cst_27 {fastmathFlags = #llvm.fastmath<none>} : f64
        %408 = arith.mulf %357, %cst_28 {fastmathFlags = #llvm.fastmath<none>} : f64
        %409 = arith.subf %408, %407 {fastmathFlags = #llvm.fastmath<none>} : f64
        %410 = arith.addf %409, %cst_29 {fastmathFlags = #llvm.fastmath<none>} : f64
        %411 = arith.mulf %354, %410 {fastmathFlags = #llvm.fastmath<none>} : f64
        %412 = arith.mulf %357, %cst_30 {fastmathFlags = #llvm.fastmath<none>} : f64
        %413 = arith.subf %cst_31, %412 {fastmathFlags = #llvm.fastmath<none>} : f64
        %414 = arith.mulf %357, %413 {fastmathFlags = #llvm.fastmath<none>} : f64
        %415 = arith.addf %414, %411 {fastmathFlags = #llvm.fastmath<none>} : f64
        %416 = arith.addf %415, %cst_32 {fastmathFlags = #llvm.fastmath<none>} : f64
        %417 = arith.mulf %354, %416 {fastmathFlags = #llvm.fastmath<none>} : f64
        %418 = arith.mulf %357, %cst_33 {fastmathFlags = #llvm.fastmath<none>} : f64
        %419 = arith.subf %cst_34, %418 {fastmathFlags = #llvm.fastmath<none>} : f64
        %420 = arith.mulf %357, %419 {fastmathFlags = #llvm.fastmath<none>} : f64
        %421 = arith.addf %420, %cst_35 {fastmathFlags = #llvm.fastmath<none>} : f64
        %422 = arith.mulf %357, %421 {fastmathFlags = #llvm.fastmath<none>} : f64
        %423 = arith.addf %422, %417 {fastmathFlags = #llvm.fastmath<none>} : f64
        %424 = arith.addf %423, %cst_36 {fastmathFlags = #llvm.fastmath<none>} : f64
        %425 = arith.mulf %354, %424 {fastmathFlags = #llvm.fastmath<none>} : f64
        %426 = arith.mulf %357, %cst_37 {fastmathFlags = #llvm.fastmath<none>} : f64
        %427 = arith.subf %cst_38, %426 {fastmathFlags = #llvm.fastmath<none>} : f64
        %428 = arith.mulf %357, %427 {fastmathFlags = #llvm.fastmath<none>} : f64
        %429 = arith.addf %428, %cst_39 {fastmathFlags = #llvm.fastmath<none>} : f64
        %430 = arith.mulf %357, %429 {fastmathFlags = #llvm.fastmath<none>} : f64
        %431 = arith.addf %430, %cst_40 {fastmathFlags = #llvm.fastmath<none>} : f64
        %432 = arith.mulf %357, %431 {fastmathFlags = #llvm.fastmath<none>} : f64
        %433 = arith.addf %432, %425 {fastmathFlags = #llvm.fastmath<none>} : f64
        %434 = arith.addf %433, %cst_41 {fastmathFlags = #llvm.fastmath<none>} : f64
        %435 = arith.mulf %354, %434 {fastmathFlags = #llvm.fastmath<none>} : f64
        %436 = arith.mulf %357, %cst_42 {fastmathFlags = #llvm.fastmath<none>} : f64
        %437 = arith.subf %cst_43, %436 {fastmathFlags = #llvm.fastmath<none>} : f64
        %438 = arith.mulf %357, %437 {fastmathFlags = #llvm.fastmath<none>} : f64
        %439 = arith.addf %438, %cst_44 {fastmathFlags = #llvm.fastmath<none>} : f64
        %440 = arith.mulf %357, %439 {fastmathFlags = #llvm.fastmath<none>} : f64
        %441 = arith.addf %440, %cst_45 {fastmathFlags = #llvm.fastmath<none>} : f64
        %442 = arith.mulf %357, %441 {fastmathFlags = #llvm.fastmath<none>} : f64
        %443 = arith.addf %442, %cst_46 {fastmathFlags = #llvm.fastmath<none>} : f64
        %444 = arith.mulf %357, %443 {fastmathFlags = #llvm.fastmath<none>} : f64
        %445 = arith.addf %444, %435 {fastmathFlags = #llvm.fastmath<none>} : f64
        %446 = arith.addf %445, %cst_47 {fastmathFlags = #llvm.fastmath<none>} : f64
        %447 = arith.mulf %354, %446 {fastmathFlags = #llvm.fastmath<none>} : f64
        %448 = arith.mulf %357, %cst_48 {fastmathFlags = #llvm.fastmath<none>} : f64
        %449 = arith.subf %cst_49, %448 {fastmathFlags = #llvm.fastmath<none>} : f64
        %450 = arith.mulf %357, %449 {fastmathFlags = #llvm.fastmath<none>} : f64
        %451 = arith.addf %450, %cst_50 {fastmathFlags = #llvm.fastmath<none>} : f64
        %452 = arith.mulf %357, %451 {fastmathFlags = #llvm.fastmath<none>} : f64
        %453 = arith.addf %452, %cst_51 {fastmathFlags = #llvm.fastmath<none>} : f64
        %454 = arith.mulf %357, %453 {fastmathFlags = #llvm.fastmath<none>} : f64
        %455 = arith.addf %454, %cst_52 {fastmathFlags = #llvm.fastmath<none>} : f64
        %456 = arith.mulf %357, %455 {fastmathFlags = #llvm.fastmath<none>} : f64
        %457 = arith.addf %456, %cst_53 {fastmathFlags = #llvm.fastmath<none>} : f64
        %458 = arith.mulf %357, %457 {fastmathFlags = #llvm.fastmath<none>} : f64
        %459 = arith.addf %458, %447 {fastmathFlags = #llvm.fastmath<none>} : f64
        %460 = arith.addf %459, %cst_54 {fastmathFlags = #llvm.fastmath<none>} : f64
        %461 = arith.addf %406, %460 {fastmathFlags = #llvm.fastmath<none>} : f64
        %462 = arith.subf %461, %cst_55 {fastmathFlags = #llvm.fastmath<none>} : f64
        %463 = arith.mulf %462, %cst_56 {fastmathFlags = #llvm.fastmath<none>} : f64
        %464 = arith.divf %463, %cst_55 {fastmathFlags = #llvm.fastmath<none>} : f64
        %465 = arith.negf %464 {fastmathFlags = #llvm.fastmath<none>} : f64
        %466 = arith.addf %350, %465 {fastmathFlags = #llvm.fastmath<none>} : f64
        %467 = arith.mulf %466, %cst_57 {fastmathFlags = #llvm.fastmath<none>} : f64
        %468 = affine.load %arg2[-%arg7 + 27] {alignment = 8 : i64, ordering = 0 : i64 } : memref<35xf64, 1>
        %469 = arith.mulf %468, %467 {fastmathFlags = #llvm.fastmath<none>} : f64
        %470 = arith.subf %235, %469 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %470, %arg0[-%arg7 + 25, %arg5 + 6, %arg6 + 6] : memref<34x104x194xf64, 1>
      }
    }
    return
  }
}
