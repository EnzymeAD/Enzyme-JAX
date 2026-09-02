// RUN:enzymexlamlir-opt %s --raise-affine-to-stablehlo | FileCheck %s

#set = affine_set<(d0, d1) : (d1 + d0 * 16 - 36 >= 0)>
#set1 = affine_set<(d0) : (-d0 >= 0)>
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {
  func.func private @"##call__Z44gpu_solve_batched_tridiagonal_system_kernel_16CompilerMetadataI10StaticSizeI8_36__18_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E12_46__28__61_EE20AcousticTridiagLower23AcousticTridiagDiagonal20AcousticTridiagUpperSC_SA_IS9_Li3ELi1E12_36__18__50_E21LatitudeLongitudeGridI15CuTracedRNumberIS9_Li1EE7BoundedSK_SK_38TerrainFollowingVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_61__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_60__EESN_SP_11LinearDecayIS9_S8_IS9_Li3ESA_IS9_Li3ELi1E11_46__28__1_EES8_IS9_Li3ESA_IS9_Li3ELi1E11_47__28__1_EES8_IS9_Li3ESA_IS9_Li3ELi1E11_46__29__1_EEEES9_S9_S8_IS9_Li1ESA_IS9_Li1ELi1E5_47__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_46__EES9_S9_S8_IS9_Li1ESA_IS9_Li1ELi1E5_29__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_28__EES16_S16_S16_S16_S9_S9_v5Int64vEv5TupleIS8_IS9_Li3ESA_IS9_Li3ELi1E12_46__28__60_EES1B_S1B_SJ_SJ_SJ_vE10ZDirection#983$par172"(%arg0: memref<61x28x46xf32, 1>, %arg1: memref<50x18x36xf32, 1>, %arg2: memref<61xf32, 1>, %arg3: memref<60xf32, 1>, %arg4: memref<1x28x46xf32, 1>, %arg5: memref<60x28x46xf32, 1>, %arg6: memref<60x28x46xf32, 1>, %arg7: memref<60x28x46xf32, 1>, %arg8: memref<f32, 1>, %arg9: memref<f32, 1>, %arg10: memref<f32, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c-51_i64 = arith.constant -51 : i64
    %c-49_i64 = arith.constant -49 : i64
    %false = arith.constant false
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c4_i64 = arith.constant 4 : i64
    %c-37_i64 = arith.constant -37 : i64
    %c-36_i64 = arith.constant -36 : i64
    %c18_i64 = arith.constant 18 : i64
    %cst = arith.constant -4.99925263E-5 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c-50_i64 = arith.constant -50 : i64
    %cst_3 = arith.constant 1.1920929E-6 : f32
    affine.parallel (%arg11, %arg12, %arg13, %arg14) = (0, 0, 0, 0) to (2, 16, 3, 16) {
      %0 = arith.muli %arg12, %c16 overflow<nuw> : index
      %1 = arith.addi %0, %arg14 : index
      %2 = arith.addi %1, %c1 : index
      %3 = arith.index_cast %arg13 : index to i64
      %4 = arith.index_castui %2 : index to i64
      %5 = arith.muli %arg12, %c16 overflow<nuw> : index
      %6 = arith.addi %5, %arg14 : index
      %7 = arith.shrui %6, %c4 : index
      %8 = arith.index_castui %7 : index to i64
      %9 = arith.subi %3, %8 : i64
      %10 = arith.shli %9, %c4_i64 : i64
      %11 = arith.addi %10, %4 : i64
      %12 = arith.shli %arg11, %c4 : index
      %13 = arith.addi %12, %7 : index
      %14 = arith.addi %13, %c1 : index
      %15 = arith.index_castui %14 : index to i64
      %16 = arith.addi %11, %c-37_i64 : i64
      %17 = arith.cmpi ult, %16, %c-36_i64 : i64
      %18 = arith.cmpi ugt, %15, %c18_i64 : i64
      %19 = arith.ori %18, %17 : i1
      %20 = arith.xori %19, %true : i1
      scf.if %20 {
        %21 = affine.load %arg2[6] {alignment = 8 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<61xf32, 1>
        %22 = affine.load %arg4[%arg11 + (%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448, %arg11 * -12 + %arg12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<1x28x46xf32, 1>
        %23 = arith.mulf %22, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
        %24 = arith.addf %23, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %25 = arith.mulf %21, %24 {fastmathFlags = #llvm.fastmath<none>} : f32
        %26 = arith.divf %cst_0, %25 {fastmathFlags = #llvm.fastmath<none>} : f32
        %27 = affine.load %arg3[5] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
        %28 = arith.mulf %27, %24 {fastmathFlags = #llvm.fastmath<none>} : f32
        %29 = arith.divf %cst_0, %28 {fastmathFlags = #llvm.fastmath<none>} : f32
        %30 = affine.load %arg3[4] {alignment = 16 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
        %31 = arith.mulf %30, %24 {fastmathFlags = #llvm.fastmath<none>} : f32
        %32 = arith.divf %cst_0, %31 {fastmathFlags = #llvm.fastmath<none>} : f32
        %33 = affine.load %arg7[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %34 = affine.load %arg5[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %35 = arith.mulf %33, %34 {fastmathFlags = #llvm.fastmath<none>} : f32
        %36 = affine.load %arg7[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %37 = affine.load %arg5[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %38 = arith.mulf %36, %37 {fastmathFlags = #llvm.fastmath<none>} : f32
        %39 = affine.load %arg6[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %40 = affine.load %arg6[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %41 = affine.if #set(%arg13, %arg14) -> f32 {
          affine.yield %40 : f32
        } else {
          affine.yield %39 : f32
        }
        %42 = arith.addf %41, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
        %43 = arith.mulf %42, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
        %44 = affine.load %arg9[] : memref<f32, 1>
        %45 = arith.mulf %44, %44 {fastmathFlags = #llvm.fastmath<none>} : f32
        %46 = arith.mulf %29, %35 {fastmathFlags = #llvm.fastmath<none>} : f32
        %47 = arith.mulf %32, %38 {fastmathFlags = #llvm.fastmath<none>} : f32
        %48 = arith.addf %46, %47 {fastmathFlags = #llvm.fastmath<none>} : f32
        %49 = arith.mulf %45, %43 {fastmathFlags = #llvm.fastmath<none>} : f32
        %50 = arith.mulf %48, %49 {fastmathFlags = #llvm.fastmath<none>} : f32
        %51 = arith.mulf %26, %50 {fastmathFlags = #llvm.fastmath<none>} : f32
        %52 = arith.subf %29, %32 {fastmathFlags = #llvm.fastmath<none>} : f32
        %53 = affine.load %arg8[] : memref<f32, 1>
        %54 = arith.mulf %45, %53 {fastmathFlags = #llvm.fastmath<none>} : f32
        %55 = arith.mulf %52, %54 {fastmathFlags = #llvm.fastmath<none>} : f32
        %56 = arith.mulf %55, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
        %57 = arith.addf %29, %32 {fastmathFlags = #llvm.fastmath<none>} : f32
        %58 = affine.load %arg10[] : memref<f32, 1>
        %59 = arith.mulf %57, %58 {fastmathFlags = #llvm.fastmath<none>} : f32
        %60 = arith.mulf %26, %59 {fastmathFlags = #llvm.fastmath<none>} : f32
        %61 = arith.addf %56, %51 {fastmathFlags = #llvm.fastmath<none>} : f32
        %62 = arith.addf %60, %61 {fastmathFlags = #llvm.fastmath<none>} : f32
        %63 = math.copysign %cst_2, %62 : f32
        %64 = arith.addf %63, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %65 = affine.load %arg0[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        %66 = arith.divf %65, %64 {fastmathFlags = #llvm.fastmath<none>} : f32
        affine.store %66, %arg0[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        %67 = affine.load %arg0[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        %68 = affine.load %arg4[%arg11 + (%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448, %arg11 * -12 + %arg12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<1x28x46xf32, 1>
        %69 = arith.mulf %68, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
        %70 = arith.addf %69, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %71 = affine.if #set(%arg13, %arg14) -> i1 {
          affine.yield %true : i1
        } else {
          affine.yield %false : i1
        }
        %72 = affine.if #set(%arg13, %arg14) -> i1 {
          affine.yield %true : i1
        } else {
          affine.yield %false : i1
        }
        %73 = affine.load %arg9[] : memref<f32, 1>
        %74 = arith.mulf %73, %73 {fastmathFlags = #llvm.fastmath<none>} : f32
        %75 = arith.negf %74 {fastmathFlags = #llvm.fastmath<none>} : f32
        %76 = affine.load %arg8[] : memref<f32, 1>
        %77 = arith.mulf %76, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
        %78 = affine.load %arg10[] : memref<f32, 1>
        %79 = arith.negf %78 {fastmathFlags = #llvm.fastmath<none>} : f32
        %80 = arith.mulf %74, %76 {fastmathFlags = #llvm.fastmath<none>} : f32
        %81 = affine.if #set(%arg13, %arg14) -> i1 {
          affine.yield %true : i1
        } else {
          affine.yield %false : i1
        }
        %82:2 = affine.for %arg15 = 0 to 49 iter_args(%arg16 = %64, %arg17 = %67) -> (f32, f32) {
          %83 = arith.index_cast %arg15 : index to i64
          %84 = affine.load %arg2[%arg15 + 6] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<61xf32, 1>
          %85 = arith.mulf %84, %70 {fastmathFlags = #llvm.fastmath<none>} : f32
          %86 = arith.divf %cst_0, %85 {fastmathFlags = #llvm.fastmath<none>} : f32
          %87 = affine.load %arg3[%arg15 + 5] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
          %88 = arith.mulf %87, %70 {fastmathFlags = #llvm.fastmath<none>} : f32
          %89 = arith.divf %cst_0, %88 {fastmathFlags = #llvm.fastmath<none>} : f32
          %90 = affine.load %arg7[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %91 = affine.load %arg5[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %92 = arith.mulf %90, %91 {fastmathFlags = #llvm.fastmath<none>} : f32
          %93 = affine.load %arg6[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %94 = affine.load %arg6[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %95 = arith.addi %83, %c-49_i64 : i64
          %96 = arith.cmpi ult, %95, %c-50_i64 : i64
          %97 = arith.ori %71, %96 : i1
          %98 = arith.addi %83, %c-50_i64 : i64
          %99 = arith.cmpi ult, %98, %c-50_i64 : i64
          %100 = arith.ori %72, %99 : i1
          %101 = arith.select %97, %94, %93 {fastmathFlags = #llvm.fastmath<none>} : f32
          %102 = arith.select %100, %101, %94 {fastmathFlags = #llvm.fastmath<none>} : f32
          %103 = arith.addf %101, %102 {fastmathFlags = #llvm.fastmath<none>} : f32
          %104 = arith.mulf %103, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %105 = arith.mulf %92, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
          %106 = arith.mulf %105, %104 {fastmathFlags = #llvm.fastmath<none>} : f32
          %107 = arith.mulf %89, %106 {fastmathFlags = #llvm.fastmath<none>} : f32
          %108 = arith.mulf %86, %107 {fastmathFlags = #llvm.fastmath<none>} : f32
          %109 = arith.mulf %89, %77 {fastmathFlags = #llvm.fastmath<none>} : f32
          %110 = arith.mulf %109, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %111 = arith.mulf %89, %79 {fastmathFlags = #llvm.fastmath<none>} : f32
          %112 = arith.mulf %86, %111 {fastmathFlags = #llvm.fastmath<none>} : f32
          %113 = arith.addf %110, %108 {fastmathFlags = #llvm.fastmath<none>} : f32
          %114 = arith.addf %112, %113 {fastmathFlags = #llvm.fastmath<none>} : f32
          %115 = math.copysign %cst_2, %114 : f32
          %116 = affine.if #set1(%arg15) -> f32 {
            affine.yield %115 : f32
          } else {
            affine.yield %114 : f32
          }
          %117 = affine.load %arg2[%arg15 + 7] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<61xf32, 1>
          %118 = arith.mulf %70, %117 {fastmathFlags = #llvm.fastmath<none>} : f32
          %119 = arith.divf %cst_0, %118 {fastmathFlags = #llvm.fastmath<none>} : f32
          %120 = affine.load %arg3[%arg15 + 6] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
          %121 = arith.mulf %70, %120 {fastmathFlags = #llvm.fastmath<none>} : f32
          %122 = arith.divf %cst_0, %121 {fastmathFlags = #llvm.fastmath<none>} : f32
          %123 = affine.load %arg7[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %124 = affine.load %arg5[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %125 = arith.mulf %123, %124 {fastmathFlags = #llvm.fastmath<none>} : f32
          %126 = arith.mulf %122, %125 {fastmathFlags = #llvm.fastmath<none>} : f32
          %127 = arith.mulf %92, %89 {fastmathFlags = #llvm.fastmath<none>} : f32
          %128 = arith.addf %127, %126 {fastmathFlags = #llvm.fastmath<none>} : f32
          %129 = arith.mulf %74, %104 {fastmathFlags = #llvm.fastmath<none>} : f32
          %130 = arith.mulf %129, %128 {fastmathFlags = #llvm.fastmath<none>} : f32
          %131 = arith.mulf %119, %130 {fastmathFlags = #llvm.fastmath<none>} : f32
          %132 = arith.subf %122, %89 {fastmathFlags = #llvm.fastmath<none>} : f32
          %133 = arith.mulf %80, %132 {fastmathFlags = #llvm.fastmath<none>} : f32
          %134 = arith.mulf %133, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %135 = arith.addf %89, %122 {fastmathFlags = #llvm.fastmath<none>} : f32
          %136 = arith.mulf %78, %135 {fastmathFlags = #llvm.fastmath<none>} : f32
          %137 = arith.mulf %119, %136 {fastmathFlags = #llvm.fastmath<none>} : f32
          %138 = arith.addf %134, %131 {fastmathFlags = #llvm.fastmath<none>} : f32
          %139 = arith.addf %137, %138 {fastmathFlags = #llvm.fastmath<none>} : f32
          %140 = arith.addf %139, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
          %141 = affine.load %arg6[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %142 = arith.addi %83, %c-51_i64 : i64
          %143 = arith.cmpi ult, %142, %c-50_i64 : i64
          %144 = arith.ori %81, %143 : i1
          %145 = arith.select %100, %141, %94 {fastmathFlags = #llvm.fastmath<none>} : f32
          %146 = arith.select %144, %145, %141 {fastmathFlags = #llvm.fastmath<none>} : f32
          %147 = arith.addf %145, %146 {fastmathFlags = #llvm.fastmath<none>} : f32
          %148 = arith.mulf %147, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %149 = arith.mulf %105, %148 {fastmathFlags = #llvm.fastmath<none>} : f32
          %150 = arith.mulf %89, %149 {fastmathFlags = #llvm.fastmath<none>} : f32
          %151 = arith.mulf %119, %150 {fastmathFlags = #llvm.fastmath<none>} : f32
          %152 = arith.mulf %89, %80 {fastmathFlags = #llvm.fastmath<none>} : f32
          %153 = arith.mulf %152, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %154 = arith.mulf %111, %119 {fastmathFlags = #llvm.fastmath<none>} : f32
          %155 = arith.addf %153, %151 {fastmathFlags = #llvm.fastmath<none>} : f32
          %156 = arith.addf %154, %155 {fastmathFlags = #llvm.fastmath<none>} : f32
          %157 = arith.divf %116, %arg16 {fastmathFlags = #llvm.fastmath<none>} : f32
          affine.store %157, %arg1[%arg15 + %arg11 + (%arg14 - %arg11 * 32 + %arg12 * 16 + ((%arg14 + %arg13 * 16) floordiv 36) * 16) floordiv 288 + 1, %arg11 * -2 + %arg12 + (%arg14 + %arg13 * 16) floordiv 36 - ((%arg14 - %arg11 * 32 + %arg12 * 16 + ((%arg14 + %arg13 * 16) floordiv 36) * 16) floordiv 288) * 18, (%arg14 + %arg13 * 16) mod 36] : memref<50x18x36xf32, 1>
          %158 = arith.mulf %157, %156 {fastmathFlags = #llvm.fastmath<none>} : f32
          %159 = arith.subf %140, %158 {fastmathFlags = #llvm.fastmath<none>} : f32
          %160 = affine.load %arg0[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          %161 = math.absf %159 : f32
          %162 = arith.cmpf ule, %161, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f32
          %163 = arith.mulf %arg17, %156 {fastmathFlags = #llvm.fastmath<none>} : f32
          %164 = arith.subf %160, %163 {fastmathFlags = #llvm.fastmath<none>} : f32
          %165 = arith.divf %164, %159 {fastmathFlags = #llvm.fastmath<none>} : f32
          %166 = affine.load %arg0[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          %167 = arith.select %162, %166, %165 {fastmathFlags = #llvm.fastmath<none>} : f32
          affine.store %167, %arg0[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          affine.yield %159, %167 : f32, f32
        }
      }
    }
    return
  }
  func.func private @"##call__Z44gpu_solve_batched_tridiagonal_system_kernel_16CompilerMetadataI10StaticSizeI8_36__18_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E12_46__28__61_EE20AcousticTridiagLower23AcousticTridiagDiagonal20AcousticTridiagUpperSC_SA_IS9_Li3ELi1E12_36__18__50_E21LatitudeLongitudeGridI15CuTracedRNumberIS9_Li1EE7BoundedSK_SK_38TerrainFollowingVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_61__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_60__EESN_SP_11LinearDecayIS9_S8_IS9_Li3ESA_IS9_Li3ELi1E11_46__28__1_EES8_IS9_Li3ESA_IS9_Li3ELi1E11_47__28__1_EES8_IS9_Li3ESA_IS9_Li3ELi1E11_46__29__1_EEEES9_S9_S8_IS9_Li1ESA_IS9_Li1ELi1E5_47__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_46__EES9_S9_S8_IS9_Li1ESA_IS9_Li1ELi1E5_29__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_28__EES16_S16_S16_S16_S9_S9_v5Int64vEv5TupleIS8_IS9_Li3ESA_IS9_Li3ELi1E12_46__28__60_EES1B_S1B_SJ_SJ_SJ_vE10ZDirection#1427$par394"(%arg0: memref<61x28x46xf32, 1>, %arg1: memref<50x18x36xf32, 1>, %arg2: memref<61xf32, 1>, %arg3: memref<60xf32, 1>, %arg4: memref<1x28x46xf32, 1>, %arg5: memref<60x28x46xf32, 1>, %arg6: memref<60x28x46xf32, 1>, %arg7: memref<60x28x46xf32, 1>, %arg8: memref<f32, 1>, %arg9: memref<f32, 1>, %arg10: memref<f32, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c-51_i64 = arith.constant -51 : i64
    %c-49_i64 = arith.constant -49 : i64
    %false = arith.constant false
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c4_i64 = arith.constant 4 : i64
    %c-37_i64 = arith.constant -37 : i64
    %c-36_i64 = arith.constant -36 : i64
    %c18_i64 = arith.constant 18 : i64
    %cst = arith.constant -4.99925263E-5 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c-50_i64 = arith.constant -50 : i64
    %cst_3 = arith.constant 1.1920929E-6 : f32
    affine.parallel (%arg11, %arg12, %arg13, %arg14) = (0, 0, 0, 0) to (2, 16, 3, 16) {
      %0 = arith.muli %arg12, %c16 overflow<nuw> : index
      %1 = arith.addi %0, %arg14 : index
      %2 = arith.addi %1, %c1 : index
      %3 = arith.index_cast %arg13 : index to i64
      %4 = arith.index_castui %2 : index to i64
      %5 = arith.muli %arg12, %c16 overflow<nuw> : index
      %6 = arith.addi %5, %arg14 : index
      %7 = arith.shrui %6, %c4 : index
      %8 = arith.index_castui %7 : index to i64
      %9 = arith.subi %3, %8 : i64
      %10 = arith.shli %9, %c4_i64 : i64
      %11 = arith.addi %10, %4 : i64
      %12 = arith.shli %arg11, %c4 : index
      %13 = arith.addi %12, %7 : index
      %14 = arith.addi %13, %c1 : index
      %15 = arith.index_castui %14 : index to i64
      %16 = arith.addi %11, %c-37_i64 : i64
      %17 = arith.cmpi ult, %16, %c-36_i64 : i64
      %18 = arith.cmpi ugt, %15, %c18_i64 : i64
      %19 = arith.ori %18, %17 : i1
      %20 = arith.xori %19, %true : i1
      scf.if %20 {
        %21 = affine.load %arg2[6] {alignment = 8 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<61xf32, 1>
        %22 = affine.load %arg4[%arg11 + (%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448, %arg11 * -12 + %arg12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<1x28x46xf32, 1>
        %23 = arith.mulf %22, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
        %24 = arith.addf %23, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %25 = arith.mulf %21, %24 {fastmathFlags = #llvm.fastmath<none>} : f32
        %26 = arith.divf %cst_0, %25 {fastmathFlags = #llvm.fastmath<none>} : f32
        %27 = affine.load %arg3[5] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
        %28 = arith.mulf %27, %24 {fastmathFlags = #llvm.fastmath<none>} : f32
        %29 = arith.divf %cst_0, %28 {fastmathFlags = #llvm.fastmath<none>} : f32
        %30 = affine.load %arg3[4] {alignment = 16 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
        %31 = arith.mulf %30, %24 {fastmathFlags = #llvm.fastmath<none>} : f32
        %32 = arith.divf %cst_0, %31 {fastmathFlags = #llvm.fastmath<none>} : f32
        %33 = affine.load %arg7[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %34 = affine.load %arg5[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %35 = arith.mulf %33, %34 {fastmathFlags = #llvm.fastmath<none>} : f32
        %36 = affine.load %arg7[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %37 = affine.load %arg5[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %38 = arith.mulf %36, %37 {fastmathFlags = #llvm.fastmath<none>} : f32
        %39 = affine.load %arg6[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %40 = affine.load %arg6[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %41 = affine.if #set(%arg13, %arg14) -> f32 {
          affine.yield %40 : f32
        } else {
          affine.yield %39 : f32
        }
        %42 = arith.addf %41, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
        %43 = arith.mulf %42, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
        %44 = affine.load %arg9[] : memref<f32, 1>
        %45 = arith.mulf %44, %44 {fastmathFlags = #llvm.fastmath<none>} : f32
        %46 = arith.mulf %29, %35 {fastmathFlags = #llvm.fastmath<none>} : f32
        %47 = arith.mulf %32, %38 {fastmathFlags = #llvm.fastmath<none>} : f32
        %48 = arith.addf %46, %47 {fastmathFlags = #llvm.fastmath<none>} : f32
        %49 = arith.mulf %45, %43 {fastmathFlags = #llvm.fastmath<none>} : f32
        %50 = arith.mulf %48, %49 {fastmathFlags = #llvm.fastmath<none>} : f32
        %51 = arith.mulf %26, %50 {fastmathFlags = #llvm.fastmath<none>} : f32
        %52 = arith.subf %29, %32 {fastmathFlags = #llvm.fastmath<none>} : f32
        %53 = affine.load %arg8[] : memref<f32, 1>
        %54 = arith.mulf %45, %53 {fastmathFlags = #llvm.fastmath<none>} : f32
        %55 = arith.mulf %52, %54 {fastmathFlags = #llvm.fastmath<none>} : f32
        %56 = arith.mulf %55, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
        %57 = arith.addf %29, %32 {fastmathFlags = #llvm.fastmath<none>} : f32
        %58 = affine.load %arg10[] : memref<f32, 1>
        %59 = arith.mulf %57, %58 {fastmathFlags = #llvm.fastmath<none>} : f32
        %60 = arith.mulf %26, %59 {fastmathFlags = #llvm.fastmath<none>} : f32
        %61 = arith.addf %56, %51 {fastmathFlags = #llvm.fastmath<none>} : f32
        %62 = arith.addf %60, %61 {fastmathFlags = #llvm.fastmath<none>} : f32
        %63 = math.copysign %cst_2, %62 : f32
        %64 = arith.addf %63, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %65 = affine.load %arg0[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        %66 = arith.divf %65, %64 {fastmathFlags = #llvm.fastmath<none>} : f32
        affine.store %66, %arg0[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        %67 = affine.load %arg0[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        %68 = affine.load %arg4[%arg11 + (%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448, %arg11 * -12 + %arg12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<1x28x46xf32, 1>
        %69 = arith.mulf %68, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
        %70 = arith.addf %69, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %71 = affine.if #set(%arg13, %arg14) -> i1 {
          affine.yield %true : i1
        } else {
          affine.yield %false : i1
        }
        %72 = affine.if #set(%arg13, %arg14) -> i1 {
          affine.yield %true : i1
        } else {
          affine.yield %false : i1
        }
        %73 = affine.load %arg9[] : memref<f32, 1>
        %74 = arith.mulf %73, %73 {fastmathFlags = #llvm.fastmath<none>} : f32
        %75 = arith.negf %74 {fastmathFlags = #llvm.fastmath<none>} : f32
        %76 = affine.load %arg8[] : memref<f32, 1>
        %77 = arith.mulf %76, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
        %78 = affine.load %arg10[] : memref<f32, 1>
        %79 = arith.negf %78 {fastmathFlags = #llvm.fastmath<none>} : f32
        %80 = arith.mulf %74, %76 {fastmathFlags = #llvm.fastmath<none>} : f32
        %81 = affine.if #set(%arg13, %arg14) -> i1 {
          affine.yield %true : i1
        } else {
          affine.yield %false : i1
        }
        %82:2 = affine.for %arg15 = 0 to 49 iter_args(%arg16 = %64, %arg17 = %67) -> (f32, f32) {
          %83 = arith.index_cast %arg15 : index to i64
          %84 = affine.load %arg2[%arg15 + 6] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<61xf32, 1>
          %85 = arith.mulf %84, %70 {fastmathFlags = #llvm.fastmath<none>} : f32
          %86 = arith.divf %cst_0, %85 {fastmathFlags = #llvm.fastmath<none>} : f32
          %87 = affine.load %arg3[%arg15 + 5] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
          %88 = arith.mulf %87, %70 {fastmathFlags = #llvm.fastmath<none>} : f32
          %89 = arith.divf %cst_0, %88 {fastmathFlags = #llvm.fastmath<none>} : f32
          %90 = affine.load %arg7[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %91 = affine.load %arg5[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %92 = arith.mulf %90, %91 {fastmathFlags = #llvm.fastmath<none>} : f32
          %93 = affine.load %arg6[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %94 = affine.load %arg6[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %95 = arith.addi %83, %c-49_i64 : i64
          %96 = arith.cmpi ult, %95, %c-50_i64 : i64
          %97 = arith.ori %71, %96 : i1
          %98 = arith.addi %83, %c-50_i64 : i64
          %99 = arith.cmpi ult, %98, %c-50_i64 : i64
          %100 = arith.ori %72, %99 : i1
          %101 = arith.select %97, %94, %93 {fastmathFlags = #llvm.fastmath<none>} : f32
          %102 = arith.select %100, %101, %94 {fastmathFlags = #llvm.fastmath<none>} : f32
          %103 = arith.addf %101, %102 {fastmathFlags = #llvm.fastmath<none>} : f32
          %104 = arith.mulf %103, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %105 = arith.mulf %92, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
          %106 = arith.mulf %105, %104 {fastmathFlags = #llvm.fastmath<none>} : f32
          %107 = arith.mulf %89, %106 {fastmathFlags = #llvm.fastmath<none>} : f32
          %108 = arith.mulf %86, %107 {fastmathFlags = #llvm.fastmath<none>} : f32
          %109 = arith.mulf %89, %77 {fastmathFlags = #llvm.fastmath<none>} : f32
          %110 = arith.mulf %109, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %111 = arith.mulf %89, %79 {fastmathFlags = #llvm.fastmath<none>} : f32
          %112 = arith.mulf %86, %111 {fastmathFlags = #llvm.fastmath<none>} : f32
          %113 = arith.addf %110, %108 {fastmathFlags = #llvm.fastmath<none>} : f32
          %114 = arith.addf %112, %113 {fastmathFlags = #llvm.fastmath<none>} : f32
          %115 = math.copysign %cst_2, %114 : f32
          %116 = affine.if #set1(%arg15) -> f32 {
            affine.yield %115 : f32
          } else {
            affine.yield %114 : f32
          }
          %117 = affine.load %arg2[%arg15 + 7] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<61xf32, 1>
          %118 = arith.mulf %70, %117 {fastmathFlags = #llvm.fastmath<none>} : f32
          %119 = arith.divf %cst_0, %118 {fastmathFlags = #llvm.fastmath<none>} : f32
          %120 = affine.load %arg3[%arg15 + 6] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
          %121 = arith.mulf %70, %120 {fastmathFlags = #llvm.fastmath<none>} : f32
          %122 = arith.divf %cst_0, %121 {fastmathFlags = #llvm.fastmath<none>} : f32
          %123 = affine.load %arg7[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %124 = affine.load %arg5[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %125 = arith.mulf %123, %124 {fastmathFlags = #llvm.fastmath<none>} : f32
          %126 = arith.mulf %122, %125 {fastmathFlags = #llvm.fastmath<none>} : f32
          %127 = arith.mulf %92, %89 {fastmathFlags = #llvm.fastmath<none>} : f32
          %128 = arith.addf %127, %126 {fastmathFlags = #llvm.fastmath<none>} : f32
          %129 = arith.mulf %74, %104 {fastmathFlags = #llvm.fastmath<none>} : f32
          %130 = arith.mulf %129, %128 {fastmathFlags = #llvm.fastmath<none>} : f32
          %131 = arith.mulf %119, %130 {fastmathFlags = #llvm.fastmath<none>} : f32
          %132 = arith.subf %122, %89 {fastmathFlags = #llvm.fastmath<none>} : f32
          %133 = arith.mulf %80, %132 {fastmathFlags = #llvm.fastmath<none>} : f32
          %134 = arith.mulf %133, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %135 = arith.addf %89, %122 {fastmathFlags = #llvm.fastmath<none>} : f32
          %136 = arith.mulf %78, %135 {fastmathFlags = #llvm.fastmath<none>} : f32
          %137 = arith.mulf %119, %136 {fastmathFlags = #llvm.fastmath<none>} : f32
          %138 = arith.addf %134, %131 {fastmathFlags = #llvm.fastmath<none>} : f32
          %139 = arith.addf %137, %138 {fastmathFlags = #llvm.fastmath<none>} : f32
          %140 = arith.addf %139, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
          %141 = affine.load %arg6[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %142 = arith.addi %83, %c-51_i64 : i64
          %143 = arith.cmpi ult, %142, %c-50_i64 : i64
          %144 = arith.ori %81, %143 : i1
          %145 = arith.select %100, %141, %94 {fastmathFlags = #llvm.fastmath<none>} : f32
          %146 = arith.select %144, %145, %141 {fastmathFlags = #llvm.fastmath<none>} : f32
          %147 = arith.addf %145, %146 {fastmathFlags = #llvm.fastmath<none>} : f32
          %148 = arith.mulf %147, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %149 = arith.mulf %105, %148 {fastmathFlags = #llvm.fastmath<none>} : f32
          %150 = arith.mulf %89, %149 {fastmathFlags = #llvm.fastmath<none>} : f32
          %151 = arith.mulf %119, %150 {fastmathFlags = #llvm.fastmath<none>} : f32
          %152 = arith.mulf %89, %80 {fastmathFlags = #llvm.fastmath<none>} : f32
          %153 = arith.mulf %152, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %154 = arith.mulf %111, %119 {fastmathFlags = #llvm.fastmath<none>} : f32
          %155 = arith.addf %153, %151 {fastmathFlags = #llvm.fastmath<none>} : f32
          %156 = arith.addf %154, %155 {fastmathFlags = #llvm.fastmath<none>} : f32
          %157 = arith.divf %116, %arg16 {fastmathFlags = #llvm.fastmath<none>} : f32
          affine.store %157, %arg1[%arg15 + %arg11 + (%arg14 - %arg11 * 32 + %arg12 * 16 + ((%arg14 + %arg13 * 16) floordiv 36) * 16) floordiv 288 + 1, %arg11 * -2 + %arg12 + (%arg14 + %arg13 * 16) floordiv 36 - ((%arg14 - %arg11 * 32 + %arg12 * 16 + ((%arg14 + %arg13 * 16) floordiv 36) * 16) floordiv 288) * 18, (%arg14 + %arg13 * 16) mod 36] : memref<50x18x36xf32, 1>
          %158 = arith.mulf %157, %156 {fastmathFlags = #llvm.fastmath<none>} : f32
          %159 = arith.subf %140, %158 {fastmathFlags = #llvm.fastmath<none>} : f32
          %160 = affine.load %arg0[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          %161 = math.absf %159 : f32
          %162 = arith.cmpf ule, %161, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f32
          %163 = arith.mulf %arg17, %156 {fastmathFlags = #llvm.fastmath<none>} : f32
          %164 = arith.subf %160, %163 {fastmathFlags = #llvm.fastmath<none>} : f32
          %165 = arith.divf %164, %159 {fastmathFlags = #llvm.fastmath<none>} : f32
          %166 = affine.load %arg0[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          %167 = arith.select %162, %166, %165 {fastmathFlags = #llvm.fastmath<none>} : f32
          affine.store %167, %arg0[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          affine.yield %159, %167 : f32, f32
        }
        affine.for %arg15 = 0 to 49 {
          %83 = affine.load %arg0[-%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 53, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          %84 = affine.load %arg1[-%arg15 + %arg11 + (%arg14 - %arg11 * 32 + %arg12 * 16 + ((%arg14 + %arg13 * 16) floordiv 36) * 16) floordiv 288 + 49, %arg11 * -2 + %arg12 + (%arg14 + %arg13 * 16) floordiv 36 - ((%arg14 - %arg11 * 32 + %arg12 * 16 + ((%arg14 + %arg13 * 16) floordiv 36) * 16) floordiv 288) * 18, (%arg14 + %arg13 * 16) mod 36] : memref<50x18x36xf32, 1>
          %85 = affine.load %arg0[-%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 54, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          %86 = arith.mulf %84, %85 {fastmathFlags = #llvm.fastmath<none>} : f32
          %87 = arith.subf %83, %86 {fastmathFlags = #llvm.fastmath<none>} : f32
          affine.store %87, %arg0[-%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 53, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        }
      }
    }
    return
  }
  func.func private @"##call__Z44gpu_solve_batched_tridiagonal_system_kernel_16CompilerMetadataI10StaticSizeI8_36__18_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E12_46__28__61_EE20AcousticTridiagLower23AcousticTridiagDiagonal20AcousticTridiagUpperSC_SA_IS9_Li3ELi1E12_36__18__50_E21LatitudeLongitudeGridI15CuTracedRNumberIS9_Li1EE7BoundedSK_SK_38TerrainFollowingVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_61__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_60__EESN_SP_11LinearDecayIS9_S8_IS9_Li3ESA_IS9_Li3ELi1E11_46__28__1_EES8_IS9_Li3ESA_IS9_Li3ELi1E11_47__28__1_EES8_IS9_Li3ESA_IS9_Li3ELi1E11_46__29__1_EEEES9_S9_S8_IS9_Li1ESA_IS9_Li1ELi1E5_47__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_46__EES9_S9_S8_IS9_Li1ESA_IS9_Li1ELi1E5_29__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_28__EES16_S16_S16_S16_S9_S9_v5Int64vEv5TupleIS8_IS9_Li3ESA_IS9_Li3ELi1E12_46__28__60_EES1B_S1B_SJ_SJ_SJ_vE10ZDirection#1871$par616"(%arg0: memref<61x28x46xf32, 1>, %arg1: memref<50x18x36xf32, 1>, %arg2: memref<61xf32, 1>, %arg3: memref<60xf32, 1>, %arg4: memref<1x28x46xf32, 1>, %arg5: memref<60x28x46xf32, 1>, %arg6: memref<60x28x46xf32, 1>, %arg7: memref<60x28x46xf32, 1>, %arg8: memref<f32, 1>, %arg9: memref<f32, 1>, %arg10: memref<f32, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c-51_i64 = arith.constant -51 : i64
    %c-49_i64 = arith.constant -49 : i64
    %false = arith.constant false
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %c4_i64 = arith.constant 4 : i64
    %c-37_i64 = arith.constant -37 : i64
    %c-36_i64 = arith.constant -36 : i64
    %c18_i64 = arith.constant 18 : i64
    %cst = arith.constant -4.99925263E-5 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c-50_i64 = arith.constant -50 : i64
    %cst_3 = arith.constant 1.1920929E-6 : f32
    affine.parallel (%arg11, %arg12, %arg13, %arg14) = (0, 0, 0, 0) to (2, 16, 3, 16) {
      %0 = arith.muli %arg12, %c16 overflow<nuw> : index
      %1 = arith.addi %0, %arg14 : index
      %2 = arith.addi %1, %c1 : index
      %3 = arith.index_cast %arg13 : index to i64
      %4 = arith.index_castui %2 : index to i64
      %5 = arith.muli %arg12, %c16 overflow<nuw> : index
      %6 = arith.addi %5, %arg14 : index
      %7 = arith.shrui %6, %c4 : index
      %8 = arith.index_castui %7 : index to i64
      %9 = arith.subi %3, %8 : i64
      %10 = arith.shli %9, %c4_i64 : i64
      %11 = arith.addi %10, %4 : i64
      %12 = arith.shli %arg11, %c4 : index
      %13 = arith.addi %12, %7 : index
      %14 = arith.addi %13, %c1 : index
      %15 = arith.index_castui %14 : index to i64
      %16 = arith.addi %11, %c-37_i64 : i64
      %17 = arith.cmpi ult, %16, %c-36_i64 : i64
      %18 = arith.cmpi ugt, %15, %c18_i64 : i64
      %19 = arith.ori %18, %17 : i1
      %20 = arith.xori %19, %true : i1
      scf.if %20 {
        %21 = affine.load %arg2[6] {alignment = 8 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<61xf32, 1>
        %22 = affine.load %arg4[%arg11 + (%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448, %arg11 * -12 + %arg12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<1x28x46xf32, 1>
        %23 = arith.mulf %22, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
        %24 = arith.addf %23, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %25 = arith.mulf %21, %24 {fastmathFlags = #llvm.fastmath<none>} : f32
        %26 = arith.divf %cst_0, %25 {fastmathFlags = #llvm.fastmath<none>} : f32
        %27 = affine.load %arg3[5] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
        %28 = arith.mulf %27, %24 {fastmathFlags = #llvm.fastmath<none>} : f32
        %29 = arith.divf %cst_0, %28 {fastmathFlags = #llvm.fastmath<none>} : f32
        %30 = affine.load %arg3[4] {alignment = 16 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
        %31 = arith.mulf %30, %24 {fastmathFlags = #llvm.fastmath<none>} : f32
        %32 = arith.divf %cst_0, %31 {fastmathFlags = #llvm.fastmath<none>} : f32
        %33 = affine.load %arg7[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %34 = affine.load %arg5[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %35 = arith.mulf %33, %34 {fastmathFlags = #llvm.fastmath<none>} : f32
        %36 = affine.load %arg7[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %37 = affine.load %arg5[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %38 = arith.mulf %36, %37 {fastmathFlags = #llvm.fastmath<none>} : f32
        %39 = affine.load %arg6[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %40 = affine.load %arg6[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
        %41 = affine.if #set(%arg13, %arg14) -> f32 {
          affine.yield %40 : f32
        } else {
          affine.yield %39 : f32
        }
        %42 = arith.addf %41, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
        %43 = arith.mulf %42, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
        %44 = affine.load %arg9[] : memref<f32, 1>
        %45 = arith.mulf %44, %44 {fastmathFlags = #llvm.fastmath<none>} : f32
        %46 = arith.mulf %29, %35 {fastmathFlags = #llvm.fastmath<none>} : f32
        %47 = arith.mulf %32, %38 {fastmathFlags = #llvm.fastmath<none>} : f32
        %48 = arith.addf %46, %47 {fastmathFlags = #llvm.fastmath<none>} : f32
        %49 = arith.mulf %45, %43 {fastmathFlags = #llvm.fastmath<none>} : f32
        %50 = arith.mulf %48, %49 {fastmathFlags = #llvm.fastmath<none>} : f32
        %51 = arith.mulf %26, %50 {fastmathFlags = #llvm.fastmath<none>} : f32
        %52 = arith.subf %29, %32 {fastmathFlags = #llvm.fastmath<none>} : f32
        %53 = affine.load %arg8[] : memref<f32, 1>
        %54 = arith.mulf %45, %53 {fastmathFlags = #llvm.fastmath<none>} : f32
        %55 = arith.mulf %52, %54 {fastmathFlags = #llvm.fastmath<none>} : f32
        %56 = arith.mulf %55, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
        %57 = arith.addf %29, %32 {fastmathFlags = #llvm.fastmath<none>} : f32
        %58 = affine.load %arg10[] : memref<f32, 1>
        %59 = arith.mulf %57, %58 {fastmathFlags = #llvm.fastmath<none>} : f32
        %60 = arith.mulf %26, %59 {fastmathFlags = #llvm.fastmath<none>} : f32
        %61 = arith.addf %56, %51 {fastmathFlags = #llvm.fastmath<none>} : f32
        %62 = arith.addf %60, %61 {fastmathFlags = #llvm.fastmath<none>} : f32
        %63 = math.copysign %cst_2, %62 : f32
        %64 = arith.addf %63, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %65 = affine.load %arg0[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        %66 = arith.divf %65, %64 {fastmathFlags = #llvm.fastmath<none>} : f32
        affine.store %66, %arg0[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        %67 = affine.load %arg0[%arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        %68 = affine.load %arg4[%arg11 + (%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448, %arg11 * -12 + %arg12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 - %arg11 * 192 + %arg12 * 16 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<1x28x46xf32, 1>
        %69 = arith.mulf %68, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
        %70 = arith.addf %69, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %71 = affine.if #set(%arg13, %arg14) -> i1 {
          affine.yield %true : i1
        } else {
          affine.yield %false : i1
        }
        %72 = affine.if #set(%arg13, %arg14) -> i1 {
          affine.yield %true : i1
        } else {
          affine.yield %false : i1
        }
        %73 = affine.load %arg9[] : memref<f32, 1>
        %74 = arith.mulf %73, %73 {fastmathFlags = #llvm.fastmath<none>} : f32
        %75 = arith.negf %74 {fastmathFlags = #llvm.fastmath<none>} : f32
        %76 = affine.load %arg8[] : memref<f32, 1>
        %77 = arith.mulf %76, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
        %78 = affine.load %arg10[] : memref<f32, 1>
        %79 = arith.negf %78 {fastmathFlags = #llvm.fastmath<none>} : f32
        %80 = arith.mulf %74, %76 {fastmathFlags = #llvm.fastmath<none>} : f32
        %81 = affine.if #set(%arg13, %arg14) -> i1 {
          affine.yield %true : i1
        } else {
          affine.yield %false : i1
        }
        %82:2 = affine.for %arg15 = 0 to 49 iter_args(%arg16 = %64, %arg17 = %67) -> (f32, f32) {
          %83 = arith.index_cast %arg15 : index to i64
          %84 = affine.load %arg2[%arg15 + 6] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<61xf32, 1>
          %85 = arith.mulf %84, %70 {fastmathFlags = #llvm.fastmath<none>} : f32
          %86 = arith.divf %cst_0, %85 {fastmathFlags = #llvm.fastmath<none>} : f32
          %87 = affine.load %arg3[%arg15 + 5] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
          %88 = arith.mulf %87, %70 {fastmathFlags = #llvm.fastmath<none>} : f32
          %89 = arith.divf %cst_0, %88 {fastmathFlags = #llvm.fastmath<none>} : f32
          %90 = affine.load %arg7[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %91 = affine.load %arg5[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %92 = arith.mulf %90, %91 {fastmathFlags = #llvm.fastmath<none>} : f32
          %93 = affine.load %arg6[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %94 = affine.load %arg6[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 5, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %95 = arith.addi %83, %c-49_i64 : i64
          %96 = arith.cmpi ult, %95, %c-50_i64 : i64
          %97 = arith.ori %71, %96 : i1
          %98 = arith.addi %83, %c-50_i64 : i64
          %99 = arith.cmpi ult, %98, %c-50_i64 : i64
          %100 = arith.ori %72, %99 : i1
          %101 = arith.select %97, %94, %93 {fastmathFlags = #llvm.fastmath<none>} : f32
          %102 = arith.select %100, %101, %94 {fastmathFlags = #llvm.fastmath<none>} : f32
          %103 = arith.addf %101, %102 {fastmathFlags = #llvm.fastmath<none>} : f32
          %104 = arith.mulf %103, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %105 = arith.mulf %92, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
          %106 = arith.mulf %105, %104 {fastmathFlags = #llvm.fastmath<none>} : f32
          %107 = arith.mulf %89, %106 {fastmathFlags = #llvm.fastmath<none>} : f32
          %108 = arith.mulf %86, %107 {fastmathFlags = #llvm.fastmath<none>} : f32
          %109 = arith.mulf %89, %77 {fastmathFlags = #llvm.fastmath<none>} : f32
          %110 = arith.mulf %109, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %111 = arith.mulf %89, %79 {fastmathFlags = #llvm.fastmath<none>} : f32
          %112 = arith.mulf %86, %111 {fastmathFlags = #llvm.fastmath<none>} : f32
          %113 = arith.addf %110, %108 {fastmathFlags = #llvm.fastmath<none>} : f32
          %114 = arith.addf %112, %113 {fastmathFlags = #llvm.fastmath<none>} : f32
          %115 = math.copysign %cst_2, %114 : f32
          %116 = affine.if #set1(%arg15) -> f32 {
            affine.yield %115 : f32
          } else {
            affine.yield %114 : f32
          }
          %117 = affine.load %arg2[%arg15 + 7] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<61xf32, 1>
          %118 = arith.mulf %70, %117 {fastmathFlags = #llvm.fastmath<none>} : f32
          %119 = arith.divf %cst_0, %118 {fastmathFlags = #llvm.fastmath<none>} : f32
          %120 = affine.load %arg3[%arg15 + 6] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<60xf32, 1>
          %121 = arith.mulf %70, %120 {fastmathFlags = #llvm.fastmath<none>} : f32
          %122 = arith.divf %cst_0, %121 {fastmathFlags = #llvm.fastmath<none>} : f32
          %123 = affine.load %arg7[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %124 = affine.load %arg5[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %125 = arith.mulf %123, %124 {fastmathFlags = #llvm.fastmath<none>} : f32
          %126 = arith.mulf %122, %125 {fastmathFlags = #llvm.fastmath<none>} : f32
          %127 = arith.mulf %92, %89 {fastmathFlags = #llvm.fastmath<none>} : f32
          %128 = arith.addf %127, %126 {fastmathFlags = #llvm.fastmath<none>} : f32
          %129 = arith.mulf %74, %104 {fastmathFlags = #llvm.fastmath<none>} : f32
          %130 = arith.mulf %129, %128 {fastmathFlags = #llvm.fastmath<none>} : f32
          %131 = arith.mulf %119, %130 {fastmathFlags = #llvm.fastmath<none>} : f32
          %132 = arith.subf %122, %89 {fastmathFlags = #llvm.fastmath<none>} : f32
          %133 = arith.mulf %80, %132 {fastmathFlags = #llvm.fastmath<none>} : f32
          %134 = arith.mulf %133, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %135 = arith.addf %89, %122 {fastmathFlags = #llvm.fastmath<none>} : f32
          %136 = arith.mulf %78, %135 {fastmathFlags = #llvm.fastmath<none>} : f32
          %137 = arith.mulf %119, %136 {fastmathFlags = #llvm.fastmath<none>} : f32
          %138 = arith.addf %134, %131 {fastmathFlags = #llvm.fastmath<none>} : f32
          %139 = arith.addf %137, %138 {fastmathFlags = #llvm.fastmath<none>} : f32
          %140 = arith.addf %139, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
          %141 = affine.load %arg6[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 4, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<60x28x46xf32, 1>
          %142 = arith.addi %83, %c-51_i64 : i64
          %143 = arith.cmpi ult, %142, %c-50_i64 : i64
          %144 = arith.ori %81, %143 : i1
          %145 = arith.select %100, %141, %94 {fastmathFlags = #llvm.fastmath<none>} : f32
          %146 = arith.select %144, %145, %141 {fastmathFlags = #llvm.fastmath<none>} : f32
          %147 = arith.addf %145, %146 {fastmathFlags = #llvm.fastmath<none>} : f32
          %148 = arith.mulf %147, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %149 = arith.mulf %105, %148 {fastmathFlags = #llvm.fastmath<none>} : f32
          %150 = arith.mulf %89, %149 {fastmathFlags = #llvm.fastmath<none>} : f32
          %151 = arith.mulf %119, %150 {fastmathFlags = #llvm.fastmath<none>} : f32
          %152 = arith.mulf %89, %80 {fastmathFlags = #llvm.fastmath<none>} : f32
          %153 = arith.mulf %152, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
          %154 = arith.mulf %111, %119 {fastmathFlags = #llvm.fastmath<none>} : f32
          %155 = arith.addf %153, %151 {fastmathFlags = #llvm.fastmath<none>} : f32
          %156 = arith.addf %154, %155 {fastmathFlags = #llvm.fastmath<none>} : f32
          %157 = arith.divf %116, %arg16 {fastmathFlags = #llvm.fastmath<none>} : f32
          affine.store %157, %arg1[%arg15 + %arg11 + (%arg14 - %arg11 * 32 + %arg12 * 16 + ((%arg14 + %arg13 * 16) floordiv 36) * 16) floordiv 288 + 1, %arg11 * -2 + %arg12 + (%arg14 + %arg13 * 16) floordiv 36 - ((%arg14 - %arg11 * 32 + %arg12 * 16 + ((%arg14 + %arg13 * 16) floordiv 36) * 16) floordiv 288) * 18, (%arg14 + %arg13 * 16) mod 36] : memref<50x18x36xf32, 1>
          %158 = arith.mulf %157, %156 {fastmathFlags = #llvm.fastmath<none>} : f32
          %159 = arith.subf %140, %158 {fastmathFlags = #llvm.fastmath<none>} : f32
          %160 = affine.load %arg0[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          %161 = math.absf %159 : f32
          %162 = arith.cmpf ule, %161, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f32
          %163 = arith.mulf %arg17, %156 {fastmathFlags = #llvm.fastmath<none>} : f32
          %164 = arith.subf %160, %163 {fastmathFlags = #llvm.fastmath<none>} : f32
          %165 = arith.divf %164, %159 {fastmathFlags = #llvm.fastmath<none>} : f32
          %166 = affine.load %arg0[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          %167 = arith.select %162, %166, %165 {fastmathFlags = #llvm.fastmath<none>} : f32
          affine.store %167, %arg0[%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 6, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          affine.yield %159, %167 : f32, f32
        }
        affine.for %arg15 = 0 to 49 {
          %83 = affine.load %arg0[-%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 53, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          %84 = affine.load %arg1[-%arg15 + %arg11 + (%arg14 - %arg11 * 32 + %arg12 * 16 + ((%arg14 + %arg13 * 16) floordiv 36) * 16) floordiv 288 + 49, %arg11 * -2 + %arg12 + (%arg14 + %arg13 * 16) floordiv 36 - ((%arg14 - %arg11 * 32 + %arg12 * 16 + ((%arg14 + %arg13 * 16) floordiv 36) * 16) floordiv 288) * 18, (%arg14 + %arg13 * 16) mod 36] : memref<50x18x36xf32, 1>
          %85 = affine.load %arg0[-%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 54, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
          %86 = arith.mulf %84, %85 {fastmathFlags = #llvm.fastmath<none>} : f32
          %87 = arith.subf %83, %86 {fastmathFlags = #llvm.fastmath<none>} : f32
          affine.store %87, %arg0[-%arg15 + %arg11 + (%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448 + 53, %arg12 - %arg11 * 12 + (%arg14 + %arg13 * 16 + 5) floordiv 46 - ((%arg14 + %arg12 * 16 - %arg11 * 192 + ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 16 + 80) floordiv 448) * 28 + 5, %arg14 + %arg13 * 16 - ((%arg14 + %arg13 * 16 + 5) floordiv 46) * 46 + 5] : memref<61x28x46xf32, 1>
        }
      }
    }
    return
  }
}

// The yielded accumulators re-align (as permuting broadcasts) onto the
// carried layout before the return.
// CHECK:      %[[ALIGN1:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [0, 3, 1, 2] : (tensor<2x16x16x3xf32>) -> tensor<2x16x3x16xf32>
// CHECK:      %[[ALIGN2:.+]] = stablehlo.broadcast_in_dim %{{.+}}, dims = [0, 3, 1, 2] : (tensor<2x16x16x3xf32>) -> tensor<2x16x3x16xf32>
// CHECK:      stablehlo.return %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %iterArg{{.+}} : tensor<i64>, tensor<2x16x3x16xf32>, tensor<2x16x3x16xf32>, tensor<61x28x46xf32>, tensor<50x18x36xf32>, tensor<61xf32>, tensor<60xf32>, tensor<1x28x46xf32>, tensor<60x28x46xf32>, tensor<60x28x46xf32>, tensor<60x28x46xf32>, tensor<f32>, tensor<f32>, tensor<f32>
// CHECK-NEXT:    }
