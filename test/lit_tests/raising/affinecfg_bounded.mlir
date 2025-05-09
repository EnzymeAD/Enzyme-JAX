// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

func.func private @"##call__Z39gpu__compute_integrated_ab2_tendencies_16CompilerMetadataI10StaticSizeI8_62__62_E12DynamicCheckvv7NDRangeILi2ES0_I6_4__4_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float32Li3E13CuTracedArrayISC_Li3ELi1E11_78__78__1_EESC_vvvES8_ISA_S9_vvvvSF_SC_vvvE21LatitudeLongitudeGridI15CuTracedRNumberISC_Li1EE7BoundedSL_SL_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_32__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_31__EESO_SQ_ESC_SC_SB_ISC_Li1E12StepRangeLenISC_7Float64ST_5Int64EESW_SC_SC_SW_SW_SB_ISC_Li1ESD_ISC_Li1ELi1E5_78__EESY_SY_SY_SC_SC_vSU_ESB_ISC_Li3ESD_ISC_Li3ELi1E12_78__78__31_EES11_S11_S11_SC_#258$par4"(%arg0: memref<1x78x78xf32, 1>, %arg1: memref<1x78x78xf32, 1>, %arg2: memref<31xf32, 1>, %arg3: memref<31x78x78xf32, 1>, %arg4: memref<31x78x78xf32, 1>, %arg5: memref<31x78x78xf32, 1>, %arg6: memref<31x78x78xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c-4 = arith.constant -4 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c-1_i64 = arith.constant -1 : i64
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c62_i64 = arith.constant 62 : i64
  %cst = arith.constant 1.600000e+00 : f32
  %cst_0 = arith.constant 6.000000e-01 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c2_i64 = arith.constant 2 : i64
  %c15_i64 = arith.constant 15 : i64
  affine.parallel (%arg7, %arg8, %arg9, %arg10) = (0, 0, 0, 0) to (4, 16, 4, 16) {
    %0 = arith.muli %arg8, %c16 overflow<nuw> : index
    %1 = arith.addi %0, %arg10 : index
    %2 = arith.addi %1, %c1 : index
    %3 = arith.muli %arg7, %c-4 : index
    %4 = arith.muli %arg7, %c4 overflow<nuw> : index
    %5 = arith.addi %4, %arg9 : index
    %6 = arith.addi %3, %5 : index
    %7 = arith.index_castui %2 : index to i64
    %8 = arith.subi %c0, %arg8 : index
    %9 = arith.addi %8, %6 : index
    %10 = arith.addi %9, %c1 : index
    %11 = arith.index_cast %10 : index to i64
    %12 = arith.addi %11, %c-1_i64 : i64
    %13 = arith.muli %12, %c16_i64 : i64
    %14 = arith.addi %7, %13 : i64
    %15 = arith.muli %arg7, %c16 : index
    %16 = arith.addi %15, %arg8 : index
    %17 = arith.index_castui %16 : index to i64
    %18 = arith.addi %17, %c1_i64 : i64
    affine.if affine_set<(d0, d1, d2, d3) : (d0 + d1 * 16 + 1 >= 0, -d0 - d1 * 16 + 61 >= 0, d3 + d2 * 16 >= 0, d2 * -16 - d3 + 61 >= 0)>(%arg8, %arg7, %arg9, %arg10) {
      %19 = affine.load %arg2[8] {alignment = 32 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<31xf32, 1>
      %20 = affine.load %arg5[8, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<31x78x78xf32, 1>
      %21 = arith.mulf %20, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %22 = affine.load %arg3[8, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<31x78x78xf32, 1>
      %23 = arith.mulf %22, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
      %24 = arith.subf %21, %23 {fastmathFlags = #llvm.fastmath<none>} : f32
      %25 = arith.addi %14, %c-1_i64 : i64
      %26 = arith.cmpi ult, %25, %c1_i64 : i64
      %27 = arith.cmpi sgt, %25, %c62_i64 : i64
      %28 = arith.ori %26, %27 : i1
      %29 = arith.select %28, %cst_1, %24 : f32
      %30 = arith.mulf %19, %29 {fastmathFlags = #llvm.fastmath<none>} : f32
      affine.store %30, %arg0[0, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<1x78x78xf32, 1>
      %31 = affine.load %arg2[8] {alignment = 32 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<31xf32, 1>
      %32 = affine.load %arg6[8, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<31x78x78xf32, 1>
      %33 = arith.mulf %32, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %34 = affine.load %arg4[8, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<31x78x78xf32, 1>
      %35 = arith.mulf %34, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
      %36 = arith.subf %33, %35 {fastmathFlags = #llvm.fastmath<none>} : f32
      %37 = arith.cmpi ult, %17, %c1_i64 : i64
      %38 = arith.cmpi sgt, %17, %c62_i64 : i64
      %39 = arith.ori %37, %38 : i1
      %40 = arith.select %39, %cst_1, %36 : f32
      %41 = arith.mulf %31, %40 {fastmathFlags = #llvm.fastmath<none>} : f32
      affine.store %41, %arg1[0, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<1x78x78xf32, 1>
      %42 = arith.cmpi sgt, %14, %c62_i64 : i64
      %43 = arith.cmpi sgt, %18, %c62_i64 : i64
      %44 = arith.ori %42, %43 : i1
      %45 = arith.ori %28, %43 : i1
      %46 = arith.ori %42, %39 : i1
      affine.for %arg11 = 0 to 14 {
        %47 = arith.index_cast %arg11 : index to i64
        %48 = arith.addi %47, %c2_i64 : i64
        %49 = affine.load %arg0[0, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<1x78x78xf32, 1>
        %50 = affine.load %arg2[%arg11 + 9] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<31xf32, 1>
        %51 = affine.load %arg5[%arg11 + 9, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<31x78x78xf32, 1>
        %52 = arith.mulf %51, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
        %53 = affine.load %arg3[%arg11 + 9, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<31x78x78xf32, 1>
        %54 = arith.mulf %53, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %55 = arith.subf %52, %54 {fastmathFlags = #llvm.fastmath<none>} : f32
        %56 = arith.cmpi slt, %48, %c1_i64 : i64
        %57 = arith.cmpi sgt, %48, %c15_i64 : i64
        %58 = arith.ori %56, %57 : i1
        %59 = arith.ori %44, %58 : i1
        %60 = arith.ori %45, %58 : i1
        %61 = arith.ori %59, %60 : i1
        %62 = arith.select %61, %cst_1, %55 : f32
        %63 = arith.mulf %50, %62 {fastmathFlags = #llvm.fastmath<none>} : f32
        %64 = arith.addf %49, %63 {fastmathFlags = #llvm.fastmath<none>} : f32
        affine.store %64, %arg0[0, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<1x78x78xf32, 1>
        %65 = affine.load %arg1[0, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<1x78x78xf32, 1>
        %66 = affine.load %arg2[%arg11 + 9] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<31xf32, 1>
        %67 = affine.load %arg6[%arg11 + 9, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<31x78x78xf32, 1>
        %68 = arith.mulf %67, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
        %69 = affine.load %arg4[%arg11 + 9, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<31x78x78xf32, 1>
        %70 = arith.mulf %69, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
        %71 = arith.subf %68, %70 {fastmathFlags = #llvm.fastmath<none>} : f32
        %72 = arith.ori %46, %58 : i1
        %73 = arith.ori %59, %72 : i1
        %74 = arith.select %73, %cst_1, %71 : f32
        %75 = arith.mulf %66, %74 {fastmathFlags = #llvm.fastmath<none>} : f32
        %76 = arith.addf %65, %75 {fastmathFlags = #llvm.fastmath<none>} : f32
        affine.store %76, %arg1[0, %arg7 * 16 + %arg8 + 8, %arg10 + %arg9 * 16 + 8] : memref<1x78x78xf32, 1>
      }
    }
  }
  return
}

// CHECK: #set = affine_set<(d0) : (d0 - 1 >= 0, -d0 + 62 >= 0)>
// CHECK: #set1 = affine_set<(d0, d1, d2) : (-d2 + 61 >= 0, -d1 + 61 >= 0, d0 + 1 >= 0, -d0 + 13 >= 0, -d1 + 61 >= 0, d0 + 1 >= 0, -d0 + 13 >= 0, d2 - 1 >= 0, -d2 + 62 >= 0)>
// CHECK: #set2 = affine_set<(d0, d1, d2) : (-d2 + 61 >= 0, -d0 + 61 >= 0, d1 + 1 >= 0, -d1 + 13 >= 0, -d2 + 61 >= 0, d1 + 1 >= 0, -d1 + 13 >= 0, d0 - 1 >= 0, -d0 + 62 >= 0)>

// CHECK:  func.func private @"##call__Z39gpu__compute_integrated_ab2_tendencies_16CompilerMetadataI10StaticSizeI8_62__62_E12DynamicCheckvv7NDRangeILi2ES0_I6_4__4_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float32Li3E13CuTracedArrayISC_Li3ELi1E11_78__78__1_EESC_vvvES8_ISA_S9_vvvvSF_SC_vvvE21LatitudeLongitudeGridI15CuTracedRNumberISC_Li1EE7BoundedSL_SL_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_32__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_31__EESO_SQ_ESC_SC_SB_ISC_Li1E12StepRangeLenISC_7Float64ST_5Int64EESW_SC_SC_SW_SW_SB_ISC_Li1ESD_ISC_Li1ELi1E5_78__EESY_SY_SY_SC_SC_vSU_ESB_ISC_Li3ESD_ISC_Li3ELi1E12_78__78__31_EES11_S11_S11_SC_#258$par4"(%arg0: memref<1x78x78xf32, 1>, %arg1: memref<1x78x78xf32, 1>, %arg2: memref<31xf32, 1>, %arg3: memref<31x78x78xf32, 1>, %arg4: memref<31x78x78xf32, 1>, %arg5: memref<31x78x78xf32, 1>, %arg6: memref<31x78x78xf32, 1>) {
// CHECK-NEXT:    %cst = arith.constant 1.600000e+00 : f32
// CHECK-NEXT:    %cst_0 = arith.constant 6.000000e-01 : f32
// CHECK-NEXT:    %cst_1 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    affine.parallel (%arg7, %arg8) = (0, 0) to (62, 62) {
// CHECK-NEXT:      %0 = affine.load %arg2[8] {alignment = 32 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<31xf32, 1>
// CHECK-NEXT:      %1 = affine.load %arg5[8, %arg7 + 8, %arg8 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:      %2 = arith.mulf %1, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %3 = affine.load %arg3[8, %arg7 + 8, %arg8 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:      %4 = arith.mulf %3, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %5 = arith.subf %2, %4 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %6 = affine.if #set(%arg8) -> f32 {
// CHECK-NEXT:        affine.yield %5 : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.yield %cst_1 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %7 = arith.mulf %0, %6 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      affine.store %7, %arg0[0, %arg7 + 8, %arg8 + 8] : memref<1x78x78xf32, 1>
// CHECK-NEXT:      %8 = affine.load %arg2[8] {alignment = 32 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<31xf32, 1>
// CHECK-NEXT:      %9 = affine.load %arg6[8, %arg7 + 8, %arg8 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:      %10 = arith.mulf %9, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %11 = affine.load %arg4[8, %arg7 + 8, %arg8 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:      %12 = arith.mulf %11, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %13 = arith.subf %10, %12 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %14 = affine.if #set(%arg7) -> f32 {
// CHECK-NEXT:        affine.yield %13 : f32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.yield %cst_1 : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %15 = arith.mulf %8, %14 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      affine.store %15, %arg1[0, %arg7 + 8, %arg8 + 8] : memref<1x78x78xf32, 1>
// CHECK-NEXT:      %16 = affine.load %arg1[0, %arg7 + 8, %arg8 + 8] : memref<1x78x78xf32, 1>
// CHECK-NEXT:      %17 = affine.load %arg0[0, %arg7 + 8, %arg8 + 8] : memref<1x78x78xf32, 1>
// CHECK-NEXT:      %18:2 = affine.parallel (%arg9) = (0) to (14) reduce ("addf", "addf") -> (f32, f32) {
// CHECK-NEXT:        %21 = affine.load %arg2[%arg9 + 9] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<31xf32, 1>
// CHECK-NEXT:        %22 = affine.load %arg5[%arg9 + 9, %arg7 + 8, %arg8 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:        %23 = arith.mulf %22, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %24 = affine.load %arg3[%arg9 + 9, %arg7 + 8, %arg8 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:        %25 = arith.mulf %24, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %26 = arith.subf %23, %25 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %27 = affine.if #set1(%arg9, %arg7, %arg8) -> f32 {
// CHECK-NEXT:          affine.yield %26 : f32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %cst_1 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %28 = arith.mulf %21, %27 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %29 = affine.load %arg2[%arg9 + 9] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<31xf32, 1>
// CHECK-NEXT:        %30 = affine.load %arg6[%arg9 + 9, %arg7 + 8, %arg8 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:        %31 = arith.mulf %30, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %32 = affine.load %arg4[%arg9 + 9, %arg7 + 8, %arg8 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:        %33 = arith.mulf %32, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %34 = arith.subf %31, %33 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        %35 = affine.if #set2(%arg7, %arg9, %arg8) -> f32 {
// CHECK-NEXT:          affine.yield %34 : f32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          affine.yield %cst_1 : f32
// CHECK-NEXT:        }
// CHECK-NEXT:        %36 = arith.mulf %29, %35 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:        affine.yield %36, %28 : f32, f32
// CHECK-NEXT:      }
// CHECK-NEXT:      %19 = arith.addf %16, %18#0 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %20 = arith.addf %17, %18#1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      affine.store %20, %arg0[0, %arg7 + 8, %arg8 + 8] : memref<1x78x78xf32, 1>
// CHECK-NEXT:      affine.store %19, %arg1[0, %arg7 + 8, %arg8 + 8] : memref<1x78x78xf32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
