// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

#set = affine_set<(d0) : (d0 - 1 >= 0, -d0 + 20 >= 0)>
#set1 = affine_set<(d0, d1) : (-d0 + 19 >= 0, d1 + 1 >= 0, -d1 + 8 >= 0, d0 - 1 >= 0, -d0 + 20 >= 0, d1 + 1 >= 0, -d1 + 8 >= 0)>
#set2 = affine_set<(d0) : (d0 - 44 == 0)>
#set3 = affine_set<(d0) : (d0 == 0)>
#set4 = affine_set<(d0) : (d0 - 19 == 0)>
#set5 = affine_set<(d0) : (-d0 + 18 >= 0, d0 - 1 >= 0, -d0 + 19 >= 0)>
#set6 = affine_set<(d0) : (-d0 + 17 >= 0, d0 - 2 >= 0, -d0 + 18 >= 0)>
#set7 = affine_set<(d0) : (-d0 + 7 >= 0, d0 - 1 >= 0, -d0 + 8 >= 0)>
#set8 = affine_set<(d0) : (-d0 + 6 >= 0, d0 - 2 >= 0, -d0 + 7 >= 0)>
#set9 = affine_set<(d0) : (-d0 + 8 >= 0, d0 - 2 >= 0)>
#set10 = affine_set<(d0) : (-d0 + 7 >= 0, d0 - 3 >= 0, -d0 + 8 >= 0)>
#set11 = affine_set<(d0) : (d0 - 1 >= 0, -d0 + 18 >= 0)>
#set12 = affine_set<(d0) : (d0 - 9 == 0)>
#set13 = affine_set<(d0) : (d0 - 2 >= 0, -d0 + 18 >= 0)>
#set14 = affine_set<(d0) : (-d0 + 18 >= 0, d0 - 2 >= 0, -d0 + 19 >= 0)>
#set15 = affine_set<(d0) : (-d0 + 17 >= 0, d0 - 3 >= 0, -d0 + 18 >= 0)>
#set16 = affine_set<(d0) : (d0 - 3 >= 0, -d0 + 17 >= 0)>
#set17 = affine_set<(d0) : (-d0 + 17 >= 0, d0 - 1 >= 0, -d0 + 18 >= 0)>
#set18 = affine_set<(d0) : (-d0 + 16 >= 0, d0 - 2 >= 0, -d0 + 17 >= 0)>
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module @"reactant_loop!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"##call__Z29gpu__compute_barotropic_mode_16CompilerMetadataI10StaticSizeI8_45__20_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E11_59__34__1_EESC_vvvES8_ISA_S9_vvvvSB_ISC_Li3ESD_ISC_Li3ELi1E11_59__35__1_EESC_vvvE21LatitudeLongitudeGridISC_8Periodic7BoundedSM_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_25__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_24__EESP_SR_ESC_SC_SB_ISC_Li1E12StepRangeLenISC_14TwicePrecisionISC_ESV_5Int64EESY_SC_SC_SY_SY_SB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EES10_S10_S10_SC_SC_vSW_EvSB_ISC_Li3ESD_ISC_Li3ELi1E12_59__34__24_EESB_ISC_Li3ESD_ISC_Li3ELi1E12_59__35__24_EESF_#555$par153"(%arg0: memref<1x34x59xf64, 1>, %arg1: memref<1x35x59xf64, 1>, %arg2: memref<24xf64, 1>, %arg3: memref<24x34x59xf64, 1>, %arg4: memref<24x35x59xf64, 1>) {
    affine.parallel (%arg5, %arg6) = (0, 0) to (20, 45) {
      %0 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
      %1 = affine.load %arg3[7, %arg5 + 7, %arg6 + 7] : memref<24x34x59xf64, 1>
      %2 = arith.mulf %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %2, %arg0[0, %arg5 + 7, %arg6 + 7] : memref<1x34x59xf64, 1>
      %3 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
      %4 = affine.load %arg4[7, %arg5 + 7, %arg6 + 7] : memref<24x35x59xf64, 1>
      %5 = arith.mulf %3, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %5, %arg1[0, %arg5 + 7, %arg6 + 7] : memref<1x35x59xf64, 1>
      %6 = affine.load %arg0[0, %arg5 + 7, %arg6 + 7] : memref<1x34x59xf64, 1>
      %7 = affine.load %arg1[0, %arg5 + 7, %arg6 + 7] : memref<1x35x59xf64, 1>
      %8:2 = affine.for %arg7 = 0 to 9 iter_args(%arg8 = %6, %arg9 = %7) -> (f64, f64) {
        %9 = affine.load %arg2[%arg7 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
        %10 = affine.load %arg3[%arg7 + 8, %arg5 + 7, %arg6 + 7] : memref<24x34x59xf64, 1>
        %11 = arith.mulf %9, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
        %12 = arith.addf %arg8, %11 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %12, %arg0[0, %arg5 + 7, %arg6 + 7] : memref<1x34x59xf64, 1>
        %13 = affine.load %arg2[%arg7 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
        %14 = affine.load %arg4[%arg7 + 8, %arg5 + 7, %arg6 + 7] : memref<24x35x59xf64, 1>
        %15 = arith.mulf %13, %14 {fastmathFlags = #llvm.fastmath<none>} : f64
        %16 = arith.addf %arg9, %15 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %16, %arg1[0, %arg5 + 7, %arg6 + 7] : memref<1x35x59xf64, 1>
        affine.yield %12, %16 : f64, f64
      }
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z29gpu__compute_barotropic_mode_16CompilerMetadataI10StaticSizeI8_45__20_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E11_59__34__1_EESC_vvvES8_ISA_S9_vvvvSB_ISC_Li3ESD_ISC_Li3ELi1E11_59__35__1_EESC_vvvE21LatitudeLongitudeGridISC_8Periodic7BoundedSM_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_25__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_24__EESP_SR_ESC_SC_SB_ISC_Li1E12StepRangeLenISC_14TwicePrecisionISC_ESV_5Int64EESY_SC_SC_SY_SY_SB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EES10_S10_S10_SC_SC_vSW_EvSB_ISC_Li3ESD_ISC_Li3ELi1E12_59__34__24_EESB_ISC_Li3ESD_ISC_Li3ELi1E12_59__35__24_EESF_#555$par153"(%arg0: memref<1x34x59xf64, 1>, %arg1: memref<1x35x59xf64, 1>, %arg2: memref<24xf64, 1>, %arg3: memref<24x34x59xf64, 1>, %arg4: memref<24x35x59xf64, 1>) {
// CHECK-NEXT:    affine.parallel (%arg5, %arg6) = (0, 0) to (20, 45) {
// CHECK-NEXT:      %0 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
// CHECK-NEXT:      %1 = affine.load %arg3[7, %arg5 + 7, %arg6 + 7] : memref<24x34x59xf64, 1>
// CHECK-NEXT:      %2 = arith.mulf %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %2, %arg0[0, %arg5 + 7, %arg6 + 7] : memref<1x34x59xf64, 1>
// CHECK-NEXT:      %3 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
// CHECK-NEXT:      %4 = affine.load %arg4[7, %arg5 + 7, %arg6 + 7] : memref<24x35x59xf64, 1>
// CHECK-NEXT:      %5 = arith.mulf %3, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %5, %arg1[0, %arg5 + 7, %arg6 + 7] : memref<1x35x59xf64, 1>
// CHECK-NEXT:      %6 = affine.load %arg0[0, %arg5 + 7, %arg6 + 7] : memref<1x34x59xf64, 1>
// CHECK-NEXT:      %7 = affine.load %arg1[0, %arg5 + 7, %arg6 + 7] : memref<1x35x59xf64, 1>
// CHECK-NEXT:      %8:2 = affine.parallel (%arg7) = (0) to (9) reduce ("addf", "addf") -> (f64, f64) {
// CHECK-NEXT:        %11 = affine.load %arg2[%arg7 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
// CHECK-NEXT:        %12 = affine.load %arg3[%arg7 + 8, %arg5 + 7, %arg6 + 7] : memref<24x34x59xf64, 1>
// CHECK-NEXT:        %13 = arith.mulf %11, %12 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %14 = affine.load %arg2[%arg7 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
// CHECK-NEXT:        %15 = affine.load %arg4[%arg7 + 8, %arg5 + 7, %arg6 + 7] : memref<24x35x59xf64, 1>
// CHECK-NEXT:        %16 = arith.mulf %14, %15 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        affine.yield %13, %16 : f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %9 = arith.addf %6, %8#0 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %10 = arith.addf %7, %8#1 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %10, %arg1[0, %arg5 + 7, %arg6 + 7] : memref<1x35x59xf64, 1>
// CHECK-NEXT:      affine.store %9, %arg0[0, %arg5 + 7, %arg6 + 7] : memref<1x34x59xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:}

// -----

#set = affine_set<(d0) : (d0 - 1 >= 0, -d0 + 20 >= 0)>
#set1 = affine_set<(d0, d1) : (-d0 + 19 >= 0, d1 + 1 >= 0, -d1 + 8 >= 0, d0 - 1 >= 0, -d0 + 20 >= 0, d1 + 1 >= 0, -d1 + 8 >= 0)>
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {
  func.func private @"##call__Z39gpu__compute_integrated_ab2_tendencies_16CompilerMetadataI10StaticSizeI8_45__20_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E11_59__34__1_EESC_vvvES8_ISA_S9_vvvvSB_ISC_Li3ESD_ISC_Li3ELi1E11_59__35__1_EESC_vvvE21LatitudeLongitudeGridISC_8Periodic7BoundedSM_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_25__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_24__EESP_SR_ESC_SC_SB_ISC_Li1E12StepRangeLenISC_14TwicePrecisionISC_ESV_5Int64EESY_SC_SC_SY_SY_SB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EES10_S10_S10_SC_SC_vSW_EvSB_ISC_Li3ESD_ISC_Li3ELi1E12_59__34__24_EESB_ISC_Li3ESD_ISC_Li3ELi1E12_59__35__24_EES13_S15_SC_#445$par98"(%arg0: memref<1x34x59xf64, 1>, %arg1: memref<1x35x59xf64, 1>, %arg2: memref<24xf64, 1>, %arg3: memref<24x34x59xf64, 1>, %arg4: memref<24x35x59xf64, 1>, %arg5: memref<24x34x59xf64, 1>, %arg6: memref<24x35x59xf64, 1>) {
    %cst = arith.constant 1.600000e+00 : f64
    %cst_0 = arith.constant 6.000000e-01 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    affine.parallel (%arg7, %arg8) = (0, 0) to (20, 45) {
      %0 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
      %1 = affine.load %arg5[7, %arg7 + 7, %arg8 + 7] : memref<24x34x59xf64, 1>
      %2 = arith.mulf %1, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      %3 = affine.load %arg3[7, %arg7 + 7, %arg8 + 7] : memref<24x34x59xf64, 1>
      %4 = arith.mulf %3, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
      %5 = arith.subf %2, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
      %6 = arith.mulf %0, %5 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %6, %arg0[0, %arg7 + 7, %arg8 + 7] : memref<1x34x59xf64, 1>
      %7 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
      %8 = affine.load %arg6[7, %arg7 + 7, %arg8 + 7] : memref<24x35x59xf64, 1>
      %9 = arith.mulf %8, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      %10 = affine.load %arg4[7, %arg7 + 7, %arg8 + 7] : memref<24x35x59xf64, 1>
      %11 = arith.mulf %10, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
      %12 = arith.subf %9, %11 {fastmathFlags = #llvm.fastmath<none>} : f64
      %13 = affine.if #set(%arg7) -> f64 {
        affine.yield %12 : f64
      } else {
        affine.yield %cst_1 : f64
      }
      %14 = arith.mulf %7, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %14, %arg1[0, %arg7 + 7, %arg8 + 7] : memref<1x35x59xf64, 1>
      %15 = affine.load %arg0[0, %arg7 + 7, %arg8 + 7] : memref<1x34x59xf64, 1>
      %16 = affine.load %arg1[0, %arg7 + 7, %arg8 + 7] : memref<1x35x59xf64, 1>
      %17:2 = affine.for %arg9 = 0 to 9 iter_args(%arg10 = %15, %arg11 = %16) -> (f64, f64) {
        %18 = affine.load %arg2[%arg9 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
        %19 = affine.load %arg5[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<24x34x59xf64, 1>
        %20 = arith.mulf %19, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
        %21 = affine.load %arg3[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<24x34x59xf64, 1>
        %22 = arith.mulf %21, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
        %23 = arith.subf %20, %22 {fastmathFlags = #llvm.fastmath<none>} : f64
        %24 = arith.mulf %18, %23 {fastmathFlags = #llvm.fastmath<none>} : f64
        %25 = arith.addf %arg10, %24 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %25, %arg0[0, %arg7 + 7, %arg8 + 7] : memref<1x34x59xf64, 1>
        %26 = affine.load %arg2[%arg9 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
        %27 = affine.load %arg6[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<24x35x59xf64, 1>
        %28 = arith.mulf %27, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
        %29 = affine.load %arg4[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<24x35x59xf64, 1>
        %30 = arith.mulf %29, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
        %31 = arith.subf %28, %30 {fastmathFlags = #llvm.fastmath<none>} : f64
        %32 = affine.if #set1(%arg7, %arg9) -> f64 {
          affine.yield %31 : f64
        } else {
          affine.yield %cst_1 : f64
        }
        %33 = arith.mulf %26, %32 {fastmathFlags = #llvm.fastmath<none>} : f64
        %34 = arith.addf %arg11, %33 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %34, %arg1[0, %arg7 + 7, %arg8 + 7] : memref<1x35x59xf64, 1>
        affine.yield %25, %34 : f64, f64
      }
    }
    return
  }
}

//CHECK:  func.func private @"##call__Z39gpu__compute_integrated_ab2_tendencies_16CompilerMetadataI10StaticSizeI8_45__20_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E11_59__34__1_EESC_vvvES8_ISA_S9_vvvvSB_ISC_Li3ESD_ISC_Li3ELi1E11_59__35__1_EESC_vvvE21LatitudeLongitudeGridISC_8Periodic7BoundedSM_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_25__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_24__EESP_SR_ESC_SC_SB_ISC_Li1E12StepRangeLenISC_14TwicePrecisionISC_ESV_5Int64EESY_SC_SC_SY_SY_SB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EES10_S10_S10_SC_SC_vSW_EvSB_ISC_Li3ESD_ISC_Li3ELi1E12_59__34__24_EESB_ISC_Li3ESD_ISC_Li3ELi1E12_59__35__24_EES13_S15_SC_#445$par98"(%arg0: memref<1x34x59xf64, 1>, %arg1: memref<1x35x59xf64, 1>, %arg2: memref<24xf64, 1>, %arg3: memref<24x34x59xf64, 1>, %arg4: memref<24x35x59xf64, 1>, %arg5: memref<24x34x59xf64, 1>, %arg6: memref<24x35x59xf64, 1>) {
//CHECK-NEXT:    %cst = arith.constant 1.600000e+00 : f64
//CHECK-NEXT:    %cst_0 = arith.constant 6.000000e-01 : f64
//CHECK-NEXT:    %cst_1 = arith.constant 0.000000e+00 : f64
//CHECK-NEXT:    affine.parallel (%arg7, %arg8) = (0, 0) to (20, 45) {
//CHECK-NEXT:      %0 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
//CHECK-NEXT:      %1 = affine.load %arg5[7, %arg7 + 7, %arg8 + 7] : memref<24x34x59xf64, 1>
//CHECK-NEXT:      %2 = arith.mulf %1, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:      %3 = affine.load %arg3[7, %arg7 + 7, %arg8 + 7] : memref<24x34x59xf64, 1>
//CHECK-NEXT:      %4 = arith.mulf %3, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:      %5 = arith.subf %2, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:      %6 = arith.mulf %0, %5 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:      affine.store %6, %arg0[0, %arg7 + 7, %arg8 + 7] : memref<1x34x59xf64, 1>
//CHECK-NEXT:      %7 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
//CHECK-NEXT:      %8 = affine.load %arg6[7, %arg7 + 7, %arg8 + 7] : memref<24x35x59xf64, 1>
//CHECK-NEXT:      %9 = arith.mulf %8, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:      %10 = affine.load %arg4[7, %arg7 + 7, %arg8 + 7] : memref<24x35x59xf64, 1>
//CHECK-NEXT:      %11 = arith.mulf %10, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:      %12 = arith.subf %9, %11 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:      %13 = affine.if #set(%arg7) -> f64 {
//CHECK-NEXT:        affine.yield %12 : f64
//CHECK-NEXT:      } else {
//CHECK-NEXT:        affine.yield %cst_1 : f64
//CHECK-NEXT:      }
//CHECK-NEXT:      %14 = arith.mulf %7, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:      affine.store %14, %arg1[0, %arg7 + 7, %arg8 + 7] : memref<1x35x59xf64, 1>
//CHECK-NEXT:      %15 = affine.load %arg0[0, %arg7 + 7, %arg8 + 7] : memref<1x34x59xf64, 1>
//CHECK-NEXT:      %16 = affine.load %arg1[0, %arg7 + 7, %arg8 + 7] : memref<1x35x59xf64, 1>
//CHECK-NEXT:      %17:2 = affine.parallel (%arg9) = (0) to (9) reduce ("addf", "addf") -> (f64, f64) {
//CHECK-NEXT:        %20 = affine.load %arg2[%arg9 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
//CHECK-NEXT:        %21 = affine.load %arg5[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<24x34x59xf64, 1>
//CHECK-NEXT:        %22 = arith.mulf %21, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:        %23 = affine.load %arg3[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<24x34x59xf64, 1>
//CHECK-NEXT:        %24 = arith.mulf %23, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:        %25 = arith.subf %22, %24 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:        %26 = arith.mulf %20, %25 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:        %27 = affine.load %arg2[%arg9 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
//CHECK-NEXT:        %28 = affine.load %arg6[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<24x35x59xf64, 1>
//CHECK-NEXT:        %29 = arith.mulf %28, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:        %30 = affine.load %arg4[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<24x35x59xf64, 1>
//CHECK-NEXT:        %31 = arith.mulf %30, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:        %32 = arith.subf %29, %31 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:        %33 = affine.if #set1(%arg7, %arg9) -> f64 {
//CHECK-NEXT:          affine.yield %32 : f64
//CHECK-NEXT:        } else {
//CHECK-NEXT:          affine.yield %cst_1 : f64
//CHECK-NEXT:        }
//CHECK-NEXT:        %34 = arith.mulf %27, %33 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:        affine.yield %26, %34 : f64, f64
//CHECK-NEXT:      }
//CHECK-NEXT:      %18 = arith.addf %15, %17#0 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:      %19 = arith.addf %16, %17#1 {fastmathFlags = #llvm.fastmath<none>} : f64
//CHECK-NEXT:      affine.store %19, %arg1[0, %arg7 + 7, %arg8 + 7] : memref<1x35x59xf64, 1>
//CHECK-NEXT:      affine.store %18, %arg0[0, %arg7 + 7, %arg8 + 7] : memref<1x34x59xf64, 1>
//CHECK-NEXT:    }
//CHECK-NEXT:    return
//CHECK-NEXT:  }