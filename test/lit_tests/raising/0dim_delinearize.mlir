// RUN: enzymexlamlir-opt %s --delinearize-indexing -allow-unregistered-dialect --canonicalize | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {
  func.func private @"##call__Z43gpu__interpolate_primary_atmospheric_state_16CompilerMetadataI16OffsetStaticSizeI13_0_181__0_91_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__6_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE10NamedTupleI35__u___v___T___p___q___Qs___Q____Mp_S7_I11OffsetArrayI7Float64Li3E13CuTracedArrayISG_Li3ELi1E13_194__104__1_EESJ_SJ_SJ_SJ_SJ_SJ_SJ_EESE_I8__i___j_S7_I5FieldI6CenterSN_vvvvSJ_SG_vvvESO_EE16TimeInterpolatorI15CuTracedRNumberISG_Li1EESS_IS8_Li1EESU_S8_E20ImmersedBoundaryGridISG_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISG_SX_SY_SZ_28StaticVerticalDiscretizationISF_ISG_Li1ESH_ISG_Li1ELi1E5_35__EESF_ISG_Li1ESH_ISG_Li1ELi1E5_34__EES13_S15_E8TripolarIS8_S8_S8_ESF_ISG_Li2ESH_ISG_Li2ELi1E10_194__104_EES1A_S1A_S1A_vE16GridFittedBottomISO_23CenterImmersedConditionEvvvESE_I8__u___v_S7_ISF_ISG_Li4ESH_ISG_Li4ELi1E17_366__186__1__24_EES1H_EESE_I8__T___q_S1I_ES1H_SE_I23__shortwave___longwave_S1I_ESE_I14__rain___snow_S1I_E8InMemoryIvE5Clamp#1401$par285"(%arg0: memref<1x104x194xf64, 1>, %arg1: memref<1x104x194xf64, 1>, %arg2: memref<1x104x194xf64, 1>, %arg3: memref<1x104x194xf64, 1>, %arg4: memref<1x104x194xf64, 1>, %arg5: memref<1x104x194xf64, 1>, %arg6: memref<1x104x194xf64, 1>, %arg7: memref<1x104x194xf64, 1>, %arg8: memref<1x104x194xf64, 1>, %arg9: memref<1x104x194xf64, 1>, %arg10: memref<f64, 1>, %arg11: memref<i64, 1>, %arg12: memref<i64, 1>, %arg13: memref<104x194xf64, 1>, %arg14: memref<104x194xf64, 1>, %arg15: memref<104x194xf64, 1>, %arg16: memref<24x1x186x366xf64, 1>, %arg17: memref<24x1x186x366xf64, 1>, %arg18: memref<24x1x186x366xf64, 1>, %arg19: memref<24x1x186x366xf64, 1>, %arg20: memref<24x1x186x366xf64, 1>, %arg21: memref<24x1x186x366xf64, 1>, %arg22: memref<24x1x186x366xf64, 1>, %arg23: memref<24x1x186x366xf64, 1>, %arg24: memref<24x1x186x366xf64, 1>) {
    %1 = "enzymexla.memref2pointer"(%arg11) : (memref<i64, 1>) -> !llvm.ptr<1>
    affine.parallel (%arg25, %arg26) = (0, 0) to (92, 182) {
      %21 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr<1>) -> memref<?xi64, 1>
      %22 = affine.load %21[0] {alignment = 128 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi64, 1>
      "test.use"(%22) : (i64) -> ()
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z43gpu__interpolate_primary_atmospheric_state_16CompilerMetadataI16OffsetStaticSizeI13_0_181__0_91_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__6_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE10NamedTupleI35__u___v___T___p___q___Qs___Q____Mp_S7_I11OffsetArrayI7Float64Li3E13CuTracedArrayISG_Li3ELi1E13_194__104__1_EESJ_SJ_SJ_SJ_SJ_SJ_SJ_EESE_I8__i___j_S7_I5FieldI6CenterSN_vvvvSJ_SG_vvvESO_EE16TimeInterpolatorI15CuTracedRNumberISG_Li1EESS_IS8_Li1EESU_S8_E20ImmersedBoundaryGridISG_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISG_SX_SY_SZ_28StaticVerticalDiscretizationISF_ISG_Li1ESH_ISG_Li1ELi1E5_35__EESF_ISG_Li1ESH_ISG_Li1ELi1E5_34__EES13_S15_E8TripolarIS8_S8_S8_ESF_ISG_Li2ESH_ISG_Li2ELi1E10_194__104_EES1A_S1A_S1A_vE16GridFittedBottomISO_23CenterImmersedConditionEvvvESE_I8__u___v_S7_ISF_ISG_Li4ESH_ISG_Li4ELi1E17_366__186__1__24_EES1H_EESE_I8__T___q_S1I_ES1H_SE_I23__shortwave___longwave_S1I_ESE_I14__rain___snow_S1I_E8InMemoryIvE5Clamp#1401$par285"(%arg0: memref<1x104x194xf64, 1>, %arg1: memref<1x104x194xf64, 1>, %arg2: memref<1x104x194xf64, 1>, %arg3: memref<1x104x194xf64, 1>, %arg4: memref<1x104x194xf64, 1>, %arg5: memref<1x104x194xf64, 1>, %arg6: memref<1x104x194xf64, 1>, %arg7: memref<1x104x194xf64, 1>, %arg8: memref<1x104x194xf64, 1>, %arg9: memref<1x104x194xf64, 1>, %arg10: memref<f64, 1>, %arg11: memref<i64, 1>, %arg12: memref<i64, 1>, %arg13: memref<104x194xf64, 1>, %arg14: memref<104x194xf64, 1>, %arg15: memref<104x194xf64, 1>, %arg16: memref<24x1x186x366xf64, 1>, %arg17: memref<24x1x186x366xf64, 1>, %arg18: memref<24x1x186x366xf64, 1>, %arg19: memref<24x1x186x366xf64, 1>, %arg20: memref<24x1x186x366xf64, 1>, %arg21: memref<24x1x186x366xf64, 1>, %arg22: memref<24x1x186x366xf64, 1>, %arg23: memref<24x1x186x366xf64, 1>, %arg24: memref<24x1x186x366xf64, 1>) {
// CHECK-NEXT:    affine.parallel (%arg25, %arg26) = (0, 0) to (92, 182) {
// CHECK-NEXT:      %0 = affine.load %arg11[] : memref<i64, 1>
// CHECK-NEXT:      "test.use"(%0) : (i64) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
