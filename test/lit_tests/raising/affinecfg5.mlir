// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

#set2 = affine_set<(d0, d1) : (-(d0 floordiv 16) >= 0, d0 mod 16 + d1 * 16 >= 0, d1 * -16 - d0 mod 16 + 179 >= 0)>
module {
  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI12_1_180__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__187__1_EEv17BoundaryConditionI6ZipperS8_ES7_I6Center4FacevE20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SQ_SR_SS_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EESW_SY_ESE_ISF_Li2ESG_ISF_Li2ELi1E10_194__187_EE8TripolarIS8_S8_S8_EvE16GridFittedBottomI5FieldISM_SM_vvvvSI_SF_vvvE23CenterImmersedConditionEvvvES7_I10NamedTupleI53__time___last__t___last_stage__t___iteration___stage_S7_ISF_SF_SF_S8_S8_EES1B_I36__u___v___w_______U___V___T___S___e_S7_ISE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__34_EES1F_SE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__35_EESI_S16_ISN_SM_vvvvSI_SF_vvvES16_ISM_SN_vvvvSI_SF_vvvES1F_S1F_S1F_EEE#1357$par86"(%arg0: memref<1x187x194xf64, 1>) {
    %cst = arith.constant -1.000000e+00 : f64
    affine.parallel (%arg1, %arg2) = (0, 0) to (12, 256) {
      affine.if #set2(%arg2, %arg1) {
        affine.for %arg3 = 0 to 50 {
          %0 = affine.load %arg0[0, -%arg3 + 136, -%arg2 - %arg1 * 16 + 186] : memref<1x187x194xf64, 1>
          %1 = arith.mulf %0, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
          affine.store %1, %arg0[0, %arg3 + 135, %arg2 + %arg1 * 16 + 7] : memref<1x187x194xf64, 1>
        }
      }
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI12_1_180__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__187__1_EEv17BoundaryConditionI6ZipperS8_ES7_I6Center4FacevE20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SQ_SR_SS_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EESW_SY_ESE_ISF_Li2ESG_ISF_Li2ELi1E10_194__187_EE8TripolarIS8_S8_S8_EvE16GridFittedBottomI5FieldISM_SM_vvvvSI_SF_vvvE23CenterImmersedConditionEvvvES7_I10NamedTupleI53__time___last__t___last_stage__t___iteration___stage_S7_ISF_SF_SF_S8_S8_EES1B_I36__u___v___w_______U___V___T___S___e_S7_ISE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__34_EES1F_SE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__35_EESI_S16_ISN_SM_vvvvSI_SF_vvvES16_ISM_SN_vvvvSI_SF_vvvES1F_S1F_S1F_EEE#1357$par86"(%[[memref:.+]]: memref<1x187x194xf64, 1>) {
// CHECK-NEXT:    %[[mone:.+]] = arith.constant -1.000000e+00 : f64
// CHECK-NEXT:    affine.parallel (%[[iv1:.+]]) = (0) to (180) {
// CHECK-NEXT:      affine.for %[[iv2:.+]] = 0 to 50 {
// CHECK-NEXT:        %[[loaded:.+]] = affine.load %[[memref]][0, -%[[iv2]] + 136, -%[[iv1]] + 186] : memref<1x187x194xf64, 1>
// CHECK-NEXT:        %[[stored:.+]] = arith.mulf %[[loaded]], %[[mone]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        affine.store %[[stored]], %arg0[0, %[[iv2]] + 135, %[[iv1]] + 7] : memref<1x187x194xf64, 1>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
