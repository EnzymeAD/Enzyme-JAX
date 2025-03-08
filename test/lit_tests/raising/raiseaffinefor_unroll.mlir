// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo
#set = affine_set<(d0) : (-d0 + 89 >= 0)>
module {
    func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI14_1_180__21_21_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I8_180__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__187__1_EE17BoundaryConditionI4FluxvESJ_I6ZipperS8_ES7_I6CenterSO_4FaceE20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SS_ST_SU_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EESY_S10_ESE_ISF_Li2ESG_ISF_Li2ELi1E10_194__187_EE8TripolarIS8_S8_S8_EvE16GridFittedBottomI5FieldISO_SO_vvvvSI_SF_vvvE23CenterImmersedConditionEvvvES7_I10NamedTupleI53__time___last__t___last_stage__t___iteration___stage_S7_ISF_SF_SF_S8_S8_EES1D_I36__u___v___w_______U___V___T___S___e_S7_ISE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__34_EES1H_SE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__35_EESI_S18_ISP_SO_vvvvSI_SF_vvvES18_ISO_SP_vvvvSI_SF_vvvES1H_S1H_S1H_EEE#814$par16"(%arg0: memref<1x187x194xf64, 1>) {
        affine.parallel (%arg1) = (0) to (180) {
        %0 = affine.load %arg0[0, 51, %arg1 + 7] : memref<1x187x194xf64, 1>
        affine.store %0, %arg0[0, 50, %arg1 + 7] : memref<1x187x194xf64, 1>
        affine.for %arg2 = 0 to 50 {
            %4 = affine.load %arg0[0, -%arg2 + 134, -%arg1 + 186] : memref<1x187x194xf64, 1>
            affine.store %4, %arg0[0, %arg2 + 136, %arg1 + 7] : memref<1x187x194xf64, 1>
        }
        %1 = affine.load %arg0[0, 135, -%arg1 + 186] : memref<1x187x194xf64, 1>
        %2 = affine.load %arg0[0, 135, %arg1 + 7] : memref<1x187x194xf64, 1>
        %3 = affine.if #set(%arg1) -> f64 {
            affine.yield %2 : f64
        } else {
            affine.yield %1 : f64
        }
        affine.store %3, %arg0[0, 135, %arg1 + 7] : memref<1x187x194xf64, 1>
        }
        return
    }
}
