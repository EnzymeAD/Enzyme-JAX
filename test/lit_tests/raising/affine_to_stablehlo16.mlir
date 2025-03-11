// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0 * -194 + d1 * 19206 + d2 * 307296 + 151908)>
#map1 = affine_map<(d0, d1)[s0] -> (-d0 + s0 - 1 - d1 * 16)>
#set = affine_set<(d0, d1, d2, d3) : (-d0 - d1 * 16 + 19 >= 0, d3 + d2 * 16 >= 0, d2 * -16 - d3 + 179 >= 0)>
#set1 = affine_set<(d0, d1) : (d1 + d0 * 16 - 1 >= 0)>
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {
  func.func @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI13_1_180__1_20_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__2_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEES7_I11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__99__34_EESI_SI_SI_SI_ES7_I17BoundaryConditionI4FluxvEvSM_SM_SM_ES7_ISK_I6ZipperS8_ESP_SP_SP_SP_ES7_IS7_I4Face6CenterSS_ES7_ISS_SR_SS_ES7_ISS_SS_SS_ESV_SV_E20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SY_SZ_S10_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EES14_S16_E8TripolarIS8_S8_S8_ESE_ISF_Li2ESG_ISF_Li2ELi1E9_194__99_EES1B_S1B_S1B_vE16GridFittedBottomI5FieldISS_SS_vvvvSE_ISF_Li3ESG_ISF_Li3ELi1E12_194__99__1_EESF_vvvE23CenterImmersedConditionEvvvES7_I10NamedTupleI53__time___last__t___last_stage__t___iteration___stage_S7_ISF_SF_SF_S8_S8_EES1L_I36__u___v___w_______U___V___T___S___e_S7_ISI_SI_SE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__35_EESE_ISF_Li3ESG_ISF_Li3ELi1E13_194__123__1_EES1E_ISR_SS_vvvvS1R_SF_vvvES1E_ISS_SR_vvvvS1R_SF_vvvESI_SI_SI_EEE#9395$par78"(%arg0: memref<194xf64, 1>, %arg1: memref<34x99x194xf64, 1>, %arg2: memref<34x99x194xf64, 1>, %arg3: memref<34x99x194xf64, 1>, %arg4: memref<34x99x194xf64, 1>) {
   %c-1_i64 = arith.constant -1 : i64
   %c1_i64 = arith.constant 1 : i64
   %c2_i64 = arith.constant 2 : i64
   %c182_i64 = arith.constant 182 : i64
   %c0 = arith.constant 0 : index
   %c6 = arith.constant 6 : index
    affine.parallel (%arg6, %arg7, %arg8, %arg9) = (0, 0, 0, 0) to (2, 16, 12, 16) {
          %8 = affine.apply #map1(%arg9, %arg9)[%c6]
          %10 = memref.load %arg0[%8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<194xf64, 1>
          affine.store %10, %arg4[%arg7 + 7, 92, %arg9 + 7] : memref<34x99x194xf64, 1>
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI13_1_180__1_20_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__2_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEES7_I11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__99__34_EESI_SI_SI_SI_ES7_I17BoundaryConditionI4FluxvEvSM_SM_SM_ES7_ISK_I6ZipperS8_ESP_SP_SP_SP_ES7_IS7_I4Face6CenterSS_ES7_ISS_SR_SS_ES7_ISS_SS_SS_ESV_SV_E20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SY_SZ_S10_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EES14_S16_E8TripolarIS8_S8_S8_ESE_ISF_Li2ESG_ISF_Li2ELi1E9_194__99_EES1B_S1B_S1B_vE16GridFittedBottomI5FieldISS_SS_vvvvSE_ISF_Li3ESG_ISF_Li3ELi1E12_194__99__1_EESF_vvvE23CenterImmersedConditionEvvvES7_I10NamedTupleI53__time___last__t___last_stage__t___iteration___stage_S7_ISF_SF_SF_S8_S8_EES1L_I36__u___v___w_______U___V___T___S___e_S7_ISI_SI_SE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__35_EESE_ISF_Li3ESG_ISF_Li3ELi1E13_194__123__1_EES1E_ISR_SS_vvvvS1R_SF_vvvES1E_ISS_SR_vvvvS1R_SF_vvvESI_SI_SI_EEE#9395$par78_raised"(%arg0: tensor<194xf64>, %arg1: tensor<34x99x194xf64>, %arg2: tensor<34x99x194xf64>, %arg3: tensor<34x99x194xf64>, %arg4: tensor<34x99x194xf64>) -> (tensor<194xf64>, tensor<34x99x194xf64>, tensor<34x99x194xf64>, tensor<34x99x194xf64>, tensor<34x99x194xf64>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<182> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<2xi64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<2xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c_5 : tensor<2xi64>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<1> : tensor<2xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_6 : tensor<2xi64>
// CHECK-NEXT:    %3 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %4 = stablehlo.add %3, %c_7 : tensor<16xi64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %5 = stablehlo.multiply %4, %c_8 : tensor<16xi64>
// CHECK-NEXT:    %6 = stablehlo.iota dim = 0 : tensor<12xi64>
// CHECK-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<12xi64>
// CHECK-NEXT:    %7 = stablehlo.add %6, %c_9 : tensor<12xi64>
// CHECK-NEXT:    %c_10 = stablehlo.constant dense<1> : tensor<12xi64>
// CHECK-NEXT:    %8 = stablehlo.multiply %7, %c_10 : tensor<12xi64>
// CHECK-NEXT:    %9 = stablehlo.iota dim = 0 : tensor<16xi64>
// CHECK-NEXT:    %c_11 = stablehlo.constant dense<0> : tensor<16xi64>
// CHECK-NEXT:    %10 = stablehlo.add %9, %c_11 : tensor<16xi64>
// CHECK-NEXT:    %c_12 = stablehlo.constant dense<1> : tensor<16xi64>
// CHECK-NEXT:    %11 = stablehlo.multiply %10, %c_12 : tensor<16xi64>
// CHECK-NEXT:    %c_13 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %12 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<16xi64>) -> tensor<16xi64>
// CHECK-NEXT:    %13 = stablehlo.broadcast_in_dim %c_13, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %14 = arith.muli %12, %13 overflow<nsw> : tensor<16xi64>
// CHECK-NEXT:    %15 = stablehlo.broadcast_in_dim %14, dims = [0] : (tensor<16xi64>) -> tensor<16xi64>
// CHECK-NEXT:    %16 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %17 = arith.addi %15, %16 : tensor<16xi64>
// CHECK-NEXT:    %c_14 = stablehlo.constant dense<-16> : tensor<i64>
// CHECK-NEXT:    %18 = stablehlo.broadcast_in_dim %11, dims = [0] : (tensor<16xi64>) -> tensor<16xi64>
// CHECK-NEXT:    %19 = stablehlo.broadcast_in_dim %c_14, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %20 = arith.muli %18, %19 overflow<nsw> : tensor<16xi64>
// CHECK-NEXT:    %21 = stablehlo.broadcast_in_dim %17, dims = [0] : (tensor<16xi64>) -> tensor<16xi64>
// CHECK-NEXT:    %22 = stablehlo.broadcast_in_dim %20, dims = [0] : (tensor<16xi64>) -> tensor<16xi64>
// CHECK-NEXT:    %23 = arith.addi %21, %22 : tensor<16xi64>
// CHECK-NEXT:    %c_15 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %24 = stablehlo.broadcast_in_dim %23, dims = [0] : (tensor<16xi64>) -> tensor<16xi64>
// CHECK-NEXT:    %25 = stablehlo.broadcast_in_dim %c_15, dims = [] : (tensor<i64>) -> tensor<16xi64>
// CHECK-NEXT:    %26 = arith.addi %24, %25 : tensor<16xi64>
// CHECK-NEXT:    %27 = stablehlo.reshape %26 : (tensor<16xi64>) -> tensor<16x1xi64>
// CHECK-NEXT:    %28 = "stablehlo.gather"(%arg0, %27) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<194xf64>, tensor<16x1xi64>) -> tensor<16xf64>
// CHECK-NEXT:    %29 = stablehlo.reshape %28 : (tensor<16xf64>) -> tensor<16xf64>
// CHECK-NEXT:    %c_16 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_17 = stablehlo.constant dense<92> : tensor<i64>
// CHECK-NEXT:    %c_18 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %30 = stablehlo.broadcast_in_dim %29, dims = [2] : (tensor<16xf64>) -> tensor<16x1x16xf64>
// CHECK-NEXT:    %31 = stablehlo.reverse %30, dims = [] : tensor<16x1x16xf64>
// CHECK-NEXT:    %32 = stablehlo.dynamic_update_slice %arg4, %31, %c_16, %c_17, %c_18 : (tensor<34x99x194xf64>, tensor<16x1x16xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<34x99x194xf64>
// CHECK-NEXT:    return %arg0, %arg1, %arg2, %arg3, %32 : tensor<194xf64>, tensor<34x99x194xf64>, tensor<34x99x194xf64>, tensor<34x99x194xf64>, tensor<34x99x194xf64>
// CHECK-NEXT:  }

