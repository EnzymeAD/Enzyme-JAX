// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "julia_iterate_interface_fluxes_1507321">
#alias_scope_domain1 = #llvm.alias_scope_domain<id = distinct[1]<>, description = "julia_iterate_interface_fluxes_1507321">
#set = affine_set<(d0) : (-d0 + 89 >= 0)>
#set1 = affine_set<(d0) : (d0 - 1 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (-d0 - d1 * 16 + 19 >= 0, d3 + d2 * 16 >= 0, d2 * -16 - d3 + 179 >= 0)>
#set3 = affine_set<(d0, d1) : (d1 + d0 * 16 - 1 >= 0)>
#set4 = affine_set<(d0, d1) : (d0 * -16 - d1 + 89 >= 0)>
#set5 = affine_set<(d0) : (d0 + 1 >= 0)>
#set6 = affine_set<(d0) : (-d0 + 18 >= 0)>
#set7 = affine_set<(d0) : (d0 >= 0)>
#set8 = affine_set<(d0) : (-d0 + 19 >= 0)>
#set9 = affine_set<(d0) : (-d0 + 17 >= 0)>
#set10 = affine_set<(d0) : (d0 - 19 == 0)>
#set11 = affine_set<(d0, d1) : (-d0 - d1 * 16 + 102 >= 0)>
#set12 = affine_set<(d0) : (d0 - 179 == 0)>
#set13 = affine_set<(d0, d1) : (d0 == 0, d1 == 0)>
#set14 = affine_set<(d0, d1, d2, d3) : (-d0 - d1 * 16 + 102 >= 0, d3 + d2 * 16 >= 0, d2 * -16 - d3 + 179 >= 0)>
#set15 = affine_set<(d0, d1) : (d1 == 0, d0 == 0)>
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#alias_scope = #llvm.alias_scope<id = distinct[2]<>, domain = #alias_scope_domain>
#alias_scope1 = #llvm.alias_scope<id = distinct[3]<>, domain = #alias_scope_domain1>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {
  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI13_1_180__1_20_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__2_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEES7_I11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__99__34_EESI_SI_SI_SI_ES7_I17BoundaryConditionI4FluxvEvSM_SM_SM_ES7_ISK_I6ZipperS8_ESP_SP_SP_SP_ES7_IS7_I4Face6CenterSS_ES7_ISS_SR_SS_ES7_ISS_SS_SS_ESV_SV_E20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SY_SZ_S10_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EES14_S16_E8TripolarIS8_S8_S8_ESE_ISF_Li2ESG_ISF_Li2ELi1E9_194__99_EES1B_S1B_S1B_vE16GridFittedBottomI5FieldISS_SS_vvvvSE_ISF_Li3ESG_ISF_Li3ELi1E12_194__99__1_EESF_vvvE23CenterImmersedConditionEvvvES7_I10NamedTupleI53__time___last__t___last_stage__t___iteration___stage_S7_ISF_SF_SF_S8_S8_EES1L_I36__u___v___w_______U___V___T___S___e_S7_ISI_SI_SE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__35_EESE_ISF_Li3ESG_ISF_Li3ELi1E13_194__123__1_EES1E_ISR_SS_vvvvS1R_SF_vvvES1E_ISS_SR_vvvvS1R_SF_vvvESI_SI_SI_EEE#9395$par78"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34x99x194xf64, 1>, %arg2: memref<34x99x194xf64, 1>, %arg3: memref<34x99x194xf64, 1>, %arg4: memref<34x99x194xf64, 1>, %c : i1) {
    %c17660_i64 = arith.constant 17660 : i64
    %c90_i64 = arith.constant 90 : i64
    %c0 = arith.constant 0 : index
    %c-12 = arith.constant -12 : index
    %c12 = arith.constant 12 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c2_i64 = arith.constant 2 : i64
    %c182_i64 = arith.constant 182 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c6_i64 = arith.constant 6 : i64
    %c19206_i64 = arith.constant 19206 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c194_i64 = arith.constant 194 : i64
    %c7_i64 = arith.constant 7 : i64
    %cst = arith.constant -1.000000e+00 : f64
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<34x99x194xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%arg5, %arg6, %arg7, %arg8) = (0, 0, 0, 0) to (2, 16, 12, 16) {
      %1 = arith.muli %arg6, %c16 overflow<nuw> : index
      %2 = arith.addi %1, %arg8 : index
      %3 = arith.addi %2, %c1 : index
      %4 = arith.muli %arg5, %c-12 : index
      %5 = arith.muli %arg5, %c12 overflow<nuw> : index
      %6 = arith.addi %5, %arg7 : index
      %7 = arith.addi %4, %6 : index
      %8 = arith.index_castui %3 : index to i64
      %9 = arith.subi %c0, %arg6 : index
      %10 = arith.addi %9, %7 : index
      %11 = arith.addi %10, %c1 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = arith.addi %12, %c-1_i64 : i64
      %14 = arith.muli %13, %c16_i64 : i64
      %15 = arith.addi %8, %14 : i64
      %16 = arith.muli %arg5, %c16 : index
      %17 = arith.addi %16, %arg6 : index
      %18 = arith.index_castui %17 : index to i64
      affine.if #set2(%arg6, %arg5, %arg7, %arg8) {
        %19 = arith.addi %18, %c7_i64 : i64
        %20 = arith.muli %19, %c19206_i64 : i64
        %21 = affine.load %arg0[%arg5 * 16 + %arg6 + 7, 7, %arg8 + %arg7 * 16 + 7] : memref<34x99x194xf64, 1>
        affine.store %21, %arg0[%arg5 * 16 + %arg6 + 7, 6, %arg8 + %arg7 * 16 + 7] : memref<34x99x194xf64, 1>
        %22 = arith.subi %c182_i64, %15 : i64
        %23 = affine.if #set3(%arg7, %arg8) -> i64 {
          affine.yield %c-1_i64 : i64
        } else {
          affine.yield %c1_i64 : i64
        }
        %24 = arith.subi %c2_i64, %15 : i64
        %25 = arith.select %c, %22, %24 : i64
        %26 = arith.addi %25, %20 : i64
        %27 = arith.sitofp %23 : i64 to f64
        affine.for %arg9 = 0 to 6 {
          %46 = arith.index_cast %arg9 : index to i64
          %47 = arith.subi %c90_i64, %46 : i64
          %48 = arith.muli %47, %c194_i64 : i64
          %49 = arith.addi %48, %26 : i64
          %50 = arith.addi %49, %c6_i64 : i64
          %51 = llvm.getelementptr inbounds %0[%50] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
          %52 = llvm.load %51 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
          %53 = arith.mulf %27, %52 {fastmathFlags = #llvm.fastmath<none>} : f64
          affine.store %53, %arg0[%arg5 * 16 + %arg6 + 7, %arg9 + 92, %arg8 + %arg7 * 16 + 7] : memref<34x99x194xf64, 1>
        }
      }
    }
    return
  }
}

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4)[s0] -> (d0 * -194 + s0 + 151907 + d1 * 19206 + d2 * 307296 - d3 - d4 * 16)>
// CHECK-NEXT: #set = affine_set<(d0, d1, d2, d3) : (-d0 - d1 * 16 + 19 >= 0, d3 + d2 * 16 >= 0, d2 * -16 - d3 + 179 >= 0)>
// CHECK-NEXT: #set1 = affine_set<(d0, d1) : (d1 + d0 * 16 - 1 >= 0)>
// CHECK-NEXT: #tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
// CHECK-NEXT: #tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
// CHECK-NEXT: #tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI13_1_180__1_20_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__2_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEES7_I11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__99__34_EESI_SI_SI_SI_ES7_I17BoundaryConditionI4FluxvEvSM_SM_SM_ES7_ISK_I6ZipperS8_ESP_SP_SP_SP_ES7_IS7_I4Face6CenterSS_ES7_ISS_SR_SS_ES7_ISS_SS_SS_ESV_SV_E20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SY_SZ_S10_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EES14_S16_E8TripolarIS8_S8_S8_ESE_ISF_Li2ESG_ISF_Li2ELi1E9_194__99_EES1B_S1B_S1B_vE16GridFittedBottomI5FieldISS_SS_vvvvSE_ISF_Li3ESG_ISF_Li3ELi1E12_194__99__1_EESF_vvvE23CenterImmersedConditionEvvvES7_I10NamedTupleI53__time___last__t___last_stage__t___iteration___stage_S7_ISF_SF_SF_S8_S8_EES1L_I36__u___v___w_______U___V___T___S___e_S7_ISI_SI_SE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__35_EESE_ISF_Li3ESG_ISF_Li3ELi1E13_194__123__1_EES1E_ISR_SS_vvvvS1R_SF_vvvES1E_ISS_SR_vvvvS1R_SF_vvvESI_SI_SI_EEE#9395$par78"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34x99x194xf64, 1>, %arg2: memref<34x99x194xf64, 1>, %arg3: memref<34x99x194xf64, 1>, %arg4: memref<34x99x194xf64, 1>, %arg5: i1) {
// CHECK-DAG:     %c-1_i64 = arith.constant -1 : i64
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-DAG:     %c2_i64 = arith.constant 2 : i64
// CHECK-DAG:     %c182_i64 = arith.constant 182 : i64
// CHECK-NEXT:     %0 = arith.select %arg5, %c182_i64, %c2_i64 : i64
// CHECK-NEXT:     %1 = arith.index_cast %0 : i64 to index
// CHECK-NEXT:     %2 = "enzymexla.memref2pointer"(%arg0) : (memref<34x99x194xf64, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:     affine.parallel (%arg6, %arg7, %arg8, %arg9) = (0, 0, 0, 0) to (2, 16, 12, 16) {
// CHECK-NEXT:       affine.if #set(%arg7, %arg6, %arg8, %arg9) {
// CHECK-NEXT:         %3 = affine.load %arg0[%arg6 * 16 + %arg7 + 7, 7, %arg9 + %arg8 * 16 + 7] : memref<34x99x194xf64, 1>
// CHECK-NEXT:         affine.store %3, %arg0[%arg6 * 16 + %arg7 + 7, 6, %arg9 + %arg8 * 16 + 7] : memref<34x99x194xf64, 1>
// CHECK-NEXT:         %4 = affine.if #set1(%arg8, %arg9) -> i64 {
// CHECK-NEXT:           affine.yield %c-1_i64 : i64
// CHECK-NEXT:         } else {
// CHECK-NEXT:           affine.yield %c1_i64 : i64
// CHECK-NEXT:         }
// CHECK-NEXT:         %5 = arith.sitofp %4 : i64 to f64
// CHECK-NEXT:         affine.for %arg10 = 0 to 6 {
// CHECK-NEXT:           %6 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
// CHECK-NEXT:           %7 = affine.apply #map(%arg10, %arg7, %arg6, %arg9, %arg8)[%1]
// CHECK-NEXT:           %8 = memref.load %6[%7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
// CHECK-NEXT:           %9 = arith.mulf %5, %8 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:           affine.store %9, %arg0[%arg6 * 16 + %arg7 + 7, %arg10 + 92, %arg9 + %arg8 * 16 + 7] : memref<34x99x194xf64, 1>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
