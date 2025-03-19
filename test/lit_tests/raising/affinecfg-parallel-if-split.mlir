// RUN: enzymexlamlir-opt %s --affine-cfg
// COM: enzymexlamlir-opt %s --affine-cfg | FileCheck %s


// TODO This can also work but needs special handling for else blocks

// CHECK:     affine.parallel (%arg1) = (1) to (90)
// CHECK:     affine.parallel (%arg1) = (90) to (180)

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "julia_iterate_interface_fluxes_150839">
#alias_scope_domain1 = #llvm.alias_scope_domain<id = distinct[1]<>, description = "julia_iterate_interface_fluxes_150839">
#alias_scope_domain2 = #llvm.alias_scope_domain<id = distinct[2]<>, description = "julia_iterate_interface_fluxes_150839">
#set = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0) : (-d0 + 89 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (-d0 - d1 * 16 + 19 >= 0, d3 + d2 * 16 >= 0, d2 * -16 - d3 + 179 >= 0)>
#set3 = affine_set<(d0, d1) : (d1 + d0 * 16 - 1 >= 0)>
#set4 = affine_set<(d0, d1) : (d0 * -16 - d1 + 89 >= 0)>
#set5 = affine_set<(d0) : (d0 - 19 == 0)>
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#alias_scope = #llvm.alias_scope<id = distinct[3]<>, domain = #alias_scope_domain>
#alias_scope1 = #llvm.alias_scope<id = distinct[4]<>, domain = #alias_scope_domain1>
#alias_scope2 = #llvm.alias_scope<id = distinct[5]<>, domain = #alias_scope_domain2>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module @"reactant_run!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.func local_unnamed_addr @__nv_isnand(f64) -> i32 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @__nv_fmax(f64, f64) -> f64 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @__nv_pow(f64, f64) -> f64 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @__nv_log(f64) -> f64 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @__nv_exp(f64) -> f64 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @__nv_fmin(f64, f64) -> f64 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @__nv_atan(f64) -> f64 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @__nv_cbrt(f64) -> f64 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @__nv_cos(f64) -> f64 attributes {sym_visibility = "private"}
  llvm.func local_unnamed_addr @__nv_llabs(i64) -> i64 attributes {sym_visibility = "private"}
  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI12_1_180__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__123__1_EE17BoundaryConditionI4FluxvESJ_I6ZipperS8_ES7_I4Face6CentervE20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SS_ST_SU_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EESY_S10_E8TripolarIS8_S8_S8_ESE_ISF_Li2ESG_ISF_Li2ELi1E10_194__123_EES15_S15_S15_vE16GridFittedBottomI5FieldISP_SP_vvvvSI_SF_vvvE23CenterImmersedConditionEvvvES7_I10NamedTupleI53__time___last__t___last_stage__t___iteration___stage_S7_ISF_SF_SF_S8_S8_EES1D_I36__u___v___w_______U___V___T___S___e_S7_ISE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__34_EES1H_SE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__35_EESI_S18_ISO_SP_vvvvSI_SF_vvvES18_ISP_SO_vvvvSI_SF_vvvES1H_S1H_S1H_EEE#721$par18"(%arg0: memref<1x123x194xf64, 1>) {
    %c16496_i64 = arith.constant 16496 : i64
    %c16690_i64 = arith.constant 16690 : i64
    %c16884_i64 = arith.constant 16884 : i64
    %c17078_i64 = arith.constant 17078 : i64
    %c17272_i64 = arith.constant 17272 : i64
    %c17466_i64 = arith.constant 17466 : i64
    %c17660_i64 = arith.constant 17660 : i64
    %c17854_i64 = arith.constant 17854 : i64
    %c18048_i64 = arith.constant 18048 : i64
    %c18242_i64 = arith.constant 18242 : i64
    %c18436_i64 = arith.constant 18436 : i64
    %c18630_i64 = arith.constant 18630 : i64
    %c18824_i64 = arith.constant 18824 : i64
    %c19018_i64 = arith.constant 19018 : i64
    %c19212_i64 = arith.constant 19212 : i64
    %c19406_i64 = arith.constant 19406 : i64
    %c19600_i64 = arith.constant 19600 : i64
    %c19794_i64 = arith.constant 19794 : i64
    %c19988_i64 = arith.constant 19988 : i64
    %c1 = arith.constant 1 : index
    %c2_i64 = arith.constant 2 : i64
    %c182_i64 = arith.constant 182 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-1_i64 = arith.constant -1 : i64
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<1x123x194xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%arg1) = (0) to (180) {
      %1 = arith.addi %arg1, %c1 : index
      %2 = arith.index_castui %1 : index to i64
      %3 = affine.load %arg0[0, 19, %arg1 + 7] : memref<1x123x194xf64, 1>
      affine.store %3, %arg0[0, 18, %arg1 + 7] : memref<1x123x194xf64, 1>
      %4 = arith.subi %c182_i64, %2 : i64
      %5 = affine.if #set(%arg1) -> i64 {
        affine.yield %c-1_i64 : i64
      } else {
        affine.yield %c1_i64 : i64
      }
      %6 = arith.subi %c2_i64, %2 : i64
      %7 = affine.if #set(%arg1) -> i64 {
        affine.yield %4 : i64
      } else {
        affine.yield %6 : i64
      }
      %8 = arith.sitofp %5 : i64 to f64
      %9 = arith.addi %7, %c19794_i64 : i64
      %10 = llvm.getelementptr inbounds %0[%9] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %11 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %12 = arith.mulf %8, %11 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %12, %arg0[0, 104, %arg1 + 7] : memref<1x123x194xf64, 1>
      %13 = arith.addi %7, %c19600_i64 : i64
      %14 = llvm.getelementptr inbounds %0[%13] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %15 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %16 = arith.mulf %8, %15 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %16, %arg0[0, 105, %arg1 + 7] : memref<1x123x194xf64, 1>
      %17 = arith.addi %7, %c19406_i64 : i64
      %18 = llvm.getelementptr inbounds %0[%17] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %19 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %20 = arith.mulf %8, %19 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %20, %arg0[0, 106, %arg1 + 7] : memref<1x123x194xf64, 1>
      %21 = arith.addi %7, %c19212_i64 : i64
      %22 = llvm.getelementptr inbounds %0[%21] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %23 = llvm.load %22 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %24 = arith.mulf %8, %23 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %24, %arg0[0, 107, %arg1 + 7] : memref<1x123x194xf64, 1>
      %25 = arith.addi %7, %c19018_i64 : i64
      %26 = llvm.getelementptr inbounds %0[%25] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %27 = llvm.load %26 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %28 = arith.mulf %8, %27 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %28, %arg0[0, 108, %arg1 + 7] : memref<1x123x194xf64, 1>
      %29 = arith.addi %7, %c18824_i64 : i64
      %30 = llvm.getelementptr inbounds %0[%29] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %31 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %32 = arith.mulf %8, %31 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %32, %arg0[0, 109, %arg1 + 7] : memref<1x123x194xf64, 1>
      %33 = arith.addi %7, %c18630_i64 : i64
      %34 = llvm.getelementptr inbounds %0[%33] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %35 = llvm.load %34 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %36 = arith.mulf %8, %35 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %36, %arg0[0, 110, %arg1 + 7] : memref<1x123x194xf64, 1>
      %37 = arith.addi %7, %c18436_i64 : i64
      %38 = llvm.getelementptr inbounds %0[%37] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %39 = llvm.load %38 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %40 = arith.mulf %8, %39 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %40, %arg0[0, 111, %arg1 + 7] : memref<1x123x194xf64, 1>
      %41 = arith.addi %7, %c18242_i64 : i64
      %42 = llvm.getelementptr inbounds %0[%41] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %43 = llvm.load %42 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %44 = arith.mulf %8, %43 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %44, %arg0[0, 112, %arg1 + 7] : memref<1x123x194xf64, 1>
      %45 = arith.addi %7, %c18048_i64 : i64
      %46 = llvm.getelementptr inbounds %0[%45] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %47 = llvm.load %46 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %48 = arith.mulf %8, %47 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %48, %arg0[0, 113, %arg1 + 7] : memref<1x123x194xf64, 1>
      %49 = arith.addi %7, %c17854_i64 : i64
      %50 = llvm.getelementptr inbounds %0[%49] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %51 = llvm.load %50 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %52 = arith.mulf %8, %51 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %52, %arg0[0, 114, %arg1 + 7] : memref<1x123x194xf64, 1>
      %53 = arith.addi %7, %c17660_i64 : i64
      %54 = llvm.getelementptr inbounds %0[%53] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %55 = llvm.load %54 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %56 = arith.mulf %8, %55 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %56, %arg0[0, 115, %arg1 + 7] : memref<1x123x194xf64, 1>
      %57 = arith.addi %7, %c17466_i64 : i64
      %58 = llvm.getelementptr inbounds %0[%57] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %59 = llvm.load %58 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %60 = arith.mulf %8, %59 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %60, %arg0[0, 116, %arg1 + 7] : memref<1x123x194xf64, 1>
      %61 = arith.addi %7, %c17272_i64 : i64
      %62 = llvm.getelementptr inbounds %0[%61] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %63 = llvm.load %62 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %64 = arith.mulf %8, %63 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %64, %arg0[0, 117, %arg1 + 7] : memref<1x123x194xf64, 1>
      %65 = arith.addi %7, %c17078_i64 : i64
      %66 = llvm.getelementptr inbounds %0[%65] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %67 = llvm.load %66 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %68 = arith.mulf %8, %67 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %68, %arg0[0, 118, %arg1 + 7] : memref<1x123x194xf64, 1>
      %69 = arith.addi %7, %c16884_i64 : i64
      %70 = llvm.getelementptr inbounds %0[%69] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %71 = llvm.load %70 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %72 = arith.mulf %8, %71 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %72, %arg0[0, 119, %arg1 + 7] : memref<1x123x194xf64, 1>
      %73 = arith.addi %7, %c16690_i64 : i64
      %74 = llvm.getelementptr inbounds %0[%73] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %75 = llvm.load %74 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %76 = arith.mulf %8, %75 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %76, %arg0[0, 120, %arg1 + 7] : memref<1x123x194xf64, 1>
      %77 = arith.addi %7, %c16496_i64 : i64
      %78 = llvm.getelementptr inbounds %0[%77] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %79 = llvm.load %78 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %80 = arith.mulf %8, %79 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %80, %arg0[0, 121, %arg1 + 7] : memref<1x123x194xf64, 1>
      %81 = arith.addi %7, %c19988_i64 : i64
      %82 = llvm.getelementptr inbounds %0[%81] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %83 = llvm.load %82 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %84 = arith.mulf %8, %83 {fastmathFlags = #llvm.fastmath<none>} : f64
      %85 = affine.load %arg0[0, 103, %arg1 + 7] : memref<1x123x194xf64, 1>
      %86 = affine.if #set1(%arg1) -> f64 {
        affine.yield %85 : f64
      } else {
        affine.yield %84 : f64
      }
      affine.store %86, %arg0[0, 103, %arg1 + 7] : memref<1x123x194xf64, 1>
    }
    return
  }
}

