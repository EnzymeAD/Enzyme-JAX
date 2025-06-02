// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --enzyme-hlo-opt | FileCheck %s

// CHECK-NOT: affine

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "julia_iterate_interface_fluxes_150839">
#alias_scope_domain1 = #llvm.alias_scope_domain<id = distinct[1]<>, description = "julia_iterate_interface_fluxes_150839">
#alias_scope_domain2 = #llvm.alias_scope_domain<id = distinct[2]<>, description = "julia_iterate_interface_fluxes_150839">
#map = affine_map<(d0, d1) -> (-d0 - d1 * 194 + 19793)>
#map1 = affine_map<(d0) -> (-d0 + 19987)>
#map2 = affine_map<(d0, d1, d2) -> (d0 * -194 + d1 * 19206 - d2 + 151907)>
#map3 = affine_map<(d0, d1) -> (d0 * 19206 - d1 + 152101)>
#set = affine_set<(d0) : (-d0 + 89 >= 0)>
#set1 = affine_set<(d0) : (d0 - 1 >= 0)>
#set2 = affine_set<(d0) : (d0 - 19 == 0)>
#set3 = affine_set<(d0) : (d0 == 0)>
#set4 = affine_set<(d0) : (d0 - 179 == 0)>
#set5 = affine_set<(d0) : (-d0 + 18 >= 0)>
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#alias_scope = #llvm.alias_scope<id = distinct[3]<>, domain = #alias_scope_domain>
#alias_scope1 = #llvm.alias_scope<id = distinct[4]<>, domain = #alias_scope_domain1>
#alias_scope2 = #llvm.alias_scope<id = distinct[5]<>, domain = #alias_scope_domain2>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module @"reactant_run!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"##call__Z44gpu_solve_batched_tridiagonal_system_kernel_16CompilerMetadataI10StaticSizeI9_180__85_E12DynamicCheckvv7NDRangeILi2ES0_I7_12__6_ES0_I8_16__16_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE40VerticallyImplicitDiffusionLowerDiagonal35VerticallyImplicitDiffusionDiagonal40VerticallyImplicitDiffusionUpperDiagonalSC_SA_IS9_Li3ELi1E13_180__85__20_E20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESO_SQ_E8TripolarI5Int64ST_ST_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EESW_SW_SW_vE16GridFittedBottomI5FieldI6CenterS10_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvEv5TupleI24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthIS9_ES9_v13CATKEEquationIS9_EE22CATKEDiffusivityFieldsIS8_IS9_Li3ESA_IS9_Li3ELi1E13_194__99__35_EESC_S13_S9_10NamedTupleI8__u___v_S17_ISC_SC_EES1I_I12__T___S___e_S17_IS1H_S1H_S1H_EES1I_I12__T___S___e_S17_I9ZeroFieldIST_Li3EES1O_SC_EEEv4FaceS10_S10_S1I_I53__time___last__t___last_stage__t___iteration___stage_S17_IS9_S9_S9_ST_ST_EES9_5_z___E10ZDirection#883$par87"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<20x85x180xf64, 1>, %arg2: memref<34xf64, 1>, %arg3: memref<35xf64, 1>, %arg4: memref<34xf64, 1>, %arg5: memref<1x99x194xf64, 1>, %arg6: memref<35x99x194xf64, 1>) {
    %c3_i64 = arith.constant 3 : i64
    %c1_i64 = arith.constant 1 : i64
    %true = arith.constant true
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant -1.200000e+03 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %c2_i64 = arith.constant 2 : i64
    %c20_i64 = arith.constant 20 : i64
    %cst_3 = arith.constant 2.2204460492503131E-15 : f64
    affine.parallel (%arg7, %arg8) = (0, 0) to (85, 180) {
      %0 = affine.load %arg2[8] {alignment = 64 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
      %1 = affine.load %arg5[0, %arg7 + 7, %arg8 + 7] : memref<1x99x194xf64, 1>
      %2 = arith.cmpf ole, %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
      %3 = affine.load %arg2[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
      %4 = arith.cmpf ole, %3, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
      %5 = arith.ori %2, %4 : i1
      %6 = affine.load %arg5[0, %arg7 + 7, %arg8 + 6] : memref<1x99x194xf64, 1>
      %7 = arith.cmpf ole, %0, %6 {fastmathFlags = #llvm.fastmath<none>} : f64
      %8 = arith.cmpf ole, %3, %6 {fastmathFlags = #llvm.fastmath<none>} : f64
      %9 = arith.ori %7, %8 : i1
      %10 = arith.ori %5, %9 : i1
      %11 = affine.load %arg6[8, %arg7 + 7, %arg8 + 6] : memref<35x99x194xf64, 1>
      %12 = affine.load %arg6[8, %arg7 + 7, %arg8 + 7] : memref<35x99x194xf64, 1>
      %13 = arith.addf %11, %12 {fastmathFlags = #llvm.fastmath<none>} : f64
      %14 = arith.mulf %13, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
      %15 = affine.load %arg4[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
      %16 = affine.load %arg3[9] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<35xf64, 1>
      %17 = arith.mulf %14, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
      %18 = arith.mulf %15, %16 {fastmathFlags = #llvm.fastmath<none>} : f64
      %19 = arith.divf %17, %18 {fastmathFlags = #llvm.fastmath<none>} : f64
      %20 = arith.select %10, %cst_1, %19 : f64
      %21 = arith.subf %cst_2, %20 {fastmathFlags = #llvm.fastmath<none>} : f64
      %22 = affine.load %arg0[7, %arg7 + 7, %arg8 + 7] : memref<34x99x194xf64, 1>
      %23 = arith.divf %22, %21 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %23, %arg0[7, %arg7 + 7, %arg8 + 7] : memref<34x99x194xf64, 1>
      %24 = affine.for %arg9 = 0 to 18 iter_args(%arg10 = %21) -> (f64) {
        %25 = arith.index_cast %arg9 : index to i64
        %26 = arith.addi %25, %c2_i64 : i64
        %27 = arith.addi %25, %c1_i64 : i64
        %28 = affine.load %arg2[%arg9 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
        %29 = affine.load %arg5[0, %arg7 + 7, %arg8 + 7] : memref<1x99x194xf64, 1>
        %30 = arith.cmpf ole, %28, %29 {fastmathFlags = #llvm.fastmath<none>} : f64
        %31 = arith.cmpi slt, %26, %c1_i64 : i64
        %32 = arith.cmpi sgt, %26, %c20_i64 : i64
        %33 = arith.ori %31, %32 : i1
        %34 = arith.ori %30, %33 : i1
        %35 = affine.load %arg2[%arg9 + 7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
        %36 = arith.cmpf ole, %35, %29 {fastmathFlags = #llvm.fastmath<none>} : f64
        %37 = arith.cmpi slt, %27, %c1_i64 : i64
        %38 = arith.cmpi sgt, %27, %c20_i64 : i64
        %39 = arith.ori %37, %38 : i1
        %40 = arith.ori %36, %39 : i1
        %41 = arith.ori %34, %40 : i1
        %42 = affine.load %arg5[0, %arg7 + 7, %arg8 + 6] : memref<1x99x194xf64, 1>
        %43 = arith.cmpf ole, %28, %42 {fastmathFlags = #llvm.fastmath<none>} : f64
        %44 = arith.ori %43, %33 : i1
        %45 = arith.cmpf ole, %35, %42 {fastmathFlags = #llvm.fastmath<none>} : f64
        %46 = arith.ori %45, %39 : i1
        %47 = arith.ori %44, %46 : i1
        %48 = arith.ori %41, %47 : i1
        %49 = arith.ori %33, %39 : i1
        %50 = arith.xori %49, %true : i1
        %51 = arith.andi %48, %50 : i1
        %52 = affine.load %arg6[%arg9 + 8, %arg7 + 7, %arg8 + 6] : memref<35x99x194xf64, 1>
        %53 = affine.load %arg6[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<35x99x194xf64, 1>
        %54 = arith.addf %52, %53 {fastmathFlags = #llvm.fastmath<none>} : f64
        %55 = arith.mulf %54, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
        %56 = affine.load %arg4[%arg9 + 7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
        %57 = affine.load %arg3[%arg9 + 9] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<35xf64, 1>
        %58 = arith.mulf %55, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
        %59 = arith.mulf %56, %57 {fastmathFlags = #llvm.fastmath<none>} : f64
        %60 = arith.divf %58, %59 {fastmathFlags = #llvm.fastmath<none>} : f64
        %61 = arith.select %51, %cst_1, %60 : f64
        %62 = arith.addi %25, %c3_i64 : i64
        %63 = affine.load %arg2[%arg9 + 9] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
        %64 = arith.cmpf ole, %63, %29 {fastmathFlags = #llvm.fastmath<none>} : f64
        %65 = arith.cmpi slt, %62, %c1_i64 : i64
        %66 = arith.cmpi sgt, %62, %c20_i64 : i64
        %67 = arith.ori %65, %66 : i1
        %68 = arith.ori %64, %67 : i1
        %69 = arith.ori %68, %34 : i1
        %70 = arith.cmpf ole, %63, %42 {fastmathFlags = #llvm.fastmath<none>} : f64
        %71 = arith.ori %70, %67 : i1
        %72 = arith.ori %71, %44 : i1
        %73 = arith.ori %69, %72 : i1
        %74 = arith.ori %67, %33 : i1
        %75 = arith.xori %74, %true : i1
        %76 = arith.andi %73, %75 : i1
        %77 = affine.load %arg6[%arg9 + 9, %arg7 + 7, %arg8 + 6] : memref<35x99x194xf64, 1>
        %78 = affine.load %arg6[%arg9 + 9, %arg7 + 7, %arg8 + 7] : memref<35x99x194xf64, 1>
        %79 = arith.addf %77, %78 {fastmathFlags = #llvm.fastmath<none>} : f64
        %80 = arith.mulf %79, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
        %81 = affine.load %arg4[%arg9 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
        %82 = affine.load %arg3[%arg9 + 10] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<35xf64, 1>
        %83 = arith.mulf %80, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
        %84 = arith.mulf %81, %82 {fastmathFlags = #llvm.fastmath<none>} : f64
        %85 = arith.divf %83, %84 {fastmathFlags = #llvm.fastmath<none>} : f64
        %86 = arith.select %76, %cst_1, %85 : f64
        %87 = arith.subf %cst_2, %86 {fastmathFlags = #llvm.fastmath<none>} : f64
        %88 = arith.mulf %81, %57 {fastmathFlags = #llvm.fastmath<none>} : f64
        %89 = arith.divf %58, %88 {fastmathFlags = #llvm.fastmath<none>} : f64
        %90 = arith.select %51, %cst_1, %89 : f64
        %91 = arith.subf %87, %90 {fastmathFlags = #llvm.fastmath<none>} : f64
        %92 = arith.divf %61, %arg10 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %92, %arg1[%arg9 + 1, %arg7, %arg8] : memref<20x85x180xf64, 1>
        %93 = arith.mulf %92, %90 {fastmathFlags = #llvm.fastmath<none>} : f64
        %94 = arith.subf %91, %93 {fastmathFlags = #llvm.fastmath<none>} : f64
        %95 = affine.load %arg0[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<34x99x194xf64, 1>
        %96 = math.absf %94 : f64
        %97 = arith.cmpf olt, %cst_3, %96 {fastmathFlags = #llvm.fastmath<none>} : f64
        %98 = affine.load %arg0[%arg9 + 7, %arg7 + 7, %arg8 + 7] : memref<34x99x194xf64, 1>
        %99 = arith.mulf %90, %98 {fastmathFlags = #llvm.fastmath<none>} : f64
        %100 = arith.subf %95, %99 {fastmathFlags = #llvm.fastmath<none>} : f64
        %101 = arith.divf %100, %94 {fastmathFlags = #llvm.fastmath<none>} : f64
        %102 = arith.select %97, %101, %95 : f64
        affine.store %102, %arg0[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<34x99x194xf64, 1>
        affine.yield %94 : f64
      }
    }
    return
  }
}
