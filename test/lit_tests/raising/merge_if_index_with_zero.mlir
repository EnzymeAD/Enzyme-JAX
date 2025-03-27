// RUN: enzymexlamlir-opt --affine-cfg --allow-unregistered-dialect %s | FileCheck %s

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
  func.func private @"##call__Z40gpu__split_explicit_barotropic_velocity_16CompilerMetadataI16OffsetStaticSizeI14_1_180__1_103_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__7_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE7Float6420ImmersedBoundaryGridISE_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISE_SG_SH_SI_28StaticVerticalDiscretizationI11OffsetArrayISE_Li1E13CuTracedArrayISE_Li1ELi1E5_35__EESL_ISE_Li1ESM_ISE_Li1ELi1E5_34__EESO_SQ_E8TripolarIS8_S8_S8_ESL_ISE_Li2ESM_ISE_Li2ELi1E10_194__123_EESV_SV_SV_vE16GridFittedBottomI5FieldI6CenterSZ_vvvvSL_ISE_Li3ESM_ISE_Li3ELi1E13_194__123__1_EESE_vvvE23CenterImmersedConditionEvvvESE_S11_SY_I4FaceSZ_vvvvS11_SE_vvvESY_ISZ_S16_vvvvS11_SE_vvvES11_S17_S18_S17_S18_SE_21ForwardBackwardScheme#9485$par117"(%arg0: memref<35xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<123x194xf64, 1>, %arg3: memref<123x194xf64, 1>, %arg4: memref<1x123x194xf64, 1>, %arg5: memref<1x123x194xf64, 1>, %arg6: memref<1x123x194xf64, 1>, %arg7: memref<1x123x194xf64, 1>, %arg8: memref<1x123x194xf64, 1>, %arg9: memref<1x123x194xf64, 1>, %arg10: memref<1x123x194xf64, 1>, %arg11: memref<1x123x194xf64, 1>, %arg12: memref<1x123x194xf64, 1>) {
    %c16 = arith.constant 16 : index
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0x7FF8000000000000 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant -9.8066499999999994 : f64
    %cst_2 = arith.constant 9.600000e+01 : f64
    %cst_3 = arith.constant -0.0037661355580298995 : f64
    affine.parallel (%arg13, %arg14, %arg15, %arg16) = (0, 0, 0, 0) to (7, 16, 12, 16) {
      %0 = arith.muli %arg13, %c16 : index
      %1 = arith.addi %0, %arg14 : index
      %2 = arith.index_castui %1 : index to i64
      affine.if #set14(%arg14, %arg13, %arg15, %arg16) {
        %3 = affine.load %arg0[27] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<35xf64, 1>
        %4 = affine.load %arg4[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 6] : memref<1x123x194xf64, 1>
        %5 = arith.subf %3, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
        %6 = affine.load %arg4[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %7 = arith.subf %3, %6 {fastmathFlags = #llvm.fastmath<none>} : f64
        %8 = math.isnan %5 : f64
        %9 = math.isnan %7 : f64
        %10 = arith.ori %8, %9 : i1
        %11 = arith.minnumf %5, %7 : f64
        %12 = arith.select %10, %cst, %11 : f64
        %13 = affine.load %arg4[0, %arg13 * 16 + %arg14 + 18, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %14 = arith.subf %3, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
        %15 = math.isnan %14 : f64
        %16 = arith.ori %15, %9 : i1
        %17 = arith.minnumf %14, %7 : f64
        %18 = arith.select %16, %cst, %17 : f64
        %19 = affine.load %arg6[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %20 = affine.load %arg1[26] {alignment = 16 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
        %21 = arith.cmpf ole, %20, %6 {fastmathFlags = #llvm.fastmath<none>} : f64
        %22 = arith.cmpf ole, %20, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
        %23 = arith.ori %21, %22 : i1
        %24 = affine.load %arg5[0, %arg13 * 16 + %arg14 + 19, 7] : memref<1x123x194xf64, 1>
        %25 = affine.load %arg5[0, %arg13 * 16 + %arg14 + 19, 186] : memref<1x123x194xf64, 1>
        %26 = arith.subf %24, %25 {fastmathFlags = #llvm.fastmath<none>} : f64
        %27 = affine.load %arg5[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %28 = affine.load %arg5[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 6] : memref<1x123x194xf64, 1>
        %29 = arith.subf %27, %28 {fastmathFlags = #llvm.fastmath<none>} : f64
        %30 = affine.if #set15(%arg15, %arg16) -> f64 {
          affine.yield %26 : f64
        } else {
          affine.yield %29 : f64
        }
        %31 = affine.load %arg2[%arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<123x194xf64, 1>
        %32 = arith.divf %30, %31 {fastmathFlags = #llvm.fastmath<none>} : f64
        %33 = arith.select %23, %cst_0, %32 : f64
        %34 = arith.mulf %12, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
        %35 = arith.mulf %34, %33 {fastmathFlags = #llvm.fastmath<none>} : f64
        %36 = affine.load %arg11[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %37 = arith.addf %36, %35 {fastmathFlags = #llvm.fastmath<none>} : f64
        %38 = arith.mulf %37, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
        %39 = arith.addf %19, %38 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %39, %arg6[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %40 = affine.load %arg7[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %41 = affine.load %arg4[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %42 = affine.load %arg1[26] {alignment = 16 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
        %43 = arith.cmpf ole, %42, %41 {fastmathFlags = #llvm.fastmath<none>} : f64
        %44 = affine.load %arg4[0, %arg13 * 16 + %arg14 + 18, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %45 = arith.cmpi ult, %2, %c1_i64 : i64
        %46 = arith.cmpf ole, %42, %44 {fastmathFlags = #llvm.fastmath<none>} : f64
        %47 = arith.ori %45, %46 : i1
        %48 = arith.ori %43, %47 : i1
        %49 = affine.load %arg5[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %50 = affine.load %arg5[0, %arg13 * 16 + %arg14 + 18, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %51 = arith.subf %49, %50 {fastmathFlags = #llvm.fastmath<none>} : f64
        %52 = affine.if #set13(%arg14, %arg13) -> f64 {
          affine.yield %cst_0 : f64
        } else {
          affine.yield %51 : f64
        }
        %53 = affine.load %arg3[%arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<123x194xf64, 1>
        %54 = arith.divf %52, %53 {fastmathFlags = #llvm.fastmath<none>} : f64
        %55 = arith.select %48, %cst_0, %54 : f64
        %56 = arith.mulf %18, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
        %57 = arith.mulf %56, %55 {fastmathFlags = #llvm.fastmath<none>} : f64
        %58 = affine.load %arg12[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %59 = arith.addf %58, %57 {fastmathFlags = #llvm.fastmath<none>} : f64
        %60 = arith.mulf %59, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
        %61 = arith.addf %40, %60 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %61, %arg7[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %62 = affine.load %arg8[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %63 = affine.load %arg5[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %64 = arith.mulf %63, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
        %65 = arith.addf %62, %64 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %65, %arg8[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %66 = affine.load %arg9[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %67 = affine.load %arg6[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %68 = arith.mulf %67, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
        %69 = arith.addf %66, %68 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %69, %arg9[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %70 = affine.load %arg10[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %71 = affine.load %arg7[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
        %72 = arith.mulf %71, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
        %73 = arith.addf %70, %72 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %73, %arg10[0, %arg13 * 16 + %arg14 + 19, %arg16 + %arg15 * 16 + 7] : memref<1x123x194xf64, 1>
      }
    }
    return
  }
}


// CHECK: #set = affine_set<(d0) : (d0 == 0)>
// CHECK:  func.func private @"##call__Z40gpu__split_explicit_barotropic_velocity_16CompilerMetadataI16OffsetStaticSizeI14_1_180__1_103_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__7_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE7Float6420ImmersedBoundaryGridISE_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISE_SG_SH_SI_28StaticVerticalDiscretizationI11OffsetArrayISE_Li1E13CuTracedArrayISE_Li1ELi1E5_35__EESL_ISE_Li1ESM_ISE_Li1ELi1E5_34__EESO_SQ_E8TripolarIS8_S8_S8_ESL_ISE_Li2ESM_ISE_Li2ELi1E10_194__123_EESV_SV_SV_vE16GridFittedBottomI5FieldI6CenterSZ_vvvvSL_ISE_Li3ESM_ISE_Li3ELi1E13_194__123__1_EESE_vvvE23CenterImmersedConditionEvvvESE_S11_SY_I4FaceSZ_vvvvS11_SE_vvvESY_ISZ_S16_vvvvS11_SE_vvvES11_S17_S18_S17_S18_SE_21ForwardBackwardScheme#9485$par117"(%arg0: memref<35xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<123x194xf64, 1>, %arg3: memref<123x194xf64, 1>, %arg4: memref<1x123x194xf64, 1>, %arg5: memref<1x123x194xf64, 1>, %arg6: memref<1x123x194xf64, 1>, %arg7: memref<1x123x194xf64, 1>, %arg8: memref<1x123x194xf64, 1>, %arg9: memref<1x123x194xf64, 1>, %arg10: memref<1x123x194xf64, 1>, %arg11: memref<1x123x194xf64, 1>, %arg12: memref<1x123x194xf64, 1>) {
// CHECK-DAG:    %c0 = arith.constant 0 : index
// CHECK-DAG:    %cst = arith.constant 0x7FF8000000000000 : f64
// CHECK-DAG:    %cst_0 = arith.constant 0.000000e+00 : f64
// CHECK-DAG:    %cst_1 = arith.constant -9.8066499999999994 : f64
// CHECK-DAG:    %cst_2 = arith.constant 9.600000e+01 : f64
// CHECK-DAG:    %cst_3 = arith.constant -0.0037661355580298995 : f64
// CHECK:    affine.parallel (%arg13, %arg14) = (0, 0) to (103, 180) {
// CHECK-NEXT:      %[[a1:.+]] = affine.load %arg0[27] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<35xf64, 1>
// CHECK-NEXT:      %[[a2:.+]] = affine.load %arg4[0, %arg13 + 19, %arg14 + 6] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a3:.+]] = arith.subf %[[a1]], %[[a2]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a4:.+]] = affine.load %arg4[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a5:.+]] = arith.subf %[[a1]], %[[a4]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a6:.+]] = math.isnan %[[a3]] : f64
// CHECK-NEXT:      %[[a7:.+]] = math.isnan %[[a5]] : f64
// CHECK-NEXT:      %[[a8:.+]] = arith.ori %[[a6]], %[[a7]] : i1
// CHECK-NEXT:      %[[a9:.+]] = arith.minnumf %[[a3]], %[[a5]] : f64
// CHECK-NEXT:      %[[a10:.+]] = arith.select %[[a8]], %cst, %[[a9]] : f64
// CHECK-NEXT:      %[[a11:.+]] = affine.load %arg4[0, %arg13 + 18, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a12:.+]] = arith.subf %[[a1]], %[[a11]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a13:.+]] = math.isnan %[[a12]] : f64
// CHECK-NEXT:      %[[a14:.+]] = arith.ori %[[a13]], %[[a7]] : i1
// CHECK-NEXT:      %[[a15:.+]] = arith.minnumf %[[a12]], %[[a5]] : f64
// CHECK-NEXT:      %[[a16:.+]] = arith.select %[[a14]], %cst, %[[a15]] : f64
// CHECK-NEXT:      %[[a17:.+]] = affine.load %arg6[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a18:.+]] = affine.load %arg1[26] {alignment = 16 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
// CHECK-NEXT:      %[[a19:.+]] = arith.cmpf ole, %[[a18]], %[[a4]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a20:.+]] = arith.cmpf ole, %[[a18]], %[[a2]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a21:.+]] = arith.ori %[[a19]], %[[a20]] : i1
// CHECK-NEXT:      %[[a22:.+]] = affine.load %arg5[0, %arg13 + 19, 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a23:.+]] = affine.load %arg5[0, %arg13 + 19, 186] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a24:.+]] = affine.load %arg5[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a25:.+]] = affine.load %arg5[0, %arg13 + 19, %arg14 + 6] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a26:.+]]:2 = affine.if #set(%arg14) -> (f64, f64) {
// CHECK-NEXT:       affine.yield %[[a22]], %[[a23]] : f64, f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:       affine.yield %[[a24]], %[[a25]] : f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[a27:.+]] = arith.subf %[[a26]]#0, %[[a26]]#1 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a28:.+]] = affine.load %arg2[%arg13 + 19, %arg14 + 7] : memref<123x194xf64, 1>
// CHECK-NEXT:      %[[a29:.+]] = arith.divf %[[a27]], %[[a28]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a30:.+]] = arith.select %[[a21]], %cst_0, %[[a29]] : f64
// CHECK-NEXT:      %[[a31:.+]] = arith.mulf %[[a10]], %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a32:.+]] = arith.mulf %[[a31]], %[[a30]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a33:.+]] = affine.load %arg11[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a34:.+]] = arith.addf %[[a33]], %[[a32]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a35:.+]] = arith.mulf %[[a34]], %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a36:.+]] = arith.addf %[[a17]], %[[a35]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %[[a36]], %arg6[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a37:.+]] = affine.load %arg7[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a38:.+]] = affine.load %arg4[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a39:.+]] = affine.load %arg1[26] {alignment = 16 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<34xf64, 1>
// CHECK-NEXT:      %[[a40:.+]] = arith.cmpf ole, %[[a39]], %[[a38]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a41:.+]] = affine.load %arg4[0, %arg13 + 18, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a42:.+]] = arith.cmpi eq, %arg13, %c0 : index
// CHECK-NEXT:      %[[a43:.+]] = arith.cmpf ole, %[[a39]], %[[a41]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a44:.+]] = arith.ori %[[a42]], %[[a43]] : i1
// CHECK-NEXT:      %[[a45:.+]] = arith.ori %[[a40]], %[[a44]] : i1
// CHECK-NEXT:      %[[a46:.+]] = affine.load %arg5[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a47:.+]] = affine.load %arg5[0, %arg13 + 18, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a48:.+]] = arith.subf %[[a46]], %[[a47]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a49:.+]] = affine.if #set(%arg13) -> f64 {
// CHECK-NEXT:       affine.yield %cst_0 : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:       affine.yield %[[a48]] : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[a50:.+]] = affine.load %arg3[%arg13 + 19, %arg14 + 7] : memref<123x194xf64, 1>
// CHECK-NEXT:      %[[a51:.+]] = arith.divf %[[a49]], %[[a50]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a52:.+]] = arith.select %[[a45]], %cst_0, %[[a51]] : f64
// CHECK-NEXT:      %[[a53:.+]] = arith.mulf %[[a16]], %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a54:.+]] = arith.mulf %[[a53]], %[[a52]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a55:.+]] = affine.load %arg12[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a56:.+]] = arith.addf %[[a55]], %[[a54]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a57:.+]] = arith.mulf %[[a56]], %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a58:.+]] = arith.addf %[[a37]], %[[a57]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %[[a58]], %arg7[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a59:.+]] = affine.load %arg8[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a60:.+]] = affine.load %arg5[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a61:.+]] = arith.mulf %[[a60]], %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a62:.+]] = arith.addf %[[a59]], %[[a61]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %[[a62]], %arg8[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a63:.+]] = affine.load %arg9[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a64:.+]] = affine.load %arg6[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a65:.+]] = arith.mulf %[[a64]], %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a66:.+]] = arith.addf %[[a63]], %[[a65]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %[[a66]], %arg9[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a67:.+]] = affine.load %arg10[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a68:.+]] = affine.load %arg7[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:      %[[a69:.+]] = arith.mulf %[[a68]], %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %[[a70:.+]] = arith.addf %[[a67]], %[[a69]] {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %[[a70]], %arg10[0, %arg13 + 19, %arg14 + 7] : memref<1x123x194xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
