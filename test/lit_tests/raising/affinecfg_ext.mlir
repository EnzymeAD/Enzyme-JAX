// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

func.func private @"##call__Z40gpu_compute_hydrostatic_free_surface_Gv_16CompilerMetadataI10StaticSizeI12_62__62__15_E12DynamicCheckvv7NDRangeILi3ES0_I10_4__4__15_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E12_78__78__31_EE21LatitudeLongitudeGridI15CuTracedRNumberIS9_Li1EE7BoundedSG_SG_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_32__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_31__EESJ_SL_ES9_S9_S8_IS9_Li1E12StepRangeLenIS9_7Float64SO_5Int64EESR_S9_S9_SR_SR_S8_IS9_Li1ESA_IS9_Li1ELi1E5_78__EEST_ST_ST_S9_S9_vSP_E5TupleI15VectorInvariantILi1ES9_Lb0E19EnstrophyConservingIS9_E15VelocityStencil16EnergyConservingIS9_ES11_S11_17OnlySelfUpwindingIS11_15FunctionStencilI21divergence_smoothnessES15_S13_I12u_smoothnessES13_I12v_smoothnessEEE28HydrostaticSphericalCoriolisISY_S9_E17ScalarDiffusivityI26ExplicitTimeDiscretization19VerticalFormulation10NamedTupleI16__T___S___e_____SV_IS9_S9_S9_S9_EES9_S1J_E17BoundaryConditionI4FluxvES1H_I12__u___v___w_SV_ISC_SC_SC_EE24SplitExplicitFreeSurfaceIS8_IS9_Li3ESA_IS9_Li3ELi1E11_78__78__1_EES1H_I8__U___V_SV_I5FieldI4Face6CentervvvvS1S_S9_vvvES1T_IS1V_S1U_vvvvS1S_S9_vvvEEES1H_I12______U___V_SV_IS1S_S1W_S1X_EES9_v18FixedSubstepNumberIS9_SV_IS9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_EE21ForwardBackwardSchemeES1H_I16__T___S___e_____SV_ISC_SC_SC_SC_EE13BuoyancyForceI16SeawaterBuoyancyIS9_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialIS9_ES9_EvvE18NegativeZDirectionEvSC_S1H_I2__SV_IJEEE11ZCoordinateS1H_I53__time___last__t___last_stage__t___iteration___stage_SV_I13TracedRNumberIS9_ES9_S9_S2L_ISP_ESP_EE11zeroforcingE#406$par78"(%arg0: memref<31x78x78xf32, 1>, %arg1: memref<32xf32, 1>, %arg2: memref<31xf32, 1>, %arg3: memref<78xf32, 1>, %arg4: memref<78xf32, 1>, %arg5: memref<78xf32, 1>, %arg6: memref<78xf32, 1>, %arg7: memref<78xf32, 1>, %arg8: memref<31x78x78xf32, 1>, %arg9: memref<31x78x78xf32, 1>, %arg10: memref<31x78x78xf32, 1>, %arg11: memref<31x78x78xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c-1_i64 = arith.constant -1 : i64
  %c1 = arith.constant 1 : index
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c62_i64 = arith.constant 62 : i64
  %true = arith.constant true
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant 107607.984 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 5.000000e-01 : f32
  %cst_2 = arith.constant 9.292990e-06 : f32
  %cst_3 = arith.constant 2.000000e+00 : f32
  %cst_4 = arith.constant 0.96774193548387099 : f64
  %cst_5 = arith.constant 7.258064516129032 : f64
  %cst_6 = arith.constant 3.14159274 : f32
  %cst_7 = arith.constant 1.800000e+02 : f32
  %cst_8 = arith.constant 1.45842307E-4 : f32
  %cst_9 = arith.constant 0.000000e+00 : f32
  %cst_10 = arith.constant 0.00999999977 : f32
  affine.parallel (%arg12, %arg13, %arg14) = (0, 0, 0) to (240, 16, 16) {
    %0 = arith.muli %arg13, %c16 overflow<nuw> : index
    %1 = arith.addi %0, %arg14 : index
    %2 = arith.addi %1, %c1 : index
    %3 = arith.index_castui %arg12 : index to i64
    %4 = arith.divui %arg12, %c4 : index
    %5 = arith.muli %4, %c4 : index
    %6 = arith.index_castui %5 : index to i64
    %7 = arith.subi %3, %6 : i64
    %8 = arith.remui %4, %c4 : index
    %9 = arith.index_castui %2 : index to i64
    %10 = arith.subi %c0, %arg13 : index
    %11 = arith.index_cast %10 : index to i64
    %12 = arith.addi %7, %11 : i64
    %13 = arith.muli %12, %c16_i64 : i64
    %14 = arith.addi %9, %13 : i64
    %15 = arith.muli %8, %c16 : index
    %16 = arith.addi %15, %arg13 : index
    %17 = arith.index_castui %16 : index to i64
    affine.if affine_set<(d0, d1, d2) : (d2 + (d1 mod 4) * 16 >= 0, -d2 - (d1 mod 4) * 16 + 61 >= 0, d0 + ((d1 floordiv 4) mod 4) * 16 >= 0, -d0 - ((d1 floordiv 4) mod 4) * 16 + 61 >= 0)>(%arg13, %arg12, %arg14) {
      %18 = arith.addi %17, %c8_i64 : i64
      %19 = affine.load %arg9[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %20 = arith.mulf %19, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %21 = affine.load %arg9[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 7] : memref<31x78x78xf32, 1>
      %22 = arith.mulf %21, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %23 = arith.subf %20, %22 {fastmathFlags = #llvm.fastmath<none>} : f32
      %24 = affine.load %arg4[%arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<78xf32, 1>
      %25 = affine.load %arg8[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %26 = arith.mulf %24, %25 {fastmathFlags = #llvm.fastmath<none>} : f32
      %27 = affine.load %arg4[%arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 7] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<78xf32, 1>
      %28 = affine.load %arg8[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 7, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %29 = arith.mulf %27, %28 {fastmathFlags = #llvm.fastmath<none>} : f32
      %30 = arith.subf %26, %29 {fastmathFlags = #llvm.fastmath<none>} : f32
      %31 = arith.subf %23, %30 {fastmathFlags = #llvm.fastmath<none>} : f32
      %32 = affine.load %arg7[%arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<78xf32, 1>
      %33 = arith.divf %cst_0, %32 {fastmathFlags = #llvm.fastmath<none>} : f32
      %34 = arith.mulf %31, %33 {fastmathFlags = #llvm.fastmath<none>} : f32
      %35 = arith.addi %14, %c1_i64 : i64
      %36 = affine.load %arg9[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 9] : memref<31x78x78xf32, 1>
      %37 = arith.mulf %36, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %38 = arith.subf %37, %20 {fastmathFlags = #llvm.fastmath<none>} : f32
      %39 = affine.load %arg8[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 9] : memref<31x78x78xf32, 1>
      %40 = arith.mulf %24, %39 {fastmathFlags = #llvm.fastmath<none>} : f32
      %41 = affine.load %arg8[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 7, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 9] : memref<31x78x78xf32, 1>
      %42 = arith.mulf %27, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
      %43 = arith.subf %40, %42 {fastmathFlags = #llvm.fastmath<none>} : f32
      %44 = arith.subf %38, %43 {fastmathFlags = #llvm.fastmath<none>} : f32
      %45 = arith.mulf %44, %33 {fastmathFlags = #llvm.fastmath<none>} : f32
      %46 = arith.addf %34, %45 {fastmathFlags = #llvm.fastmath<none>} : f32
      %47 = arith.mulf %46, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %48 = arith.mulf %28, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %49 = arith.mulf %41, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %50 = arith.addf %48, %49 {fastmathFlags = #llvm.fastmath<none>} : f32
      %51 = arith.mulf %50, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %52 = arith.mulf %25, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %53 = arith.mulf %39, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %54 = arith.addf %52, %53 {fastmathFlags = #llvm.fastmath<none>} : f32
      %55 = arith.mulf %54, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %56 = arith.addf %51, %55 {fastmathFlags = #llvm.fastmath<none>} : f32
      %57 = arith.mulf %56, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %58 = arith.mulf %47, %57 {fastmathFlags = #llvm.fastmath<none>} : f32
      %59 = arith.mulf %58, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f32
      %60 = affine.load %arg5[%arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 7] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<78xf32, 1>
      %61 = affine.load %arg10[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 7, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %62 = arith.mulf %60, %61 {fastmathFlags = #llvm.fastmath<none>} : f32
      %63 = affine.load %arg5[%arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<78xf32, 1>
      %64 = affine.load %arg10[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %65 = arith.mulf %63, %64 {fastmathFlags = #llvm.fastmath<none>} : f32
      %66 = arith.addf %62, %65 {fastmathFlags = #llvm.fastmath<none>} : f32
      %67 = arith.mulf %66, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %68 = affine.load %arg9[%arg12 floordiv 16 + 7, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %69 = arith.subf %19, %68 {fastmathFlags = #llvm.fastmath<none>} : f32
      %70 = affine.load %arg1[%arg12 floordiv 16 + 9] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<32xf32, 1>
      %71 = arith.divf %cst_0, %70 {fastmathFlags = #llvm.fastmath<none>} : f32
      %72 = arith.mulf %69, %71 {fastmathFlags = #llvm.fastmath<none>} : f32
      %73 = arith.mulf %67, %72 {fastmathFlags = #llvm.fastmath<none>} : f32
      %74 = affine.load %arg10[%arg12 floordiv 16 + 9, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 7, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %75 = arith.mulf %60, %74 {fastmathFlags = #llvm.fastmath<none>} : f32
      %76 = affine.load %arg10[%arg12 floordiv 16 + 9, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %77 = arith.mulf %63, %76 {fastmathFlags = #llvm.fastmath<none>} : f32
      %78 = arith.addf %75, %77 {fastmathFlags = #llvm.fastmath<none>} : f32
      %79 = arith.mulf %78, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %80 = affine.load %arg9[%arg12 floordiv 16 + 9, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %81 = arith.subf %80, %19 {fastmathFlags = #llvm.fastmath<none>} : f32
      %82 = affine.load %arg1[%arg12 floordiv 16 + 10] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<32xf32, 1>
      %83 = arith.divf %cst_0, %82 {fastmathFlags = #llvm.fastmath<none>} : f32
      %84 = arith.mulf %81, %83 {fastmathFlags = #llvm.fastmath<none>} : f32
      %85 = arith.mulf %79, %84 {fastmathFlags = #llvm.fastmath<none>} : f32
      %86 = arith.addf %73, %85 {fastmathFlags = #llvm.fastmath<none>} : f32
      %87 = arith.mulf %86, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %88 = affine.load %arg6[%arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<78xf32, 1>
      %89 = arith.divf %cst_0, %88 {fastmathFlags = #llvm.fastmath<none>} : f32
      %90 = arith.mulf %87, %89 {fastmathFlags = #llvm.fastmath<none>} : f32
      %91 = arith.mulf %25, %25 {fastmathFlags = #llvm.fastmath<none>} : f32
      %92 = arith.mulf %39, %39 {fastmathFlags = #llvm.fastmath<none>} : f32
      %93 = arith.addf %91, %92 {fastmathFlags = #llvm.fastmath<none>} : f32
      %94 = arith.mulf %93, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %95 = arith.mulf %19, %19 {fastmathFlags = #llvm.fastmath<none>} : f32
      %96 = affine.load %arg9[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 9, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %97 = arith.mulf %96, %96 {fastmathFlags = #llvm.fastmath<none>} : f32
      %98 = arith.addf %95, %97 {fastmathFlags = #llvm.fastmath<none>} : f32
      %99 = arith.mulf %98, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %100 = arith.addf %94, %99 {fastmathFlags = #llvm.fastmath<none>} : f32
      %101 = arith.divf %100, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f32
      %102 = arith.mulf %28, %28 {fastmathFlags = #llvm.fastmath<none>} : f32
      %103 = arith.mulf %41, %41 {fastmathFlags = #llvm.fastmath<none>} : f32
      %104 = arith.addf %102, %103 {fastmathFlags = #llvm.fastmath<none>} : f32
      %105 = arith.mulf %104, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %106 = affine.load %arg9[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 7, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %107 = arith.mulf %106, %106 {fastmathFlags = #llvm.fastmath<none>} : f32
      %108 = arith.addf %107, %95 {fastmathFlags = #llvm.fastmath<none>} : f32
      %109 = arith.mulf %108, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %110 = arith.addf %105, %109 {fastmathFlags = #llvm.fastmath<none>} : f32
      %111 = arith.divf %110, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f32
      %112 = arith.subf %101, %111 {fastmathFlags = #llvm.fastmath<none>} : f32
      %113 = arith.mulf %112, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f32
      %114 = arith.addf %59, %90 {fastmathFlags = #llvm.fastmath<none>} : f32
      %115 = arith.addf %114, %113 {fastmathFlags = #llvm.fastmath<none>} : f32
      %116 = arith.negf %115 {fastmathFlags = #llvm.fastmath<none>} : f32
      %117 = arith.sitofp %18 : i64 to f64
      %118 = arith.mulf %117, %cst_4 {fastmathFlags = #llvm.fastmath<none>} : f64
      %119 = arith.addf %118, %cst_5 {fastmathFlags = #llvm.fastmath<none>} : f64
      %120 = arith.truncf %119 : f64 to f32
      %121 = arith.mulf %120, %cst_6 {fastmathFlags = #llvm.fastmath<none>} : f32
      %122 = arith.divf %121, %cst_7 {fastmathFlags = #llvm.fastmath<none>} : f32
      %123 = math.sin %122 : f32
      %124 = arith.mulf %123, %cst_8 {fastmathFlags = #llvm.fastmath<none>} : f32
      %125 = arith.addf %124, %124 {fastmathFlags = #llvm.fastmath<none>} : f32
      %126 = arith.mulf %125, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %127 = arith.cmpi ult, %17, %c1_i64 : i64
      %128 = arith.cmpi sgt, %17, %c62_i64 : i64
      %129 = arith.ori %127, %128 : i1
      %130 = arith.addi %14, %c-1_i64 : i64
      %131 = arith.cmpi ult, %130, %c1_i64 : i64
      %132 = arith.cmpi sgt, %130, %c62_i64 : i64
      %133 = arith.ori %131, %132 : i1
      %134 = arith.ori %133, %129 : i1
      %135 = arith.xori %134, %true : i1
      %136 = arith.cmpi sgt, %35, %c62_i64 : i64
      %137 = arith.ori %136, %129 : i1
      %138 = arith.xori %137, %true : i1
      %139 = arith.extui %135 : i1 to i64
      %140 = arith.extui %138 : i1 to i64
      %141 = arith.addi %140, %139 : i64
      %142 = arith.sitofp %141 : i64 to f32
      %143 = arith.mulf %142, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %144 = arith.xori %133, %true : i1
      %145 = arith.cmpi sle, %35, %c62_i64 : i64
      %146 = arith.extui %144 : i1 to i64
      %147 = arith.extui %145 : i1 to i64
      %148 = arith.addi %147, %146 : i64
      %149 = arith.sitofp %148 : i64 to f32
      %150 = arith.mulf %149, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %151 = arith.addf %143, %150 {fastmathFlags = #llvm.fastmath<none>} : f32
      %152 = arith.mulf %151, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %153 = arith.cmpf oeq, %152, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f32
      %154 = arith.divf %57, %152 {fastmathFlags = #llvm.fastmath<none>} : f32
      %155 = arith.select %153, %cst_9, %154 : f32
      %156 = arith.mulf %126, %155 {fastmathFlags = #llvm.fastmath<none>} : f32
      %157 = arith.mulf %156, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f32
      %158 = arith.subf %116, %157 {fastmathFlags = #llvm.fastmath<none>} : f32
      %159 = affine.load %arg11[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %160 = affine.load %arg11[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 7, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
      %161 = arith.subf %159, %160 {fastmathFlags = #llvm.fastmath<none>} : f32
      %162 = arith.mulf %161, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f32
      %163 = arith.subf %158, %162 {fastmathFlags = #llvm.fastmath<none>} : f32
      %164 = affine.load %arg2[%arg12 floordiv 16 + 8] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<31xf32, 1>
      %165 = arith.mulf %88, %164 {fastmathFlags = #llvm.fastmath<none>} : f32
      %166 = arith.divf %cst_0, %165 {fastmathFlags = #llvm.fastmath<none>} : f32
      %167 = arith.mulf %164, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %168 = arith.mulf %167, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f32
      %169 = arith.subf %168, %168 {fastmathFlags = #llvm.fastmath<none>} : f32
      %170 = affine.load %arg3[%arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<78xf32, 1>
      %171 = arith.mulf %170, %164 {fastmathFlags = #llvm.fastmath<none>} : f32
      %172 = arith.mulf %171, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f32
      %173 = affine.load %arg3[%arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 7] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : memref<78xf32, 1>
      %174 = arith.mulf %173, %164 {fastmathFlags = #llvm.fastmath<none>} : f32
      %175 = arith.mulf %174, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f32
      %176 = arith.subf %172, %175 {fastmathFlags = #llvm.fastmath<none>} : f32
      %177 = arith.mulf %84, %cst_10 {fastmathFlags = #llvm.fastmath<none>} : f32
      %178 = arith.negf %177 {fastmathFlags = #llvm.fastmath<none>} : f32
      %179 = arith.mulf %88, %178 {fastmathFlags = #llvm.fastmath<none>} : f32
      %180 = arith.mulf %72, %cst_10 {fastmathFlags = #llvm.fastmath<none>} : f32
      %181 = arith.negf %180 {fastmathFlags = #llvm.fastmath<none>} : f32
      %182 = arith.mulf %88, %181 {fastmathFlags = #llvm.fastmath<none>} : f32
      %183 = arith.subf %179, %182 {fastmathFlags = #llvm.fastmath<none>} : f32
      %184 = arith.addf %169, %176 {fastmathFlags = #llvm.fastmath<none>} : f32
      %185 = arith.addf %184, %183 {fastmathFlags = #llvm.fastmath<none>} : f32
      %186 = arith.mulf %166, %185 {fastmathFlags = #llvm.fastmath<none>} : f32
      %187 = arith.subf %163, %186 {fastmathFlags = #llvm.fastmath<none>} : f32
      %188 = arith.addf %187, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f32
      affine.store %188, %arg0[%arg12 floordiv 16 + 8, %arg13 + (%arg12 floordiv 4) * 16 - (%arg12 floordiv 16) * 64 + 8, %arg12 * 16 + %arg14 - (%arg12 floordiv 4) * 64 + 8] : memref<31x78x78xf32, 1>
    }
  }
  return
}

// CHECK: #set = affine_set<(d0, d1) : (d1 - 1 >= 0, -d1 + 62 >= 0, d0 - 1 >= 0, -d0 + 62 >= 0)>
// CHECK: #set1 = affine_set<(d0, d1) : (-d1 + 60 >= 0, d0 - 1 >= 0, -d0 + 62 >= 0)>
// CHECK: #set2 = affine_set<(d0) : (d0 - 1 >= 0, -d0 + 62 >= 0)>
// CHECK: #set3 = affine_set<(d0) : (-d0 + 60 >= 0)>
// CHECK:   func.func private @"##call__Z40gpu_compute_hydrostatic_free_surface_Gv_16CompilerMetadataI10StaticSizeI12_62__62__15_E12DynamicCheckvv7NDRangeILi3ES0_I10_4__4__15_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E12_78__78__31_EE21LatitudeLongitudeGridI15CuTracedRNumberIS9_Li1EE7BoundedSG_SG_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_32__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_31__EESJ_SL_ES9_S9_S8_IS9_Li1E12StepRangeLenIS9_7Float64SO_5Int64EESR_S9_S9_SR_SR_S8_IS9_Li1ESA_IS9_Li1ELi1E5_78__EEST_ST_ST_S9_S9_vSP_E5TupleI15VectorInvariantILi1ES9_Lb0E19EnstrophyConservingIS9_E15VelocityStencil16EnergyConservingIS9_ES11_S11_17OnlySelfUpwindingIS11_15FunctionStencilI21divergence_smoothnessES15_S13_I12u_smoothnessES13_I12v_smoothnessEEE28HydrostaticSphericalCoriolisISY_S9_E17ScalarDiffusivityI26ExplicitTimeDiscretization19VerticalFormulation10NamedTupleI16__T___S___e_____SV_IS9_S9_S9_S9_EES9_S1J_E17BoundaryConditionI4FluxvES1H_I12__u___v___w_SV_ISC_SC_SC_EE24SplitExplicitFreeSurfaceIS8_IS9_Li3ESA_IS9_Li3ELi1E11_78__78__1_EES1H_I8__U___V_SV_I5FieldI4Face6CentervvvvS1S_S9_vvvES1T_IS1V_S1U_vvvvS1S_S9_vvvEEES1H_I12______U___V_SV_IS1S_S1W_S1X_EES9_v18FixedSubstepNumberIS9_SV_IS9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_EE21ForwardBackwardSchemeES1H_I16__T___S___e_____SV_ISC_SC_SC_SC_EE13BuoyancyForceI16SeawaterBuoyancyIS9_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialIS9_ES9_EvvE18NegativeZDirectionEvSC_S1H_I2__SV_IJEEE11ZCoordinateS1H_I53__time___last__t___last_stage__t___iteration___stage_SV_I13TracedRNumberIS9_ES9_S9_S2L_ISP_ESP_EE11zeroforcingE#406$par78"(%arg0: memref<31x78x78xf32, 1>, %arg1: memref<32xf32, 1>, %arg2: memref<31xf32, 1>, %arg3: memref<78xf32, 1>, %arg4: memref<78xf32, 1>, %arg5: memref<78xf32, 1>, %arg6: memref<78xf32, 1>, %arg7: memref<78xf32, 1>, %arg8: memref<31x78x78xf32, 1>, %arg9: memref<31x78x78xf32, 1>, %arg10: memref<31x78x78xf32, 1>, %arg11: memref<31x78x78xf32, 1>) {
// CHECK-NEXT:     %c8 = arith.constant 8 : index
// CHECK-NEXT:     %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %cst = arith.constant 107607.984 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_1 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:     %cst_2 = arith.constant 9.292990e-06 : f32
// CHECK-NEXT:     %cst_3 = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:     %cst_4 = arith.constant 0.96774193548387099 : f64
// CHECK-NEXT:     %cst_5 = arith.constant 7.258064516129032 : f64
// CHECK-NEXT:     %cst_6 = arith.constant 3.14159274 : f32
// CHECK-NEXT:     %cst_7 = arith.constant 1.800000e+02 : f32
// CHECK-NEXT:     %cst_8 = arith.constant 1.45842307E-4 : f32
// CHECK-NEXT:     %cst_9 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %cst_10 = arith.constant 0.00999999977 : f32
// CHECK-NEXT:     affine.parallel (%arg12, %arg13, %arg14) = (0, 0, 0) to (15, 62, 62) {
// CHECK-NEXT:       %0 = arith.addi %arg13, %c8 : index
// CHECK-NEXT:       %1 = arith.index_castui %0 : index to i64
// CHECK-NEXT:       %2 = affine.load %arg9[%arg12 + 8, %arg13 + 8, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %3 = arith.mulf %2, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %4 = affine.load %arg9[%arg12 + 8, %arg13 + 8, %arg14 + 7] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %5 = arith.mulf %4, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %6 = arith.subf %3, %5 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %7 = affine.load %arg4[%arg13 + 8] : memref<78xf32, 1>
// CHECK-NEXT:       %8 = affine.load %arg8[%arg12 + 8, %arg13 + 8, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %9 = arith.mulf %7, %8 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %10 = affine.load %arg4[%arg13 + 7] : memref<78xf32, 1>
// CHECK-NEXT:       %11 = affine.load %arg8[%arg12 + 8, %arg13 + 7, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %12 = arith.mulf %10, %11 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %13 = arith.subf %9, %12 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %14 = arith.subf %6, %13 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %15 = affine.load %arg7[%arg13 + 8] : memref<78xf32, 1>
// CHECK-NEXT:       %16 = arith.divf %cst_0, %15 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %17 = arith.mulf %14, %16 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %18 = affine.load %arg9[%arg12 + 8, %arg13 + 8, %arg14 + 9] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %19 = arith.mulf %18, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %20 = arith.subf %19, %3 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %21 = affine.load %arg8[%arg12 + 8, %arg13 + 8, %arg14 + 9] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %22 = arith.mulf %7, %21 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %23 = affine.load %arg8[%arg12 + 8, %arg13 + 7, %arg14 + 9] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %24 = arith.mulf %10, %23 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %25 = arith.subf %22, %24 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %26 = arith.subf %20, %25 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %27 = arith.mulf %26, %16 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %28 = arith.addf %17, %27 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %29 = arith.mulf %28, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %30 = arith.mulf %11, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %31 = arith.mulf %23, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %32 = arith.addf %30, %31 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %33 = arith.mulf %32, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %34 = arith.mulf %8, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %35 = arith.mulf %21, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %36 = arith.addf %34, %35 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %37 = arith.mulf %36, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %38 = arith.addf %33, %37 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %39 = arith.mulf %38, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %40 = arith.mulf %29, %39 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %41 = arith.mulf %40, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %42 = affine.load %arg5[%arg13 + 7] : memref<78xf32, 1>
// CHECK-NEXT:       %43 = affine.load %arg10[%arg12 + 8, %arg13 + 7, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %44 = arith.mulf %42, %43 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %45 = affine.load %arg5[%arg13 + 8] : memref<78xf32, 1>
// CHECK-NEXT:       %46 = affine.load %arg10[%arg12 + 8, %arg13 + 8, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %47 = arith.mulf %45, %46 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %48 = arith.addf %44, %47 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %49 = arith.mulf %48, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %50 = affine.load %arg9[%arg12 + 7, %arg13 + 8, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %51 = arith.subf %2, %50 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %52 = affine.load %arg1[%arg12 + 9] : memref<32xf32, 1>
// CHECK-NEXT:       %53 = arith.divf %cst_0, %52 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %54 = arith.mulf %51, %53 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %55 = arith.mulf %49, %54 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %56 = affine.load %arg10[%arg12 + 9, %arg13 + 7, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %57 = arith.mulf %42, %56 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %58 = affine.load %arg10[%arg12 + 9, %arg13 + 8, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %59 = arith.mulf %45, %58 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %60 = arith.addf %57, %59 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %61 = arith.mulf %60, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %62 = affine.load %arg9[%arg12 + 9, %arg13 + 8, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %63 = arith.subf %62, %2 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %64 = affine.load %arg1[%arg12 + 10] : memref<32xf32, 1>
// CHECK-NEXT:       %65 = arith.divf %cst_0, %64 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %66 = arith.mulf %63, %65 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %67 = arith.mulf %61, %66 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %68 = arith.addf %55, %67 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %69 = arith.mulf %68, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %70 = affine.load %arg6[%arg13 + 8] : memref<78xf32, 1>
// CHECK-NEXT:       %71 = arith.divf %cst_0, %70 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %72 = arith.mulf %69, %71 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %73 = arith.mulf %8, %8 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %74 = arith.mulf %21, %21 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %75 = arith.addf %73, %74 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %76 = arith.mulf %75, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %77 = arith.mulf %2, %2 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %78 = affine.load %arg9[%arg12 + 8, %arg13 + 9, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %79 = arith.mulf %78, %78 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %80 = arith.addf %77, %79 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %81 = arith.mulf %80, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %82 = arith.addf %76, %81 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %83 = arith.divf %82, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %84 = arith.mulf %11, %11 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %85 = arith.mulf %23, %23 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %86 = arith.addf %84, %85 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %87 = arith.mulf %86, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %88 = affine.load %arg9[%arg12 + 8, %arg13 + 7, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %89 = arith.mulf %88, %88 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %90 = arith.addf %89, %77 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %91 = arith.mulf %90, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %92 = arith.addf %87, %91 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %93 = arith.divf %92, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %94 = arith.subf %83, %93 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %95 = arith.mulf %94, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %96 = arith.addf %41, %72 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %97 = arith.addf %96, %95 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %98 = arith.negf %97 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %99 = arith.sitofp %1 : i64 to f64
// CHECK-NEXT:       %100 = arith.mulf %99, %cst_4 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:       %101 = arith.addf %100, %cst_5 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:       %102 = arith.truncf %101 : f64 to f32
// CHECK-NEXT:       %103 = arith.mulf %102, %cst_6 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %104 = arith.divf %103, %cst_7 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %105 = math.sin %104 : f32
// CHECK-NEXT:       %106 = arith.mulf %105, %cst_8 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %107 = arith.addf %106, %106 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %108 = arith.mulf %107, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %109 = affine.if #set(%arg13, %arg14) -> i64 {
// CHECK-NEXT:         affine.yield %c1_i64 : i64
// CHECK-NEXT:       } else {
// CHECK-NEXT:         affine.yield %c0_i64 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:       %110 = affine.if #set1(%arg13, %arg14) -> i64 {
// CHECK-NEXT:         affine.yield %c1_i64 : i64
// CHECK-NEXT:       } else {
// CHECK-NEXT:         affine.yield %c0_i64 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:       %111 = arith.addi %110, %109 : i64
// CHECK-NEXT:       %112 = arith.sitofp %111 : i64 to f32
// CHECK-NEXT:       %113 = arith.mulf %112, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %114 = affine.if #set2(%arg14) -> i64 {
// CHECK-NEXT:         affine.yield %c1_i64 : i64
// CHECK-NEXT:       } else {
// CHECK-NEXT:         affine.yield %c0_i64 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:       %115 = affine.if #set3(%arg14) -> i64 {
// CHECK-NEXT:         affine.yield %c1_i64 : i64
// CHECK-NEXT:       } else {
// CHECK-NEXT:         affine.yield %c0_i64 : i64
// CHECK-NEXT:       }
// CHECK-NEXT:       %116 = arith.addi %115, %114 : i64
// CHECK-NEXT:       %117 = arith.sitofp %116 : i64 to f32
// CHECK-NEXT:       %118 = arith.mulf %117, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %119 = arith.addf %113, %118 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %120 = arith.mulf %119, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %121 = arith.cmpf oeq, %120, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %122 = arith.divf %39, %120 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %123 = arith.select %121, %cst_9, %122 : f32
// CHECK-NEXT:       %124 = arith.mulf %108, %123 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %125 = arith.mulf %124, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %126 = arith.subf %98, %125 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %127 = affine.load %arg11[%arg12 + 8, %arg13 + 8, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %128 = affine.load %arg11[%arg12 + 8, %arg13 + 7, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:       %129 = arith.subf %127, %128 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %130 = arith.mulf %129, %cst_2 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %131 = arith.subf %126, %130 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %132 = affine.load %arg2[%arg12 + 8] : memref<31xf32, 1>
// CHECK-NEXT:       %133 = arith.mulf %70, %132 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %134 = arith.divf %cst_0, %133 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %135 = arith.mulf %132, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %136 = arith.mulf %135, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %137 = arith.subf %136, %136 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %138 = affine.load %arg3[%arg13 + 8] : memref<78xf32, 1>
// CHECK-NEXT:       %139 = arith.mulf %138, %132 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %140 = arith.mulf %139, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %141 = affine.load %arg3[%arg13 + 7] : memref<78xf32, 1>
// CHECK-NEXT:       %142 = arith.mulf %141, %132 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %143 = arith.mulf %142, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %144 = arith.subf %140, %143 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %145 = arith.mulf %66, %cst_10 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %146 = arith.negf %145 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %147 = arith.mulf %70, %146 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %148 = arith.mulf %54, %cst_10 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %149 = arith.negf %148 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %150 = arith.mulf %70, %149 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %151 = arith.subf %147, %150 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %152 = arith.addf %137, %144 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %153 = arith.addf %152, %151 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %154 = arith.mulf %134, %153 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %155 = arith.subf %131, %154 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       %156 = arith.addf %155, %cst_9 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:       affine.store %156, %arg0[%arg12 + 8, %arg13 + 8, %arg14 + 8] : memref<31x78x78xf32, 1>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

