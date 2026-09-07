// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file --canonicalize | FileCheck %s

module {
 func.func private @"##call__Z40gpu_compute_hydrostatic_free_surface_Gc_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SE_SF_SG_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESK_SM_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SR_SR_EvE16GridFittedBottomI5FieldI6CenterSW_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvEv5TupleI3ValILi3EES14_I2_eEv24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthIS9_ES9_v13CATKEEquationIS9_EE24DefaultBoundaryConditionI17BoundaryConditionI4FluxvEE13BuoyancyForceI16SeawaterBuoyancyIS9_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialIS9_ES9_EvvE18NegativeZDirectionEv10NamedTupleI12__u___v___w_S13_ISC_SC_S8_IS9_Li3ESA_IS9_Li3ELi1E13_194__99__35_EEEE24SplitExplicitFreeSurfaceIS8_IS9_Li3ESA_IS9_Li3ELi1E13_194__187__1_EES1S_I8__U___V_S13_ISV_I4FaceSW_vvvvS1Z_S9_vvvESV_ISW_S20_vvvvS1Z_S9_vvvEEES1S_I12______U___V_S13_IS1Z_S21_S22_EES9_v18FixedSubstepNumberIS9_S13_IS9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_EE21ForwardBackwardSchemeES1S_I12__T___S___e_S13_ISC_SC_SC_EES1S_I141___u____c____e___Le___J____previous_compute_time___previous_velocities____tupled_tracer_diffusivities____tupled_implicit_linear_coefficients_S13_IS1U_S1U_S1U_SC_SZ_16ReactantRefValueIS9_ES1S_I8__u___v_S13_ISC_SC_EES1S_I12__T___S___e_S13_IS1U_S1U_S1U_EES1S_I12__T___S___e_S13_I9ZeroFieldISR_Li3EES2L_SC_EEEES1S_I2__S13_ES1S_I53__time___last__t___last_stage__t___iteration___stage_S13_IS9_S9_S9_SR_SR_EE11zeroforcingE#860$par244"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<35xf64, 1>, %arg3: memref<34xf64, 1>, %arg4: memref<99x194xf64, 1>, %arg5: memref<99x194xf64, 1>, %arg6: memref<99x194xf64, 1>, %arg7: memref<1x99x194xf64, 1>, %arg8: memref<34x99x194xf64, 1>, %arg9: memref<35x99x194xf64, 1>) {
  %c2 = arith.constant 2 : index
  %c1_i64 = arith.constant 1 : i64
  %c20_i64 = arith.constant 20 : i64
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 0.000000e+00 : f64
  affine.parallel (%arg10, %arg11, %arg12) = (0, 0, 0) to (20, 85, 180) {
    %0 = arith.index_castui %arg10 : index to i64
    %1 = affine.load %arg6[%arg11 + 7, %arg12 + 7] : memref<99x194xf64, 1>
    %2 = affine.load %arg3[%arg10 + 7] : memref<34xf64, 1>
    %3 = arith.mulf %1, %2 {fastmathFlags = #llvm.fastmath<none>} : f64
    %4 = arith.divf %cst, %3 {fastmathFlags = #llvm.fastmath<none>} : f64
    %5 = affine.load %arg5[%arg11 + 7, %arg12 + 8] : memref<99x194xf64, 1>
    %6 = arith.mulf %5, %2 {fastmathFlags = #llvm.fastmath<none>} : f64
    %7 = arith.mulf %6, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
    %8 = affine.load %arg5[%arg11 + 7, %arg12 + 7] : memref<99x194xf64, 1>
    %9 = arith.mulf %8, %2 {fastmathFlags = #llvm.fastmath<none>} : f64
    %10 = arith.mulf %9, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
    %11 = arith.subf %7, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
    %12 = affine.load %arg4[%arg11 + 8, %arg12 + 7] : memref<99x194xf64, 1>
    %13 = arith.mulf %12, %2 {fastmathFlags = #llvm.fastmath<none>} : f64
    %14 = arith.mulf %13, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
    %15 = affine.load %arg4[%arg11 + 7, %arg12 + 7] : memref<99x194xf64, 1>
    %16 = arith.mulf %15, %2 {fastmathFlags = #llvm.fastmath<none>} : f64
    %17 = arith.mulf %16, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
    %18 = arith.subf %14, %17 {fastmathFlags = #llvm.fastmath<none>} : f64
    %19 = arith.addi %arg10, %c2 : index
    %20 = arith.index_castui %19 : index to i64
    %21 = affine.load %arg9[%arg10 + 8, %arg11 + 7, %arg12 + 7] : memref<35x99x194xf64, 1>
    %22 = arith.negf %21 {fastmathFlags = #llvm.fastmath<none>} : f64
    %23 = affine.load %arg1[%arg10 + 8] : memref<34xf64, 1>
    %24 = affine.load %arg7[0, %arg11 + 7, %arg12 + 7] : memref<1x99x194xf64, 1>
    %25 = arith.cmpf ole, %23, %24 {fastmathFlags = #llvm.fastmath<none>} : f64
    %26 = arith.cmpi sgt, %20, %c20_i64 : i64
    %27 = arith.ori %26, %25 : i1
    %28 = arith.cmpi sle, %20, %c20_i64 : i64
    %29 = arith.andi %28, %27 : i1
    %30 = affine.load %arg1[%arg10 + 7] : memref<34xf64, 1>
    %31 = arith.cmpf ole, %30, %24 {fastmathFlags = #llvm.fastmath<none>} : f64
    %32 = arith.ori %29, %31 : i1
    %33 = affine.load %arg8[%arg10 + 8, %arg11 + 7, %arg12 + 7] : memref<34x99x194xf64, 1>
    %34 = affine.load %arg8[%arg10 + 7, %arg11 + 7, %arg12 + 7] : memref<34x99x194xf64, 1>
    %35 = arith.subf %33, %34 {fastmathFlags = #llvm.fastmath<none>} : f64
    %36 = arith.select %32, %cst_0, %35 : f64
    %37 = affine.load %arg2[%arg10 + 9] : memref<35xf64, 1>
    %38 = arith.divf %36, %37 {fastmathFlags = #llvm.fastmath<none>} : f64
    %39 = arith.mulf %22, %38 {fastmathFlags = #llvm.fastmath<none>} : f64
    %40 = affine.if affine_set<(d0) : (d0 - 19 == 0)>(%arg10) -> f64 {
      affine.yield %39 : f64
    } else {
      affine.yield %cst_0 : f64
    }
    %41 = arith.ori %27, %31 : i1
    %42 = arith.andi %28, %41 : i1
    %43 = arith.select %42, %cst_0, %40 : f64
    %44 = arith.mulf %1, %43 {fastmathFlags = #llvm.fastmath<none>} : f64
    %45 = affine.load %arg9[%arg10 + 7, %arg11 + 7, %arg12 + 7] : memref<35x99x194xf64, 1>
    %46 = arith.negf %45 {fastmathFlags = #llvm.fastmath<none>} : f64
    %47 = affine.load %arg1[%arg10 + 6] : memref<34xf64, 1>
    %48 = arith.cmpf ole, %47, %24 {fastmathFlags = #llvm.fastmath<none>} : f64
    %49 = arith.cmpi ult, %0, %c1_i64 : i64
    %50 = arith.ori %49, %48 : i1
    %51 = arith.cmpi uge, %0, %c1_i64 : i64
    %52 = arith.andi %51, %50 : i1
    %53 = arith.ori %31, %52 : i1
    %54 = affine.load %arg8[%arg10 + 6, %arg11 + 7, %arg12 + 7] : memref<34x99x194xf64, 1>
    %55 = arith.subf %34, %54 {fastmathFlags = #llvm.fastmath<none>} : f64
    %56 = arith.select %53, %cst_0, %55 : f64
    %57 = affine.load %arg2[%arg10 + 8] : memref<35xf64, 1>
    %58 = arith.divf %56, %57 {fastmathFlags = #llvm.fastmath<none>} : f64
    %59 = arith.mulf %46, %58 {fastmathFlags = #llvm.fastmath<none>} : f64
    %60 = affine.if affine_set<(d0) : (d0 == 0)>(%arg10) -> f64 {
      affine.yield %59 : f64
    } else {
      affine.yield %cst_0 : f64
    }
    %61 = arith.ori %31, %50 : i1
    %62 = arith.andi %51, %61 : i1
    %63 = arith.select %62, %cst_0, %60 : f64
    %64 = arith.mulf %1, %63 {fastmathFlags = #llvm.fastmath<none>} : f64
    %65 = arith.subf %44, %64 {fastmathFlags = #llvm.fastmath<none>} : f64
    %66 = arith.addf %11, %18 {fastmathFlags = #llvm.fastmath<none>} : f64
    %67 = arith.addf %66, %65 {fastmathFlags = #llvm.fastmath<none>} : f64
    %68 = arith.mulf %4, %67 {fastmathFlags = #llvm.fastmath<none>} : f64
    %69 = arith.negf %68 : f64
    affine.store %69, %arg0[%arg10 + 7, %arg11 + 7, %arg12 + 7] : memref<34x99x194xf64, 1>
  }
  return
}
}

// CHECK:    func.func private @"##call__Z40gpu_compute_hydrostatic_free_surface_Gc_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SE_SF_SG_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESK_SM_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SR_SR_EvE16GridFittedBottomI5FieldI6CenterSW_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvEv5TupleI3ValILi3EES14_I2_eEv24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthIS9_ES9_v13CATKEEquationIS9_EE24DefaultBoundaryConditionI17BoundaryConditionI4FluxvEE13BuoyancyForceI16SeawaterBuoyancyIS9_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialIS9_ES9_EvvE18NegativeZDirectionEv10NamedTupleI12__u___v___w_S13_ISC_SC_S8_IS9_Li3ESA_IS9_Li3ELi1E13_194__99__35_EEEE24SplitExplicitFreeSurfaceIS8_IS9_Li3ESA_IS9_Li3ELi1E13_194__187__1_EES1S_I8__U___V_S13_ISV_I4FaceSW_vvvvS1Z_S9_vvvESV_ISW_S20_vvvvS1Z_S9_vvvEEES1S_I12______U___V_S13_IS1Z_S21_S22_EES9_v18FixedSubstepNumberIS9_S13_IS9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_EE21ForwardBackwardSchemeES1S_I12__T___S___e_S13_ISC_SC_SC_EES1S_I141___u____c____e___Le___J____previous_compute_time___previous_velocities____tupled_tracer_diffusivities____tupled_implicit_linear_coefficients_S13_IS1U_S1U_S1U_SC_SZ_16ReactantRefValueIS9_ES1S_I8__u___v_S13_ISC_SC_EES1S_I12__T___S___e_S13_IS1U_S1U_S1U_EES1S_I12__T___S___e_S13_I9ZeroFieldISR_Li3EES2L_SC_EEEES1S_I2__S13_ES1S_I53__time___last__t___last_stage__t___iteration___stage_S13_IS9_S9_S9_SR_SR_EE11zeroforcingE#860$par244_raised"(%[[a1:.+]]: tensor<34x99x194xf64>, %[[a2:.+]]: tensor<34xf64>, %[[a3:.+]]: tensor<35xf64>, %[[a4:.+]]: tensor<34xf64>, %[[a5:.+]]: tensor<99x194xf64>, %[[a6:.+]]: tensor<99x194xf64>, %[[a7:.+]]: tensor<99x194xf64>, %[[a8:.+]]: tensor<1x99x194xf64>, %[[a9:.+]]: tensor<34x99x194xf64>, %[[a10:.+]]: tensor<35x99x194xf64>) -> (tensor<34x99x194xf64>, tensor<34xf64>, tensor<35xf64>, tensor<34xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<1x99x194xf64>, tensor<34x99x194xf64>, tensor<35x99x194xf64>) {
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.constant dense<-19> : tensor<i64>
// CHECK-NEXT:    %[[a13:.+]] = stablehlo.constant dense<1> : tensor<20xi64>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.constant dense<0> : tensor<20xi64>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.constant dense<20> : tensor<i64>
// CHECK-NEXT:    %[[a18:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a19:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a20:.+]] = stablehlo.iota dim = 0 : tensor<20xi64>
// CHECK-NEXT:    %[[a21:.+]] = stablehlo.add %[[a20]], %[[a14]] : tensor<20xi64>
// CHECK-NEXT:    %[[a22:.+]] = stablehlo.multiply %[[a21]], %[[a13]] : tensor<20xi64>
// CHECK-NEXT:    %[[a23:.+]] = stablehlo.slice %[[a7]] [7:92, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[a24:.+]] = stablehlo.slice %[[a4]] [7:27] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[a25:.+]] = stablehlo.broadcast_in_dim %[[a23]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a26:.+]] = stablehlo.broadcast_in_dim %[[a24]], dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a27:.+]] = arith.mulf %[[a25]], %[[a26]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a28:.+]] = stablehlo.broadcast_in_dim %[[a18]], dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a29:.+]] = arith.divf %[[a28]], %[[a27]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a30:.+]] = stablehlo.slice %[[a6]] [7:92, 8:188] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[a31:.+]] = stablehlo.broadcast_in_dim %[[a30]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a32:.+]] = stablehlo.broadcast_in_dim %[[a24]], dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a33:.+]] = arith.mulf %[[a31]], %[[a32]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a34:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a35:.+]] = arith.mulf %[[a33]], %[[a34]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a36:.+]] = stablehlo.slice %[[a6]] [7:92, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[a37:.+]] = stablehlo.broadcast_in_dim %[[a36]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a38:.+]] = stablehlo.broadcast_in_dim %[[a24]], dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a39:.+]] = arith.mulf %[[a37]], %[[a38]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a40:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a41:.+]] = arith.mulf %[[a39]], %[[a40]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a42:.+]] = arith.subf %[[a35]], %[[a41]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a43:.+]] = stablehlo.slice %[[a5]] [8:93, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[a44:.+]] = stablehlo.broadcast_in_dim %[[a43]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a45:.+]] = stablehlo.broadcast_in_dim %[[a24]], dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a46:.+]] = arith.mulf %[[a44]], %[[a45]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a47:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a48:.+]] = arith.mulf %[[a46]], %[[a47]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a49:.+]] = stablehlo.slice %[[a5]] [7:92, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[a50:.+]] = stablehlo.broadcast_in_dim %[[a49]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a51:.+]] = stablehlo.broadcast_in_dim %[[a24]], dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a52:.+]] = arith.mulf %[[a50]], %[[a51]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a53:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a54:.+]] = arith.mulf %[[a52]], %[[a53]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a55:.+]] = arith.subf %[[a48]], %[[a54]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a56:.+]] = stablehlo.broadcast_in_dim %[[a15]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[a57:.+]] = arith.addi %[[a22]], %[[a56]] : tensor<20xi64>
// CHECK-NEXT:    %[[a58:.+]] = stablehlo.slice %[[a10]] [8:28, 7:92, 7:187] : (tensor<35x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a59:.+]] = arith.negf %[[a58]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a60:.+]] = stablehlo.slice %[[a2]] [8:28] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[a61:.+]] = stablehlo.slice %[[a8]] [0:1, 7:92, 7:187] : (tensor<1x99x194xf64>) -> tensor<1x85x180xf64>
// CHECK-NEXT:    %[[a62:.+]] = stablehlo.reshape %[[a61]] : (tensor<1x85x180xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[a63:.+]] = stablehlo.broadcast_in_dim %[[a60]], dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a64:.+]] = stablehlo.broadcast_in_dim %[[a62]], dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a65:.+]] = arith.cmpf ole, %[[a63]], %[[a64]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a66:.+]] = stablehlo.broadcast_in_dim %[[a17]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[a67:.+]] = arith.cmpi sgt, %[[a57]], %[[a66]] : tensor<20xi64>
// CHECK-NEXT:    %[[a68:.+]] = stablehlo.broadcast_in_dim %[[a67]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a69:.+]] = arith.ori %[[a68]], %[[a65]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a70:.+]] = stablehlo.broadcast_in_dim %[[a17]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[a71:.+]] = arith.cmpi sle, %[[a57]], %[[a70]] : tensor<20xi64>
// CHECK-NEXT:    %[[a72:.+]] = stablehlo.broadcast_in_dim %[[a71]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a73:.+]] = arith.andi %[[a72]], %[[a69]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a74:.+]] = stablehlo.slice %[[a2]] [7:27] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[a75:.+]] = stablehlo.broadcast_in_dim %[[a74]], dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a76:.+]] = stablehlo.broadcast_in_dim %[[a62]], dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a77:.+]] = arith.cmpf ole, %[[a75]], %[[a76]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a78:.+]] = arith.ori %[[a73]], %[[a77]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a79:.+]] = stablehlo.slice %[[a9]] [8:28, 7:92, 7:187] : (tensor<34x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a80:.+]] = stablehlo.slice %[[a9]] [7:27, 7:92, 7:187] : (tensor<34x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a81:.+]] = arith.subf %[[a79]], %[[a80]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a82:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a83:.+]] = arith.select %[[a78]], %[[a82]], %[[a81]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a84:.+]] = stablehlo.slice %[[a3]] [9:29] : (tensor<35xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[a85:.+]] = stablehlo.broadcast_in_dim %[[a84]], dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a86:.+]] = arith.divf %[[a83]], %[[a85]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a87:.+]] = arith.mulf %[[a59]], %[[a86]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a88:.+]] = stablehlo.broadcast_in_dim %[[a12]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[a89:.+]] = stablehlo.add %[[a22]], %[[a88]] : tensor<20xi64>
// CHECK-NEXT:    %[[a90:.+]] = stablehlo.compare EQ, %[[a89]], %[[a14]] : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
// CHECK-NEXT:    %[[a91:.+]] = stablehlo.broadcast_in_dim %[[a90]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a92:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a93:.+]] = stablehlo.select %[[a91]], %[[a87]], %[[a92]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a94:.+]] = arith.ori %[[a69]], %[[a77]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a95:.+]] = stablehlo.broadcast_in_dim %[[a71]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a96:.+]] = arith.andi %[[a95]], %[[a94]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a97:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a98:.+]] = arith.select %[[a96]], %[[a97]], %[[a93]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a99:.+]] = stablehlo.broadcast_in_dim %[[a23]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a100:.+]] = stablehlo.broadcast_in_dim %[[a98]], dims = [2, 0, 1] : (tensor<20x85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a101:.+]] = arith.mulf %[[a99]], %[[a100]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a102:.+]] = stablehlo.slice %[[a10]] [7:27, 7:92, 7:187] : (tensor<35x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a103:.+]] = arith.negf %[[a102]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a104:.+]] = stablehlo.slice %[[a2]] [6:26] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[a105:.+]] = stablehlo.broadcast_in_dim %[[a104]], dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a106:.+]] = stablehlo.broadcast_in_dim %[[a62]], dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a107:.+]] = arith.cmpf ole, %[[a105]], %[[a106]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a108:.+]] = stablehlo.broadcast_in_dim %[[a16]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[a109:.+]] = arith.cmpi ult, %[[a22]], %[[a108]] : tensor<20xi64>
// CHECK-NEXT:    %[[a110:.+]] = stablehlo.broadcast_in_dim %[[a109]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a111:.+]] = arith.ori %[[a110]], %[[a107]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a112:.+]] = stablehlo.broadcast_in_dim %[[a16]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[a113:.+]] = arith.cmpi uge, %[[a22]], %[[a112]] : tensor<20xi64>
// CHECK-NEXT:    %[[a114:.+]] = stablehlo.broadcast_in_dim %[[a113]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a115:.+]] = arith.andi %[[a114]], %[[a111]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a116:.+]] = arith.ori %[[a77]], %[[a115]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a117:.+]] = stablehlo.slice %[[a9]] [6:26, 7:92, 7:187] : (tensor<34x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a118:.+]] = arith.subf %[[a80]], %[[a117]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a119:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a120:.+]] = arith.select %[[a116]], %[[a119]], %[[a118]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a121:.+]] = stablehlo.slice %[[a3]] [8:28] : (tensor<35xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[a122:.+]] = stablehlo.broadcast_in_dim %[[a121]], dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a123:.+]] = arith.divf %[[a120]], %[[a122]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a124:.+]] = arith.mulf %[[a103]], %[[a123]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a125:.+]] = stablehlo.compare EQ, %[[a22]], %[[a14]] : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
// CHECK-NEXT:    %[[a126:.+]] = stablehlo.broadcast_in_dim %[[a125]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a127:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a128:.+]] = stablehlo.select %[[a126]], %[[a124]], %[[a127]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a129:.+]] = arith.ori %[[a77]], %[[a111]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a130:.+]] = stablehlo.broadcast_in_dim %[[a113]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a131:.+]] = arith.andi %[[a130]], %[[a129]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[a132:.+]] = stablehlo.broadcast_in_dim %[[a19]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a133:.+]] = arith.select %[[a131]], %[[a132]], %[[a128]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a134:.+]] = stablehlo.broadcast_in_dim %[[a23]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a135:.+]] = stablehlo.broadcast_in_dim %[[a133]], dims = [2, 0, 1] : (tensor<20x85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a136:.+]] = arith.mulf %[[a134]], %[[a135]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a137:.+]] = arith.subf %[[a101]], %[[a136]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a138:.+]] = arith.addf %[[a42]], %[[a55]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a139:.+]] = arith.addf %[[a138]], %[[a137]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a140:.+]] = arith.mulf %[[a29]], %[[a139]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a141:.+]] = arith.negf %[[a140]] : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[a142:.+]] = stablehlo.broadcast_in_dim %[[a141]], dims = [1, 2, 0] : (tensor<85x180x20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[a143:.+]] = stablehlo.dynamic_update_slice %[[a1]], %[[a142]], %[[a11]], %[[a11]], %[[a11]] : (tensor<34x99x194xf64>, tensor<20x85x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<34x99x194xf64>
// CHECK-NEXT:    return %[[a143]], %[[a2]], %[[a3]], %[[a4]], %[[a5]], %[[a6]], %[[a7]], %[[a8]], %[[a9]], %[[a10]] : tensor<34x99x194xf64>, tensor<34xf64>, tensor<35xf64>, tensor<34xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<1x99x194xf64>, tensor<34x99x194xf64>, tensor<35x99x194xf64>
// CHECK-NEXT:  }

// -----

#set = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0) : (-d0 + 89 >= 0)>
module {
  func.func private @par6(%arg0: memref<1x104x194xf64, 1>) {
    %c2_i64 = arith.constant 2 : i64
    %c182_i64 = arith.constant 182 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-1_i64 = arith.constant -1 : i64
    affine.parallel (%arg1) = (0) to (180) {
      %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x104x194xf64, 1>
      affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x104x194xf64, 1>
      %1:10 = affine.if #set(%arg1) -> (i64, i64, f64, f64, f64, f64, f64, f64, f64, f64) {
        %13 = affine.load %arg0[0, 96, -%arg1 + 187] : memref<1x104x194xf64, 1>
        %14 = affine.load %arg0[0, 89, -%arg1 + 187] : memref<1x104x194xf64, 1>
        %15 = affine.load %arg0[0, 90, -%arg1 + 187] : memref<1x104x194xf64, 1>
        %16 = affine.load %arg0[0, 91, -%arg1 + 187] : memref<1x104x194xf64, 1>
        %17 = affine.load %arg0[0, 92, -%arg1 + 187] : memref<1x104x194xf64, 1>
        %18 = affine.load %arg0[0, 93, -%arg1 + 187] : memref<1x104x194xf64, 1>
        %19 = affine.load %arg0[0, 94, -%arg1 + 187] : memref<1x104x194xf64, 1>
        %20 = affine.load %arg0[0, 95, -%arg1 + 187] : memref<1x104x194xf64, 1>
        affine.yield %c-1_i64, %c182_i64, %13, %14, %15, %16, %17, %18, %19, %20 : i64, i64, f64, f64, f64, f64, f64, f64, f64, f64
      } else {
        %13 = affine.load %arg0[0, 96, 7] : memref<1x104x194xf64, 1>
        %14 = affine.load %arg0[0, 89, 7] : memref<1x104x194xf64, 1>
        %15 = affine.load %arg0[0, 90, 7] : memref<1x104x194xf64, 1>
        %16 = affine.load %arg0[0, 91, 7] : memref<1x104x194xf64, 1>
        %17 = affine.load %arg0[0, 92, 7] : memref<1x104x194xf64, 1>
        %18 = affine.load %arg0[0, 93, 7] : memref<1x104x194xf64, 1>
        %19 = affine.load %arg0[0, 94, 7] : memref<1x104x194xf64, 1>
        %20 = affine.load %arg0[0, 95, 7] : memref<1x104x194xf64, 1>
        affine.yield %c1_i64, %c2_i64, %13, %14, %15, %16, %17, %18, %19, %20 : i64, i64, f64, f64, f64, f64, f64, f64, f64, f64
      }
      %2 = arith.sitofp %1#0 : i64 to f64
      %3 = arith.mulf %2, %1#9 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %3, %arg0[0, 97, %arg1 + 7] : memref<1x104x194xf64, 1>
      %4 = arith.mulf %2, %1#8 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %4, %arg0[0, 98, %arg1 + 7] : memref<1x104x194xf64, 1>
      %5 = arith.mulf %2, %1#7 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %5, %arg0[0, 99, %arg1 + 7] : memref<1x104x194xf64, 1>
      %6 = arith.mulf %2, %1#6 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %6, %arg0[0, 100, %arg1 + 7] : memref<1x104x194xf64, 1>
      %7 = arith.mulf %2, %1#5 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %7, %arg0[0, 101, %arg1 + 7] : memref<1x104x194xf64, 1>
      %8 = arith.mulf %2, %1#4 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %8, %arg0[0, 102, %arg1 + 7] : memref<1x104x194xf64, 1>
      %9 = arith.mulf %2, %1#3 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %9, %arg0[0, 103, %arg1 + 7] : memref<1x104x194xf64, 1>
      %10 = arith.mulf %2, %1#2 {fastmathFlags = #llvm.fastmath<none>} : f64
      %11 = affine.load %arg0[0, 96, %arg1 + 7] : memref<1x104x194xf64, 1>
      %12 = affine.if #set1(%arg1) -> f64 {
        affine.yield %11 : f64
      } else {
        affine.yield %10 : f64
      }
      affine.store %12, %arg0[0, 96, %arg1 + 7] : memref<1x104x194xf64, 1>
    }
    return
  }
}

// CHECK:    func.func private @par6_raised(%[[a1:.+]]: tensor<1x104x194xf64>) -> tensor<1x104x194xf64> {
// CHECK-NEXT:    %[[a2:.+]] = stablehlo.constant dense<96> : tensor<i64>
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.constant dense<89> : tensor<i64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.constant dense<103> : tensor<i64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.constant dense<102> : tensor<i64>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.constant dense<101> : tensor<i64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.constant dense<100> : tensor<i64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.constant dense<99> : tensor<i64>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.constant dense<98> : tensor<i64>
// CHECK-NEXT:    %[[a10:.+]] = stablehlo.constant dense<97> : tensor<i64>
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:    %[[a13:.+]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.constant dense<1> : tensor<180xi64>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.constant dense<0> : tensor<180xi64>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.constant dense<-1> : tensor<i64>
// CHECK-NEXT:    %[[a18:.+]] = stablehlo.iota dim = 0 : tensor<180xi64>
// CHECK-NEXT:    %[[a19:.+]] = stablehlo.add %[[a18]], %[[a15]] : tensor<180xi64>
// CHECK-NEXT:    %[[a20:.+]] = stablehlo.multiply %[[a19]], %[[a14]] : tensor<180xi64>
// CHECK-NEXT:    %[[a21:.+]] = stablehlo.slice %[[a1]] [0:1, 7:8, 7:187] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a22:.+]] = stablehlo.reshape %[[a21]] : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a23:.+]] = stablehlo.broadcast_in_dim %[[a22]], dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a24:.+]] = stablehlo.dynamic_update_slice %[[a1]], %[[a23]], %[[a13]], %[[a12]], %[[a11]] : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK-NEXT:    %[[a25:.+]] = stablehlo.broadcast_in_dim %[[a17]], dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK-NEXT:    %[[a26:.+]] = stablehlo.add %[[a20]], %[[a25]] : tensor<180xi64>
// CHECK-NEXT:    %[[a27:.+]] = stablehlo.compare GE, %[[a26]], %[[a15]] : (tensor<180xi64>, tensor<180xi64>) -> tensor<180xi1>
// CHECK-NEXT:    %[[a28:.+]] = stablehlo.slice %[[a24]] [0:1, 96:97, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a29:.+]] = stablehlo.reverse %[[a28]], dims = [2] : tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a30:.+]] = stablehlo.reshape %[[a29]] : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a31:.+]] = stablehlo.slice %[[a24]] [0:1, 89:90, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a32:.+]] = stablehlo.reverse %[[a31]], dims = [2] : tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a33:.+]] = stablehlo.reshape %[[a32]] : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a34:.+]] = stablehlo.slice %[[a24]] [0:1, 90:91, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a35:.+]] = stablehlo.reverse %[[a34]], dims = [2] : tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a36:.+]] = stablehlo.reshape %[[a35]] : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a37:.+]] = stablehlo.slice %[[a24]] [0:1, 91:92, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a38:.+]] = stablehlo.reverse %[[a37]], dims = [2] : tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a39:.+]] = stablehlo.reshape %[[a38]] : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a40:.+]] = stablehlo.slice %[[a24]] [0:1, 92:93, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a41:.+]] = stablehlo.reverse %[[a40]], dims = [2] : tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a42:.+]] = stablehlo.reshape %[[a41]] : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a43:.+]] = stablehlo.slice %[[a24]] [0:1, 93:94, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a44:.+]] = stablehlo.reverse %[[a43]], dims = [2] : tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a45:.+]] = stablehlo.reshape %[[a44]] : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a46:.+]] = stablehlo.slice %[[a24]] [0:1, 94:95, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a47:.+]] = stablehlo.reverse %[[a46]], dims = [2] : tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a48:.+]] = stablehlo.reshape %[[a47]] : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a49:.+]] = stablehlo.slice %[[a24]] [0:1, 95:96, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a50:.+]] = stablehlo.reverse %[[a49]], dims = [2] : tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a51:.+]] = stablehlo.reshape %[[a50]] : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a52:.+]] = stablehlo.slice %[[a24]] [0:1, 96:97, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %[[a53:.+]] = stablehlo.reshape %[[a52]] : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[a54:.+]] = stablehlo.slice %[[a24]] [0:1, 89:90, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %[[a55:.+]] = stablehlo.reshape %[[a54]] : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[a56:.+]] = stablehlo.slice %[[a24]] [0:1, 90:91, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %[[a57:.+]] = stablehlo.reshape %[[a56]] : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[a58:.+]] = stablehlo.slice %[[a24]] [0:1, 91:92, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %[[a59:.+]] = stablehlo.reshape %[[a58]] : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[a60:.+]] = stablehlo.slice %[[a24]] [0:1, 92:93, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %[[a61:.+]] = stablehlo.reshape %[[a60]] : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[a62:.+]] = stablehlo.slice %[[a24]] [0:1, 93:94, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %[[a63:.+]] = stablehlo.reshape %[[a62]] : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[a64:.+]] = stablehlo.slice %[[a24]] [0:1, 94:95, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %[[a65:.+]] = stablehlo.reshape %[[a64]] : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[a66:.+]] = stablehlo.slice %[[a24]] [0:1, 95:96, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK-NEXT:    %[[a67:.+]] = stablehlo.reshape %[[a66]] : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK-NEXT:    %[[a68:.+]] = stablehlo.broadcast_in_dim %[[a17]], dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK-NEXT:    %[[a69:.+]] = stablehlo.broadcast_in_dim %[[a16]], dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK-NEXT:    %[[a70:.+]] = stablehlo.select %[[a27]], %[[a68]], %[[a69]] : tensor<180xi1>, tensor<180xi64>
// CHECK-NEXT:    %[[a71:.+]] = stablehlo.broadcast_in_dim %[[a53]], dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a72:.+]] = stablehlo.select %[[a27]], %[[a30]], %[[a71]] : tensor<180xi1>, tensor<180xf64>
// CHECK-NEXT:    %[[a73:.+]] = stablehlo.broadcast_in_dim %[[a55]], dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a74:.+]] = stablehlo.select %[[a27]], %[[a33]], %[[a73]] : tensor<180xi1>, tensor<180xf64>
// CHECK-NEXT:    %[[a75:.+]] = stablehlo.broadcast_in_dim %[[a57]], dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a76:.+]] = stablehlo.select %[[a27]], %[[a36]], %[[a75]] : tensor<180xi1>, tensor<180xf64>
// CHECK-NEXT:    %[[a77:.+]] = stablehlo.broadcast_in_dim %[[a59]], dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a78:.+]] = stablehlo.select %[[a27]], %[[a39]], %[[a77]] : tensor<180xi1>, tensor<180xf64>
// CHECK-NEXT:    %[[a79:.+]] = stablehlo.broadcast_in_dim %[[a61]], dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a80:.+]] = stablehlo.select %[[a27]], %[[a42]], %[[a79]] : tensor<180xi1>, tensor<180xf64>
// CHECK-NEXT:    %[[a81:.+]] = stablehlo.broadcast_in_dim %[[a63]], dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a82:.+]] = stablehlo.select %[[a27]], %[[a45]], %[[a81]] : tensor<180xi1>, tensor<180xf64>
// CHECK-NEXT:    %[[a83:.+]] = stablehlo.broadcast_in_dim %[[a65]], dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a84:.+]] = stablehlo.select %[[a27]], %[[a48]], %[[a83]] : tensor<180xi1>, tensor<180xf64>
// CHECK-NEXT:    %[[a85:.+]] = stablehlo.broadcast_in_dim %[[a67]], dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a86:.+]] = stablehlo.select %[[a27]], %[[a51]], %[[a85]] : tensor<180xi1>, tensor<180xf64>
// CHECK-NEXT:    %[[a87:.+]] = arith.sitofp %[[a70]] : tensor<180xi64> to tensor<180xf64>
// CHECK-NEXT:    %[[a88:.+]] = arith.mulf %[[a87]], %[[a86]] {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK-NEXT:    %[[a89:.+]] = stablehlo.broadcast_in_dim %[[a88]], dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a90:.+]] = stablehlo.dynamic_update_slice %[[a24]], %[[a89]], %[[a13]], %[[a10]], %[[a11]] : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK-NEXT:    %[[a91:.+]] = arith.mulf %[[a87]], %[[a84]] {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK-NEXT:    %[[a92:.+]] = stablehlo.broadcast_in_dim %[[a91]], dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a93:.+]] = stablehlo.dynamic_update_slice %[[a90]], %[[a92]], %[[a13]], %[[a9]], %[[a11]] : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK-NEXT:    %[[a94:.+]] = arith.mulf %[[a87]], %[[a82]] {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK-NEXT:    %[[a95:.+]] = stablehlo.broadcast_in_dim %[[a94]], dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a96:.+]] = stablehlo.dynamic_update_slice %[[a93]], %[[a95]], %[[a13]], %[[a8]], %[[a11]] : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK-NEXT:    %[[a97:.+]] = arith.mulf %[[a87]], %[[a80]] {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK-NEXT:    %[[a98:.+]] = stablehlo.broadcast_in_dim %[[a97]], dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a99:.+]] = stablehlo.dynamic_update_slice %[[a96]], %[[a98]], %[[a13]], %[[a7]], %[[a11]] : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK-NEXT:    %[[a100:.+]] = arith.mulf %[[a87]], %[[a78]] {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK-NEXT:    %[[a101:.+]] = stablehlo.broadcast_in_dim %[[a100]], dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a102:.+]] = stablehlo.dynamic_update_slice %[[a99]], %[[a101]], %[[a13]], %[[a6]], %[[a11]] : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK-NEXT:    %[[a103:.+]] = arith.mulf %[[a87]], %[[a76]] {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK-NEXT:    %[[a104:.+]] = stablehlo.broadcast_in_dim %[[a103]], dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a105:.+]] = stablehlo.dynamic_update_slice %[[a102]], %[[a104]], %[[a13]], %[[a5]], %[[a11]] : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK-NEXT:    %[[a106:.+]] = arith.mulf %[[a87]], %[[a74]] {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK-NEXT:    %[[a107:.+]] = stablehlo.broadcast_in_dim %[[a106]], dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a108:.+]] = stablehlo.dynamic_update_slice %[[a105]], %[[a107]], %[[a13]], %[[a4]], %[[a11]] : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK-NEXT:    %[[a109:.+]] = arith.mulf %[[a87]], %[[a72]] {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK-NEXT:    %[[a110:.+]] = stablehlo.slice %[[a108]] [0:1, 96:97, 7:187] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a111:.+]] = stablehlo.reshape %[[a110]] : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK-NEXT:    %[[a112:.+]] = stablehlo.broadcast_in_dim %[[a17]], dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK-NEXT:    %[[a113:.+]] = stablehlo.multiply %[[a20]], %[[a112]] : tensor<180xi64>
// CHECK-NEXT:    %[[a114:.+]] = stablehlo.broadcast_in_dim %[[a3]], dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK-NEXT:    %[[a115:.+]] = stablehlo.add %[[a113]], %[[a114]] : tensor<180xi64>
// CHECK-NEXT:    %[[a116:.+]] = stablehlo.compare GE, %[[a115]], %[[a15]] : (tensor<180xi64>, tensor<180xi64>) -> tensor<180xi1>
// CHECK-NEXT:    %[[a117:.+]] = stablehlo.select %[[a116]], %[[a111]], %[[a109]] : tensor<180xi1>, tensor<180xf64>
// CHECK-NEXT:    %[[a118:.+]] = stablehlo.broadcast_in_dim %[[a117]], dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK-NEXT:    %[[a119:.+]] = stablehlo.dynamic_update_slice %[[a108]], %[[a118]], %[[a13]], %[[a2]], %[[a11]] : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK-NEXT:    return %[[a119]] : tensor<1x104x194xf64>
// CHECK-NEXT:  }
