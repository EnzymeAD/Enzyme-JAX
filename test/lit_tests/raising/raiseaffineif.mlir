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

// CHECK:  func.func private @"##call__Z40gpu_compute_hydrostatic_free_surface_Gc_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SE_SF_SG_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESK_SM_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SR_SR_EvE16GridFittedBottomI5FieldI6CenterSW_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvEv5TupleI3ValILi3EES14_I2_eEv24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthIS9_ES9_v13CATKEEquationIS9_EE24DefaultBoundaryConditionI17BoundaryConditionI4FluxvEE13BuoyancyForceI16SeawaterBuoyancyIS9_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialIS9_ES9_EvvE18NegativeZDirectionEv10NamedTupleI12__u___v___w_S13_ISC_SC_S8_IS9_Li3ESA_IS9_Li3ELi1E13_194__99__35_EEEE24SplitExplicitFreeSurfaceIS8_IS9_Li3ESA_IS9_Li3ELi1E13_194__187__1_EES1S_I8__U___V_S13_ISV_I4FaceSW_vvvvS1Z_S9_vvvESV_ISW_S20_vvvvS1Z_S9_vvvEEES1S_I12______U___V_S13_IS1Z_S21_S22_EES9_v18FixedSubstepNumberIS9_S13_IS9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_EE21ForwardBackwardSchemeES1S_I12__T___S___e_S13_ISC_SC_SC_EES1S_I141___u____c____e___Le___J____previous_compute_time___previous_velocities____tupled_tracer_diffusivities____tupled_implicit_linear_coefficients_S13_IS1U_S1U_S1U_SC_SZ_16ReactantRefValueIS9_ES1S_I8__u___v_S13_ISC_SC_EES1S_I12__T___S___e_S13_IS1U_S1U_S1U_EES1S_I12__T___S___e_S13_I9ZeroFieldISR_Li3EES2L_SC_EEEES1S_I2__S13_ES1S_I53__time___last__t___last_stage__t___iteration___stage_S13_IS9_S9_S9_SR_SR_EE11zeroforcingE#860$par244_raised"(%arg0: tensor<34x99x194xf64>, %arg1: tensor<34xf64>, %arg2: tensor<35xf64>, %arg3: tensor<34xf64>, %arg4: tensor<99x194xf64>, %arg5: tensor<99x194xf64>, %arg6: tensor<99x194xf64>, %arg7: tensor<1x99x194xf64>, %arg8: tensor<34x99x194xf64>, %arg9: tensor<35x99x194xf64>) -> (tensor<34x99x194xf64>, tensor<34xf64>, tensor<35xf64>, tensor<34xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<1x99x194xf64>, tensor<34x99x194xf64>, tensor<35x99x194xf64>) {
// CHECK-NEXT:    %[[v0:.+]] = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %[[v1:.+]] = stablehlo.constant dense<-19> : tensor<i64>
// CHECK-NEXT:    %[[v2:.+]] = stablehlo.constant dense<1> : tensor<20xi64>
// CHECK-NEXT:    %[[v3:.+]] = stablehlo.constant dense<0> : tensor<20xi64>
// CHECK-NEXT:    %[[v4:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:    %[[v5:.+]] = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %[[v6:.+]] = stablehlo.constant dense<20> : tensor<i64>
// CHECK-NEXT:    %[[v7:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[v8:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[v9:.+]] = stablehlo.iota dim = 0 : tensor<20xi64>
// CHECK-NEXT:    %[[v10:.+]] = stablehlo.add %[[v9]], %[[v3]] : tensor<20xi64>
// CHECK-NEXT:    %[[v11:.+]] = stablehlo.multiply %[[v10]], %[[v2]] : tensor<20xi64>
// CHECK-NEXT:    %[[v12:.+]] = stablehlo.slice %arg6 [7:92, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v13:.+]] = stablehlo.reshape %[[v12]] : (tensor<85x180xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v14:.+]] = stablehlo.slice %arg3 [7:27] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v15:.+]] = stablehlo.reshape %[[v14]] : (tensor<20xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v16:.+]] = stablehlo.broadcast_in_dim %[[v13]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v17:.+]] = stablehlo.broadcast_in_dim %[[v15]], dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v18:.+]] = arith.mulf %[[v16]], %[[v17]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v19:.+]] = stablehlo.broadcast_in_dim %[[v7]], dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v20:.+]] = arith.divf %[[v19]], %[[v18]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v21:.+]] = stablehlo.slice %arg5 [7:92, 8:188] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v22:.+]] = stablehlo.reshape %[[v21]] : (tensor<85x180xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v23:.+]] = stablehlo.broadcast_in_dim %[[v22]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v24:.+]] = stablehlo.broadcast_in_dim %[[v15]], dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v25:.+]] = arith.mulf %[[v23]], %[[v24]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v26:.+]] = stablehlo.broadcast_in_dim %[[v8]], dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v27:.+]] = arith.mulf %[[v25]], %[[v26]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v28:.+]] = stablehlo.slice %arg5 [7:92, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v29:.+]] = stablehlo.reshape %[[v28]] : (tensor<85x180xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v30:.+]] = stablehlo.broadcast_in_dim %[[v29]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v31:.+]] = stablehlo.broadcast_in_dim %[[v15]], dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v32:.+]] = arith.mulf %[[v30]], %[[v31]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v33:.+]] = stablehlo.broadcast_in_dim %[[v8]], dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v34:.+]] = arith.mulf %[[v32]], %[[v33]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v35:.+]] = arith.subf %[[v27]], %[[v34]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v36:.+]] = stablehlo.slice %arg4 [8:93, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v37:.+]] = stablehlo.reshape %[[v36]] : (tensor<85x180xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v38:.+]] = stablehlo.broadcast_in_dim %[[v37]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v39:.+]] = stablehlo.broadcast_in_dim %[[v15]], dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v40:.+]] = arith.mulf %[[v38]], %[[v39]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v41:.+]] = stablehlo.broadcast_in_dim %[[v8]], dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v42:.+]] = arith.mulf %[[v40]], %[[v41]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v43:.+]] = stablehlo.slice %arg4 [7:92, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v44:.+]] = stablehlo.reshape %[[v43]] : (tensor<85x180xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v45:.+]] = stablehlo.broadcast_in_dim %[[v44]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v46:.+]] = stablehlo.broadcast_in_dim %[[v15]], dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v47:.+]] = arith.mulf %[[v45]], %[[v46]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v48:.+]] = stablehlo.broadcast_in_dim %[[v8]], dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v49:.+]] = arith.mulf %[[v47]], %[[v48]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v50:.+]] = arith.subf %[[v42]], %[[v49]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v51:.+]] = stablehlo.broadcast_in_dim %[[v4]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[v52:.+]] = arith.addi %[[v11]], %[[v51]] : tensor<20xi64>
// CHECK-NEXT:    %[[v53:.+]] = stablehlo.slice %arg9 [8:28, 7:92, 7:187] : (tensor<35x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v54:.+]] = stablehlo.reshape %[[v53]] : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v55:.+]] = arith.negf %[[v54]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v56:.+]] = stablehlo.slice %arg1 [8:28] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v57:.+]] = stablehlo.reshape %[[v56]] : (tensor<20xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v58:.+]] = stablehlo.slice %arg7 [0:1, 7:92, 7:187] : (tensor<1x99x194xf64>) -> tensor<1x85x180xf64>
// CHECK-NEXT:    %[[v59:.+]] = stablehlo.reshape %[[v58]] : (tensor<1x85x180xf64>) -> tensor<85x180xf64>
// CHECK-NEXT:    %[[v60:.+]] = stablehlo.broadcast_in_dim %[[v57]], dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v61:.+]] = stablehlo.broadcast_in_dim %[[v59]], dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v62:.+]] = arith.cmpf ole, %[[v60]], %[[v61]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v63:.+]] = stablehlo.broadcast_in_dim %[[v6]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[v64:.+]] = arith.cmpi sgt, %[[v52]], %[[v63]] : tensor<20xi64>
// CHECK-NEXT:    %[[v65:.+]] = stablehlo.broadcast_in_dim %[[v64]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v66:.+]] = arith.ori %[[v65]], %[[v62]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v67:.+]] = stablehlo.broadcast_in_dim %[[v6]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[v68:.+]] = arith.cmpi sle, %[[v52]], %[[v67]] : tensor<20xi64>
// CHECK-NEXT:    %[[v69:.+]] = stablehlo.broadcast_in_dim %[[v68]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v70:.+]] = arith.andi %[[v69]], %[[v66]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v71:.+]] = stablehlo.slice %arg1 [7:27] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v72:.+]] = stablehlo.reshape %[[v71]] : (tensor<20xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v73:.+]] = stablehlo.broadcast_in_dim %[[v72]], dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v74:.+]] = stablehlo.broadcast_in_dim %[[v59]], dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v75:.+]] = arith.cmpf ole, %[[v73]], %[[v74]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v76:.+]] = arith.ori %[[v70]], %[[v75]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v77:.+]] = stablehlo.slice %arg8 [8:28, 7:92, 7:187] : (tensor<34x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v78:.+]] = stablehlo.reshape %[[v77]] : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v79:.+]] = stablehlo.slice %arg8 [7:27, 7:92, 7:187] : (tensor<34x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v80:.+]] = stablehlo.reshape %[[v79]] : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v81:.+]] = arith.subf %[[v78]], %[[v80]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v82:.+]] = stablehlo.broadcast_in_dim %[[v8]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v83:.+]] = arith.select %[[v76]], %[[v82]], %[[v81]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v84:.+]] = stablehlo.slice %arg2 [9:29] : (tensor<35xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v85:.+]] = stablehlo.reshape %[[v84]] : (tensor<20xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v86:.+]] = stablehlo.broadcast_in_dim %[[v85]], dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v87:.+]] = arith.divf %[[v83]], %[[v86]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v88:.+]] = arith.mulf %[[v55]], %[[v87]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v89:.+]] = stablehlo.broadcast_in_dim %[[v1]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[v90:.+]] = stablehlo.add %[[v11]], %[[v89]] : tensor<20xi64>
// CHECK-NEXT:    %[[v91:.+]] = stablehlo.compare  EQ, %[[v90]], %[[v3]] : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
// CHECK-NEXT:    %[[v92:.+]] = stablehlo.broadcast_in_dim %[[v91]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v93:.+]] = stablehlo.broadcast_in_dim %[[v8]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v94:.+]] = stablehlo.select %[[v92]], %[[v88]], %[[v93]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v95:.+]] = arith.ori %[[v66]], %[[v75]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v96:.+]] = stablehlo.broadcast_in_dim %[[v68]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v97:.+]] = arith.andi %[[v96]], %[[v95]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v98:.+]] = stablehlo.broadcast_in_dim %[[v8]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v99:.+]] = arith.select %[[v97]], %[[v98]], %[[v94]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v100:.+]] = stablehlo.broadcast_in_dim %[[v13]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v101:.+]] = stablehlo.broadcast_in_dim %[[v99]], dims = [2, 0, 1] : (tensor<20x85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v102:.+]] = arith.mulf %[[v100]], %[[v101]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v103:.+]] = stablehlo.slice %arg9 [7:27, 7:92, 7:187] : (tensor<35x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v104:.+]] = stablehlo.reshape %[[v103]] : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v105:.+]] = arith.negf %[[v104]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v106:.+]] = stablehlo.slice %arg1 [6:26] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v107:.+]] = stablehlo.reshape %[[v106]] : (tensor<20xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v108:.+]] = stablehlo.broadcast_in_dim %[[v107]], dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v109:.+]] = stablehlo.broadcast_in_dim %[[v59]], dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v110:.+]] = arith.cmpf ole, %[[v108]], %[[v109]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v111:.+]] = stablehlo.broadcast_in_dim %[[v5]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[v112:.+]] = arith.cmpi ult, %[[v11]], %[[v111]] : tensor<20xi64>
// CHECK-NEXT:    %[[v113:.+]] = stablehlo.broadcast_in_dim %[[v112]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v114:.+]] = arith.ori %[[v113]], %[[v110]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v115:.+]] = stablehlo.broadcast_in_dim %[[v5]], dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK-NEXT:    %[[v116:.+]] = arith.cmpi uge, %[[v11]], %[[v115]] : tensor<20xi64>
// CHECK-NEXT:    %[[v117:.+]] = stablehlo.broadcast_in_dim %[[v116]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v118:.+]] = arith.andi %[[v117]], %[[v114]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v119:.+]] = arith.ori %[[v75]], %[[v118]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v120:.+]] = stablehlo.slice %arg8 [6:26, 7:92, 7:187] : (tensor<34x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v121:.+]] = stablehlo.reshape %[[v120]] : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v122:.+]] = arith.subf %[[v80]], %[[v121]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v123:.+]] = stablehlo.broadcast_in_dim %[[v8]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v124:.+]] = arith.select %[[v119]], %[[v123]], %[[v122]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v125:.+]] = stablehlo.slice %arg2 [8:28] : (tensor<35xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v126:.+]] = stablehlo.reshape %[[v125]] : (tensor<20xf64>) -> tensor<20xf64>
// CHECK-NEXT:    %[[v127:.+]] = stablehlo.broadcast_in_dim %[[v126]], dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v128:.+]] = arith.divf %[[v124]], %[[v127]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v129:.+]] = arith.mulf %[[v105]], %[[v128]] {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v130:.+]] = stablehlo.compare  EQ, %[[v11]], %[[v3]] : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
// CHECK-NEXT:    %[[v131:.+]] = stablehlo.broadcast_in_dim %[[v130]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v132:.+]] = stablehlo.broadcast_in_dim %[[v8]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v133:.+]] = stablehlo.select %[[v131]], %[[v129]], %[[v132]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v134:.+]] = arith.ori %[[v75]], %[[v114]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v135:.+]] = stablehlo.broadcast_in_dim %[[v116]], dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v136:.+]] = arith.andi %[[v135]], %[[v134]] : tensor<20x85x180xi1>
// CHECK-NEXT:    %[[v137:.+]] = stablehlo.broadcast_in_dim %[[v8]], dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v138:.+]] = arith.select %[[v136]], %[[v137]], %[[v133]] : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v139:.+]] = stablehlo.broadcast_in_dim %[[v13]], dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v140:.+]] = stablehlo.broadcast_in_dim %[[v138]], dims = [2, 0, 1] : (tensor<20x85x180xf64>) -> tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v141:.+]] = arith.mulf %[[v139]], %[[v140]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v142:.+]] = arith.subf %[[v102]], %[[v141]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v143:.+]] = arith.addf %[[v35]], %[[v50]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v144:.+]] = arith.addf %[[v143]], %[[v142]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v145:.+]] = arith.mulf %[[v20]], %[[v144]] {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v146:.+]] = arith.negf %[[v145]] : tensor<85x180x20xf64>
// CHECK-NEXT:    %[[v147:.+]] = stablehlo.broadcast_in_dim %[[v146]], dims = [1, 2, 0] : (tensor<85x180x20xf64>) -> tensor<20x85x180xf64>
// CHECK-NEXT:    %[[v148:.+]] = stablehlo.dynamic_update_slice %arg0, %[[v147]], %[[v0]], %[[v0]], %[[v0]] : (tensor<34x99x194xf64>, tensor<20x85x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<34x99x194xf64>
// CHECK-NEXT:    return %[[v148]], %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : tensor<34x99x194xf64>, tensor<34xf64>, tensor<35xf64>, tensor<34xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<1x99x194xf64>, tensor<34x99x194xf64>, tensor<35x99x194xf64>
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
      // CHECK: stablehlo.slice %[[arg0:.+]] [0:1, 7:8, 7:187]
      // CHECK: %[[argu:.+]] = stablehlo.dynamic_update_slice %[[arg0]]
      %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x104x194xf64, 1>
      affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x104x194xf64, 1>
      %1:10 = affine.if #set(%arg1) -> (i64, i64, f64, f64, f64, f64, f64, f64, f64, f64) {
        // CHECK: stablehlo.slice %[[argu]] [0:1, 96:97, 8:188]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 89:90, 8:188]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 90:91, 8:188]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 91:92, 8:188]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 92:93, 8:188]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 93:94, 8:188]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 94:95, 8:188]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 95:96, 8:188]
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
        // CHECK: stablehlo.slice %[[argu]] [0:1, 96:97, 7:8]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 89:90, 7:8]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 90:91, 7:8]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 91:92, 7:8]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 92:93, 7:8]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 93:94, 7:8]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 94:95, 7:8]
        // CHECK: stablehlo.slice %[[argu]] [0:1, 95:96, 7:8]
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
