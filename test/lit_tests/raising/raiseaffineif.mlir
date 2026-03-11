// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --split-input-file | FileCheck %s

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





// CHECK:  func.func private @"##call__Z40gpu_compute_hydrostatic_free_surface_Gc_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SE_SF_SG_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESK_SM_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SR_SR_EvE16GridFittedBottomI5FieldI6CenterSW_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvEv5TupleI3ValILi3EES14_I2_eEv24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthIS9_ES9_v13CATKEEquationIS9_EE24DefaultBoundaryConditionI17BoundaryConditionI4FluxvEE13BuoyancyForceI16SeawaterBuoyancyIS9_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialIS9_ES9_EvvE18NegativeZDirectionEv10NamedTupleI12__u___v___w_S13_ISC_SC_S8_IS9_Li3ESA_IS9_Li3ELi1E13_194__99__35_EEEE24SplitExplicitFreeSurfaceIS8_IS9_Li3ESA_IS9_Li3ELi1E13_194__187__1_EES1S_I8__U___V_S13_ISV_I4FaceSW_vvvvS1Z_S9_vvvESV_ISW_S20_vvvvS1Z_S9_vvvEEES1S_I12______U___V_S13_IS1Z_S21_S22_EES9_v18FixedSubstepNumberIS9_S13_IS9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_EE21ForwardBackwardSchemeES1S_I12__T___S___e_S13_ISC_SC_SC_EES1S_I141___u____c____e___Le___J____previous_compute_time___previous_velocities____tupled_tracer_diffusivities____tupled_implicit_linear_coefficients_S13_IS1U_S1U_S1U_SC_SZ_16ReactantRefValueIS9_ES1S_I8__u___v_S13_ISC_SC_EES1S_I12__T___S___e_S13_IS1U_S1U_S1U_EES1S_I12__T___S___e_S13_I9ZeroFieldISR_Li3EES2L_SC_EEEES1S_I2__S13_ES1S_I53__time___last__t___last_stage__t___iteration___stage_S13_IS9_S9_S9_SR_SR_EE11zeroforcingE#860$par244_raised"(%arg0: tensor<34x99x194xf64>, %arg1: tensor<34xf64>, %arg2: tensor<35xf64>, %arg3: tensor<34xf64>, %arg4: tensor<99x194xf64>, %arg5: tensor<99x194xf64>, %arg6: tensor<99x194xf64>, %arg7: tensor<1x99x194xf64>, %arg8: tensor<34x99x194xf64>, %arg9: tensor<35x99x194xf64>) -> (tensor<34x99x194xf64>, tensor<34xf64>, tensor<35xf64>, tensor<34xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<1x99x194xf64>, tensor<34x99x194xf64>, tensor<35x99x194xf64>) {
// CHECK:    %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK:    %c_1 = stablehlo.constant dense<20> : tensor<i64>
// CHECK:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CHECK:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:    %0 = stablehlo.iota dim = 0 : tensor<20xi64>
// CHECK:    %c_3 = stablehlo.constant dense<0> : tensor<20xi64>
// CHECK:    %1 = stablehlo.add %0, %c_3 : tensor<20xi64>
// CHECK:    %c_4 = stablehlo.constant dense<1> : tensor<20xi64>
// CHECK:    %2 = stablehlo.multiply %1, %c_4 : tensor<20xi64>
// CHECK:    %c_5 = stablehlo.constant dense<20> : tensor<1xi64>
// CHECK:    %3 = stablehlo.iota dim = 0 : tensor<85xi64>
// CHECK:    %c_6 = stablehlo.constant dense<0> : tensor<85xi64>
// CHECK:    %4 = stablehlo.add %3, %c_6 : tensor<85xi64>
// CHECK:    %c_7 = stablehlo.constant dense<1> : tensor<85xi64>
// CHECK:    %5 = stablehlo.multiply %4, %c_7 : tensor<85xi64>
// CHECK:    %c_8 = stablehlo.constant dense<85> : tensor<1xi64>
// CHECK:    %6 = stablehlo.iota dim = 0 : tensor<180xi64>
// CHECK:    %c_9 = stablehlo.constant dense<0> : tensor<180xi64>
// CHECK:    %7 = stablehlo.add %6, %c_9 : tensor<180xi64>
// CHECK:    %c_10 = stablehlo.constant dense<1> : tensor<180xi64>
// CHECK:    %8 = stablehlo.multiply %7, %c_10 : tensor<180xi64>
// CHECK:    %c_11 = stablehlo.constant dense<180> : tensor<1xi64>
// CHECK:    %9 = stablehlo.slice %arg6 [7:92, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK:    %10 = stablehlo.reshape %9 : (tensor<85x180xf64>) -> tensor<85x180xf64>
// CHECK:    %11 = stablehlo.slice %arg3 [7:27] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK:    %12 = stablehlo.reshape %11 : (tensor<20xf64>) -> tensor<20xf64>
// CHECK:    %13 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK:    %14 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK:    %15 = arith.mulf %13, %14 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %16 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK:    %17 = arith.divf %16, %15 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %18 = stablehlo.slice %arg5 [7:92, 8:188] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK:    %19 = stablehlo.reshape %18 : (tensor<85x180xf64>) -> tensor<85x180xf64>
// CHECK:    %20 = stablehlo.broadcast_in_dim %19, dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK:    %21 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK:    %22 = arith.mulf %20, %21 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %23 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK:    %24 = arith.mulf %22, %23 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %25 = stablehlo.slice %arg5 [7:92, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK:    %26 = stablehlo.reshape %25 : (tensor<85x180xf64>) -> tensor<85x180xf64>
// CHECK:    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK:    %28 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK:    %29 = arith.mulf %27, %28 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %30 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK:    %31 = arith.mulf %29, %30 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %32 = arith.subf %24, %31 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %33 = stablehlo.slice %arg4 [8:93, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK:    %34 = stablehlo.reshape %33 : (tensor<85x180xf64>) -> tensor<85x180xf64>
// CHECK:    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK:    %36 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK:    %37 = arith.mulf %35, %36 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %38 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK:    %39 = arith.mulf %37, %38 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %40 = stablehlo.slice %arg4 [7:92, 7:187] : (tensor<99x194xf64>) -> tensor<85x180xf64>
// CHECK:    %41 = stablehlo.reshape %40 : (tensor<85x180xf64>) -> tensor<85x180xf64>
// CHECK:    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK:    %43 = stablehlo.broadcast_in_dim %12, dims = [2] : (tensor<20xf64>) -> tensor<85x180x20xf64>
// CHECK:    %44 = arith.mulf %42, %43 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %45 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<85x180x20xf64>
// CHECK:    %46 = arith.mulf %44, %45 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %47 = arith.subf %39, %46 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %48 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK:    %49 = arith.addi %2, %48 : tensor<20xi64>
// CHECK:    %50 = stablehlo.slice %arg9 [8:28, 7:92, 7:187] : (tensor<35x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK:    %51 = stablehlo.reshape %50 : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK:    %52 = arith.negf %51 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %53 = stablehlo.slice %arg1 [8:28] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK:    %54 = stablehlo.reshape %53 : (tensor<20xf64>) -> tensor<20xf64>
// CHECK:    %55 = stablehlo.slice %arg7 [0:1, 7:92, 7:187] : (tensor<1x99x194xf64>) -> tensor<1x85x180xf64>
// CHECK:    %56 = stablehlo.reshape %55 : (tensor<1x85x180xf64>) -> tensor<85x180xf64>
// CHECK:    %57 = stablehlo.broadcast_in_dim %54, dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK:    %58 = stablehlo.broadcast_in_dim %56, dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK:    %59 = arith.cmpf ole, %57, %58 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %60 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK:    %61 = arith.cmpi sgt, %49, %60 : tensor<20xi64>
// CHECK:    %62 = stablehlo.broadcast_in_dim %61, dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK:    %63 = arith.ori %62, %59 : tensor<20x85x180xi1>
// CHECK:    %64 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK:    %65 = arith.cmpi sle, %49, %64 : tensor<20xi64>
// CHECK:    %66 = stablehlo.broadcast_in_dim %65, dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK:    %67 = arith.andi %66, %63 : tensor<20x85x180xi1>
// CHECK:    %68 = stablehlo.slice %arg1 [7:27] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK:    %69 = stablehlo.reshape %68 : (tensor<20xf64>) -> tensor<20xf64>
// CHECK:    %70 = stablehlo.broadcast_in_dim %69, dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK:    %71 = stablehlo.broadcast_in_dim %56, dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK:    %72 = arith.cmpf ole, %70, %71 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %73 = arith.ori %67, %72 : tensor<20x85x180xi1>
// CHECK:    %74 = stablehlo.slice %arg8 [8:28, 7:92, 7:187] : (tensor<34x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK:    %75 = stablehlo.reshape %74 : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK:    %76 = stablehlo.slice %arg8 [7:27, 7:92, 7:187] : (tensor<34x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK:    %77 = stablehlo.reshape %76 : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK:    %78 = arith.subf %75, %77 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %79 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK:    %80 = arith.select %73, %79, %78 : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK:    %81 = stablehlo.slice %arg2 [9:29] : (tensor<35xf64>) -> tensor<20xf64>
// CHECK:    %82 = stablehlo.reshape %81 : (tensor<20xf64>) -> tensor<20xf64>
// CHECK:    %83 = stablehlo.broadcast_in_dim %82, dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK:    %84 = arith.divf %80, %83 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %85 = arith.mulf %52, %84 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %c_12 = stablehlo.constant dense<-19> : tensor<i64>
// CHECK:    %86 = stablehlo.broadcast_in_dim %c_12, dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK:    %87 = stablehlo.add %2, %86 : tensor<20xi64>
// CHECK:    %c_13 = stablehlo.constant dense<0> : tensor<20xi64>
// CHECK:    %88 = stablehlo.compare  EQ, %87, %c_13 : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
// CHECK:    %89 = stablehlo.broadcast_in_dim %88, dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK:    %90 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK:    %91 = stablehlo.select %89, %85, %90 : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK:    %92 = arith.ori %63, %72 : tensor<20x85x180xi1>
// CHECK:    %93 = stablehlo.broadcast_in_dim %65, dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK:    %94 = arith.andi %93, %92 : tensor<20x85x180xi1>
// CHECK:    %95 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK:    %96 = arith.select %94, %95, %91 : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK:    %97 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK:    %98 = stablehlo.broadcast_in_dim %96, dims = [2, 0, 1] : (tensor<20x85x180xf64>) -> tensor<85x180x20xf64>
// CHECK:    %99 = arith.mulf %97, %98 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %100 = stablehlo.slice %arg9 [7:27, 7:92, 7:187] : (tensor<35x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK:    %101 = stablehlo.reshape %100 : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK:    %102 = arith.negf %101 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %103 = stablehlo.slice %arg1 [6:26] : (tensor<34xf64>) -> tensor<20xf64>
// CHECK:    %104 = stablehlo.reshape %103 : (tensor<20xf64>) -> tensor<20xf64>
// CHECK:    %105 = stablehlo.broadcast_in_dim %104, dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK:    %106 = stablehlo.broadcast_in_dim %56, dims = [1, 2] : (tensor<85x180xf64>) -> tensor<20x85x180xf64>
// CHECK:    %107 = arith.cmpf ole, %105, %106 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %108 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK:    %109 = arith.cmpi ult, %2, %108 : tensor<20xi64>
// CHECK:    %110 = stablehlo.broadcast_in_dim %109, dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK:    %111 = arith.ori %110, %107 : tensor<20x85x180xi1>
// CHECK:    %112 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<20xi64>
// CHECK:    %113 = arith.cmpi uge, %2, %112 : tensor<20xi64>
// CHECK:    %114 = stablehlo.broadcast_in_dim %113, dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK:    %115 = arith.andi %114, %111 : tensor<20x85x180xi1>
// CHECK:    %116 = arith.ori %72, %115 : tensor<20x85x180xi1>
// CHECK:    %117 = stablehlo.slice %arg8 [6:26, 7:92, 7:187] : (tensor<34x99x194xf64>) -> tensor<20x85x180xf64>
// CHECK:    %118 = stablehlo.reshape %117 : (tensor<20x85x180xf64>) -> tensor<20x85x180xf64>
// CHECK:    %119 = arith.subf %77, %118 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %120 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK:    %121 = arith.select %116, %120, %119 : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK:    %122 = stablehlo.slice %arg2 [8:28] : (tensor<35xf64>) -> tensor<20xf64>
// CHECK:    %123 = stablehlo.reshape %122 : (tensor<20xf64>) -> tensor<20xf64>
// CHECK:    %124 = stablehlo.broadcast_in_dim %123, dims = [0] : (tensor<20xf64>) -> tensor<20x85x180xf64>
// CHECK:    %125 = arith.divf %121, %124 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %126 = arith.mulf %102, %125 {fastmathFlags = #llvm.fastmath<none>} : tensor<20x85x180xf64>
// CHECK:    %c_14 = stablehlo.constant dense<0> : tensor<20xi64>
// CHECK:    %127 = stablehlo.compare  EQ, %2, %c_14 : (tensor<20xi64>, tensor<20xi64>) -> tensor<20xi1>
// CHECK:    %128 = stablehlo.broadcast_in_dim %127, dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK:    %129 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK:    %130 = stablehlo.select %128, %126, %129 : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK:    %131 = arith.ori %72, %111 : tensor<20x85x180xi1>
// CHECK:    %132 = stablehlo.broadcast_in_dim %113, dims = [0] : (tensor<20xi1>) -> tensor<20x85x180xi1>
// CHECK:    %133 = arith.andi %132, %131 : tensor<20x85x180xi1>
// CHECK:    %134 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f64>) -> tensor<20x85x180xf64>
// CHECK:    %135 = arith.select %133, %134, %130 : tensor<20x85x180xi1>, tensor<20x85x180xf64>
// CHECK:    %136 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<85x180xf64>) -> tensor<85x180x20xf64>
// CHECK:    %137 = stablehlo.broadcast_in_dim %135, dims = [2, 0, 1] : (tensor<20x85x180xf64>) -> tensor<85x180x20xf64>
// CHECK:    %138 = arith.mulf %136, %137 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %139 = arith.subf %99, %138 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %140 = arith.addf %32, %47 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %141 = arith.addf %140, %139 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %142 = arith.mulf %17, %141 {fastmathFlags = #llvm.fastmath<none>} : tensor<85x180x20xf64>
// CHECK:    %143 = arith.negf %142 : tensor<85x180x20xf64>
// CHECK:    %c_15 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %c_16 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %c_17 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %144 = stablehlo.broadcast_in_dim %143, dims = [1, 2, 0] : (tensor<85x180x20xf64>) -> tensor<20x85x180xf64>
// CHECK:    %145 = stablehlo.dynamic_update_slice %arg0, %144, %c_15, %c_16, %c_17 : (tensor<34x99x194xf64>, tensor<20x85x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<34x99x194xf64>
// CHECK:    return %145, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : tensor<34x99x194xf64>, tensor<34xf64>, tensor<35xf64>, tensor<34xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<99x194xf64>, tensor<1x99x194xf64>, tensor<34x99x194xf64>, tensor<35x99x194xf64>
// CHECK:// -----
// CHECK:  func.func private @par6_raised(%arg0: tensor<1x104x194xf64>) -> tensor<1x104x194xf64> {
// CHECK:    %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK:    %c_0 = stablehlo.constant dense<182> : tensor<i64>
// CHECK:    %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK:    %c_2 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK:    %0 = stablehlo.iota dim = 0 : tensor<180xi64>
// CHECK:    %c_3 = stablehlo.constant dense<0> : tensor<180xi64>
// CHECK:    %1 = stablehlo.add %0, %c_3 : tensor<180xi64>
// CHECK:    %c_4 = stablehlo.constant dense<1> : tensor<180xi64>
// CHECK:    %2 = stablehlo.multiply %1, %c_4 : tensor<180xi64>
// CHECK:    %c_5 = stablehlo.constant dense<180> : tensor<1xi64>
// CHECK:    %3 = stablehlo.slice %arg0 [0:1, 7:8, 7:187] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK:    %4 = stablehlo.reshape %3 : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK:    %c_6 = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %c_7 = stablehlo.constant dense<6> : tensor<i64>
// CHECK:    %c_8 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %5 = stablehlo.broadcast_in_dim %4, dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK:    %6 = stablehlo.dynamic_update_slice %arg0, %5, %c_6, %c_7, %c_8 : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK:    %c_9 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK:    %7 = stablehlo.broadcast_in_dim %c_9, dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK:    %8 = stablehlo.add %2, %7 : tensor<180xi64>
// CHECK:    %c_10 = stablehlo.constant dense<0> : tensor<180xi64>
// CHECK:    %9 = stablehlo.compare  GE, %8, %c_10 : (tensor<180xi64>, tensor<180xi64>) -> tensor<180xi1>
// CHECK:    %10 = stablehlo.slice %6 [0:1, 96:97, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK:    %11 = stablehlo.reverse %10, dims = [2] : tensor<1x1x180xf64>
// CHECK:    %12 = stablehlo.reshape %11 : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK:    %13 = stablehlo.slice %6 [0:1, 89:90, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK:    %14 = stablehlo.reverse %13, dims = [2] : tensor<1x1x180xf64>
// CHECK:    %15 = stablehlo.reshape %14 : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK:    %16 = stablehlo.slice %6 [0:1, 90:91, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK:    %17 = stablehlo.reverse %16, dims = [2] : tensor<1x1x180xf64>
// CHECK:    %18 = stablehlo.reshape %17 : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK:    %19 = stablehlo.slice %6 [0:1, 91:92, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK:    %20 = stablehlo.reverse %19, dims = [2] : tensor<1x1x180xf64>
// CHECK:    %21 = stablehlo.reshape %20 : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK:    %22 = stablehlo.slice %6 [0:1, 92:93, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK:    %23 = stablehlo.reverse %22, dims = [2] : tensor<1x1x180xf64>
// CHECK:    %24 = stablehlo.reshape %23 : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK:    %25 = stablehlo.slice %6 [0:1, 93:94, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK:    %26 = stablehlo.reverse %25, dims = [2] : tensor<1x1x180xf64>
// CHECK:    %27 = stablehlo.reshape %26 : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK:    %28 = stablehlo.slice %6 [0:1, 94:95, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK:    %29 = stablehlo.reverse %28, dims = [2] : tensor<1x1x180xf64>
// CHECK:    %30 = stablehlo.reshape %29 : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK:    %31 = stablehlo.slice %6 [0:1, 95:96, 8:188] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK:    %32 = stablehlo.reverse %31, dims = [2] : tensor<1x1x180xf64>
// CHECK:    %33 = stablehlo.reshape %32 : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK:    %34 = stablehlo.slice %6 [0:1, 96:97, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK:    %35 = stablehlo.reshape %34 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK:    %36 = stablehlo.slice %6 [0:1, 89:90, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK:    %37 = stablehlo.reshape %36 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK:    %38 = stablehlo.slice %6 [0:1, 90:91, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK:    %39 = stablehlo.reshape %38 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK:    %40 = stablehlo.slice %6 [0:1, 91:92, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK:    %41 = stablehlo.reshape %40 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK:    %42 = stablehlo.slice %6 [0:1, 92:93, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK:    %43 = stablehlo.reshape %42 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK:    %44 = stablehlo.slice %6 [0:1, 93:94, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK:    %45 = stablehlo.reshape %44 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK:    %46 = stablehlo.slice %6 [0:1, 94:95, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK:    %47 = stablehlo.reshape %46 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK:    %48 = stablehlo.slice %6 [0:1, 95:96, 7:8] : (tensor<1x104x194xf64>) -> tensor<1x1x1xf64>
// CHECK:    %49 = stablehlo.reshape %48 : (tensor<1x1x1xf64>) -> tensor<f64>
// CHECK:    %50 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK:    %51 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK:    %52 = stablehlo.select %9, %50, %51 : tensor<180xi1>, tensor<180xi64>
// CHECK:    %53 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK:    %54 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK:    %55 = stablehlo.select %9, %53, %54 : tensor<180xi1>, tensor<180xi64>
// CHECK:    %56 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK:    %57 = stablehlo.select %9, %12, %56 : tensor<180xi1>, tensor<180xf64>
// CHECK:    %58 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK:    %59 = stablehlo.select %9, %15, %58 : tensor<180xi1>, tensor<180xf64>
// CHECK:    %60 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK:    %61 = stablehlo.select %9, %18, %60 : tensor<180xi1>, tensor<180xf64>
// CHECK:    %62 = stablehlo.broadcast_in_dim %41, dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK:    %63 = stablehlo.select %9, %21, %62 : tensor<180xi1>, tensor<180xf64>
// CHECK:    %64 = stablehlo.broadcast_in_dim %43, dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK:    %65 = stablehlo.select %9, %24, %64 : tensor<180xi1>, tensor<180xf64>
// CHECK:    %66 = stablehlo.broadcast_in_dim %45, dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK:    %67 = stablehlo.select %9, %27, %66 : tensor<180xi1>, tensor<180xf64>
// CHECK:    %68 = stablehlo.broadcast_in_dim %47, dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK:    %69 = stablehlo.select %9, %30, %68 : tensor<180xi1>, tensor<180xf64>
// CHECK:    %70 = stablehlo.broadcast_in_dim %49, dims = [] : (tensor<f64>) -> tensor<180xf64>
// CHECK:    %71 = stablehlo.select %9, %33, %70 : tensor<180xi1>, tensor<180xf64>
// CHECK:    %72 = arith.sitofp %52 : tensor<180xi64> to tensor<180xf64>
// CHECK:    %73 = arith.mulf %72, %71 {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK:    %c_11 = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %c_12 = stablehlo.constant dense<97> : tensor<i64>
// CHECK:    %c_13 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %74 = stablehlo.broadcast_in_dim %73, dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK:    %75 = stablehlo.dynamic_update_slice %6, %74, %c_11, %c_12, %c_13 : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK:    %76 = arith.mulf %72, %69 {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK:    %c_14 = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %c_15 = stablehlo.constant dense<98> : tensor<i64>
// CHECK:    %c_16 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %77 = stablehlo.broadcast_in_dim %76, dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK:    %78 = stablehlo.dynamic_update_slice %75, %77, %c_14, %c_15, %c_16 : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK:    %79 = arith.mulf %72, %67 {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK:    %c_17 = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %c_18 = stablehlo.constant dense<99> : tensor<i64>
// CHECK:    %c_19 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %80 = stablehlo.broadcast_in_dim %79, dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK:    %81 = stablehlo.dynamic_update_slice %78, %80, %c_17, %c_18, %c_19 : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK:    %82 = arith.mulf %72, %65 {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK:    %c_20 = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %c_21 = stablehlo.constant dense<100> : tensor<i64>
// CHECK:    %c_22 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %83 = stablehlo.broadcast_in_dim %82, dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK:    %84 = stablehlo.dynamic_update_slice %81, %83, %c_20, %c_21, %c_22 : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK:    %85 = arith.mulf %72, %63 {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK:    %c_23 = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %c_24 = stablehlo.constant dense<101> : tensor<i64>
// CHECK:    %c_25 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %86 = stablehlo.broadcast_in_dim %85, dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK:    %87 = stablehlo.dynamic_update_slice %84, %86, %c_23, %c_24, %c_25 : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK:    %88 = arith.mulf %72, %61 {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK:    %c_26 = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %c_27 = stablehlo.constant dense<102> : tensor<i64>
// CHECK:    %c_28 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %89 = stablehlo.broadcast_in_dim %88, dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK:    %90 = stablehlo.dynamic_update_slice %87, %89, %c_26, %c_27, %c_28 : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK:    %91 = arith.mulf %72, %59 {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK:    %c_29 = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %c_30 = stablehlo.constant dense<103> : tensor<i64>
// CHECK:    %c_31 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %92 = stablehlo.broadcast_in_dim %91, dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK:    %93 = stablehlo.dynamic_update_slice %90, %92, %c_29, %c_30, %c_31 : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK:    %94 = arith.mulf %72, %57 {fastmathFlags = #llvm.fastmath<none>} : tensor<180xf64>
// CHECK:    %95 = stablehlo.slice %93 [0:1, 96:97, 7:187] : (tensor<1x104x194xf64>) -> tensor<1x1x180xf64>
// CHECK:    %96 = stablehlo.reshape %95 : (tensor<1x1x180xf64>) -> tensor<180xf64>
// CHECK:    %c_32 = stablehlo.constant dense<-1> : tensor<i64>
// CHECK:    %97 = stablehlo.broadcast_in_dim %c_32, dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK:    %98 = stablehlo.multiply %2, %97 : tensor<180xi64>
// CHECK:    %c_33 = stablehlo.constant dense<89> : tensor<i64>
// CHECK:    %99 = stablehlo.broadcast_in_dim %c_33, dims = [] : (tensor<i64>) -> tensor<180xi64>
// CHECK:    %100 = stablehlo.add %98, %99 : tensor<180xi64>
// CHECK:    %c_34 = stablehlo.constant dense<0> : tensor<180xi64>
// CHECK:    %101 = stablehlo.compare  GE, %100, %c_34 : (tensor<180xi64>, tensor<180xi64>) -> tensor<180xi1>
// CHECK:    %102 = stablehlo.select %101, %96, %94 : tensor<180xi1>, tensor<180xf64>
// CHECK:    %c_35 = stablehlo.constant dense<0> : tensor<i64>
// CHECK:    %c_36 = stablehlo.constant dense<96> : tensor<i64>
// CHECK:    %c_37 = stablehlo.constant dense<7> : tensor<i64>
// CHECK:    %103 = stablehlo.broadcast_in_dim %102, dims = [2] : (tensor<180xf64>) -> tensor<1x1x180xf64>
// CHECK:    %104 = stablehlo.dynamic_update_slice %93, %103, %c_35, %c_36, %c_37 : (tensor<1x104x194xf64>, tensor<1x1x180xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x104x194xf64>
// CHECK:    return %104 : tensor<1x104x194xf64>
// CHECK:// -----
// CHECK:// -----
// CHECK:// -----
// CHECK:// -----
