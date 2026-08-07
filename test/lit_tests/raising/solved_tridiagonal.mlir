// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
func.func private @"##call__Z44gpu_solve_batched_tridiagonal_system_kernel_16CompilerMetadataI10StaticSizeI9_192__96_E12DynamicCheckvv7NDRangeILi2ES0_I7_12__6_ES0_I8_16__16_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E14_208__112__36_EE40VerticallyImplicitDiffusionLowerDiagonal35VerticallyImplicitDiffusionDiagonal40VerticallyImplicitDiffusionUpperDiagonalSC_SA_IS9_Li3ELi1E13_192__96__20_E20ImmersedBoundaryGridIS9_8Periodic17RightCenterFolded7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_37__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_36__EESO_SQ_E8TripolarI5Int64ST_ST_SJ_ES8_IS9_Li2ESA_IS9_Li2ELi1E10_208__112_EESW_SW_SW_vS9_vE16GridFittedBottomIS8_IS9_Li3ESA_IS9_Li3ELi1E13_208__112__1_EE23CenterImmersedConditionEvvvEv5TupleI24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthIS9_ES9_v13CATKEEquationIS9_EE18CATKEClosureFieldsIS8_IS9_Li3ESA_IS9_Li3ELi1E14_208__112__37_EESC_5FieldI6CenterS1G_vvvvS10_S9_vvvE10NamedTupleI8__u___v_S14_ISC_SC_EES1I_I12__T___S___e_S14_IS1E_S1E_S1E_EES1I_I12__T___S___e_S14_I9ZeroFieldIST_Li3EES1O_SC_EEE3ValILi1EES1G_S1G_S1G_S9_S1I_I53__time___last__t___last_stage__t___iteration___stage_S14_I15CuTracedRNumberIS9_Li1EES1V_S1V_S1U_IST_Li1EEST_EES1I_I36__u___v___w___T___S___e_______U___V_S14_ISC_SC_S1E_SC_SC_SC_S8_IS9_Li3ESA_IS9_Li3ELi1E13_208__142__1_EES1F_I4FaceS1G_vvvvS20_S9_vvvES1F_IS1G_S21_vvvvS20_S9_vvvEEE4WENOILi4ES9_18ConvertingDivisionI7Float32E26ExplicitTimeDiscretizationvS26_ILi3ES9_S29_S2A_vS26_ILi2ES9_S29_S2A_v8CenteredILi1ES9_S2A_vES2C_ES2B_ILi2ES9_S2A_S2C_EES2B_ILi3ES9_S2A_S2E_EES1E_vE10ZDirection#4090$par971"(%arg0: memref<36x112x208xf64, 1>, %arg1: memref<20x96x192xf64, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
  %cst = arith.constant 0x7FF8000000000000 : f64
  affine.parallel (%arg2, %arg3) = (0, 0) to (96, 192) {
    affine.store %cst, %arg0[8, %arg2 + 8, %arg3 + 8] : memref<36x112x208xf64, 1>
    affine.parallel (%arg4) = (0) to (19) {
      %2 = affine.load %arg0[%arg4 + 9, %arg2 + 8, %arg3 + 8] : memref<36x112x208xf64, 1>
      affine.store %cst, %arg1[%arg4 + 1, %arg2, %arg3] : memref<20x96x192xf64, 1>
      affine.store %2, %arg0[%arg4 + 9, %arg2 + 8, %arg3 + 8] : memref<36x112x208xf64, 1>
    }
    %0 = affine.load %arg0[27, %arg2 + 8, %arg3 + 8] : memref<36x112x208xf64, 1>
    %1 = affine.for %arg4 = 0 to 19 iter_args(%arg5 = %0) -> (f64) {
      %2 = affine.load %arg0[-%arg4 + 26, %arg2 + 8, %arg3 + 8] : memref<36x112x208xf64, 1>
      %3 = affine.load %arg1[-%arg4 + 19, %arg2, %arg3] : memref<20x96x192xf64, 1>
      %4 = arith.mulf %3, %arg5 {fastmathFlags = #llvm.fastmath<none>} : f64
      %5 = arith.subf %2, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %5, %arg0[-%arg4 + 26, %arg2 + 8, %arg3 + 8] : memref<36x112x208xf64, 1>
      affine.yield %5 : f64
    }
  }
  return
}
}

// CHECK:  func.func private @"##call__Z44gpu_solve_batched_tridiagonal_system_kernel_16CompilerMetadataI10StaticSizeI9_192__96_E12DynamicCheckvv7NDRangeILi2ES0_I7_12__6_ES0_I8_16__16_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E14_208__112__36_EE40VerticallyImplicitDiffusionLowerDiagonal35VerticallyImplicitDiffusionDiagonal40VerticallyImplicitDiffusionUpperDiagonalSC_SA_IS9_Li3ELi1E13_192__96__20_E20ImmersedBoundaryGridIS9_8Periodic17RightCenterFolded7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_37__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_36__EESO_SQ_E8TripolarI5Int64ST_ST_SJ_ES8_IS9_Li2ESA_IS9_Li2ELi1E10_208__112_EESW_SW_SW_vS9_vE16GridFittedBottomIS8_IS9_Li3ESA_IS9_Li3ELi1E13_208__112__1_EE23CenterImmersedConditionEvvvEv5TupleI24CATKEVerticalDiffusivityI36VerticallyImplicitTimeDiscretization17CATKEMixingLengthIS9_ES9_v13CATKEEquationIS9_EE18CATKEClosureFieldsIS8_IS9_Li3ESA_IS9_Li3ELi1E14_208__112__37_EESC_5FieldI6CenterS1G_vvvvS10_S9_vvvE10NamedTupleI8__u___v_S14_ISC_SC_EES1I_I12__T___S___e_S14_IS1E_S1E_S1E_EES1I_I12__T___S___e_S14_I9ZeroFieldIST_Li3EES1O_SC_EEE3ValILi1EES1G_S1G_S1G_S9_S1I_I53__time___last__t___last_stage__t___iteration___stage_S14_I15CuTracedRNumberIS9_Li1EES1V_S1V_S1U_IST_Li1EEST_EES1I_I36__u___v___w___T___S___e_______U___V_S14_ISC_SC_S1E_SC_SC_SC_S8_IS9_Li3ESA_IS9_Li3ELi1E13_208__142__1_EES1F_I4FaceS1G_vvvvS20_S9_vvvES1F_IS1G_S21_vvvvS20_S9_vvvEEE4WENOILi4ES9_18ConvertingDivisionI7Float32E26ExplicitTimeDiscretizationvS26_ILi3ES9_S29_S2A_vS26_ILi2ES9_S29_S2A_v8CenteredILi1ES9_S2A_vES2C_ES2B_ILi2ES9_S2A_S2C_EES2B_ILi3ES9_S2A_S2E_EES1E_vE10ZDirection#4090$par971_raised"(%arg0: tensor<36x112x208xf64>, %arg1: tensor<20x96x192xf64>) -> (tensor<36x112x208xf64>, tensor<20x96x192xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0x7FF8000000000000> : tensor<f64>
// CHECK-NEXT:    %c = stablehlo.constant dense<26> : tensor<i64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<19> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<8> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [9:28, 8:104, 8:200] : (tensor<36x112x208xf64>) -> tensor<19x96x192xf64>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1, 0:96, 0:192] : (tensor<20x96x192xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %2 = stablehlo.pad %1, %cst, low = [0, 0, 0], high = [19, 0, 0], interior = [0, 0, 0] : (tensor<1x96x192xf64>, tensor<f64>) -> tensor<20x96x192xf64>
// CHECK-NEXT:    %3 = stablehlo.pad %0, %cst, low = [1, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<19x96x192xf64>, tensor<f64>) -> tensor<20x96x192xf64>
// CHECK-NEXT:    %4 = stablehlo.dynamic_update_slice %arg0, %3, %c_3, %c_3, %c_3 : (tensor<36x112x208xf64>, tensor<20x96x192xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<36x112x208xf64>
// CHECK-NEXT:    %5 = stablehlo.slice %arg0 [27:28, 8:104, 8:200] : (tensor<36x112x208xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %6 = stablehlo.reshape %5 : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %7:4 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %6, %iterArg_5 = %4, %iterArg_6 = %2) : tensor<i64>, tensor<96x192xf64>, tensor<36x112x208xf64>, tensor<20x96x192xf64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %8 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %8 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %8 = stablehlo.subtract %c, %iterArg {enzymexla.bounds = {{.+}}} : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.dynamic_slice %iterArg_5, %8, %c_3, %c_3, sizes = [1, 96, 192] : (tensor<36x112x208xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:      %10 = stablehlo.reshape %9 : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:      %11 = stablehlo.subtract %c_0, %iterArg {enzymexla.bounds = {{.+}}} : tensor<i64>
// CHECK-NEXT:      %12 = stablehlo.dynamic_slice %iterArg_6, %11, %c_1, %c_1, sizes = [1, 96, 192] : (tensor<20x96x192xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:      %13 = stablehlo.reshape %12 : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:      %14 = stablehlo.multiply %13, %iterArg_4 : tensor<96x192xf64>
// CHECK-NEXT:      %15 = stablehlo.subtract %10, %14 : tensor<96x192xf64>
// CHECK-NEXT:      %16 = stablehlo.reshape %15 : (tensor<96x192xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:      %17 = stablehlo.dynamic_update_slice %iterArg_5, %16, %8, %c_3, %c_3 : (tensor<36x112x208xf64>, tensor<1x96x192xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<36x112x208xf64>
// CHECK-NEXT:      %18 = stablehlo.add %iterArg, %c_2 {enzymexla.bounds = {{.+}}} : tensor<i64>
// CHECK-NEXT:      stablehlo.return %18, %15, %17, %iterArg_6 : tensor<i64>, tensor<96x192xf64>, tensor<36x112x208xf64>, tensor<20x96x192xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %7#2, %7#3 : tensor<36x112x208xf64>, tensor<20x96x192xf64>
// CHECK-NEXT:  }
