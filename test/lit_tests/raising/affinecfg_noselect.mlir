// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module {
func.func private @"##call__Z40gpu_compute_hydrostatic_free_surface_Gu_16CompilerMetadataI10StaticSizeI15_3056__1520__4_E12DynamicCheckvv7NDRangeILi3ES0_I12_191__95__4_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E16_3072__1536__20_EE21LatitudeLongitudeGridI15CuTracedRNumberIS9_Li1EE8Periodic7BoundedSH_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_21__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_20__EESK_SM_ES9_S9_S8_IS9_Li1E12StepRangeLenIS9_7Float64SP_5Int64EESS_S9_S9_SS_SS_S8_IS9_Li1ESA_IS9_Li1ELi1E7_1536__EESU_SU_SU_S9_S9_vSQ_E5TupleI15VectorInvariantILi3ES9_Lb0E4WENOILi3ES9_S9_vSY_ILi2ES9_S9_v12UpwindBiasedILi1ES9_v8CenteredILi1ES9_vEES11_ES10_ILi2ES9_S11_EE15VelocityStencilS15_S15_S15_17OnlySelfUpwindingIS14_15FunctionStencilI21divergence_smoothnessES1A_S18_I12u_smoothnessES18_I12v_smoothnessEEE28HydrostaticSphericalCoriolisI19EnstrophyConservingIS9_ES9_Ev17BoundaryConditionI4FluxvE10NamedTupleI12__u___v___w_SW_ISC_SC_SC_EE24SplitExplicitFreeSurfaceIS8_IS9_Li3ESA_IS9_Li3ELi1E15_3072__1536__1_EES1O_I8__U___V_SW_I5FieldI4Face6CentervvvvS1T_S9_vvvES1U_IS1W_S1V_vvvvS1T_S9_vvvEEES1O_I12______U___V_SW_IS1T_S1X_S1Y_EES9_v18FixedSubstepNumberIS9_SW_IS9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_EE21ForwardBackwardSchemeES1O_I8__T___S_SW_ISC_SC_EE13BuoyancyForceI16SeawaterBuoyancyIS9_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialIS9_ES9_EvvE18NegativeZDirectionEvSC_S1O_I2__SW_IJEEE11ZCoordinateS1O_I53__time___last__t___last_stage__t___iteration___stage_SW_I13TracedRNumberIS9_ES9_S9_S2M_ISQ_ESQ_EE11zeroforcingE#1036$par15"(%arg0: memref<20x1536x3072xf32, 1>, %arg1: memref<20xf32, 1>, %arg2: memref<1536xf32, 1>, %arg3: memref<1536xf32, 1>, %arg4: memref<1536xf32, 1>, %arg5: memref<1536xf32, 1>, %arg6: memref<1536xf32, 1>, %arg7: memref<20x1536x3072xf32, 1>, %arg8: memref<20x1536x3072xf32, 1>, %arg9: memref<20x1536x3072xf32, 1>, %arg10: memref<20x1536x3072xf32, 1>) {
  %c16 = arith.constant 16 : index
  %c95 = arith.constant 95 : index
  %c191 = arith.constant 191 : index
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c1520_i64 = arith.constant 1520 : i64
  %cst = arith.constant 1.000000e+00 : f32
  %c2_i64 = arith.constant 2 : i64
  %cst_0 = arith.constant 11704.7295 : f32
  %0 = "enzymexla.memref2pointer"(%arg0) : (memref<20x1536x3072xf32, 1>) -> !llvm.ptr<1>
  affine.parallel (%arg11, %arg12) = (0, 0) to (72580, 256) {
    %1 = arith.divui %arg11, %c191 : index
    %2 = arith.remui %1, %c95 : index
    %3 = arith.index_cast %2 : index to i64
    %4 = arith.divui %arg12, %c16 : index
    %5 = arith.index_castui %4 : index to i64
    %c16_1 = arith.constant 16 : index 
    %6 = arith.muli %2, %c16_1 : index
    %7 = arith.index_castui %6 : index to i64
    %8 = arith.addi %6, %4 : index
    %9 = arith.index_castui %8 : index to i64
    %10 = arith.addi %9, %c1_i64 : i64
    %11 = arith.cmpi sle, %10, %c1520_i64 : i64
    %12 = arith.cmpi uge, %10, %c2_i64 : i64
    %13 = arith.andi %12, %11 : i1
    %14 = arith.select %13, %cst, %cst_0 : f32
    llvm.store %14, %0 {alignment = 4 : i64, tbaa = [#llvm.tbaa_tag<base_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, access_type = <id = "custom_tbaa_addrspace(1)", members = {<#llvm.tbaa_root<id = "custom_tbaa">, 0>}>, offset = 0>]} : f32, !llvm.ptr<1>
  }
  return
}

}

// CHECK: #set = affine_set<(d0, d1) : (d0 floordiv 16 + (d1 floordiv 191) * 16 - (d1 floordiv 18145) * 1520 + 1 >= 0, -(d0 floordiv 16) - (d1 floordiv 191) * 16 + (d1 floordiv 18145) * 1520 >= 0)>
// CHECK:   func.func private @"##call__Z40gpu_compute_hydrostatic_free_surface_Gu_16CompilerMetadataI10StaticSizeI15_3056__1520__4_E12DynamicCheckvv7NDRangeILi3ES0_I12_191__95__4_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E16_3072__1536__20_EE21LatitudeLongitudeGridI15CuTracedRNumberIS9_Li1EE8Periodic7BoundedSH_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_21__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_20__EESK_SM_ES9_S9_S8_IS9_Li1E12StepRangeLenIS9_7Float64SP_5Int64EESS_S9_S9_SS_SS_S8_IS9_Li1ESA_IS9_Li1ELi1E7_1536__EESU_SU_SU_S9_S9_vSQ_E5TupleI15VectorInvariantILi3ES9_Lb0E4WENOILi3ES9_S9_vSY_ILi2ES9_S9_v12UpwindBiasedILi1ES9_v8CenteredILi1ES9_vEES11_ES10_ILi2ES9_S11_EE15VelocityStencilS15_S15_S15_17OnlySelfUpwindingIS14_15FunctionStencilI21divergence_smoothnessES1A_S18_I12u_smoothnessES18_I12v_smoothnessEEE28HydrostaticSphericalCoriolisI19EnstrophyConservingIS9_ES9_Ev17BoundaryConditionI4FluxvE10NamedTupleI12__u___v___w_SW_ISC_SC_SC_EE24SplitExplicitFreeSurfaceIS8_IS9_Li3ESA_IS9_Li3ELi1E15_3072__1536__1_EES1O_I8__U___V_SW_I5FieldI4Face6CentervvvvS1T_S9_vvvES1U_IS1W_S1V_vvvvS1T_S9_vvvEEES1O_I12______U___V_SW_IS1T_S1X_S1Y_EES9_v18FixedSubstepNumberIS9_SW_IS9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_S9_EE21ForwardBackwardSchemeES1O_I8__T___S_SW_ISC_SC_EE13BuoyancyForceI16SeawaterBuoyancyIS9_25BoussinesqEquationOfStateI24TEOS10SeawaterPolynomialIS9_ES9_EvvE18NegativeZDirectionEvSC_S1O_I2__SW_IJEEE11ZCoordinateS1O_I53__time___last__t___last_stage__t___iteration___stage_SW_I13TracedRNumberIS9_ES9_S9_S2M_ISQ_ESQ_EE11zeroforcingE#1036$par15"(%arg0: memref<20x1536x3072xf32, 1>, %arg1: memref<20xf32, 1>, %arg2: memref<1536xf32, 1>, %arg3: memref<1536xf32, 1>, %arg4: memref<1536xf32, 1>, %arg5: memref<1536xf32, 1>, %arg6: memref<1536xf32, 1>, %arg7: memref<20x1536x3072xf32, 1>, %arg8: memref<20x1536x3072xf32, 1>, %arg9: memref<20x1536x3072xf32, 1>, %arg10: memref<20x1536x3072xf32, 1>) {
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 11704.7295 : f32
// CHECK-NEXT:     %0 = "enzymexla.memref2pointer"(%arg0) : (memref<20x1536x3072xf32, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:     affine.parallel (%arg11, %arg12) = (0, 0) to (72580, 256) {
// CHECK-NEXT:       %1 = affine.if #set(%arg12, %arg11) -> i1 {
// CHECK-NEXT:         affine.yield %false : i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         affine.yield %true : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       %2 = arith.select %1, %cst, %cst_0 : f32
// CHECK-NEXT:       llvm.store %2, %0 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : f32, !llvm.ptr<1>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

