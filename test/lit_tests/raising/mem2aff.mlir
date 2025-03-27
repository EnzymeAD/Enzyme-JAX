func.func private @"##call__Z43gpu__interpolate_primary_atmospheric_state_16CompilerMetadataI16OffsetStaticSizeI13_0_181__0_91_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__6_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE10NamedTupleI35__u___v___T___p___q___Qs___Q____Mp_S7_I11OffsetArrayI7Float64Li3E13CuTracedArrayISG_Li3ELi1E13_194__104__1_EESJ_SJ_SJ_SJ_SJ_SJ_SJ_EESE_I8__i___j_S7_I5FieldI6CenterSN_vvvvSJ_SG_vvvESO_EE16TimeInterpolatorI15CuTracedRNumberISG_Li1EESS_IS8_Li1EESU_S8_E20ImmersedBoundaryGridISG_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISG_SX_SY_SZ_28StaticVerticalDiscretizationISF_ISG_Li1ESH_ISG_Li1ELi1E5_35__EESF_ISG_Li1ESH_ISG_Li1ELi1E5_34__EES13_S15_E8TripolarIS8_S8_S8_ESF_ISG_Li2ESH_ISG_Li2ELi1E10_194__104_EES1A_S1A_S1A_vSG_E16GridFittedBottomISO_23CenterImmersedConditionEvvvESE_I8__u___v_S7_ISF_ISG_Li4ESH_ISG_Li4ELi1E17_366__186__1__24_EES1H_EESE_I8__T___q_S1I_ES1H_SE_I23__shortwave___longwave_S1I_ESE_I14__rain___snow_S1I_E8InMemoryIvE5Clamp#2136$par448"(%arg0: memref<1x104x194xf64, 1>, %arg1: memref<1x104x194xf64, 1>, %arg2: memref<1x104x194xf64, 1>, %arg3: memref<1x104x194xf64, 1>, %arg4: memref<1x104x194xf64, 1>, %arg5: memref<1x104x194xf64, 1>, %arg6: memref<1x104x194xf64, 1>, %arg7: memref<1x104x194xf64, 1>, %arg8: memref<1x104x194xf64, 1>, %arg9: memref<1x104x194xf64, 1>, %arg10: memref<f64, 1>, %arg11: memref<i64, 1>, %arg12: memref<i64, 1>, %arg13: memref<104x194xf64, 1>, %arg14: memref<104x194xf64, 1>, %arg15: memref<104x194xf64, 1>, %arg16: memref<24x1x186x366xf64, 1>, %arg17: memref<24x1x186x366xf64, 1>, %arg18: memref<24x1x186x366xf64, 1>, %arg19: memref<24x1x186x366xf64, 1>, %arg20: memref<24x1x186x366xf64, 1>, %arg21: memref<24x1x186x366xf64, 1>, %arg22: memref<24x1x186x366xf64, 1>, %arg23: memref<24x1x186x366xf64, 1>, %arg24: memref<24x1x186x366xf64, 1>) {
  %c0 = arith.constant 0 : index
  %c186 = arith.constant 186 : index
  %c366 = arith.constant 366 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c3_i64 = arith.constant 3 : i64
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 0.000000e+00 : f64
  %c24_i64 = arith.constant 24 : i64
  %c2_i64 = arith.constant 2 : i64
  %c366_i64 = arith.constant 366 : i64
  %c-1_i64 = arith.constant -1 : i64
  %c68076_i64 = arith.constant 68076 : i64
  %cst_1 = arith.constant 0.017453292519943295 : f64
  %cst_2 = arith.constant 2.000000e+00 : f64
  affine.parallel (%arg25, %arg26) = (0, 0) to (92, 182) {
    %0 = affine.load %arg8[0, %arg25 + 6, %arg26 + 6] : memref<1x104x194xf64, 1>
    %1 = affine.load %arg9[0, %arg25 + 6, %arg26 + 6] : memref<1x104x194xf64, 1>
    %2 = affine.load %arg11[] : memref<i64, 1>
    %3 = affine.load %arg12[] : memref<i64, 1>
    %4 = arith.fptosi %0 : f64 to i64
    %5 = arith.remf %0, %cst : f64
    %6 = arith.cmpf oeq, %5, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
    %7 = math.copysign %5, %cst : f64
    %8 = arith.cmpf olt, %cst_0, %5 {fastmathFlags = #llvm.fastmath<none>} : f64
    %9 = arith.addf %5, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
    %10 = arith.select %8, %5, %9 {fastmathFlags = #llvm.fastmath<none>} : f64
    %11 = arith.select %6, %7, %10 : f64
    %12 = arith.fptosi %1 : f64 to i64
    %13 = arith.remf %1, %cst : f64
    %14 = arith.cmpf oeq, %13, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
    %15 = math.copysign %13, %cst : f64
    %16 = arith.cmpf olt, %cst_0, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
    %17 = arith.addf %13, %cst {fastmathFlags = #llvm.fastmath<none>} : f64
    %18 = arith.select %16, %13, %17 {fastmathFlags = #llvm.fastmath<none>} : f64
    %19 = arith.select %14, %15, %18 : f64
    %20 = arith.cmpi sle, %2, %c24_i64 : i64
    %21 = arith.cmpi sge, %2, %c1_i64 : i64
    %22 = arith.select %21, %2, %c1_i64 {fastmathFlags = #llvm.fastmath<none>} : i64
    %23 = arith.select %20, %22, %c24_i64 {fastmathFlags = #llvm.fastmath<none>} : i64
    %24 = arith.cmpi sle, %3, %c24_i64 : i64
    %25 = arith.cmpi sge, %3, %c1_i64 : i64
    %26 = arith.select %25, %3, %c1_i64 {fastmathFlags = #llvm.fastmath<none>} : i64
    %27 = arith.select %24, %26, %c24_i64 {fastmathFlags = #llvm.fastmath<none>} : i64
    %28 = arith.subf %cst, %11 {fastmathFlags = #llvm.fastmath<none>} : f64
    %29 = arith.subf %cst, %19 {fastmathFlags = #llvm.fastmath<none>} : f64
    %30 = arith.mulf %28, %29 {fastmathFlags = #llvm.fastmath<none>} : f64
    %31 = arith.addi %12, %c2_i64 : i64
    %32 = arith.muli %31, %c366_i64 : i64
    %33 = arith.addi %23, %c-1_i64 : i64
    %34 = arith.muli %33, %c68076_i64 : i64
    %35 = arith.index_cast %23 : i64 to index
    %45 = memref.load %arg16[%35, %c0, %c0, %c0] : memref<24x1x186x366xf64, 1>
    "test.use"(%45) : (f64) -> ()
  }
  return
}
