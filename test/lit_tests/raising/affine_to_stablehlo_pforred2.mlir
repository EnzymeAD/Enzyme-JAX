// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --arith-raise --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

func.func private @"##call__Z29gpu__compute_barotropic_mode_16CompilerMetadataI10StaticSizeI9_192__96_E12DynamicCheckvv7NDRangeILi2ES0_I7_12__6_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E13_206__140__1_EESC_vvvES8_ISA_S9_vvvvSF_SC_vvvE20ImmersedBoundaryGridISC_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISC_SJ_SK_SL_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_35__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EESP_SR_E8TripolarI5Int64SU_SU_ESB_ISC_Li2ESD_ISC_Li2ELi1E10_206__110_EESX_SX_SX_vSC_E16GridFittedBottomIS8_ISA_SA_vvvvSB_ISC_Li3ESD_ISC_Li3ELi1E13_206__110__1_EESC_vvvE23CenterImmersedConditionEvvvEvSB_ISC_Li3ESD_ISC_Li3ELi1E14_206__110__34_EES17_SF_#525$par125"(%arg0: memref<1x140x206xf64, 1>, %arg1: memref<1x140x206xf64, 1>, %arg2: memref<35xf64, 1>, %arg3: memref<34xf64, 1>, %arg4: memref<1x110x206xf64, 1>, %arg5: memref<34x110x206xf64, 1>, %arg6: memref<34x110x206xf64, 1>) {
  %cst = arith.constant 0x7FF8000000000000 : f64
  %cst_0 = arith.constant 0.000000e+00 : f64
  %cst_1 = arith.constant 1.000000e+00 : f64
  affine.parallel (%arg7, %arg8) = (0, 0) to (96, 192) {
    %0 = affine.load %arg2[27] {alignment = 8 : i64, ordering = 0 : i64} : memref<35xf64, 1>
    %1 = affine.load %arg4[0, %arg7 + 7, %arg8 + 6] : memref<1x110x206xf64, 1>
    %2 = arith.subf %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
    %3 = affine.load %arg4[0, %arg7 + 7, %arg8 + 7] : memref<1x110x206xf64, 1>
    %4 = arith.subf %0, %3 {fastmathFlags = #llvm.fastmath<none>} : f64
    %5 = math.isnan %2 : f64
    %6 = math.isnan %4 : f64
    %7 = arith.ori %5, %6 : i1
    %8 = arith.minnumf %2, %4 : f64
    %9 = arith.select %7, %cst, %8 : f64
    %10 = affine.load %arg4[0, %arg7 + 6, %arg8 + 7] : memref<1x110x206xf64, 1>
    %11 = arith.subf %0, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
    %12 = math.isnan %11 : f64
    %13 = arith.ori %12, %6 : i1
    %14 = arith.minnumf %11, %4 : f64
    %15 = arith.select %13, %cst, %14 : f64
    %16 = arith.cmpf oeq, %9, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
    %17 = arith.divf %9, %9 {fastmathFlags = #llvm.fastmath<none>} : f64
    %18 = arith.select %16, %cst_1, %17 : f64
    %19 = arith.cmpf oeq, %15, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
    %20 = arith.divf %15, %15 {fastmathFlags = #llvm.fastmath<none>} : f64
    %21 = arith.select %19, %cst_1, %20 : f64
    %22 = affine.load %arg3[7] {alignment = 8 : i64, ordering = 0 : i64} : memref<34xf64, 1>
    %23 = affine.load %arg5[7, %arg7 + 7, %arg8 + 7] : memref<34x110x206xf64, 1>
    %24 = arith.mulf %22, %23 {fastmathFlags = #llvm.fastmath<none>} : f64
    %25 = arith.mulf %18, %24 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %25, %arg0[0, %arg7 + 22, %arg8 + 7] : memref<1x140x206xf64, 1>
    %26 = affine.load %arg3[7] {alignment = 8 : i64, ordering = 0 : i64} : memref<34xf64, 1>
    %27 = affine.load %arg6[7, %arg7 + 7, %arg8 + 7] : memref<34x110x206xf64, 1>
    %28 = arith.mulf %26, %27 {fastmathFlags = #llvm.fastmath<none>} : f64
    %29 = arith.mulf %21, %28 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %29, %arg1[0, %arg7 + 22, %arg8 + 7] : memref<1x140x206xf64, 1>
    %30 = affine.load %arg0[0, %arg7 + 22, %arg8 + 7] : memref<1x140x206xf64, 1>
    %31 = affine.load %arg1[0, %arg7 + 22, %arg8 + 7] : memref<1x140x206xf64, 1>
    %32:8 = affine.parallel (%arg9) = (0) to (19) reduce ("addf", "addf", "andi", "xori", "maximumf", "maxnumf", "minimumf", "minnumf") -> (f64, f64, i1, i1, f64, f64, f64, f64) {
      %35 = affine.load %arg3[%arg9 + 8] {alignment = 8 : i64, ordering = 0 : i64} : memref<34xf64, 1>
      %36 = affine.load %arg5[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<34x110x206xf64, 1>
      %37 = arith.mulf %35, %36 {fastmathFlags = #llvm.fastmath<none>} : f64
      %38 = arith.mulf %18, %37 {fastmathFlags = #llvm.fastmath<none>} : f64
      %39 = affine.load %arg3[%arg9 + 8] {alignment = 8 : i64, ordering = 0 : i64} : memref<34xf64, 1>
      %40 = affine.load %arg6[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<34x110x206xf64, 1>
      %41 = arith.mulf %39, %40 {fastmathFlags = #llvm.fastmath<none>} : f64
      %42 = arith.mulf %21, %41 {fastmathFlags = #llvm.fastmath<none>} : f64
      %43 = arith.cmpf ogt, %38, %cst_0 : f64
      %44 = arith.cmpf olt, %42, %cst_0 : f64
      affine.yield %38, %42, %43, %44, %38, %42, %38, %42 : f64, f64, i1, i1, f64, f64, f64, f64
    }
    %33 = arith.addf %30, %32#0 {fastmathFlags = #llvm.fastmath<none>} : f64
    %34 = arith.addf %31, %32#1 {fastmathFlags = #llvm.fastmath<none>} : f64
    %45 = arith.select %32#2, %32#4, %32#6 : f64
    %46 = arith.select %32#3, %32#5, %32#7 : f64
    %47 = arith.addf %33, %45 {fastmathFlags = #llvm.fastmath<none>} : f64
    %48 = arith.addf %34, %46 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %48, %arg1[0, %arg7 + 22, %arg8 + 7] : memref<1x140x206xf64, 1>
    affine.store %47, %arg0[0, %arg7 + 22, %arg8 + 7] : memref<1x140x206xf64, 1>
  }
  return
}

// CHECK:  func.func private @"##call__Z29gpu__compute_barotropic_mode_16CompilerMetadataI10StaticSizeI9_192__96_E12DynamicCheckvv7NDRangeILi2ES0_I7_12__6_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E13_206__140__1_EESC_vvvES8_ISA_S9_vvvvSF_SC_vvvE20ImmersedBoundaryGridISC_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISC_SJ_SK_SL_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_35__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EESP_SR_E8TripolarI5Int64SU_SU_ESB_ISC_Li2ESD_ISC_Li2ELi1E10_206__110_EESX_SX_SX_vSC_E16GridFittedBottomIS8_ISA_SA_vvvvSB_ISC_Li3ESD_ISC_Li3ELi1E13_206__110__1_EESC_vvvE23CenterImmersedConditionEvvvEvSB_ISC_Li3ESD_ISC_Li3ELi1E14_206__110__34_EES17_SF_#525$par125_raised"(%arg0: tensor<1x140x206xf64>, %arg1: tensor<1x140x206xf64>, %arg2: tensor<35xf64>, %arg3: tensor<34xf64>, %arg4: tensor<1x110x206xf64>, %arg5: tensor<34x110x206xf64>, %arg6: tensor<34x110x206xf64>) -> (tensor<1x140x206xf64>, tensor<1x140x206xf64>, tensor<35xf64>, tensor<34xf64>, tensor<1x110x206xf64>, tensor<34x110x206xf64>, tensor<34x110x206xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<96x192x19xf64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<96x192xf64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<96x192xf64>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<0x7FF8000000000000> : tensor<96x192xf64>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
// CHECK-NEXT:    %cst_4 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
// CHECK-NEXT:    %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<true> : tensor<i1>
// CHECK-NEXT:    %c_6 = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_7 = stablehlo.constant dense<22> : tensor<i64>
// CHECK-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %[[a0:.+]] = stablehlo.slice %arg2 [27:28] : (tensor<35xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[a1:.+]] = stablehlo.slice %arg4 [0:1, 7:103, 6:198] : (tensor<1x110x206xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a2:.+]] = stablehlo.reshape %[[a1]] : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.broadcast_in_dim %[[a0]], dims = [0] : (tensor<1xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.subtract %[[a3]], %[[a2]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a5:.+]] = stablehlo.slice %arg4 [0:1, 7:103, 7:199] : (tensor<1x110x206xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.reshape %[[a5]] : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.subtract %[[a3]], %[[a6]] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a8:.+]] = stablehlo.is_finite %[[a4]] : (tensor<96x192xf64>) -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a9:.+]] = stablehlo.not %[[a8]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a10:.+]] = chlo.is_inf %[[a4]] : tensor<96x192xf64> -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a11:.+]] = stablehlo.not %[[a10]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a12:.+]] = stablehlo.and %[[a9]], %[[a11]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a13:.+]] = stablehlo.is_finite %[[a7]] : (tensor<96x192xf64>) -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.not %[[a13]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a15:.+]] = chlo.is_inf %[[a7]] : tensor<96x192xf64> -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a16:.+]] = stablehlo.not %[[a15]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a17:.+]] = stablehlo.and %[[a14]], %[[a16]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a18:.+]] = stablehlo.or %[[a12]], %[[a17]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a19:.+]] = stablehlo.is_finite %[[a4]] : (tensor<96x192xf64>) -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a20:.+]] = stablehlo.not %[[a19]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a21:.+]] = chlo.is_inf %[[a4]] : tensor<96x192xf64> -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a22:.+]] = stablehlo.not %[[a21]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a23:.+]] = stablehlo.and %[[a20]], %[[a22]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a24:.+]] = stablehlo.minimum %[[a4]], %[[a7]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a25:.+]] = stablehlo.select %[[a23]], %[[a7]], %[[a24]] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a26:.+]] = stablehlo.select %[[a18]], %cst_2, %[[a25]] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a27:.+]] = stablehlo.slice %arg4 [0:1, 6:102, 7:199] : (tensor<1x110x206xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a28:.+]] = stablehlo.reshape %[[a27]] : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a29:.+]] = stablehlo.subtract %[[a3]], %[[a28]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a30:.+]] = stablehlo.is_finite %[[a29]] : (tensor<96x192xf64>) -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a31:.+]] = stablehlo.not %[[a30]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a32:.+]] = chlo.is_inf %[[a29]] : tensor<96x192xf64> -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a33:.+]] = stablehlo.not %[[a32]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a34:.+]] = stablehlo.and %[[a31]], %[[a33]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a35:.+]] = stablehlo.or %[[a34]], %[[a17]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a36:.+]] = stablehlo.is_finite %[[a29]] : (tensor<96x192xf64>) -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a37:.+]] = stablehlo.not %[[a36]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a38:.+]] = chlo.is_inf %[[a29]] : tensor<96x192xf64> -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a39:.+]] = stablehlo.not %[[a38]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a40:.+]] = stablehlo.and %[[a37]], %[[a39]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a41:.+]] = stablehlo.minimum %[[a29]], %[[a7]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a42:.+]] = stablehlo.select %[[a40]], %[[a7]], %[[a41]] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a43:.+]] = stablehlo.select %[[a35]], %cst_2, %[[a42]] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a44:.+]] = stablehlo.compare EQ, %[[a26]], %cst_1, FLOAT : (tensor<96x192xf64>, tensor<96x192xf64>) -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a45:.+]] = stablehlo.divide %[[a26]], %[[a26]] {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a46:.+]] = stablehlo.select %[[a44]], %cst_0, %[[a45]] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a47:.+]] = stablehlo.compare EQ, %[[a43]], %cst_1, FLOAT : (tensor<96x192xf64>, tensor<96x192xf64>) -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a48:.+]] = stablehlo.divide %[[a43]], %[[a43]] {enzymexla.no_nan = [#enzymexla<guaranteed NOTGUARANTEED>], enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a49:.+]] = stablehlo.select %[[a47]], %cst_0, %[[a48]] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a50:.+]] = stablehlo.slice %arg3 [7:8] : (tensor<34xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[a51:.+]] = stablehlo.slice %arg5 [7:8, 7:103, 7:199] : (tensor<34x110x206xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a52:.+]] = stablehlo.reshape %[[a51]] : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a53:.+]] = stablehlo.broadcast_in_dim %[[a50]], dims = [0] : (tensor<1xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a54:.+]] = stablehlo.multiply %[[a53]], %[[a52]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a55:.+]] = stablehlo.multiply %[[a46]], %[[a54]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a56:.+]] = stablehlo.slice %arg6 [7:8, 7:103, 7:199] : (tensor<34x110x206xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a57:.+]] = stablehlo.reshape %[[a56]] : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a58:.+]] = stablehlo.multiply %[[a53]], %[[a57]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a59:.+]] = stablehlo.multiply %[[a49]], %[[a58]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a60:.+]] = stablehlo.slice %arg3 [8:27] : (tensor<34xf64>) -> tensor<19xf64>
// CHECK-NEXT:    %[[a61:.+]] = stablehlo.slice %arg5 [8:27, 7:103, 7:199] : (tensor<34x110x206xf64>) -> tensor<19x96x192xf64>
// CHECK-NEXT:    %[[a62:.+]] = stablehlo.broadcast_in_dim %[[a60]], dims = [0] : (tensor<19xf64>) -> tensor<19x96x192xf64>
// CHECK-NEXT:    %[[a63:.+]] = stablehlo.multiply %[[a62]], %[[a61]] : tensor<19x96x192xf64>
// CHECK-NEXT:    %[[a64:.+]] = stablehlo.broadcast_in_dim %[[a46]], dims = [0, 1] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<96x192xf64>) -> tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a65:.+]] = stablehlo.transpose %[[a63]], dims = [1, 2, 0] : (tensor<19x96x192xf64>) -> tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a66:.+]] = stablehlo.multiply %[[a64]], %[[a65]] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a67:.+]] = stablehlo.slice %arg6 [8:27, 7:103, 7:199] : (tensor<34x110x206xf64>) -> tensor<19x96x192xf64>
// CHECK-NEXT:    %[[a68:.+]] = stablehlo.multiply %[[a62]], %[[a67]] : tensor<19x96x192xf64>
// CHECK-NEXT:    %[[a69:.+]] = stablehlo.broadcast_in_dim %[[a49]], dims = [0, 1] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : (tensor<96x192xf64>) -> tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a70:.+]] = stablehlo.transpose %[[a68]], dims = [1, 2, 0] : (tensor<19x96x192xf64>) -> tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a71:.+]] = stablehlo.multiply %[[a69]], %[[a70]] {enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a72:.+]] = stablehlo.compare GT, %[[a66]], %cst, FLOAT : (tensor<96x192x19xf64>, tensor<96x192x19xf64>) -> tensor<96x192x19xi1>
// CHECK-NEXT:    %[[a73:.+]] = stablehlo.compare LT, %[[a71]], %cst, FLOAT : (tensor<96x192x19xf64>, tensor<96x192x19xf64>) -> tensor<96x192x19xi1>
// CHECK-NEXT:    %[[a74:.+]] = stablehlo.reduce(%[[a66]] init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<96x192x19xf64>, tensor<f64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a75:.+]] = stablehlo.reduce(%[[a71]] init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<96x192x19xf64>, tensor<f64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a76:.+]] = stablehlo.reduce(%[[a72]] init: %c_5) applies stablehlo.and across dimensions = [2] : (tensor<96x192x19xi1>, tensor<i1>) -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a77:.+]] = stablehlo.reduce(%[[a73]] init: %c) applies stablehlo.xor across dimensions = [2] : (tensor<96x192x19xi1>, tensor<i1>) -> tensor<96x192xi1>
// CHECK-NEXT:    %[[a78:.+]] = stablehlo.reduce(%[[a66]] init: %cst_4) applies stablehlo.maximum across dimensions = [2] : (tensor<96x192x19xf64>, tensor<f64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a79:.+]] = stablehlo.reduce(%[[a71]] init: %cst_4) across dimensions = [2] : (tensor<96x192x19xf64>, tensor<f64>) -> tensor<96x192xf64>
// CHECK-NEXT:    reducer(%arg7: tensor<f64>, %arg8: tensor<f64>)  {
// CHECK-NEXT:    %[[a92:.+]] = stablehlo.is_finite %arg7 : (tensor<f64>) -> tensor<i1>
// CHECK-NEXT:    %[[a93:.+]] = stablehlo.not %[[a92]] : tensor<i1>
// CHECK-NEXT:    %[[a94:.+]] = chlo.is_inf %arg7 : tensor<f64> -> tensor<i1>
// CHECK-NEXT:    %[[a95:.+]] = stablehlo.not %[[a94]] : tensor<i1>
// CHECK-NEXT:    %[[a96:.+]] = stablehlo.and %[[a93]], %[[a95]] : tensor<i1>
// CHECK-NEXT:    %[[a97:.+]] = stablehlo.maximum %arg7, %arg8 : tensor<f64>
// CHECK-NEXT:    %[[a98:.+]] = stablehlo.select %[[a96]], %arg8, %[[a97]] : tensor<i1>, tensor<f64>
// CHECK-NEXT:    stablehlo.return %[[a98]] : tensor<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a80:.+]] = stablehlo.reduce(%[[a66]] init: %cst_3) applies stablehlo.minimum across dimensions = [2] : (tensor<96x192x19xf64>, tensor<f64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a81:.+]] = stablehlo.reduce(%[[a71]] init: %cst_3) across dimensions = [2] : (tensor<96x192x19xf64>, tensor<f64>) -> tensor<96x192xf64>
// CHECK-NEXT:    reducer(%arg7: tensor<f64>, %arg8: tensor<f64>)  {
// CHECK-NEXT:    %[[a92:.+]] = stablehlo.is_finite %arg7 : (tensor<f64>) -> tensor<i1>
// CHECK-NEXT:    %[[a93:.+]] = stablehlo.not %[[a92]] : tensor<i1>
// CHECK-NEXT:    %[[a94:.+]] = chlo.is_inf %arg7 : tensor<f64> -> tensor<i1>
// CHECK-NEXT:    %[[a95:.+]] = stablehlo.not %[[a94]] : tensor<i1>
// CHECK-NEXT:    %[[a96:.+]] = stablehlo.and %[[a93]], %[[a95]] : tensor<i1>
// CHECK-NEXT:    %[[a97:.+]] = stablehlo.minimum %arg7, %arg8 : tensor<f64>
// CHECK-NEXT:    %[[a98:.+]] = stablehlo.select %[[a96]], %arg8, %[[a97]] : tensor<i1>, tensor<f64>
// CHECK-NEXT:    stablehlo.return %[[a98]] : tensor<f64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[a82:.+]] = stablehlo.add %[[a55]], %[[a74]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a83:.+]] = stablehlo.add %[[a59]], %[[a75]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a84:.+]] = stablehlo.select %[[a76]], %[[a78]], %[[a80]] : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a85:.+]] = stablehlo.select %[[a77]], %[[a79]], %[[a81]] : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a86:.+]] = stablehlo.add %[[a82]], %[[a84]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a87:.+]] = stablehlo.add %[[a83]], %[[a85]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a88:.+]] = stablehlo.reshape %[[a87]] : (tensor<96x192xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a89:.+]] = stablehlo.dynamic_update_slice %arg1, %[[a88]], %c_8, %c_7, %c_6 : (tensor<1x140x206xf64>, tensor<1x96x192xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x140x206xf64>
// CHECK-NEXT:    %[[a90:.+]] = stablehlo.reshape %[[a86]] : (tensor<96x192xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a91:.+]] = stablehlo.dynamic_update_slice %arg0, %[[a90]], %c_8, %c_7, %c_6 : (tensor<1x140x206xf64>, tensor<1x96x192xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x140x206xf64>
// CHECK-NEXT:    return %[[a91]], %[[a89]], %arg2, %arg3, %arg4, %arg5, %arg6 : tensor<1x140x206xf64>, tensor<1x140x206xf64>, tensor<35xf64>, tensor<34xf64>, tensor<1x110x206xf64>, tensor<34x110x206xf64>, tensor<34x110x206xf64>
// CHECK-NEXT:    }

