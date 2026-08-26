// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --canonicalize --enzyme-hlo-opt=max_constant_expansion=0 | FileCheck %s

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
    %32:2 = affine.parallel (%arg9) = (0) to (19) reduce ("addf", "addf") -> (f64, f64) {
      %35 = affine.load %arg3[%arg9 + 8] {alignment = 8 : i64, ordering = 0 : i64} : memref<34xf64, 1>
      %36 = affine.load %arg5[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<34x110x206xf64, 1>
      %37 = arith.mulf %35, %36 {fastmathFlags = #llvm.fastmath<none>} : f64
      %38 = arith.mulf %18, %37 {fastmathFlags = #llvm.fastmath<none>} : f64
      %39 = affine.load %arg3[%arg9 + 8] {alignment = 8 : i64, ordering = 0 : i64} : memref<34xf64, 1>
      %40 = affine.load %arg6[%arg9 + 8, %arg7 + 7, %arg8 + 7] : memref<34x110x206xf64, 1>
      %41 = arith.mulf %39, %40 {fastmathFlags = #llvm.fastmath<none>} : f64
      %42 = arith.mulf %21, %41 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.yield %38, %42 : f64, f64
    }
    %33 = arith.addf %30, %32#0 {fastmathFlags = #llvm.fastmath<none>} : f64
    %34 = arith.addf %31, %32#1 {fastmathFlags = #llvm.fastmath<none>} : f64
    affine.store %34, %arg1[0, %arg7 + 22, %arg8 + 7] : memref<1x140x206xf64, 1>
    affine.store %33, %arg0[0, %arg7 + 22, %arg8 + 7] : memref<1x140x206xf64, 1>
  }
  return
}

// CHECK:  func.func private @"##call__Z29gpu__compute_barotropic_mode_16CompilerMetadataI10StaticSizeI9_192__96_E12DynamicCheckvv7NDRangeILi2ES0_I7_12__6_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E13_206__140__1_EESC_vvvES8_ISA_S9_vvvvSF_SC_vvvE20ImmersedBoundaryGridISC_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISC_SJ_SK_SL_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_35__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EESP_SR_E8TripolarI5Int64SU_SU_ESB_ISC_Li2ESD_ISC_Li2ELi1E10_206__110_EESX_SX_SX_vSC_E16GridFittedBottomIS8_ISA_SA_vvvvSB_ISC_Li3ESD_ISC_Li3ELi1E13_206__110__1_EESC_vvvE23CenterImmersedConditionEvvvEvSB_ISC_Li3ESD_ISC_Li3ELi1E14_206__110__34_EES17_SF_#525$par125_raised"(%arg0: tensor<1x140x206xf64>, %arg1: tensor<1x140x206xf64>, %arg2: tensor<35xf64>, %arg3: tensor<34xf64>, %arg4: tensor<1x110x206xf64>, %arg5: tensor<34x110x206xf64>, %arg6: tensor<34x110x206xf64>) -> (tensor<1x140x206xf64>, tensor<1x140x206xf64>, tensor<35xf64>, tensor<34xf64>, tensor<1x110x206xf64>, tensor<34x110x206xf64>, tensor<34x110x206xf64>) {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<96x192xf64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<96x192xf64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0x7FF8000000000000> : tensor<96x192xf64>
// CHECK-NEXT:    %c = stablehlo.constant dense<7> : tensor<i64>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<22> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:    %0 = stablehlo.slice %arg2 [27:28] : (tensor<35xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[a2:.+]] = stablehlo.slice %arg4 [0:1, 7:103, 6:198] : (tensor<1x110x206xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a3:.+]] = stablehlo.reshape %[[a2]] : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a4:.+]] = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a5:.+]] = arith.subf %[[a4]], %[[a3]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a6:.+]] = stablehlo.slice %arg4 [0:1, 7:103, 7:199] : (tensor<1x110x206xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a7:.+]] = stablehlo.reshape %[[a6]] : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a8:.+]] = arith.subf %[[a4]], %[[a7]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a9:.+]] = math.isnan %[[a5]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a10:.+]] = math.isnan %[[a8]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a11:.+]] = arith.ori %[[a9]], %[[a10]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a12:.+]] = arith.minnumf %[[a5]], %[[a8]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a13:.+]] = arith.select %[[a11]], %cst_1, %[[a12]] : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a14:.+]] = stablehlo.slice %arg4 [0:1, 6:102, 7:199] : (tensor<1x110x206xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a15:.+]] = stablehlo.reshape %[[a14]] : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a16:.+]] = arith.subf %[[a4]], %[[a15]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a17:.+]] = math.isnan %[[a16]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a18:.+]] = arith.ori %[[a17]], %[[a10]] : tensor<96x192xi1>
// CHECK-NEXT:    %[[a19:.+]] = arith.minnumf %[[a16]], %[[a8]] : tensor<96x192xf64>
// CHECK-NEXT:    %[[a20:.+]] = arith.select %[[a18]], %cst_1, %[[a19]] : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a21:.+]] = arith.cmpf oeq, %[[a13]], %cst_0 {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a22:.+]] = arith.divf %[[a13]], %[[a13]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a23:.+]] = arith.select %[[a21]], %cst, %[[a22]] : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a24:.+]] = arith.cmpf oeq, %[[a20]], %cst_0 {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a25:.+]] = arith.divf %[[a20]], %[[a20]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a26:.+]] = arith.select %[[a24]], %cst, %[[a25]] : tensor<96x192xi1>, tensor<96x192xf64>
// CHECK-NEXT:    %[[a27:.+]] = stablehlo.slice %arg3 [7:8] : (tensor<34xf64>) -> tensor<1xf64>
// CHECK-NEXT:    %[[a29:.+]] = stablehlo.slice %arg5 [7:8, 7:103, 7:199] : (tensor<34x110x206xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a30:.+]] = stablehlo.reshape %[[a29]] : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a31:.+]] = stablehlo.broadcast_in_dim %[[a27]], dims = [0] : (tensor<1xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a32:.+]] = arith.mulf %[[a31]], %[[a30]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a33:.+]] = arith.mulf %[[a23]], %[[a32]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a34:.+]] = stablehlo.slice %arg6 [7:8, 7:103, 7:199] : (tensor<34x110x206xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a35:.+]] = stablehlo.reshape %[[a34]] : (tensor<1x96x192xf64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a36:.+]] = arith.mulf %[[a31]], %[[a35]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a37:.+]] = arith.mulf %[[a26]], %[[a36]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a38:.+]] = stablehlo.slice %arg3 [8:27] : (tensor<34xf64>) -> tensor<19xf64>
// CHECK-NEXT:    %[[a39:.+]] = stablehlo.slice %arg5 [8:27, 7:103, 7:199] : (tensor<34x110x206xf64>) -> tensor<19x96x192xf64>
// CHECK-NEXT:    %[[a40:.+]] = stablehlo.broadcast_in_dim %[[a38]], dims = [0] : (tensor<19xf64>) -> tensor<19x96x192xf64>
// CHECK-NEXT:    %[[a41:.+]] = arith.mulf %[[a40]], %[[a39]] {fastmathFlags = #llvm.fastmath<none>} : tensor<19x96x192xf64>
// CHECK-NEXT:    %[[a42:.+]] = stablehlo.broadcast_in_dim %[[a23]], dims = [0, 1] : (tensor<96x192xf64>) -> tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a43:.+]] = stablehlo.transpose %[[a41]], dims = [1, 2, 0] : (tensor<19x96x192xf64>) -> tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a44:.+]] = arith.mulf %[[a42]], %[[a43]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a45:.+]] = stablehlo.slice %arg6 [8:27, 7:103, 7:199] : (tensor<34x110x206xf64>) -> tensor<19x96x192xf64>
// CHECK-NEXT:    %[[a46:.+]] = arith.mulf %[[a40]], %[[a45]] {fastmathFlags = #llvm.fastmath<none>} : tensor<19x96x192xf64>
// CHECK-NEXT:    %[[a47:.+]] = stablehlo.broadcast_in_dim %[[a26]], dims = [0, 1] : (tensor<96x192xf64>) -> tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a48:.+]] = stablehlo.transpose %[[a46]], dims = [1, 2, 0] : (tensor<19x96x192xf64>) -> tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a49:.+]] = arith.mulf %[[a47]], %[[a48]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192x19xf64>
// CHECK-NEXT:    %[[a50:.+]] = stablehlo.reduce(%[[a44]] init: %cst_4) applies stablehlo.add across dimensions = [2] : (tensor<96x192x19xf64>, tensor<f64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a51:.+]] = stablehlo.reduce(%[[a49]] init: %cst_4) applies stablehlo.add across dimensions = [2] : (tensor<96x192x19xf64>, tensor<f64>) -> tensor<96x192xf64>
// CHECK-NEXT:    %[[a52:.+]] = arith.addf %[[a33]], %[[a50]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a53:.+]] = arith.addf %[[a37]], %[[a51]] {fastmathFlags = #llvm.fastmath<none>} : tensor<96x192xf64>
// CHECK-NEXT:    %[[a54:.+]] = stablehlo.reshape %[[a53]] : (tensor<96x192xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a55:.+]] = stablehlo.dynamic_update_slice %arg1, %[[a54]], %c_3, %c_2, %c : (tensor<1x140x206xf64>, tensor<1x96x192xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x140x206xf64>
// CHECK-NEXT:    %[[a56:.+]] = stablehlo.reshape %[[a52]] : (tensor<96x192xf64>) -> tensor<1x96x192xf64>
// CHECK-NEXT:    %[[a57:.+]] = stablehlo.dynamic_update_slice %arg0, %[[a56]], %c_3, %c_2, %c : (tensor<1x140x206xf64>, tensor<1x96x192xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x140x206xf64>
// CHECK-NEXT:    return %[[a57]], %[[a55]], %arg2, %arg3, %arg4, %arg5, %arg6 : tensor<1x140x206xf64>, tensor<1x140x206xf64>, tensor<35xf64>, tensor<34xf64>, tensor<1x110x206xf64>, tensor<34x110x206xf64>, tensor<34x110x206xf64>
// CHECK-NEXT:  }

