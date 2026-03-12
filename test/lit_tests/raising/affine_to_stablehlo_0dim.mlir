// RUN: enzymexlamlir-opt --raise-affine-to-stablehlo %s | FileCheck %s

module @"reactant_run!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func private @"##call__Z25gpu__mask_immersed_field_16CompilerMetadataI10StaticSizeI11_16__16__4_E12DynamicCheckvv7NDRangeILi3ES0_I9_1__1__4_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E12_30__30__18_EE5TupleI4Face6CenterSF_E20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1E12StepRangeLenIS9_14TwicePrecisionIS9_ESP_5Int64EESS_S9_S9_E8TripolarISQ_SQ_SQ_ES8_IS9_Li2ESA_IS9_Li2ELi1E8_30__30_EESX_SX_SX_vS9_E16GridFittedBottomI5FieldISF_SF_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E11_30__30__1_EES9_vvvE23CenterImmersedConditionEvvvE15CuTracedRNumberIS9_Li1EE#259$par5"(%arg0: memref<18x30x30xf64, 1>, %arg1: memref<1x30x30xf64, 1>, %arg2: memref<f64, 1>) {
    %cst = arith.constant 2.500000e-01 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.250000e-01 : f64 
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (4, 16, 16) {
      %0 = arith.index_castui %arg3 : index to i64 
      %1 = arith.sitofp %0 : i64 to f64 
      %2 = arith.mulf %1, %cst {fastmathFlags = #llvm.fastmath<none>} : f64 
      %3 = arith.mulf %1, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64 
      %4 = math.absf %2 : f64 
      %5 = arith.cmpf olt, %cst_1, %4 {fastmathFlags = #llvm.fastmath<none>} : f64 
      %6 = arith.select %5, %2, %cst_1 : f64 
      %7 = arith.select %5, %cst_1, %2 : f64 
      %8 = arith.addf %6, %7 {fastmathFlags = #llvm.fastmath<none>} : f64
      %9 = arith.subf %6, %8 {fastmathFlags = #llvm.fastmath<none>} : f64 
      %10 = arith.addf %7, %9 {fastmathFlags = #llvm.fastmath<none>} : f64 
      %11 = arith.addf %3, %10 {fastmathFlags = #llvm.fastmath<none>} : f64 
      %12 = arith.addf %8, %11 {fastmathFlags = #llvm.fastmath<none>} : f64 
      %13 = affine.load %arg1[0, %arg4 + 7, %arg5 + 7] : memref<1x30x30xf64, 1> 
      %14 = arith.cmpf ole, %12, %13 {fastmathFlags = #llvm.fastmath<none>} : f64 
      %15 = affine.load %arg1[0, %arg4 + 7, %arg5 + 6] : memref<1x30x30xf64, 1> 
      %16 = arith.cmpf ole, %12, %15 {fastmathFlags = #llvm.fastmath<none>} : f64 
      %17 = arith.ori %14, %16 : i1 
      %18 = affine.load %arg0[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<18x30x30xf64, 1> 
      %19 = scf.if %17 -> (f64) {
        %20 = affine.load %arg2[] : memref<f64, 1>
        scf.yield %20 : f64
      } else {
        scf.yield %18 : f64
      }
      affine.store %19, %arg0[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<18x30x30xf64, 1>
    } 
    return
  } 
}

// CHECK: stablehlo.dynamic_update_slice