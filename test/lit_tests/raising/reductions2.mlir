// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {
  func.func private @"##call__Z44gpu_solve_batched_tridiagonal_system_kernel_16CompilerMetadataI10StaticSizeI8_45__20_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E12_59__35__24_EE40VerticallyImplicitDiffusionLowerDiagonal35VerticallyImplicitDiffusionDiagonal40VerticallyImplicitDiffusionUpperDiagonalSC_SA_IS9_Li3ELi1E12_45__20__10_E21LatitudeLongitudeGridIS9_8Periodic7BoundedSJ_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_25__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_24__EESM_SO_ES9_S9_S8_IS9_Li1E12StepRangeLenIS9_14TwicePrecisionIS9_ESS_5Int64EESV_S9_S9_SV_SV_S8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESX_SX_SX_S9_S9_vST_Ev5TupleI17ScalarDiffusivityI36VerticallyImplicitTimeDiscretization19VerticalFormulation10NamedTupleI8__T___S_SZ_IS9_S9_EES9_S15_Evv6Center4FaceS17_S13_I53__time___last__t___last_stage__t___iteration___stage_SZ_I13TracedRNumberIS9_ES9_S9_S19_IST_EST_EES9_5_z___E10ZDirection#459$par105"(%arg0: memref<24x35x59xf64, 1>, %arg1: memref<10x20x45xf64, 1>, %arg2: memref<25xf64, 1>, %arg3: memref<24xf64, 1>) {
    %cst = arith.constant -6.000000e-03 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 2.2204460492503131E-15 : f64
    affine.parallel (%arg4, %arg5) = (0, 0) to (20, 45) {
      %0 = affine.load %arg3[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
      %1 = affine.load %arg2[9] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<25xf64, 1>
      %2 = arith.mulf %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
      %3 = arith.divf %cst, %2 {fastmathFlags = #llvm.fastmath<none>} : f64
      %4 = arith.subf %cst_0, %3 {fastmathFlags = #llvm.fastmath<none>} : f64
      %5 = affine.load %arg0[7, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
      %6 = arith.divf %5, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
      affine.store %6, %arg0[7, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
      %7 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %4) -> (f64) {
        %8 = affine.load %arg3[%arg6 + 7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
        %9 = affine.load %arg2[%arg6 + 9] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<25xf64, 1>
        %10 = arith.mulf %8, %9 {fastmathFlags = #llvm.fastmath<none>} : f64
        %11 = arith.divf %cst, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
        %12 = affine.load %arg3[%arg6 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
        %13 = affine.load %arg2[%arg6 + 10] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<25xf64, 1>
        %14 = arith.mulf %12, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
        %15 = arith.divf %cst, %14 {fastmathFlags = #llvm.fastmath<none>} : f64
        %16 = arith.subf %cst_0, %15 {fastmathFlags = #llvm.fastmath<none>} : f64
        %17 = arith.mulf %12, %9 {fastmathFlags = #llvm.fastmath<none>} : f64
        %18 = arith.divf %cst, %17 {fastmathFlags = #llvm.fastmath<none>} : f64
        %19 = arith.subf %16, %18 {fastmathFlags = #llvm.fastmath<none>} : f64
        %20 = arith.divf %11, %arg7 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %20, %arg1[%arg6 + 1, %arg4, %arg5] : memref<10x20x45xf64, 1>
        %21 = arith.mulf %20, %18 {fastmathFlags = #llvm.fastmath<none>} : f64
        %22 = arith.subf %19, %21 {fastmathFlags = #llvm.fastmath<none>} : f64
        %23 = affine.load %arg0[%arg6 + 8, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
        %24 = math.absf %22 : f64
        %25 = arith.cmpf olt, %cst_1, %24 {fastmathFlags = #llvm.fastmath<none>} : f64
        %26 = affine.load %arg0[%arg6 + 7, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
        %27 = arith.mulf %18, %26 {fastmathFlags = #llvm.fastmath<none>} : f64
        %28 = arith.subf %23, %27 {fastmathFlags = #llvm.fastmath<none>} : f64
        %29 = arith.divf %28, %22 {fastmathFlags = #llvm.fastmath<none>} : f64
        %30 = arith.select %25, %29, %23 : f64
        affine.store %30, %arg0[%arg6 + 8, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
        affine.yield %22 : f64
      }
      affine.for %arg6 = 0 to 9 {
        %8 = affine.load %arg0[-%arg6 + 15, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
        %9 = affine.load %arg1[-%arg6 + 9, %arg4, %arg5] : memref<10x20x45xf64, 1>
        %10 = affine.load %arg0[-%arg6 + 16, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
        %11 = arith.mulf %9, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
        %12 = arith.subf %8, %11 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %12, %arg0[-%arg6 + 15, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
      }
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z44gpu_solve_batched_tridiagonal_system_kernel_16CompilerMetadataI10StaticSizeI8_45__20_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E12_59__35__24_EE40VerticallyImplicitDiffusionLowerDiagonal35VerticallyImplicitDiffusionDiagonal40VerticallyImplicitDiffusionUpperDiagonalSC_SA_IS9_Li3ELi1E12_45__20__10_E21LatitudeLongitudeGridIS9_8Periodic7BoundedSJ_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_25__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_24__EESM_SO_ES9_S9_S8_IS9_Li1E12StepRangeLenIS9_14TwicePrecisionIS9_ESS_5Int64EESV_S9_S9_SV_SV_S8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESX_SX_SX_S9_S9_vST_Ev5TupleI17ScalarDiffusivityI36VerticallyImplicitTimeDiscretization19VerticalFormulation10NamedTupleI8__T___S_SZ_IS9_S9_EES9_S15_Evv6Center4FaceS17_S13_I53__time___last__t___last_stage__t___iteration___stage_SZ_I13TracedRNumberIS9_ES9_S9_S19_IST_EST_EES9_5_z___E10ZDirection#459$par105"(%arg0: memref<24x35x59xf64, 1>, %arg1: memref<10x20x45xf64, 1>, %arg2: memref<25xf64, 1>, %arg3: memref<24xf64, 1>) {
// CHECK-NEXT:    %cst = arith.constant -6.000000e-03 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %cst_1 = arith.constant 2.2204460492503131E-15 : f64
// CHECK-NEXT:    affine.parallel (%arg4, %arg5) = (0, 0) to (20, 45) {
// CHECK-NEXT:      %0 = affine.load %arg3[7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
// CHECK-NEXT:      %1 = affine.load %arg2[9] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<25xf64, 1>
// CHECK-NEXT:      %2 = arith.mulf %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %3 = arith.divf %cst, %2 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %4 = arith.subf %cst_0, %3 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %5 = affine.load %arg0[7, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
// CHECK-NEXT:      %6 = arith.divf %5, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %6, %arg0[7, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
// CHECK-NEXT:      %7 = affine.load %arg0[7, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
// CHECK-NEXT:      %8:2 = affine.for %arg6 = 0 to 8 iter_args(%arg7 = %4, %arg8 = %7) -> (f64, f64) {
// CHECK-NEXT:        %11 = affine.load %arg3[%arg6 + 7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
// CHECK-NEXT:        %12 = affine.load %arg2[%arg6 + 9] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<25xf64, 1>
// CHECK-NEXT:        %13 = arith.mulf %11, %12 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %14 = arith.divf %cst, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %15 = affine.load %arg3[%arg6 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<24xf64, 1>
// CHECK-NEXT:        %16 = affine.load %arg2[%arg6 + 10] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<25xf64, 1>
// CHECK-NEXT:        %17 = arith.mulf %15, %16 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %18 = arith.divf %cst, %17 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %19 = arith.subf %cst_0, %18 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %20 = arith.mulf %15, %12 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %21 = arith.divf %cst, %20 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %22 = arith.subf %19, %21 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %23 = arith.divf %14, %arg7 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        affine.store %23, %arg1[%arg6 + 1, %arg4, %arg5] : memref<10x20x45xf64, 1>
// CHECK-NEXT:        %24 = arith.mulf %23, %21 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %25 = arith.subf %22, %24 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %26 = affine.load %arg0[%arg6 + 8, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
// CHECK-NEXT:        %27 = math.absf %25 : f64
// CHECK-NEXT:        %28 = arith.cmpf olt, %cst_1, %27 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %29 = arith.mulf %21, %arg8 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %30 = arith.subf %26, %29 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %31 = arith.divf %30, %25 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %32 = arith.select %28, %31, %26 : f64
// CHECK-NEXT:        affine.store %32, %arg0[%arg6 + 8, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
// CHECK-NEXT:        affine.yield %25, %32 : f64, f64
// CHECK-NEXT:      }
// CHECK-NEXT:      %9 = affine.load %arg0[16, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
// CHECK-NEXT:      %10 = affine.for %arg6 = 0 to 9 iter_args(%arg7 = %9) -> (f64) {
// CHECK-NEXT:        %11 = affine.load %arg0[-%arg6 + 15, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
// CHECK-NEXT:        %12 = affine.load %arg1[-%arg6 + 9, %arg4, %arg5] : memref<10x20x45xf64, 1>
// CHECK-NEXT:        %13 = arith.mulf %12, %arg7 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %14 = arith.subf %11, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        affine.store %14, %arg0[-%arg6 + 15, %arg4 + 7, %arg5 + 7] : memref<24x35x59xf64, 1>
// CHECK-NEXT:        affine.yield %14 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
