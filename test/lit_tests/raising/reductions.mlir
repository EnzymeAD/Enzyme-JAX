// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s
module {
  func.func private @"##call__Z44gpu_solve_batched_tridiagonal_system_kernel_16CompilerMetadataI10StaticSizeI8_45__20_E12DynamicCheckvv7NDRangeILi2ES0_I6_3__2_ES0_I8_16__16_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E12_59__34__24_EE40VerticallyImplicitDiffusionLowerDiagonal35VerticallyImplicitDiffusionDiagonal40VerticallyImplicitDiffusionUpperDiagonalSC_SA_IS9_Li3ELi1E12_45__20__10_E21LatitudeLongitudeGridIS9_8Periodic7BoundedSJ_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_25__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_24__EESM_SO_ES9_S9_S8_IS9_Li1E12StepRangeLenIS9_14TwicePrecisionIS9_ESS_5Int64EESV_S9_S9_SV_SV_S8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESX_SX_SX_S9_S9_vST_Ev5TupleI17ScalarDiffusivityI36VerticallyImplicitTimeDiscretization19VerticalFormulation10NamedTupleI8__T___S_SZ_IS9_S9_EES9_S15_Evv4Face6CenterS18_S13_I53__time___last__t___last_stage__t___iteration___stage_SZ_I13TracedRNumberIS9_ES9_S9_S19_IST_EST_EES9_5_z___E10ZDirection#455$par103"(%arg0: memref<24x34x59xf64, 1>, %arg1: memref<10x20x45xf64, 1>, %arg2: memref<25xf64, 1>, %arg3: memref<24xf64, 1>) {
    %cst = arith.constant -6.000000e-03 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 2.2204460492503131E-15 : f64
    affine.parallel (%arg4, %arg5) = (0, 0) to (20, 45) {
      affine.for %arg6 = 0 to 9 {
        %8 = affine.load %arg0[-%arg6 + 15, %arg4 + 7, %arg5 + 7] : memref<24x34x59xf64, 1>
        %9 = affine.load %arg1[-%arg6 + 9, %arg4, %arg5] : memref<10x20x45xf64, 1>
        %10 = affine.load %arg0[-%arg6 + 16, %arg4 + 7, %arg5 + 7] : memref<24x34x59xf64, 1>
        %11 = arith.mulf %9, %10 {fastmathFlags = #llvm.fastmath<none>} : f64
        %12 = arith.subf %8, %11 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %12, %arg0[-%arg6 + 15, %arg4 + 7, %arg5 + 7] : memref<24x34x59xf64, 1>
      }
    }
    return
  }
}

// CHECK:    affine.parallel (%arg4, %arg5) = (0, 0) to (20, 45) {
// CHECK-NEXT:      %0 = affine.load %arg0[16, %arg4 + 7, %arg5 + 7] : memref<24x34x59xf64, 1>
// CHECK-NEXT:      %1 = affine.for %arg6 = 0 to 9 iter_args(%arg7 = %0) -> (f64) {
// CHECK-NEXT:        %2 = affine.load %arg0[-%arg6 + 15, %arg4 + 7, %arg5 + 7] : memref<24x34x59xf64, 1>
// CHECK-NEXT:        %3 = affine.load %arg1[-%arg6 + 9, %arg4, %arg5] : memref<10x20x45xf64, 1>
// CHECK-NEXT:        %4 = arith.mulf %3, %arg7 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %5 = arith.subf %2, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        affine.store %5, %arg0[-%arg6 + 15, %arg4 + 7, %arg5 + 7] : memref<24x34x59xf64, 1>
// CHECK-NEXT:        affine.yield %5 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:    }
