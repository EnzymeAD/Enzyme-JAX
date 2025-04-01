// RUN: enzymexlamlir-opt --sort-memory %s | FileCheck %s

module {
    func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_48__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_3__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float32Li3E13CuTracedArrayISF_Li3ELi1E11_62__38__1_EE17BoundaryConditionI4FluxvESL_S7_I4Face6CentervE21LatitudeLongitudeGridISF_8Periodic7BoundedSR_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_25__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_24__EESU_SW_ESF_SF_SE_ISF_Li1E12StepRangeLenISF_7Float64SZ_S8_EES11_SF_SF_S11_S11_SE_ISF_Li1ESG_ISF_Li1ELi1E5_38__EES13_S13_S13_SF_SF_vS8_ES7_IJEE#447$par99"(%arg0: memref<1x38x62xf32, 1>) {
    affine.parallel (%arg1) = (0) to (48) {
      %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x38x62xf32, 1>
      affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x38x62xf32, 1>
      %1 = affine.load %arg0[0, 30, %arg1 + 7] : memref<1x38x62xf32, 1>
      affine.store %1, %arg0[0, 31, %arg1 + 7] : memref<1x38x62xf32, 1>
    }
    return
  }
  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_48__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_3__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float32Li3E13CuTracedArrayISF_Li3ELi1E11_62__38__1_EE17BoundaryConditionI4FluxvESL_S7_I4Face6CentervE21LatitudeLongitudeGridISF_8Periodic7BoundedSR_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_25__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_24__EESU_SW_ESF_SF_SE_ISF_Li1E12StepRangeLenISF_7Float64SZ_S8_EES11_SF_SF_S11_S11_SE_ISF_Li1ESG_ISF_Li1ELi1E5_38__EES13_S13_S13_SF_SF_vS8_ES7_IJEE#447$par163"(%arg0: memref<1x38x62xf32, 1>) {
    affine.parallel (%arg1) = (0) to (48) {
      %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x38x62xf32, 1>
      affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x38x62xf32, 1>
      %1 = affine.load %arg0[0, 30, %arg1 + 7] : memref<1x38x62xf32, 1>
      affine.store %1, %arg0[0, 31, %arg1 + 7] : memref<1x38x62xf32, 1>
    }
    return
  }
  func.func private @"##call__Z37gpu_fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_38__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I7_38__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE13CuTracedArrayI7Float32Li3ELi1E11_62__38__1_E3ValILi7EES8_#449$par100"(%arg0: memref<1x38x62xf32, 1>) {
    affine.parallel (%arg1) = (0) to (38) {
      %0 = affine.load %arg0[0, %arg1, 48] : memref<1x38x62xf32, 1>
      affine.store %0, %arg0[0, %arg1, 0] : memref<1x38x62xf32, 1>
      %1 = affine.load %arg0[0, %arg1, 7] : memref<1x38x62xf32, 1>
      affine.store %1, %arg0[0, %arg1, 55] : memref<1x38x62xf32, 1>
      %2 = affine.load %arg0[0, %arg1, 49] : memref<1x38x62xf32, 1>
      affine.store %2, %arg0[0, %arg1, 1] : memref<1x38x62xf32, 1>
      %3 = affine.load %arg0[0, %arg1, 8] : memref<1x38x62xf32, 1>
      affine.store %3, %arg0[0, %arg1, 56] : memref<1x38x62xf32, 1>
      %4 = affine.load %arg0[0, %arg1, 50] : memref<1x38x62xf32, 1>
      affine.store %4, %arg0[0, %arg1, 2] : memref<1x38x62xf32, 1>
      %5 = affine.load %arg0[0, %arg1, 9] : memref<1x38x62xf32, 1>
      affine.store %5, %arg0[0, %arg1, 57] : memref<1x38x62xf32, 1>
      %6 = affine.load %arg0[0, %arg1, 51] : memref<1x38x62xf32, 1>
      affine.store %6, %arg0[0, %arg1, 3] : memref<1x38x62xf32, 1>
      %7 = affine.load %arg0[0, %arg1, 10] : memref<1x38x62xf32, 1>
      affine.store %7, %arg0[0, %arg1, 58] : memref<1x38x62xf32, 1>
      %8 = affine.load %arg0[0, %arg1, 52] : memref<1x38x62xf32, 1>
      affine.store %8, %arg0[0, %arg1, 4] : memref<1x38x62xf32, 1>
      %9 = affine.load %arg0[0, %arg1, 11] : memref<1x38x62xf32, 1>
      affine.store %9, %arg0[0, %arg1, 59] : memref<1x38x62xf32, 1>
      %10 = affine.load %arg0[0, %arg1, 53] : memref<1x38x62xf32, 1>
      affine.store %10, %arg0[0, %arg1, 5] : memref<1x38x62xf32, 1>
      %11 = affine.load %arg0[0, %arg1, 12] : memref<1x38x62xf32, 1>
      affine.store %11, %arg0[0, %arg1, 60] : memref<1x38x62xf32, 1>
      %12 = affine.load %arg0[0, %arg1, 54] : memref<1x38x62xf32, 1>
      affine.store %12, %arg0[0, %arg1, 6] : memref<1x38x62xf32, 1>
      %13 = affine.load %arg0[0, %arg1, 13] : memref<1x38x62xf32, 1>
      affine.store %13, %arg0[0, %arg1, 61] : memref<1x38x62xf32, 1>
    }
    return
  }
  func.func private @"##call__Z37gpu_fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_38__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I7_38__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE13CuTracedArrayI7Float32Li3ELi1E11_62__38__1_E3ValILi7EES8_#449$par162"(%arg0: memref<1x38x62xf32, 1>) {
    affine.parallel (%arg1) = (0) to (38) {
      %0 = affine.load %arg0[0, %arg1, 48] : memref<1x38x62xf32, 1>
      affine.store %0, %arg0[0, %arg1, 0] : memref<1x38x62xf32, 1>
      %1 = affine.load %arg0[0, %arg1, 7] : memref<1x38x62xf32, 1>
      affine.store %1, %arg0[0, %arg1, 55] : memref<1x38x62xf32, 1>
      %2 = affine.load %arg0[0, %arg1, 49] : memref<1x38x62xf32, 1>
      affine.store %2, %arg0[0, %arg1, 1] : memref<1x38x62xf32, 1>
      %3 = affine.load %arg0[0, %arg1, 8] : memref<1x38x62xf32, 1>
      affine.store %3, %arg0[0, %arg1, 56] : memref<1x38x62xf32, 1>
      %4 = affine.load %arg0[0, %arg1, 50] : memref<1x38x62xf32, 1>
      affine.store %4, %arg0[0, %arg1, 2] : memref<1x38x62xf32, 1>
      %5 = affine.load %arg0[0, %arg1, 9] : memref<1x38x62xf32, 1>
      affine.store %5, %arg0[0, %arg1, 57] : memref<1x38x62xf32, 1>
      %6 = affine.load %arg0[0, %arg1, 51] : memref<1x38x62xf32, 1>
      affine.store %6, %arg0[0, %arg1, 3] : memref<1x38x62xf32, 1>
      %7 = affine.load %arg0[0, %arg1, 10] : memref<1x38x62xf32, 1>
      affine.store %7, %arg0[0, %arg1, 58] : memref<1x38x62xf32, 1>
      %8 = affine.load %arg0[0, %arg1, 52] : memref<1x38x62xf32, 1>
      affine.store %8, %arg0[0, %arg1, 4] : memref<1x38x62xf32, 1>
      %9 = affine.load %arg0[0, %arg1, 11] : memref<1x38x62xf32, 1>
      affine.store %9, %arg0[0, %arg1, 59] : memref<1x38x62xf32, 1>
      %10 = affine.load %arg0[0, %arg1, 53] : memref<1x38x62xf32, 1>
      affine.store %10, %arg0[0, %arg1, 5] : memref<1x38x62xf32, 1>
      %11 = affine.load %arg0[0, %arg1, 12] : memref<1x38x62xf32, 1>
      affine.store %11, %arg0[0, %arg1, 60] : memref<1x38x62xf32, 1>
      %12 = affine.load %arg0[0, %arg1, 54] : memref<1x38x62xf32, 1>
      affine.store %12, %arg0[0, %arg1, 6] : memref<1x38x62xf32, 1>
      %13 = affine.load %arg0[0, %arg1, 13] : memref<1x38x62xf32, 1>
      affine.store %13, %arg0[0, %arg1, 61] : memref<1x38x62xf32, 1>
    }
    return
  }
  func.func private @"##call__Z37gpu_fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_38__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I7_38__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE13CuTracedArrayI7Float32Li3ELi1E11_62__38__1_E3ValILi7EES8_#449$par164"(%arg0: memref<1x38x62xf32, 1>) {
    affine.parallel (%arg1) = (0) to (38) {
      %0 = affine.load %arg0[0, %arg1, 48] : memref<1x38x62xf32, 1>
      affine.store %0, %arg0[0, %arg1, 0] : memref<1x38x62xf32, 1>
      %1 = affine.load %arg0[0, %arg1, 7] : memref<1x38x62xf32, 1>
      affine.store %1, %arg0[0, %arg1, 55] : memref<1x38x62xf32, 1>
      %2 = affine.load %arg0[0, %arg1, 49] : memref<1x38x62xf32, 1>
      affine.store %2, %arg0[0, %arg1, 1] : memref<1x38x62xf32, 1>
      %3 = affine.load %arg0[0, %arg1, 8] : memref<1x38x62xf32, 1>
      affine.store %3, %arg0[0, %arg1, 56] : memref<1x38x62xf32, 1>
      %4 = affine.load %arg0[0, %arg1, 50] : memref<1x38x62xf32, 1>
      affine.store %4, %arg0[0, %arg1, 2] : memref<1x38x62xf32, 1>
      %5 = affine.load %arg0[0, %arg1, 9] : memref<1x38x62xf32, 1>
      affine.store %5, %arg0[0, %arg1, 57] : memref<1x38x62xf32, 1>
      %6 = affine.load %arg0[0, %arg1, 51] : memref<1x38x62xf32, 1>
      affine.store %6, %arg0[0, %arg1, 3] : memref<1x38x62xf32, 1>
      %7 = affine.load %arg0[0, %arg1, 10] : memref<1x38x62xf32, 1>
      affine.store %7, %arg0[0, %arg1, 58] : memref<1x38x62xf32, 1>
      %8 = affine.load %arg0[0, %arg1, 52] : memref<1x38x62xf32, 1>
      affine.store %8, %arg0[0, %arg1, 4] : memref<1x38x62xf32, 1>
      %9 = affine.load %arg0[0, %arg1, 11] : memref<1x38x62xf32, 1>
      affine.store %9, %arg0[0, %arg1, 59] : memref<1x38x62xf32, 1>
      %10 = affine.load %arg0[0, %arg1, 53] : memref<1x38x62xf32, 1>
      affine.store %10, %arg0[0, %arg1, 5] : memref<1x38x62xf32, 1>
      %11 = affine.load %arg0[0, %arg1, 12] : memref<1x38x62xf32, 1>
      affine.store %11, %arg0[0, %arg1, 60] : memref<1x38x62xf32, 1>
      %12 = affine.load %arg0[0, %arg1, 54] : memref<1x38x62xf32, 1>
      affine.store %12, %arg0[0, %arg1, 6] : memref<1x38x62xf32, 1>
      %13 = affine.load %arg0[0, %arg1, 13] : memref<1x38x62xf32, 1>
      affine.store %13, %arg0[0, %arg1, 61] : memref<1x38x62xf32, 1>
    }
    return
  }
  func.func private @"##call__Z37gpu_fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_39__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I7_39__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE13CuTracedArrayI7Float32Li3ELi1E11_62__39__1_E3ValILi7EES8_#451$par101"(%arg0: memref<1x39x62xf32, 1>) {
    affine.parallel (%arg1) = (0) to (39) {
      %0 = affine.load %arg0[0, %arg1, 48] : memref<1x39x62xf32, 1>
      affine.store %0, %arg0[0, %arg1, 0] : memref<1x39x62xf32, 1>
      %1 = affine.load %arg0[0, %arg1, 7] : memref<1x39x62xf32, 1>
      affine.store %1, %arg0[0, %arg1, 55] : memref<1x39x62xf32, 1>
      %2 = affine.load %arg0[0, %arg1, 49] : memref<1x39x62xf32, 1>
      affine.store %2, %arg0[0, %arg1, 1] : memref<1x39x62xf32, 1>
      %3 = affine.load %arg0[0, %arg1, 8] : memref<1x39x62xf32, 1>
      affine.store %3, %arg0[0, %arg1, 56] : memref<1x39x62xf32, 1>
      %4 = affine.load %arg0[0, %arg1, 50] : memref<1x39x62xf32, 1>
      affine.store %4, %arg0[0, %arg1, 2] : memref<1x39x62xf32, 1>
      %5 = affine.load %arg0[0, %arg1, 9] : memref<1x39x62xf32, 1>
      affine.store %5, %arg0[0, %arg1, 57] : memref<1x39x62xf32, 1>
      %6 = affine.load %arg0[0, %arg1, 51] : memref<1x39x62xf32, 1>
      affine.store %6, %arg0[0, %arg1, 3] : memref<1x39x62xf32, 1>
      %7 = affine.load %arg0[0, %arg1, 10] : memref<1x39x62xf32, 1>
      affine.store %7, %arg0[0, %arg1, 58] : memref<1x39x62xf32, 1>
      %8 = affine.load %arg0[0, %arg1, 52] : memref<1x39x62xf32, 1>
      affine.store %8, %arg0[0, %arg1, 4] : memref<1x39x62xf32, 1>
      %9 = affine.load %arg0[0, %arg1, 11] : memref<1x39x62xf32, 1>
      affine.store %9, %arg0[0, %arg1, 59] : memref<1x39x62xf32, 1>
      %10 = affine.load %arg0[0, %arg1, 53] : memref<1x39x62xf32, 1>
      affine.store %10, %arg0[0, %arg1, 5] : memref<1x39x62xf32, 1>
      %11 = affine.load %arg0[0, %arg1, 12] : memref<1x39x62xf32, 1>
      affine.store %11, %arg0[0, %arg1, 60] : memref<1x39x62xf32, 1>
      %12 = affine.load %arg0[0, %arg1, 54] : memref<1x39x62xf32, 1>
      affine.store %12, %arg0[0, %arg1, 6] : memref<1x39x62xf32, 1>
      %13 = affine.load %arg0[0, %arg1, 13] : memref<1x39x62xf32, 1>
      affine.store %13, %arg0[0, %arg1, 61] : memref<1x39x62xf32, 1>
    }
    return
  }
  func.func private @"##call__Z37gpu_fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_39__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I7_39__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE13CuTracedArrayI7Float32Li3ELi1E11_62__39__1_E3ValILi7EES8_#451$par165"(%arg0: memref<1x39x62xf32, 1>) {
    affine.parallel (%arg1) = (0) to (39) {
      %0 = affine.load %arg0[0, %arg1, 48] : memref<1x39x62xf32, 1>
      affine.store %0, %arg0[0, %arg1, 0] : memref<1x39x62xf32, 1>
      %1 = affine.load %arg0[0, %arg1, 7] : memref<1x39x62xf32, 1>
      affine.store %1, %arg0[0, %arg1, 55] : memref<1x39x62xf32, 1>
      %2 = affine.load %arg0[0, %arg1, 49] : memref<1x39x62xf32, 1>
      affine.store %2, %arg0[0, %arg1, 1] : memref<1x39x62xf32, 1>
      %3 = affine.load %arg0[0, %arg1, 8] : memref<1x39x62xf32, 1>
      affine.store %3, %arg0[0, %arg1, 56] : memref<1x39x62xf32, 1>
      %4 = affine.load %arg0[0, %arg1, 50] : memref<1x39x62xf32, 1>
      affine.store %4, %arg0[0, %arg1, 2] : memref<1x39x62xf32, 1>
      %5 = affine.load %arg0[0, %arg1, 9] : memref<1x39x62xf32, 1>
      affine.store %5, %arg0[0, %arg1, 57] : memref<1x39x62xf32, 1>
      %6 = affine.load %arg0[0, %arg1, 51] : memref<1x39x62xf32, 1>
      affine.store %6, %arg0[0, %arg1, 3] : memref<1x39x62xf32, 1>
      %7 = affine.load %arg0[0, %arg1, 10] : memref<1x39x62xf32, 1>
      affine.store %7, %arg0[0, %arg1, 58] : memref<1x39x62xf32, 1>
      %8 = affine.load %arg0[0, %arg1, 52] : memref<1x39x62xf32, 1>
      affine.store %8, %arg0[0, %arg1, 4] : memref<1x39x62xf32, 1>
      %9 = affine.load %arg0[0, %arg1, 11] : memref<1x39x62xf32, 1>
      affine.store %9, %arg0[0, %arg1, 59] : memref<1x39x62xf32, 1>
      %10 = affine.load %arg0[0, %arg1, 53] : memref<1x39x62xf32, 1>
      affine.store %10, %arg0[0, %arg1, 5] : memref<1x39x62xf32, 1>
      %11 = affine.load %arg0[0, %arg1, 12] : memref<1x39x62xf32, 1>
      affine.store %11, %arg0[0, %arg1, 60] : memref<1x39x62xf32, 1>
      %12 = affine.load %arg0[0, %arg1, 54] : memref<1x39x62xf32, 1>
      affine.store %12, %arg0[0, %arg1, 6] : memref<1x39x62xf32, 1>
      %13 = affine.load %arg0[0, %arg1, 13] : memref<1x39x62xf32, 1>
      affine.store %13, %arg0[0, %arg1, 61] : memref<1x39x62xf32, 1>
    }
    return
  }
  func.func private @"##call__Z19gpu_ab2_step_field_16CompilerMetadataI10StaticSizeI12_48__24__10_E12DynamicCheckvv7NDRangeILi3ES0_I10_3__2__10_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E12_62__38__24_EES9_S9_SC_SC_#453$par102"(%arg0: memref<24x38x62xf32, 1>, %arg1: memref<24x38x62xf32, 1>, %arg2: memref<24x38x62xf32, 1>) {
    %cst = arith.constant 1.600000e+00 : f32
    %cst_0 = arith.constant 6.000000e-01 : f32
    %cst_1 = arith.constant 6.000000e+01 : f32
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (10, 24, 48) {
      %0 = affine.load %arg1[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<24x38x62xf32, 1>
      %1 = arith.mulf %0, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
      %2 = affine.load %arg2[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<24x38x62xf32, 1>
      %3 = arith.mulf %2, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
      %4 = arith.subf %1, %3 {fastmathFlags = #llvm.fastmath<none>} : f32
      %5 = affine.load %arg0[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<24x38x62xf32, 1>
      %6 = arith.mulf %4, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
      %7 = arith.addf %5, %6 {fastmathFlags = #llvm.fastmath<none>} : f32
      affine.store %7, %arg0[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<24x38x62xf32, 1>
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_48__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_3__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float32Li3E13CuTracedArrayISF_Li3ELi1E11_62__38__1_EE17BoundaryConditionI4FluxvESL_S7_I4Face6CentervE21LatitudeLongitudeGridISF_8Periodic7BoundedSR_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_25__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_24__EESU_SW_ESF_SF_SE_ISF_Li1E12StepRangeLenISF_7Float64SZ_S8_EES11_SF_SF_S11_S11_SE_ISF_Li1ESG_ISF_Li1ELi1E5_38__EES13_S13_S13_SF_SF_vS8_ES7_IJEE#447$par99"(%arg0: memref<1x38x62xf32, 1>) {
// CHECK-NEXT:    affine.parallel (%arg1) = (0) to (48) {
// CHECK-NEXT:      %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %1 = affine.load %arg0[0, 30, %arg1 + 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %1, %arg0[0, 31, %arg1 + 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_48__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_3__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float32Li3E13CuTracedArrayISF_Li3ELi1E11_62__38__1_EE17BoundaryConditionI4FluxvESL_S7_I4Face6CentervE21LatitudeLongitudeGridISF_8Periodic7BoundedSR_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_25__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_24__EESU_SW_ESF_SF_SE_ISF_Li1E12StepRangeLenISF_7Float64SZ_S8_EES11_SF_SF_S11_S11_SE_ISF_Li1ESG_ISF_Li1ELi1E5_38__EES13_S13_S13_SF_SF_vS8_ES7_IJEE#447$par163"(%arg0: memref<1x38x62xf32, 1>) {
// CHECK-NEXT:    affine.parallel (%arg1) = (0) to (48) {
// CHECK-NEXT:      %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %1 = affine.load %arg0[0, 30, %arg1 + 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %1, %arg0[0, 31, %arg1 + 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @"##call__Z37gpu_fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_38__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I7_38__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE13CuTracedArrayI7Float32Li3ELi1E11_62__38__1_E3ValILi7EES8_#449$par100"(%arg0: memref<1x38x62xf32, 1>) {
// CHECK-NEXT:    affine.parallel (%arg1) = (0) to (38) {
// CHECK-NEXT:      %0 = affine.load %arg0[0, %arg1, 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %1 = affine.load %arg0[0, %arg1, 8] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %2 = affine.load %arg0[0, %arg1, 9] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %3 = affine.load %arg0[0, %arg1, 10] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %4 = affine.load %arg0[0, %arg1, 11] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %5 = affine.load %arg0[0, %arg1, 12] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %6 = affine.load %arg0[0, %arg1, 13] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %7 = affine.load %arg0[0, %arg1, 48] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %8 = affine.load %arg0[0, %arg1, 49] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %9 = affine.load %arg0[0, %arg1, 50] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %10 = affine.load %arg0[0, %arg1, 51] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %11 = affine.load %arg0[0, %arg1, 52] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %12 = affine.load %arg0[0, %arg1, 53] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %13 = affine.load %arg0[0, %arg1, 54] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %7, %arg0[0, %arg1, 0] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %8, %arg0[0, %arg1, 1] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %9, %arg0[0, %arg1, 2] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %10, %arg0[0, %arg1, 3] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %11, %arg0[0, %arg1, 4] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %12, %arg0[0, %arg1, 5] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %13, %arg0[0, %arg1, 6] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %0, %arg0[0, %arg1, 55] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %1, %arg0[0, %arg1, 56] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %2, %arg0[0, %arg1, 57] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %3, %arg0[0, %arg1, 58] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %4, %arg0[0, %arg1, 59] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %5, %arg0[0, %arg1, 60] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %6, %arg0[0, %arg1, 61] : memref<1x38x62xf32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @"##call__Z37gpu_fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_38__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I7_38__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE13CuTracedArrayI7Float32Li3ELi1E11_62__38__1_E3ValILi7EES8_#449$par162"(%arg0: memref<1x38x62xf32, 1>) {
// CHECK-NEXT:    affine.parallel (%arg1) = (0) to (38) {
// CHECK-NEXT:      %0 = affine.load %arg0[0, %arg1, 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %1 = affine.load %arg0[0, %arg1, 8] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %2 = affine.load %arg0[0, %arg1, 9] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %3 = affine.load %arg0[0, %arg1, 10] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %4 = affine.load %arg0[0, %arg1, 11] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %5 = affine.load %arg0[0, %arg1, 12] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %6 = affine.load %arg0[0, %arg1, 13] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %7 = affine.load %arg0[0, %arg1, 48] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %8 = affine.load %arg0[0, %arg1, 49] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %9 = affine.load %arg0[0, %arg1, 50] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %10 = affine.load %arg0[0, %arg1, 51] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %11 = affine.load %arg0[0, %arg1, 52] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %12 = affine.load %arg0[0, %arg1, 53] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %13 = affine.load %arg0[0, %arg1, 54] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %7, %arg0[0, %arg1, 0] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %8, %arg0[0, %arg1, 1] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %9, %arg0[0, %arg1, 2] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %10, %arg0[0, %arg1, 3] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %11, %arg0[0, %arg1, 4] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %12, %arg0[0, %arg1, 5] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %13, %arg0[0, %arg1, 6] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %0, %arg0[0, %arg1, 55] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %1, %arg0[0, %arg1, 56] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %2, %arg0[0, %arg1, 57] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %3, %arg0[0, %arg1, 58] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %4, %arg0[0, %arg1, 59] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %5, %arg0[0, %arg1, 60] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %6, %arg0[0, %arg1, 61] : memref<1x38x62xf32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @"##call__Z37gpu_fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_38__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I7_38__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE13CuTracedArrayI7Float32Li3ELi1E11_62__38__1_E3ValILi7EES8_#449$par164"(%arg0: memref<1x38x62xf32, 1>) {
// CHECK-NEXT:    affine.parallel (%arg1) = (0) to (38) {
// CHECK-NEXT:      %0 = affine.load %arg0[0, %arg1, 7] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %1 = affine.load %arg0[0, %arg1, 8] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %2 = affine.load %arg0[0, %arg1, 9] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %3 = affine.load %arg0[0, %arg1, 10] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %4 = affine.load %arg0[0, %arg1, 11] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %5 = affine.load %arg0[0, %arg1, 12] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %6 = affine.load %arg0[0, %arg1, 13] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %7 = affine.load %arg0[0, %arg1, 48] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %8 = affine.load %arg0[0, %arg1, 49] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %9 = affine.load %arg0[0, %arg1, 50] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %10 = affine.load %arg0[0, %arg1, 51] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %11 = affine.load %arg0[0, %arg1, 52] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %12 = affine.load %arg0[0, %arg1, 53] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      %13 = affine.load %arg0[0, %arg1, 54] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %7, %arg0[0, %arg1, 0] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %8, %arg0[0, %arg1, 1] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %9, %arg0[0, %arg1, 2] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %10, %arg0[0, %arg1, 3] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %11, %arg0[0, %arg1, 4] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %12, %arg0[0, %arg1, 5] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %13, %arg0[0, %arg1, 6] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %0, %arg0[0, %arg1, 55] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %1, %arg0[0, %arg1, 56] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %2, %arg0[0, %arg1, 57] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %3, %arg0[0, %arg1, 58] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %4, %arg0[0, %arg1, 59] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %5, %arg0[0, %arg1, 60] : memref<1x38x62xf32, 1>
// CHECK-NEXT:      affine.store %6, %arg0[0, %arg1, 61] : memref<1x38x62xf32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @"##call__Z37gpu_fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_39__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I7_39__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE13CuTracedArrayI7Float32Li3ELi1E11_62__39__1_E3ValILi7EES8_#451$par101"(%arg0: memref<1x39x62xf32, 1>) {
// CHECK-NEXT:    affine.parallel (%arg1) = (0) to (39) {
// CHECK-NEXT:      %0 = affine.load %arg0[0, %arg1, 7] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %1 = affine.load %arg0[0, %arg1, 8] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %2 = affine.load %arg0[0, %arg1, 9] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %3 = affine.load %arg0[0, %arg1, 10] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %4 = affine.load %arg0[0, %arg1, 11] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %5 = affine.load %arg0[0, %arg1, 12] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %6 = affine.load %arg0[0, %arg1, 13] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %7 = affine.load %arg0[0, %arg1, 48] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %8 = affine.load %arg0[0, %arg1, 49] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %9 = affine.load %arg0[0, %arg1, 50] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %10 = affine.load %arg0[0, %arg1, 51] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %11 = affine.load %arg0[0, %arg1, 52] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %12 = affine.load %arg0[0, %arg1, 53] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %13 = affine.load %arg0[0, %arg1, 54] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %7, %arg0[0, %arg1, 0] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %8, %arg0[0, %arg1, 1] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %9, %arg0[0, %arg1, 2] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %10, %arg0[0, %arg1, 3] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %11, %arg0[0, %arg1, 4] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %12, %arg0[0, %arg1, 5] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %13, %arg0[0, %arg1, 6] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %0, %arg0[0, %arg1, 55] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %1, %arg0[0, %arg1, 56] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %2, %arg0[0, %arg1, 57] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %3, %arg0[0, %arg1, 58] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %4, %arg0[0, %arg1, 59] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %5, %arg0[0, %arg1, 60] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %6, %arg0[0, %arg1, 61] : memref<1x39x62xf32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @"##call__Z37gpu_fill_periodic_west_and_east_halo_16CompilerMetadataI16OffsetStaticSizeI11_1_39__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I7_39__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE13CuTracedArrayI7Float32Li3ELi1E11_62__39__1_E3ValILi7EES8_#451$par165"(%arg0: memref<1x39x62xf32, 1>) {
// CHECK-NEXT:    affine.parallel (%arg1) = (0) to (39) {
// CHECK-NEXT:      %0 = affine.load %arg0[0, %arg1, 7] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %1 = affine.load %arg0[0, %arg1, 8] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %2 = affine.load %arg0[0, %arg1, 9] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %3 = affine.load %arg0[0, %arg1, 10] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %4 = affine.load %arg0[0, %arg1, 11] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %5 = affine.load %arg0[0, %arg1, 12] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %6 = affine.load %arg0[0, %arg1, 13] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %7 = affine.load %arg0[0, %arg1, 48] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %8 = affine.load %arg0[0, %arg1, 49] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %9 = affine.load %arg0[0, %arg1, 50] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %10 = affine.load %arg0[0, %arg1, 51] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %11 = affine.load %arg0[0, %arg1, 52] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %12 = affine.load %arg0[0, %arg1, 53] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      %13 = affine.load %arg0[0, %arg1, 54] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %7, %arg0[0, %arg1, 0] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %8, %arg0[0, %arg1, 1] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %9, %arg0[0, %arg1, 2] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %10, %arg0[0, %arg1, 3] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %11, %arg0[0, %arg1, 4] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %12, %arg0[0, %arg1, 5] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %13, %arg0[0, %arg1, 6] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %0, %arg0[0, %arg1, 55] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %1, %arg0[0, %arg1, 56] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %2, %arg0[0, %arg1, 57] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %3, %arg0[0, %arg1, 58] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %4, %arg0[0, %arg1, 59] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %5, %arg0[0, %arg1, 60] : memref<1x39x62xf32, 1>
// CHECK-NEXT:      affine.store %6, %arg0[0, %arg1, 61] : memref<1x39x62xf32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @"##call__Z19gpu_ab2_step_field_16CompilerMetadataI10StaticSizeI12_48__24__10_E12DynamicCheckvv7NDRangeILi3ES0_I10_3__2__10_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float32Li3E13CuTracedArrayIS9_Li3ELi1E12_62__38__24_EES9_S9_SC_SC_#453$par102"(%arg0: memref<24x38x62xf32, 1>, %arg1: memref<24x38x62xf32, 1>, %arg2: memref<24x38x62xf32, 1>) {
// CHECK-NEXT:    %cst = arith.constant 1.600000e+00 : f32
// CHECK-NEXT:    %cst_0 = arith.constant 6.000000e-01 : f32
// CHECK-NEXT:    %cst_1 = arith.constant 6.000000e+01 : f32
// CHECK-NEXT:    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (10, 24, 48) {
// CHECK-NEXT:      %0 = affine.load %arg1[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<24x38x62xf32, 1>
// CHECK-NEXT:      %1 = affine.load %arg2[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<24x38x62xf32, 1>
// CHECK-NEXT:      %2 = affine.load %arg0[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<24x38x62xf32, 1>
// CHECK-NEXT:      %3 = arith.mulf %0, %cst {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %4 = arith.mulf %1, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %5 = arith.subf %3, %4 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %6 = arith.mulf %5, %cst_1 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      %7 = arith.addf %2, %6 {fastmathFlags = #llvm.fastmath<none>} : f32
// CHECK-NEXT:      affine.store %7, %arg0[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<24x38x62xf32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }