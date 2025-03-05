// RUN: enzymexlamlir-opt  "--pass-pipeline=builtin.module(func.func(canonicalize-loops),affine-cfg)" %s | FileCheck %s

#set1 = affine_set<(d0, d1) : (d0 mod 16 + (d1 mod 12) * 16 >= 0, -(d0 mod 16) - (d1 mod 12) * 16 + 179 >= 0, d0 floordiv 16 + ((d1 floordiv 12) mod 6) * 16 >= 0, -(d0 floordiv 16) - ((d1 floordiv 12) mod 6) * 16 + 84 >= 0)>
module {
  func.func private @"##call__Z25gpu__mask_immersed_field_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE5TupleI6Center4FaceSE_E20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESO_SQ_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SV_SV_EvE16GridFittedBottomI5FieldISE_SE_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvES9_#1335$par75"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<1x99x194xf64, 1>) {
    %c-1_i64 = arith.constant -1 : i64
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %c12_i32 = arith.constant 12 : i32
    %c6_i32 = arith.constant 6 : i32
    %c-6_i64 = arith.constant -6 : i64
    %c16_i16 = arith.constant 16 : i16
    %c16_i64 = arith.constant 16 : i64
    %cst = arith.constant 0.000000e+00 : f64
    affine.parallel (%arg3, %arg4) = (0, 0) to (1440, 256) {
      %0 = arith.addi %arg3, %c1 : index
      %1 = arith.addi %arg4, %c1 : index
      %2 = arith.index_castui %0 : index to i64
      %3 = arith.addi %2, %c-1_i64 : i64
      %4 = arith.trunci %3 : i64 to i32
      %5 = arith.divui %4, %c12_i32 : i32
      %6 = arith.extui %5 : i32 to i64
      %7 = arith.divui %5, %c6_i32 : i32
      %8 = arith.extui %7 : i32 to i64
      %9 = arith.muli %8, %c-6_i64 : i64
      %10 = arith.addi %6, %9 : i64
      %11 = arith.index_castui %1 : index to i64
      %12 = arith.addi %11, %c-1_i64 : i64
      %13 = arith.trunci %12 : i64 to i16
      %14 = arith.divui %13, %c16_i16 : i16
      %15 = arith.extui %14 : i16 to i64
      %16 = arith.muli %10, %c16_i64 : i64
      %17 = arith.addi %15, %c1_i64 : i64
      %18 = arith.addi %17, %16 : i64
      affine.if #set1(%arg4, %arg3) {
        %19 = affine.load %arg1[%arg3 floordiv 72 + 7] : memref<34xf64, 1>
        %20 = affine.load %arg2[0, %arg4 floordiv 16 + (%arg3 floordiv 12) * 16 - (%arg3 floordiv 72) * 96 + 7, %arg4 mod 16 + (%arg3 mod 12) * 16 + 7] : memref<1x99x194xf64, 1>
        %21 = arith.cmpf ole, %19, %20 {fastmathFlags = #llvm.fastmath<none>} : f64
        %22 = arith.addi %18, %c-1_i64 : i64
        %23 = affine.load %arg2[0, %arg4 floordiv 16 + (%arg3 floordiv 12) * 16 - (%arg3 floordiv 72) * 96 + 6, %arg4 mod 16 + (%arg3 mod 12) * 16 + 7] : memref<1x99x194xf64, 1>
        %24 = arith.cmpf ole, %19, %23 {fastmathFlags = #llvm.fastmath<none>} : f64
        %25 = arith.cmpi ult, %22, %c1_i64 : i64
        %26 = arith.ori %25, %24 : i1
        %27 = arith.ori %21, %26 : i1
        %28 = arith.cmpi uge, %22, %c1_i64 : i64
        %29 = arith.andi %28, %27 : i1
        %30 = affine.load %arg0[%arg3 floordiv 72 + 7, %arg4 floordiv 16 + (%arg3 floordiv 12) * 16 - (%arg3 floordiv 72) * 96 + 7, %arg4 mod 16 + (%arg3 mod 12) * 16 + 7] : memref<34x99x194xf64, 1>
        %31 = arith.select %29, %cst, %30 : f64
        affine.store %31, %arg0[%arg3 floordiv 72 + 7, %arg4 floordiv 16 + (%arg3 floordiv 12) * 16 - (%arg3 floordiv 72) * 96 + 7, %arg4 mod 16 + (%arg3 mod 12) * 16 + 7] : memref<34x99x194xf64, 1>
      }
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z25gpu__mask_immersed_field_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE5TupleI6Center4FaceSE_E20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESO_SQ_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SV_SV_EvE16GridFittedBottomI5FieldISE_SE_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvES9_#1335$par75"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<1x99x194xf64, 1>) {
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (20, 85, 180) {
// CHECK-NEXT:      %0 = affine.load %arg1[%arg3 + 7] : memref<34xf64, 1>
// CHECK-NEXT:      %1 = affine.load %arg2[0, %arg4 + 7, %arg5 + 7] : memref<1x99x194xf64, 1>
// CHECK-NEXT:      %2 = arith.cmpf ole, %0, %1 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %3 = arith.index_castui %arg4 : index to i64
// CHECK-NEXT:      %4 = affine.load %arg2[0, %arg4 + 6, %arg5 + 7] : memref<1x99x194xf64, 1>
// CHECK-NEXT:      %5 = arith.cmpf ole, %0, %4 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %6 = arith.cmpi ult, %3, %c1_i64 : i64
// CHECK-NEXT:      %7 = arith.ori %6, %5 : i1
// CHECK-NEXT:      %8 = arith.ori %2, %7 : i1
// CHECK-NEXT:      %9 = arith.cmpi uge, %3, %c1_i64 : i64
// CHECK-NEXT:      %10 = arith.andi %9, %8 : i1
// CHECK-NEXT:      %11 = affine.load %arg0[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<34x99x194xf64, 1>
// CHECK-NEXT:      %12 = arith.select %10, %cst, %11 : f64
// CHECK-NEXT:      affine.store %12, %arg0[%arg3 + 7, %arg4 + 7, %arg5 + 7] : memref<34x99x194xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
