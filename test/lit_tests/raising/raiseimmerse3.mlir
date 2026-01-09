// RUN: enzymexlamlir-opt  "--pass-pipeline=builtin.module(func.func(canonicalize-loops),affine-cfg)" %s | FileCheck %s

#set = affine_set<(d0, d1) : (d0 + d1 * 16 >= 0, -d0 - d1 * 16 + 84 >= 0)>
module {
  func.func private @"##call__Z25gpu__mask_immersed_field_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE5TupleI6Center4FaceSE_E20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESO_SQ_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SV_SV_EvE16GridFittedBottomI5FieldISE_SE_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvES9_#1335$par75"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<1x99x194xf64, 1>, %6: i1) {
    %c16 = arith.constant 16 : index
    %c6 = arith.constant 6 : index
    %c-1_i64 = arith.constant -1 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    affine.parallel (%arg3, %arg4) = (0, 0) to (16, 6) {
      %0 = arith.remui %arg4, %c6 : index
      %1 = arith.muli %0, %c16 : index
      %2 = arith.addi %1, %arg3 : index
      affine.if #set(%arg3, %arg4) {
        %8 = arith.cmpi ult, %2, %c1 : index
        %9 = arith.ori %8, %6 : i1
        %10 = arith.ori %6, %9 : i1
        %11 = arith.select %10, %cst, %cst_0 : f64
        affine.store %11, %arg0[7, %arg3 + %arg4 * 16 + 7, 7] : memref<34x99x194xf64, 1>
      }
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z25gpu__mask_immersed_field_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE5TupleI6Center4FaceSE_E20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESO_SQ_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SV_SV_EvE16GridFittedBottomI5FieldISE_SE_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvES9_#1335$par75"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<1x99x194xf64, 1>, %arg3: i1) {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    affine.parallel (%arg4) = (0) to (85) {
// CHECK-NEXT:      %0 = arith.cmpi eq, %arg4, %c0 : index
// CHECK-NEXT:      %1 = arith.ori %0, %arg3 : i1
// CHECK-NEXT:      %2 = arith.ori %arg3, %1 : i1
// CHECK-NEXT:      %3 = arith.select %2, %cst, %cst_0 : f64
// CHECK-NEXT:      affine.store %3, %arg0[7, %arg4 + 7, 7] : memref<34x99x194xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
