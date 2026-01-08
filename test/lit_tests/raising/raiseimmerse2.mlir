// RUN: enzymexlamlir-opt  "--pass-pipeline=builtin.module(func.func(canonicalize-loops),affine-cfg)" %s | FileCheck %s

#set = affine_set<(d0, d1) : (d0 + d1 * 16 >= 0, -d0 - d1 * 16 + 84 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 == 0, d1 mod 6 == 0)>
module {
  func.func private @"##call__Z25gpu__mask_immersed_field_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE5TupleI6Center4FaceSE_E20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESO_SQ_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SV_SV_EvE16GridFittedBottomI5FieldISE_SE_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvES9_#1335$par75"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<1x99x194xf64, 1>) {
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    affine.parallel (%arg3, %arg4) = (0, 0) to (16, 6) {
      affine.if #set(%arg3, %arg4) {
        %0 = affine.if #set1(%arg3, %arg4) -> f64 {
          affine.yield %cst : f64
        } else {
          affine.yield %cst_0 : f64
        }
        affine.store %0, %arg0[0, %arg3 + %arg4 * 16 + 7, 0] : memref<34x99x194xf64, 1>
      }
    }
    return
  }
}

// CHECK: #set = affine_set<(d0) : (d0 == 0)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func private @"##call__Z25gpu__mask_immersed_field_16CompilerMetadataI10StaticSizeI13_180__85__20_E12DynamicCheckvv7NDRangeILi3ES0_I11_12__6__20_ES0_I11_16__16__1_EvvEE11OffsetArrayI7Float64Li3E13CuTracedArrayIS9_Li3ELi1E13_194__99__34_EE5TupleI6Center4FaceSE_E20ImmersedBoundaryGridIS9_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridIS9_SI_SJ_SK_28StaticVerticalDiscretizationIS8_IS9_Li1ESA_IS9_Li1ELi1E5_35__EES8_IS9_Li1ESA_IS9_Li1ELi1E5_34__EESO_SQ_ES8_IS9_Li2ESA_IS9_Li2ELi1E9_194__99_EE8TripolarI5Int64SV_SV_EvE16GridFittedBottomI5FieldISE_SE_vvvvS8_IS9_Li3ESA_IS9_Li3ELi1E12_194__99__1_EES9_vvvE23CenterImmersedConditionEvvvES9_#1335$par75"(%arg0: memref<34x99x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<1x99x194xf64, 1>) {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:     %cst_0 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:     affine.parallel (%arg3) = (0) to (85) {
// CHECK-NEXT:       %0 = affine.if #set(%arg3) -> f64 {
// CHECK-NEXT:         affine.yield %cst : f64
// CHECK-NEXT:       } else {
// CHECK-NEXT:         affine.yield %cst_0 : f64
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.store %0, %arg0[0, %arg3 + 7, 0] : memref<34x99x194xf64, 1>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

