// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

#set = affine_set<(d0) : (-d0 + 89 >= 0)>
#set1 = affine_set<(d0) : (d0 - 1 >= 0)>
#map2 = affine_map<(d0, d1) -> (-d0 - d1 * 194 + 18435)>

module {
  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI12_1_180__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__104__1_EE17BoundaryConditionI4FluxvESJ_I6ZipperS8_ES7_I4Face6CentervE20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SS_ST_SU_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EESY_S10_E8TripolarIS8_S8_S8_ESE_ISF_Li2ESG_ISF_Li2ELi1E10_194__104_EES15_S15_S15_vSF_E16GridFittedBottomI5FieldISP_SP_vvvvSI_SF_vvvE23CenterImmersedConditionEvvvES7_IJEE#1632$par230"(%arg0: memref<1x104x194xf64, 1>) {
    %c0 = arith.constant 0 : index
    %c104 = arith.constant 104 : index
    %c194 = arith.constant 194 : index
    %c2_i64 = arith.constant 2 : i64
    %c182_i64 = arith.constant 182 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-1_i64 = arith.constant -1 : i64
    affine.parallel (%arg1) = (0) to (180) {
      %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x104x194xf64, 1>
      affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x104x194xf64, 1>
      %1:3 = affine.if #set1(%arg1) -> (i64, i64, f64) {
        %6 = affine.load %arg0[0, 96, -%arg1 + 187] : memref<1x104x194xf64, 1>
        affine.yield %c-1_i64, %c182_i64, %6 : i64, i64, f64
      } else {
        %6 = affine.load %arg0[0, 96, 7] : memref<1x104x194xf64, 1>
        affine.yield %c1_i64, %c2_i64, %6 : i64, i64, f64
      }
      %2 = arith.sitofp %1#0 : i64 to f64
      affine.for %arg2 = 0 to 7 {
        %6 = arith.index_cast %1#1 : i64 to index
        %7 = affine.apply #map2(%arg1, %arg2)
        %8 = arith.addi %7, %6 : index
        %9 = arith.index_cast %1#1 : i64 to index
        %13 = memref.load %arg0[%8, %9, %c0] : memref<1x104x194xf64, 1>
        %14 = arith.mulf %2, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %14, %arg0[0, %arg2 + 97, %arg1 + 7] : memref<1x104x194xf64, 1>
      }
      %3 = arith.mulf %2, %1#2 {fastmathFlags = #llvm.fastmath<none>} : f64
      %4 = affine.load %arg0[0, 96, %arg1 + 7] : memref<1x104x194xf64, 1>
      %5 = affine.if #set(%arg1) -> f64 {
        affine.yield %4 : f64
      } else {
        affine.yield %3 : f64
      }
      affine.store %5, %arg0[0, 96, %arg1 + 7] : memref<1x104x194xf64, 1>
    }
    return
  }
}

// CHECK: #set = affine_set<(d0) : (d0 - 1 >= 0)>
// CHECK-NEXT: #set1 = affine_set<(d0) : (-d0 + 89 >= 0)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI12_1_180__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__104__1_EE17BoundaryConditionI4FluxvESJ_I6ZipperS8_ES7_I4Face6CentervE20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SS_ST_SU_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EESY_S10_E8TripolarIS8_S8_S8_ESE_ISF_Li2ESG_ISF_Li2ELi1E10_194__104_EES15_S15_S15_vSF_E16GridFittedBottomI5FieldISP_SP_vvvvSI_SF_vvvE23CenterImmersedConditionEvvvES7_IJEE#1632$par230"(%arg0: memref<1x104x194xf64, 1>) {
// CHECK-NEXT:     %c2_i64 = arith.constant 2 : i64
// CHECK-NEXT:     %c182_i64 = arith.constant 182 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %c-1_i64 = arith.constant -1 : i64
// CHECK-NEXT:     affine.parallel (%arg1) = (0) to (180) {
// CHECK-NEXT:       %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x104x194xf64, 1>
// CHECK-NEXT:       affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x104x194xf64, 1>
// CHECK-NEXT:       %1:3 = affine.if #set(%arg1) -> (i64, i64, f64) {
// CHECK-NEXT:         %6 = affine.load %arg0[0, 96, -%arg1 + 187] : memref<1x104x194xf64, 1>
// CHECK-NEXT:         affine.yield %c-1_i64, %c182_i64, %6 : i64, i64, f64
// CHECK-NEXT:       } else {
// CHECK-NEXT:         %6 = affine.load %arg0[0, 96, 7] : memref<1x104x194xf64, 1>
// CHECK-NEXT:         affine.yield %c1_i64, %c2_i64, %6 : i64, i64, f64
// CHECK-NEXT:       }
// CHECK-NEXT:       %2 = arith.sitofp %1#0 : i64 to f64
// CHECK-NEXT:       affine.parallel (%arg2) = (0) to (7) {
// CHECK-NEXT:         %6:2 = affine.if #set(%arg1) -> (i64, f64) {
// CHECK-NEXT:           %8 = affine.load %arg0[-%arg1 - %arg2 * 194 + 18617, 182, 0] : memref<1x104x194xf64, 1>
// CHECK-NEXT:           affine.yield %c182_i64, %8 : i64, f64
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %8 = affine.load %arg0[%arg2 * -194 + 18437, 2, 0] : memref<1x104x194xf64, 1>
// CHECK-NEXT:           affine.yield %c2_i64, %8 : i64, f64
// CHECK-NEXT:         }
// CHECK-NEXT:         %7 = arith.mulf %2, %6#1 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:         affine.store %7, %arg0[0, %arg2 + 97, %arg1 + 7] : memref<1x104x194xf64, 1>
// CHECK-NEXT:       }
// CHECK-NEXT:       %3 = arith.mulf %2, %1#2 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:       %4 = affine.load %arg0[0, 96, %arg1 + 7] : memref<1x104x194xf64, 1>
// CHECK-NEXT:       %5 = affine.if #set1(%arg1) -> f64 {
// CHECK-NEXT:         affine.yield %4 : f64
// CHECK-NEXT:       } else {
// CHECK-NEXT:         affine.yield %3 : f64
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.store %5, %arg0[0, 96, %arg1 + 7] : memref<1x104x194xf64, 1>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
