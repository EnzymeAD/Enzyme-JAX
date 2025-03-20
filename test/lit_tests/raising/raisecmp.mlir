// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

#set = affine_set<(d0, d1) : (-(d0 floordiv 16) >= 0, d0 mod 16 + d1 * 16 >= 0, d1 * -16 - d0 mod 16 + 179 >= 0)>

module {
  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI12_1_180__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E12_194__99__1_EE17BoundaryConditionI4FluxvESJ_I6ZipperS8_ES7_I6CenterSO_vE20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SR_SS_ST_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EESX_SZ_ESE_ISF_Li2ESG_ISF_Li2ELi1E9_194__99_EE8TripolarIS8_S8_S8_EvE16GridFittedBottomI5FieldISO_SO_vvvvSI_SF_vvvE23CenterImmersedConditionEvvvES7_#842$par30"(%arg0: memref<1x99x194xf64, 1>) {
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c90_i64 = arith.constant 90 : i64
    affine.parallel (%arg1, %arg2) = (0, 0) to (12, 256) {
      %0 = arith.addi %arg1, %c1 : index
      %1 = arith.addi %arg2, %c1 : index
      %2 = arith.index_castui %0 : index to i64
      %3 = arith.index_castui %1 : index to i64
      %4 = arith.divui %arg2, %c16 : index
      %5 = arith.index_castui %4 : index to i64 // arg2 / 16
      %6 = arith.subi %c0_i64, %5 : i64       // -(arg2 / 16)
      %7 = arith.addi %6, %2 : i64	      // -(arg2 / 16) + (arg1 + 1)
      %8 = arith.addi %7, %c-1_i64 : i64      // %7 - 1 ; -(arg2/16) + arg1
      %9 = arith.muli %8, %c16_i64 : i64      // %8 * 16; -16(arg2/16) + 16 arg1
      %10 = arith.addi %3, %9 : i64          // %3 + %9 ; arg2 + 1 -16(arg2/16) + 16 arg1
      affine.if #set(%arg2, %arg1) {
        %11 = affine.load %arg0[0, 7, %arg2 + %arg1 * 16 + 7] : memref<1x99x194xf64, 1>
        affine.store %11, %arg0[0, 6, %arg2 + %arg1 * 16 + 7] : memref<1x99x194xf64, 1>
        affine.for %arg3 = 0 to 6 {
          %16 = affine.load %arg0[0, -%arg3 + 90, -%arg2 - %arg1 * 16 + 186] : memref<1x99x194xf64, 1>
          affine.store %16, %arg0[0, %arg3 + 92, %arg2 + %arg1 * 16 + 7] : memref<1x99x194xf64, 1>
        }
        %12 = affine.load %arg0[0, 91, -%arg2 - %arg1 * 16 + 186] : memref<1x99x194xf64, 1>
        %13 = affine.load %arg0[0, 91, %arg2 + %arg1 * 16 + 7] : memref<1x99x194xf64, 1>
        %14 = arith.cmpi sle, %10, %c90_i64 : i64
        %15 = arith.select %14, %13, %12 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %15, %arg0[0, 91, %arg2 + %arg1 * 16 + 7] : memref<1x99x194xf64, 1>
      }
    }
    return
  }
}

// CHECK: #set = affine_set<(d0) : (-d0 + 89 >= 0)>
// CHECK:  func.func private @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI12_1_180__1_1_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__1_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E12_194__99__1_EE17BoundaryConditionI4FluxvESJ_I6ZipperS8_ES7_I6CenterSO_vE20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SR_SS_ST_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EESX_SZ_ESE_ISF_Li2ESG_ISF_Li2ELi1E9_194__99_EE8TripolarIS8_S8_S8_EvE16GridFittedBottomI5FieldISO_SO_vvvvSI_SF_vvvE23CenterImmersedConditionEvvvES7_#842$par30"(%arg0: memref<1x99x194xf64, 1>) {
// CHECK-NEXT:    affine.parallel (%arg1) = (0) to (180) {
// CHECK-NEXT:      %0 = affine.load %arg0[0, 7, %arg1 + 7] : memref<1x99x194xf64, 1>
// CHECK-NEXT:      affine.store %0, %arg0[0, 6, %arg1 + 7] : memref<1x99x194xf64, 1>
// CHECK-NEXT:      affine.parallel (%arg2) = (0) to (6) {
// CHECK-NEXT:        %4 = affine.load %arg0[0, -%arg2 + 90, -%arg1 + 186] : memref<1x99x194xf64, 1>
// CHECK-NEXT:        affine.store %4, %arg0[0, %arg2 + 92, %arg1 + 7] : memref<1x99x194xf64, 1>
// CHECK-NEXT:      }
// CHECK-NEXT:      %1 = affine.load %arg0[0, 91, -%arg1 + 186] : memref<1x99x194xf64, 1>
// CHECK-NEXT:      %2 = affine.load %arg0[0, 91, %arg1 + 7] : memref<1x99x194xf64, 1>
// CHECK-NEXT:      %3 = affine.if #set(%arg1) -> f64 {
// CHECK-NEXT:        affine.yield %2 : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        affine.yield %1 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.store %3, %arg0[0, 91, %arg1 + 7] : memref<1x99x194xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
