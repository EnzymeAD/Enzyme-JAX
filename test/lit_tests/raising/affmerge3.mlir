// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func private @"##call__Z28gpu__mask_immersed_field_xy_16CompilerMetadataI10StaticSizeI9_180__85_E12DynamicCheckvv7NDRangeILi2ES0_I7_12__6_ES0_I8_16__16_EvvEE5FieldI6Center4Facevvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E13_194__187__1_EESC_vvvE5TupleIS9_SA_vE20ImmersedBoundaryGridISC_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISC_SK_SL_SM_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_35__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EESQ_SS_ESB_ISC_Li2ESD_ISC_Li2ELi1E10_194__187_EE8TripolarI5Int64SX_SX_EvE16GridFittedBottomIS8_IS9_S9_vvvvSF_SC_vvvE23CenterImmersedConditionEvvvESC_SX_#1339$par77"(%arg0: memref<1x187x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<1x187x194xf64, 1>) {
    %c16 = arith.constant 16 : index
    %c-1_i64 = arith.constant -1 : i64
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f64
    affine.parallel (%arg3, %arg4) = (0, 0) to (6, 16) {
      %0 = arith.addi %arg4, %c1 : index
      %1 = arith.muli %arg3, %c16 : index
      %2 = arith.addi %0, %1 : index
      %3 = arith.index_castui %2 : index to i64
      "test.op"(%3) : (i64) -> ()
    }
    return
  }
}

// CHECK:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    affine.parallel (%arg3) = (0) to (96) {
// CHECK-NEXT:      %0 = arith.addi %arg3, %c1 : index
// CHECK-NEXT:      %1 = arith.index_castui %0 : index to i64
// CHECK-NEXT:      "test.op"(%1) : (i64) -> ()
// CHECK-NEXT:    }
