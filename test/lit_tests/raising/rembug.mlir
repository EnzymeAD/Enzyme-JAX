// RUN: enzymexlamlir-opt  "--pass-pipeline=builtin.module(func.func(canonicalize-loops),llvm-to-affine-access)" %s | FileCheck %s

#set = affine_set<(d0, d1) : (d0 floordiv 16 + (d1 floordiv 12) * 16 + 1 >= 0, -(d0 floordiv 16) - (d1 floordiv 12) * 16 + 84 >= 0, d0 mod 16 + (d1 mod 12) * 16 >= 0, -(d0 mod 16) - (d1 mod 12) * 16 + 179 >= 0)>
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module @"reactant_run!" attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @bar(%arg0: tensor<1x187x194xf64>, %arg1: tensor<34xf64>, %arg2: tensor<1x187x194xf64>) -> tensor<1x187x194xf64> {
    %0 = enzymexla.jit_call @"##call__Z28gpu__mask_immersed_field_xy_16CompilerMetadataI10StaticSizeI9_180__85_E12DynamicCheckvv7NDRangeILi2ES0_I7_12__6_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E13_194__187__1_EESC_vvvE5TupleIS9_SA_vE20ImmersedBoundaryGridISC_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISC_SK_SL_SM_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_35__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EESQ_SS_ESB_ISC_Li2ESD_ISC_Li2ELi1E10_194__187_EE8TripolarI5Int64SX_SX_EvE16GridFittedBottomIS8_ISA_SA_vvvvSF_SC_vvvE23CenterImmersedConditionEvvvESC_SX_#802$par0" (%arg0, %arg1, %arg2) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<1x187x194xf64>, tensor<34xf64>, tensor<1x187x194xf64>) -> tensor<1x187x194xf64>
    return %0 : tensor<1x187x194xf64>
  }
  func.func private @"##call__Z28gpu__mask_immersed_field_xy_16CompilerMetadataI10StaticSizeI9_180__85_E12DynamicCheckvv7NDRangeILi2ES0_I7_12__6_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E13_194__187__1_EESC_vvvE5TupleIS9_SA_vE20ImmersedBoundaryGridISC_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISC_SK_SL_SM_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_35__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EESQ_SS_ESB_ISC_Li2ESD_ISC_Li2ELi1E10_194__187_EE8TripolarI5Int64SX_SX_EvE16GridFittedBottomIS8_ISA_SA_vvvvSF_SC_vvvE23CenterImmersedConditionEvvvESC_SX_#802$par0"(%arg0: memref<1x187x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<1x187x194xf64, 1>) {
    %c51_i64 = arith.constant 51 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c1 = arith.constant 1 : index
    %c12_i32 = arith.constant 12 : i32
    %c-12_i64 = arith.constant -12 : i64
    %c16_i16 = arith.constant 16 : i16
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c194_i64 = arith.constant 194 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %cst = arith.constant 0.000000e+00 : f64
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<1x187x194xf64, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.memref2pointer"(%arg1) : (memref<34xf64, 1>) -> !llvm.ptr<1>
    %2 = "enzymexla.memref2pointer"(%arg2) : (memref<1x187x194xf64, 1>) -> !llvm.ptr<1>
    affine.parallel (%arg3, %arg4) = (0, 0) to (72, 256) {
      %3 = arith.addi %arg3, %c1 : index
      %4 = arith.addi %arg4, %c1 : index
      %5 = arith.index_castui %3 : index to i64
      %6 = arith.addi %5, %c-1_i64 : i64
      %7 = arith.trunci %6 : i64 to i32
      %8 = arith.divui %7, %c12_i32 : i32
      %9 = arith.extui %8 : i32 to i64
      %10 = arith.muli %9, %c-12_i64 : i64
      %11 = arith.addi %10, %5 : i64
      %12 = arith.index_castui %4 : index to i64
      %13 = arith.addi %12, %c-1_i64 : i64
      %14 = arith.trunci %13 : i64 to i16
      %15 = arith.divui %14, %c16_i16 : i16
      %16 = arith.extui %15 : i16 to i64
      %17 = arith.subi %c0_i64, %16 : i64
      %18 = arith.addi %17, %11 : i64
      %19 = arith.addi %18, %c-1_i64 : i64
      %20 = arith.muli %19, %c16_i64 : i64
      %21 = arith.addi %12, %20 : i64
      %22 = arith.muli %9, %c16_i64 : i64
      %23 = arith.addi %22, %16 : i64
      affine.if #set(%arg4, %arg3) {
        %24 = llvm.getelementptr inbounds %1[26] : (!llvm.ptr<1>) -> !llvm.ptr<1>, f64
        %25 = llvm.load %24 {alignment = 16 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
        %26 = arith.addi %23, %c51_i64 : i64
        %27 = arith.muli %26, %c194_i64 : i64
        %28 = arith.addi %27, %21 : i64
        %29 = arith.addi %28, %c6_i64 : i64
        %30 = llvm.getelementptr inbounds %2[%29] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
        %31 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
        %32 = arith.cmpf ole, %25, %31 {fastmathFlags = #llvm.fastmath<none>} : f64
        %33 = arith.addi %27, %21 : i64
        %34 = arith.addi %33, %c5_i64 : i64
        %35 = llvm.getelementptr inbounds %2[%34] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
        %36 = llvm.load %35 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
        %37 = arith.cmpf ole, %25, %36 {fastmathFlags = #llvm.fastmath<none>} : f64
        %38 = arith.ori %32, %37 : i1
        %39 = llvm.getelementptr inbounds %0[%29] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
        %40 = llvm.load %39 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
        %41 = arith.select %38, %cst, %40 : f64
        llvm.store %41, %39 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr<1>
      }
    }
    return
  }
}

// CHECK:  func.func private @"##call__Z28gpu__mask_immersed_field_xy_16CompilerMetadataI10StaticSizeI9_180__85_E12DynamicCheckvv7NDRangeILi2ES0_I7_12__6_ES0_I8_16__16_EvvEE5FieldI4Face6Centervvvv11OffsetArrayI7Float64Li3E13CuTracedArrayISC_Li3ELi1E13_194__187__1_EESC_vvvE5TupleIS9_SA_vE20ImmersedBoundaryGridISC_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISC_SK_SL_SM_28StaticVerticalDiscretizationISB_ISC_Li1ESD_ISC_Li1ELi1E5_35__EESB_ISC_Li1ESD_ISC_Li1ELi1E5_34__EESQ_SS_ESB_ISC_Li2ESD_ISC_Li2ELi1E10_194__187_EE8TripolarI5Int64SX_SX_EvE16GridFittedBottomIS8_ISA_SA_vvvvSF_SC_vvvE23CenterImmersedConditionEvvvESC_SX_#802$par0"(%arg0: memref<1x187x194xf64, 1>, %arg1: memref<34xf64, 1>, %arg2: memref<1x187x194xf64, 1>) {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<1x187x194xf64, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:    %1 = "enzymexla.memref2pointer"(%arg1) : (memref<34xf64, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:    %2 = "enzymexla.memref2pointer"(%arg2) : (memref<1x187x194xf64, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:    affine.parallel (%arg3, %arg4) = (0, 0) to (72, 256) {
// CHECK-NEXT:      affine.if #set(%arg4, %arg3) {
// CHECK-NEXT:        %3 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr<1>) -> memref<?xf64, 1>
// CHECK-NEXT:        %4 = affine.load %3[26] {alignment = 16 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1>
// CHECK-NEXT:        %5 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr<1>) -> memref<?xf64, 1>
// CHECK-NEXT:        %6 = affine.load %5[%arg3 * 16 + %arg4 + (%arg3 floordiv 12) * 2912 + (%arg4 floordiv 16) * 178 + 9901] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1>
// CHECK-NEXT:        %7 = arith.cmpf ole, %4, %6 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %8 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr<1>) -> memref<?xf64, 1>
// CHECK-NEXT:        %9 = affine.load %8[%arg3 * 16 + %arg4 + (%arg3 floordiv 12) * 2912 + (%arg4 floordiv 16) * 178 + 9900] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1>
// CHECK-NEXT:        %10 = arith.cmpf ole, %4, %9 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:        %11 = arith.ori %7, %10 : i1
// CHECK-NEXT:        %12 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xf64, 1>
// CHECK-NEXT:        %13 = affine.load %12[%arg3 * 16 + %arg4 + (%arg3 floordiv 12) * 2912 + (%arg4 floordiv 16) * 178 + 9901] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1>
// CHECK-NEXT:        %14 = arith.select %11, %cst, %13 : f64
// CHECK-NEXT:        %15 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr<1>) -> memref<?xf64, 1>
// CHECK-NEXT:        affine.store %14, %15[%arg3 * 16 + %arg4 + (%arg3 floordiv 12) * 2912 + (%arg4 floordiv 16) * 178 + 9901] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
