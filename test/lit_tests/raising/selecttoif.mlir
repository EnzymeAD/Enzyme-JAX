// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s


// CHECK: select
// CHECK: select
// CHECK-NOT: select

#set3 = affine_set<(d0, d1, d2, d3) : (-d0 - d1 * 16 + d3 + 102 >= 0, d3 + d2 * 16 >= 0, d2 * -16 - d3 + 179 >= 0)>
module {
  func.func private @"##call__Z33gpu__split_explicit_free_surface_16CompilerMetadataI16OffsetStaticSizeI14_1_180__1_103_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI7_12__7_ES4_I8_16__16_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE20ImmersedBoundaryGridI7Float648Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SG_SH_SI_28StaticVerticalDiscretizationI11OffsetArrayISF_Li1E13CuTracedArrayISF_Li1ELi1E5_35__EESL_ISF_Li1ESM_ISF_Li1ELi1E5_34__EESO_SQ_E8TripolarIS8_S8_S8_ESL_ISF_Li2ESM_ISF_Li2ELi1E10_194__123_EESV_SV_SV_vE16GridFittedBottomI5FieldI6CenterSZ_vvvvSL_ISF_Li3ESM_ISF_Li3ELi1E13_194__123__1_EESF_vvvE23CenterImmersedConditionEvvvESF_S11_SY_I4FaceSZ_vvvvS11_SF_vvvESY_ISZ_S16_vvvvS11_SF_vvvE21ForwardBackwardScheme#1287$par125"(%arg0: memref<34xf64, 1>, %arg1: memref<123x194xf64, 1>, %arg2: memref<123x194xf64, 1>, %arg3: memref<123x194xf64, 1>, %arg4: memref<1x123x194xf64, 1>, %arg5: memref<1x123x194xf64, 1>, %arg6: memref<1x123x194xf64, 1>, %arg7: memref<1x123x194xf64, 1>) {
    %c-12 = arith.constant -12 : index 
    %c12 = arith.constant 12 : index
    %c16 = arith.constant 16 : index
    %c-1_i64 = arith.constant -1 : i64 
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c16_i64 = arith.constant 16 : i64
    %c180_i64 = arith.constant 180 : i64
    %cst = arith.constant 0.000000e+00 : f64 
    %cst_0 = arith.constant 9.600000e+01 : f64
    affine.parallel (%arg8, %arg9, %arg10, %arg11) = (0, 0, 0, 0) to (7, 16, 12, 16) {
      %0 = arith.muli %arg9, %c16 overflow<nuw> : index 
      %1 = arith.addi %0, %arg11 : index
      %2 = arith.addi %1, %c1 : index
      %3 = arith.muli %arg8, %c-12 : index
      %4 = arith.muli %arg8, %c12 overflow<nuw> : index
      %5 = arith.addi %4, %arg10 : index
      %6 = arith.addi %3, %5 : index
      %7 = arith.addi %6, %c1 : index
      %8 = arith.index_cast %7 : index to i64
      %9 = arith.index_castui %2 : index to i64
      %10 = arith.index_castui %arg9 : index to i64
      %11 = arith.subi %c0_i64, %10 : i64
      %12 = arith.addi %11, %8 : i64
      %13 = arith.addi %12, %c-1_i64 : i64
      %14 = arith.muli %13, %c16_i64 : i64
      %15 = arith.addi %9, %14 : i64 
      %16 = arith.muli %arg8, %c16 : index
      %17 = arith.addi %16, %arg9 : index
      %18 = arith.index_castui %17 : index to i64
      %19 = arith.addi %18, %c1_i64 : i64
      affine.if #set3(%arg9, %arg8, %arg10, %arg11) {
        %20 = affine.load %arg5[0, %arg8 * 16 + %arg9 + 19, %arg11 + %arg10 * 16 + 7] : memref<1x123x194xf64, 1>
        %21 = affine.load %arg0[26] {alignment = 16 : i64, ordering = 0 : i64} : memref<34xf64, 1>
        %22 = affine.load %arg4[0, %arg8 * 16 + %arg9 + 19, 7] : memref<1x123x194xf64, 1>
        %23 = arith.cmpf ole, %21, %22 {fastmathFlags = #llvm.fastmath<none>} : f64
        %24 = affine.load %arg2[%arg8 * 16 + %arg9 + 19, 7] : memref<123x194xf64, 1>
        %25 = affine.load %arg6[0, %arg8 * 16 + %arg9 + 19, 7] : memref<1x123x194xf64, 1>
        %26 = arith.mulf %24, %25 {fastmathFlags = #llvm.fastmath<none>} : f64
        %27 = arith.select %23, %cst, %26 : f64
        %28 = affine.load %arg4[0, %arg8 * 16 + %arg9 + 19, 186] : memref<1x123x194xf64, 1>
        %29 = arith.cmpf ole, %21, %28 {fastmathFlags = #llvm.fastmath<none>} : f64 
        %30 = affine.load %arg2[%arg8 * 16 + %arg9 + 19, 186] : memref<123x194xf64, 1>
        %31 = affine.load %arg6[0, %arg8 * 16 + %arg9 + 19, 186] : memref<1x123x194xf64, 1>
        %32 = arith.mulf %30, %31 {fastmathFlags = #llvm.fastmath<none>} : f64
        %33 = arith.select %29, %cst, %32 : f64
        %34 = arith.subf %27, %33 {fastmathFlags = #llvm.fastmath<none>} : f64
        %35 = affine.load %arg4[0, %arg8 * 16 + %arg9 + 19, %arg11 + %arg10 * 16 + 8] : memref<1x123x194xf64, 1>
        %36 = arith.cmpf ole, %21, %35 {fastmathFlags = #llvm.fastmath<none>} : f64
        %37 = affine.load %arg2[%arg8 * 16 + %arg9 + 19, %arg11 + %arg10 * 16 + 8] : memref<123x194xf64, 1> 
        %38 = affine.load %arg6[0, %arg8 * 16 + %arg9 + 19, %arg11 + %arg10 * 16 + 8] : memref<1x123x194xf64, 1>
        %39 = arith.mulf %37, %38 {fastmathFlags = #llvm.fastmath<none>} : f64 
        %40 = arith.select %36, %cst, %39 : f64
        %41 = affine.load %arg4[0, %arg8 * 16 + %arg9 + 19, %arg11 + %arg10 * 16 + 7] : memref<1x123x194xf64, 1>
        %42 = arith.cmpf ole, %21, %41 {fastmathFlags = #llvm.fastmath<none>} : f64
        %43 = affine.load %arg2[%arg8 * 16 + %arg9 + 19, %arg11 + %arg10 * 16 + 7] : memref<123x194xf64, 1>
        %44 = affine.load %arg6[0, %arg8 * 16 + %arg9 + 19, %arg11 + %arg10 * 16 + 7] : memref<1x123x194xf64, 1>
        %45 = arith.mulf %43, %44 {fastmathFlags = #llvm.fastmath<none>} : f64
        %46 = arith.select %42, %cst, %45 : f64
        %47 = arith.subf %40, %46 {fastmathFlags = #llvm.fastmath<none>} : f64
        %48 = arith.cmpi ne, %15, %c180_i64 : i64
        %49 = arith.select %48, %47, %34 {fastmathFlags = #llvm.fastmath<none>} : f64
        affine.store %49, %arg5[0, %arg8 * 16 + %arg9 + 19, %arg11 + %arg10 * 16 + 7] : memref<1x123x194xf64, 1>
      }   
    }     
    return
  }  
}
