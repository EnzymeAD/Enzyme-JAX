// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(affine-cfg,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},func.func(canonicalize-loops),canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},llvm-to-affine-access,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},delinearize-indexing,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},simplify-affine-exprs,affine-cfg,canonicalize)" | FileCheck %s
#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>

module {

  func.func private @"##call__Z23gpu_move_gather_kernel_16CompilerMetadataI16OffsetStaticSizeI15_1_2__1_2__1_3_E12DynamicCheckvv7NDRangeILi3E10StaticSizeI9_1__1__3_ES4_I11_16__16__1_E5TupleI5Int64S8_S8_ES0_I9_0__0__0_EEE13CuTracedArrayI7Float64Li3ELi1E9_2__2__3_ESD_ISE_Li3ELi1E9_2__2__2_ESD_I5Int32Li1ELi1E4_3__E#287$par0"(%arg0: memref<3x2x2xf64, 1>, %arg1: memref<2x2x2xf64, 1>, %arg2: memref<3xi32, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %c16_i16 = arith.constant 16 : i16
    %c16_i64 = arith.constant 16 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0_i64 = arith.constant 0 : i64
    %c4_i64 = arith.constant 4 : i64
    %c-1_i64 = arith.constant -1 : i64
    %cst = arith.constant 0.000000e+00 : f64
    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<3x2x2xf64, 1>) -> !llvm.ptr<1>
    %1 = "enzymexla.memref2pointer"(%arg1) : (memref<2x2x2xf64, 1>) -> !llvm.ptr<1>
    %2 = "enzymexla.memref2pointer"(%arg2) : (memref<3xi32, 1>) -> !llvm.ptr<1>
    affine.parallel (%arg3, %arg4) = (0, 0) to (3, 256) {
      %3 = arith.addi %arg3, %c1 : index
      %4 = arith.addi %arg4, %c1 : index
      %5 = arith.index_castui %3 : index to i64
      %6 = arith.index_castui %4 : index to i64
      %7 = arith.addi %6, %c-1_i64 : i64
      %8 = arith.trunci %7 : i64 to i16
      %9 = arith.divui %8, %c16_i16 : i16
      %10 = arith.extui %9 nneg : i16 to i64
      %11 = arith.muli %10, %c16_i64 : i64
      %12 = arith.subi %7, %11 : i64
      %13 = arith.addi %12, %c1_i64 : i64
      %14 = arith.addi %10, %c1_i64 : i64
      %15 = arith.cmpi sge, %13, %c1_i64 : i64
      %16 = arith.cmpi sle, %13, %c2_i64 : i64
      %17 = arith.andi %15, %16 : i1
      %18 = arith.cmpi sle, %14, %c2_i64 : i64
      %19 = arith.andi %17, %18 : i1
      scf.if %19 {
        %20 = arith.addi %5, %c-1_i64 : i64
        %21 = llvm.getelementptr inbounds %2[%20] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
        %22 = llvm.load %21 invariant {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> i32
        %23 = arith.extsi %22 : i32 to i64
        %24 = arith.cmpi sle, %23, %c0_i64 : i64
        %25 = arith.select %24, %c1_i64, %23 {fastmathFlags = #llvm.fastmath<none>} : i64
        %26 = arith.muli %10, %c2_i64 : i64
        %27 = arith.addi %26, %13 : i64
        %28 = arith.addi %25, %c-1_i64 : i64
        %29 = arith.muli %28, %c4_i64 : i64
        %30 = arith.addi %27, %c-1_i64 : i64
        %31 = arith.addi %30, %29 : i64
        %32 = llvm.getelementptr inbounds %1[%31] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
        %33 = llvm.load %32 invariant {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
        %34 = arith.select %24, %cst, %33 {fastmathFlags = #llvm.fastmath<none>} : f64
        %35 = arith.muli %20, %c4_i64 : i64
        %36 = arith.addi %35, %c-1_i64 : i64
        %37 = arith.addi %36, %27 : i64
        %38 = llvm.getelementptr inbounds %0[%37] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
        %39 = llvm.load %38 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
        %40 = arith.addf %39, %34 {fastmathFlags = #llvm.fastmath<none>} : f64
        llvm.store %40, %38 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr<1>
      }
    }
    return
  }

}

// CHECK:#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK:  func.func private @"##call__Z23gpu_move_gather_kernel_16CompilerMetadataI16OffsetStaticSizeI15_1_2__1_2__1_3_E12DynamicCheckvv7NDRangeILi3E10StaticSizeI9_1__1__3_ES4_I11_16__16__1_E5TupleI5Int64S8_S8_ES0_I9_0__0__0_EEE13CuTracedArrayI7Float64Li3ELi1E9_2__2__3_ESD_ISE_Li3ELi1E9_2__2__2_ESD_I5Int32Li1ELi1E4_3__E#287$par0"(%arg0: memref<3x2x2xf64, 1>, %arg1: memref<2x2x2xf64, 1>, %arg2: memref<3xi32, 1>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:    %c4_i64 = arith.constant 4 : i64
// CHECK-NEXT:    %c-1_i64 = arith.constant -1 : i64
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (3, 2, 2) {
// CHECK-NEXT:      %0 = affine.load %arg2[%arg3] {alignment = 4 : i64, invariant, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<3xi32, 1>
// CHECK-NEXT:      %1 = arith.extsi %0 : i32 to i64
// CHECK-NEXT:      %2 = arith.cmpi sle, %1, %c0_i64 : i64
// CHECK-NEXT:      %3 = arith.select %2, %c1_i64, %1 {fastmathFlags = #llvm.fastmath<none>} : i64
// CHECK-NEXT:      %4 = arith.addi %3, %c-1_i64 : i64
// CHECK-NEXT:      %5 = arith.muli %4, %c4_i64 : i64
// CHECK-NEXT:      %6 = arith.index_cast %5 : i64 to index
// CHECK-NEXT:      %7 = affine.apply #map(%arg4, %arg5)
// CHECK-NEXT:      %8 = arith.addi %7, %6 : index
// CHECK-NEXT:      %9 = arith.remui %8, %c2 : index
// CHECK-NEXT:      %10 = arith.divui %8, %c2 : index
// CHECK-NEXT:      %11 = arith.remui %10, %c2 : index
// CHECK-NEXT:      %12 = arith.divui %10, %c2 : index
// CHECK-NEXT:      %13 = memref.load %arg1[%12, %11, %9] : memref<2x2x2xf64, 1>
// CHECK-NEXT:      %14 = arith.select %2, %cst, %13 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %15 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<3x2x2xf64, 1>
// CHECK-NEXT:      %16 = arith.addf %15, %14 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      affine.store %16, %arg0[%arg3, %arg4, %arg5] : memref<3x2x2xf64, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

