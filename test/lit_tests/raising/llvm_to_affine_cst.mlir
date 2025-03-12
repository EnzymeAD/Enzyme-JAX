// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module {
  llvm.func ptx_kernelcc @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI14_1_180__21_21_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I8_180__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__123__1_EE17BoundaryConditionI4FluxvESJ_I6ZipperS8_ES7_I6CenterSO_4FaceE20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SS_ST_SU_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EESY_S10_E8TripolarIS8_S8_S8_ESE_ISF_Li2ESG_ISF_Li2ELi1E10_194__123_EES15_S15_S15_vE16GridFittedBottomI5FieldISO_SO_vvvvSI_SF_vvvE23CenterImmersedConditionEvvvES7_I10NamedTupleI53__time___last__t___last_stage__t___iteration___stage_S7_ISF_SF_SF_S8_S8_EES1D_I36__u___v___w_______U___V___T___S___e_S7_ISE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__34_EES1H_SE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__35_EESI_S18_ISP_SO_vvvvSI_SF_vvvES18_ISO_SP_vvvvSI_SF_vvvES1H_S1H_S1H_EEE#1325"(%arg0: !llvm.ptr<1> {llvm.align = 128 : i64, llvm.dereferenceable = 190896 : i64, llvm.dereferenceable_or_null = 190896 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync"], sym_visibility = "private", will_return} {
    %c19988_i64 = arith.constant 19988 : i64
    %c104_i64 = arith.constant 104 : i64
    %c102_i64 = arith.constant 102 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c181_i64 = arith.constant 181 : i64
    %c1_i64 = arith.constant 1 : i64
    %c180_i16 = arith.constant 180 : i16
    %c-180_i64 = arith.constant -180 : i64
    %c20_i64 = arith.constant 20 : i64
    %c180_i64 = arith.constant 180 : i64
    %c21_i64 = arith.constant 21 : i64
    %c23862_i64 = arith.constant 23862 : i64
    %c3692_i64 = arith.constant 3692 : i64
    %c3498_i64 = arith.constant 3498 : i64
    %c6_i64 = arith.constant 6 : i64
    %c194_i64 = arith.constant 194 : i64
    %c90_i64 = arith.constant 90 : i64
    %0 = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 1> : i32
    %1 = arith.addi %0, %c1_i32 : i32
    %2 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 180> : i32
    %3 = arith.addi %2, %c1_i32 : i32
    %4 = arith.extui %1 : i32 to i64
    %5 = arith.extui %3 : i32 to i64
    %6 = arith.addi %5, %c-1_i64 : i64
    %7 = arith.trunci %6 : i64 to i16
    %8 = arith.divui %7, %c180_i16 : i16
    %9 = arith.extui %8 : i16 to i64
    %10 = arith.muli %9, %c-180_i64 : i64
    %11 = arith.addi %5, %10 : i64
    %12 = arith.addi %9, %4 : i64
    %13 = arith.addi %12, %c20_i64 : i64
    %14 = arith.cmpi sge, %11, %c1_i64 : i64
    %15 = arith.cmpi sle, %11, %c180_i64 : i64
    %16 = arith.andi %14, %15 : i1
    %17 = arith.cmpi sge, %13, %c21_i64 : i64
    %18 = arith.cmpi sle, %13, %c21_i64 : i64
    %19 = arith.andi %17, %18 : i1
    %20 = arith.andi %19, %16 : i1
    scf.if %20 {
      %21 = arith.addi %12, %c-1_i64 : i64
      %22 = arith.muli %21, %c23862_i64 : i64
      %23 = arith.addi %22, %11 : i64
      %24 = arith.addi %23, %c3692_i64 : i64
      %25 = llvm.getelementptr inbounds %arg0[%24] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %26 = llvm.load %25 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %27 = arith.addi %22, %11 : i64
      %28 = arith.addi %27, %c3498_i64 : i64
      %29 = llvm.getelementptr inbounds %arg0[%28] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      llvm.store %26, %29 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr<1>
      %30 = arith.subi %c181_i64, %11 : i64
      %31 = arith.addi %30, %22 : i64
      %32 = arith.addi %22, %11 : i64
      affine.for %arg1 = 0 to 18 {
        %41 = arith.index_cast %arg1 : index to i64
        %42 = arith.subi %c102_i64, %41 : i64
        %43 = arith.muli %42, %c194_i64 : i64
        %44 = arith.addi %43, %31 : i64
        %45 = arith.addi %44, %c6_i64 : i64
        %46 = llvm.getelementptr inbounds %arg0[%45] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
        %47 = llvm.load %46 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
        %48 = arith.addi %41, %c104_i64 : i64
        %49 = arith.muli %48, %c194_i64 : i64
        %50 = arith.addi %49, %32 : i64
        %51 = arith.addi %50, %c6_i64 : i64
        %52 = llvm.getelementptr inbounds %arg0[%51] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
        llvm.store %47, %52 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr<1>
      }
      %33 = arith.addi %31, %c19988_i64 : i64
      %34 = llvm.getelementptr inbounds %arg0[%33] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %35 = llvm.load %34 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %36 = arith.addi %32, %c19988_i64 : i64
      %37 = llvm.getelementptr inbounds %arg0[%36] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
      %38 = llvm.load %37 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr<1> -> f64
      %39 = arith.cmpi sle, %11, %c90_i64 : i64
      %40 = arith.select %39, %38, %35 {fastmathFlags = #llvm.fastmath<none>} : f64
      llvm.store %40, %37 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr<1>
    }
    llvm.return
  }
}

// CHECK:  #map = affine_map<(d0) -> (d0 * -194 + 19975)>
// CHECK:  llvm.func ptx_kernelcc @"##call__Z31gpu__fill_south_and_north_halo_16CompilerMetadataI16OffsetStaticSizeI14_1_180__21_21_E12DynamicCheckvv7NDRangeILi2E10StaticSizeI6_1__1_ES4_I8_180__1_E5TupleI5Int64S8_E13KernelOffsetsIS9_EEE11OffsetArrayI7Float64Li3E13CuTracedArrayISF_Li3ELi1E13_194__123__1_EE17BoundaryConditionI4FluxvESJ_I6ZipperS8_ES7_I6CenterSO_4FaceE20ImmersedBoundaryGridISF_8Periodic14RightConnected7Bounded28OrthogonalSphericalShellGridISF_SS_ST_SU_28StaticVerticalDiscretizationISE_ISF_Li1ESG_ISF_Li1ELi1E5_35__EESE_ISF_Li1ESG_ISF_Li1ELi1E5_34__EESY_S10_E8TripolarIS8_S8_S8_ESE_ISF_Li2ESG_ISF_Li2ELi1E10_194__123_EES15_S15_S15_vE16GridFittedBottomI5FieldISO_SO_vvvvSI_SF_vvvE23CenterImmersedConditionEvvvES7_I10NamedTupleI53__time___last__t___last_stage__t___iteration___stage_S7_ISF_SF_SF_S8_S8_EES1D_I36__u___v___w_______U___V___T___S___e_S7_ISE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__34_EES1H_SE_ISF_Li3ESG_ISF_Li3ELi1E13_194__99__35_EESI_S18_ISP_SO_vvvvSI_SF_vvvES18_ISO_SP_vvvvSI_SF_vvvES1H_S1H_S1H_EEE#1325"(%arg0: !llvm.ptr<1> {llvm.align = 128 : i64, llvm.dereferenceable = 190896 : i64, llvm.dereferenceable_or_null = 190896 : i64, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync"], sym_visibility = "private", will_return} {
// CHECK-NEXT:    %c19988 = arith.constant 19988 : index
// CHECK-NEXT:    %c20169 = arith.constant 20169 : index
// CHECK-NEXT:    %c3692 = arith.constant 3692 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c19988_i64 = arith.constant 19988 : i64
// CHECK-NEXT:    %c104_i64 = arith.constant 104 : i64
// CHECK-NEXT:    %c-1_i64 = arith.constant -1 : i64
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %c180_i16 = arith.constant 180 : i16
// CHECK-NEXT:    %c-180_i64 = arith.constant -180 : i64
// CHECK-NEXT:    %c20_i64 = arith.constant 20 : i64
// CHECK-NEXT:    %c180_i64 = arith.constant 180 : i64
// CHECK-NEXT:    %c21_i64 = arith.constant 21 : i64
// CHECK-NEXT:    %c23862_i64 = arith.constant 23862 : i64
// CHECK-NEXT:    %c3498_i64 = arith.constant 3498 : i64
// CHECK-NEXT:    %c6_i64 = arith.constant 6 : i64
// CHECK-NEXT:    %c194_i64 = arith.constant 194 : i64
// CHECK-NEXT:    %c90_i64 = arith.constant 90 : i64
// CHECK-NEXT:    %0 = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 1> : i32
// CHECK-NEXT:    %1 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:    %2 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 180> : i32
// CHECK-NEXT:    %3 = arith.addi %2, %c1_i32 : i32
// CHECK-NEXT:    %4 = arith.extui %1 : i32 to i64
// CHECK-NEXT:    %5 = arith.extui %3 : i32 to i64
// CHECK-NEXT:    %6 = arith.addi %5, %c-1_i64 : i64
// CHECK-NEXT:    %7 = arith.trunci %6 : i64 to i16
// CHECK-NEXT:    %8 = arith.divui %7, %c180_i16 : i16
// CHECK-NEXT:    %9 = arith.extui %8 : i16 to i64
// CHECK-NEXT:    %10 = arith.muli %9, %c-180_i64 : i64
// CHECK-NEXT:    %11 = arith.addi %5, %10 : i64
// CHECK-NEXT:    %12 = arith.addi %9, %4 : i64
// CHECK-NEXT:    %13 = arith.addi %12, %c20_i64 : i64
// CHECK-NEXT:    %14 = arith.cmpi sge, %11, %c1_i64 : i64
// CHECK-NEXT:    %15 = arith.cmpi sle, %11, %c180_i64 : i64
// CHECK-NEXT:    %16 = arith.andi %14, %15 : i1
// CHECK-NEXT:    %17 = arith.cmpi sge, %13, %c21_i64 : i64
// CHECK-NEXT:    %18 = arith.cmpi sle, %13, %c21_i64 : i64
// CHECK-NEXT:    %19 = arith.andi %17, %18 : i1
// CHECK-NEXT:    %20 = arith.andi %19, %16 : i1
// CHECK-NEXT:    scf.if %20 {
// CHECK-NEXT:      %21 = arith.addi %12, %c-1_i64 : i64
// CHECK-NEXT:      %22 = arith.muli %21, %c23862_i64 : i64
// CHECK-NEXT:      %23 = arith.index_cast %10 : i64 to index
// CHECK-NEXT:      %24 = arith.index_cast %5 : i64 to index
// CHECK-NEXT:      %25 = arith.addi %24, %23 : index
// CHECK-NEXT:      %26 = arith.index_cast %22 : i64 to index
// CHECK-NEXT:      %27 = arith.addi %26, %25 : index
// CHECK-NEXT:      %28 = arith.addi %27, %c3692 : index
// CHECK-NEXT:      %29 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
// CHECK-NEXT:      %30 = memref.load %29[%28] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
// CHECK-NEXT:      %31 = arith.addi %22, %11 : i64
// CHECK-NEXT:      %32 = arith.addi %31, %c3498_i64 : i64
// CHECK-NEXT:      %33 = llvm.getelementptr inbounds %arg0[%32] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
// CHECK-NEXT:      %34 = "enzymexla.pointer2memref"(%33) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
// CHECK-NEXT:      memref.store %30, %34[%c0] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
// CHECK-NEXT:      %35 = arith.addi %22, %11 : i64
// CHECK-NEXT:      affine.for %arg1 = 0 to 18 {
// CHECK-NEXT:        %57 = arith.index_cast %arg1 : index to i64
// CHECK-NEXT:        %58 = arith.index_cast %22 : i64 to index
// CHECK-NEXT:        %59 = arith.index_cast %10 : i64 to index
// CHECK-NEXT:        %60 = arith.index_cast %5 : i64 to index
// CHECK-NEXT:        %61 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
// CHECK-NEXT:        %62 = affine.apply #map(%arg1)
// CHECK-NEXT:        %63 = arith.addi %62, %58 : index
// CHECK-NEXT:        %64 = arith.subi %63, %59 : index
// CHECK-NEXT:        %65 = arith.subi %64, %60 : index
// CHECK-NEXT:        %66 = memref.load %61[%65] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
// CHECK-NEXT:        %67 = arith.addi %57, %c104_i64 : i64
// CHECK-NEXT:        %68 = arith.muli %67, %c194_i64 : i64
// CHECK-NEXT:        %69 = arith.addi %68, %35 : i64
// CHECK-NEXT:        %70 = arith.addi %69, %c6_i64 : i64
// CHECK-NEXT:        %71 = llvm.getelementptr inbounds %arg0[%70] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
// CHECK-NEXT:        %72 = "enzymexla.pointer2memref"(%71) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
// CHECK-NEXT:        memref.store %66, %72[%c0] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
// CHECK-NEXT:      }
// CHECK-NEXT:      %36 = arith.index_cast %22 : i64 to index
// CHECK-NEXT:      %37 = arith.index_cast %10 : i64 to index
// CHECK-NEXT:      %38 = arith.index_cast %5 : i64 to index
// CHECK-NEXT:      %39 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
// CHECK-NEXT:      %40 = arith.addi %36, %c20169 : index
// CHECK-NEXT:      %41 = arith.subi %40, %37 : index
// CHECK-NEXT:      %42 = arith.subi %41, %38 : index
// CHECK-NEXT:      %43 = memref.load %39[%42] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
// CHECK-NEXT:      %44 = arith.addi %35, %c19988_i64 : i64
// CHECK-NEXT:      %45 = arith.index_cast %10 : i64 to index
// CHECK-NEXT:      %46 = arith.index_cast %5 : i64 to index
// CHECK-NEXT:      %47 = arith.addi %46, %45 : index
// CHECK-NEXT:      %48 = arith.index_cast %22 : i64 to index
// CHECK-NEXT:      %49 = arith.addi %48, %47 : index
// CHECK-NEXT:      %50 = arith.addi %49, %c19988 : index
// CHECK-NEXT:      %51 = llvm.getelementptr inbounds %arg0[%44] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f64
// CHECK-NEXT:      %52 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
// CHECK-NEXT:      %53 = memref.load %52[%50] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
// CHECK-NEXT:      %54 = arith.cmpi sle, %11, %c90_i64 : i64
// CHECK-NEXT:      %55 = arith.select %54, %53, %43 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %56 = "enzymexla.pointer2memref"(%51) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
// CHECK-NEXT:      memref.store %55, %56[%c0] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
