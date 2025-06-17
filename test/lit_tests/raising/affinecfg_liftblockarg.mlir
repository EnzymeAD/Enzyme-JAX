// RUN: enzymexlamlir-opt --affine-cfg %s | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "custom_tbaa">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "custom_tbaa_addrspace(1)", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module @"reactant_spmv!" attributes {gpu.container_module, mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Dwarf Version", 2 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>]
  gpu.module @gpumod___call__Z16gpu_spmv_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi1E5TupleI5OneToI5Int64EEE7NDRangeILi1ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li1ELi1E4_5__E22GenericSparseMatrixCSRISD_S5_SC_IS5_Li1ELi1E4_6__ESC_IS5_Li1ELi1E5_16__ESC_ISD_Li1ELi1E5_16__EESE__304 {
    gpu.func @__call__Z16gpu_spmv_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi1E5TupleI5OneToI5Int64EEE7NDRangeILi1ES0_S0_S8_S8_EE13CuTracedArrayI7Float64Li1ELi1E4_5__E22GenericSparseMatrixCSRISD_S5_SC_IS5_Li1ELi1E4_6__ESC_IS5_Li1ELi1E5_16__ESC_ISD_Li1ELi1E5_16__EESE__304(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>, %arg4: !llvm.ptr<1>) kernel {
      %c-1 = arith.constant -1 : index
      %c16_i64 = arith.constant 16 : i64
      %cst = arith.constant 0.000000e+00 : f64
      %c6_i64 = arith.constant 6 : i64
      %c5_i64 = arith.constant 5 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = ub.poison : f64
      %1 = ub.poison : i64
      %c2_i32 = arith.constant 2 : i32
      %c-1_i64 = arith.constant -1 : i64
      %true = arith.constant true
      %false = arith.constant false
      %2 = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 1> : i32
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = nvvm.read.ptx.sreg.tid.x range <i32, 0, 5> : i32
      %5 = arith.addi %4, %c1_i32 : i32
      %6 = arith.extui %3 : i32 to i64
      %7 = arith.extui %5 : i32 to i64
      %8 = arith.addi %6, %c-1_i64 : i64
      %9 = arith.muli %8, %c5_i64 : i64
      %10 = arith.addi %9, %7 : i64
      %11 = arith.index_cast %9 : i64 to index
      %12 = arith.index_cast %7 : i64 to index
      %13 = arith.addi %11, %12 : index
      %14 = arith.index_cast %9 : i64 to index
      %15 = arith.index_cast %7 : i64 to index
      %16 = arith.addi %14, %15 : index
      %17 = arith.index_cast %9 : i64 to index
      %18 = arith.index_cast %7 : i64 to index
      %19 = arith.addi %17, %18 : index
      %20 = arith.cmpi sge, %10, %c1_i64 : i64
      %21 = arith.cmpi sle, %10, %c5_i64 : i64
      %22 = arith.andi %20, %21 : i1
      %23 = arith.cmpi sgt, %10, %c5_i64 : i64
      %24 = arith.xori %22, %true : i1
      %25 = arith.andi %22, %23 : i1
      %26 = arith.ori %25, %24 : i1
        %27 = arith.addi %10, %c-1_i64 : i64
        %28 = arith.cmpi uge, %27, %c6_i64 : i64
          %29 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr<1>) -> memref<?xi64, 1>
          %30 = affine.load %29[symbol(%16) - 1] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi64, 1>
          %31 = arith.cmpi uge, %10, %c6_i64 : i64
            %32 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr<1>) -> memref<?xi64, 1>
            %33 = affine.load %32[symbol(%13)] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi64, 1>
            %34 = arith.addi %33, %c-1_i64 : i64
            %35 = arith.cmpi sgt, %30, %34 : i64
            %36 = arith.select %35, %30, %33 : i64
            %37 = arith.addi %36, %c-1_i64 : i64
            %38 = arith.cmpi slt, %37, %30 : i64
              %41:4 = scf.while (%arg5 = %30, %arg6 = %cst) : (i64, f64) -> (i64, f64, f64, i32) {
                %44 = arith.addi %arg5, %c-1_i64 : i64
                %45 = arith.index_cast %arg5 : i64 to index
                %46 = arith.addi %45, %c-1 : index
                %47 = arith.index_cast %arg5 : i64 to index
                %48 = arith.addi %47, %c-1 : index
                %49 = arith.cmpi uge, %44, %c16_i64 : i64
                  %51 = "enzymexla.pointer2memref"(%arg2) : (!llvm.ptr<1>) -> memref<?xi64, 1 : index>
                  %52 = memref.load %51[%46] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi64, 1 : index>
                  %53 = "enzymexla.pointer2memref"(%arg3) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
                  %54 = memref.load %53[%48] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
                  %55 = arith.addi %52, %c-1_i64 : i64
                  %56 = arith.index_cast %52 : i64 to index
                  %57 = arith.addi %56, %c-1 : index
                  %58 = arith.cmpi uge, %55, %c5_i64 : i64
                  %59:5 = scf.if %58 -> (i64, f64, f64, i32, i1) {
                    scf.yield %1, %0, %0, %c2_i32, %false : i64, f64, f64, i32, i1
                  } else {
                    %60 = "enzymexla.pointer2memref"(%arg4) : (!llvm.ptr<1>) -> memref<?xf64, 1 : index>
                    %61 = memref.load %60[%57] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xf64, 1 : index>
                    %62 = arith.mulf %54, %61 {fastmathFlags = #llvm.fastmath<none>} : f64
                    %63 = arith.addf %arg6, %62 {fastmathFlags = #llvm.fastmath<none>} : f64
                    %64 = arith.addi %arg5, %c1_i64 : i64
                    %65 = arith.cmpi eq, %arg5, %37 : i64
                    %66 = arith.extui %65 : i1 to i32
                    %67 = arith.cmpi ne, %arg5, %37 : i64
                    scf.yield %64, %63, %63, %66, %67 : i64, f64, f64, i32, i1
                  }
                scf.condition(%false) %1, %0, %0, %c2_i32 : i64, f64, f64, i32
              } do {
              ^bb0(%arg5: i64, %arg6: f64, %arg7: f64, %arg8: i32):
                scf.yield %arg5, %arg6 : i64, f64
              }
      gpu.return
    }
  }
}

// CHECK: gpu.func @__call__Z16gpu_spmv_kernel_16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndicesILi1E5TupleI5OneToI5Int64EEE7ND
