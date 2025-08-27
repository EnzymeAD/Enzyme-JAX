// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(libdevice-funcs-raise,canonicalize,llvm-to-memref-access,polygeist-mem2reg,canonicalize,convert-llvm-to-cf,canonicalize,polygeist-mem2reg,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops),llvm.func(canonicalize-loops),canonicalize-scf-for,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops),llvm.func(canonicalize-loops),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,llvm-to-affine-access,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize)' | FileCheck %s

module {
  llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPfS_(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: !llvm.ptr {llvm.noundef}) attributes {} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.mlir.constant(1 : i64) : i64
    %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %4 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %5 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg2, %6 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %1, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %8 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %9 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
    %10 = llvm.sdiv %9, %2 : i32
    %11 = llvm.icmp "ult" %8, %10 : i32
    llvm.cond_br %11, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    llvm.call @_Z26__enzyme_set_checkpointingm(%3) : (i64 {llvm.noundef}) -> ()
    %12 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %13 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    llvm.call fastcc @_ZL4kernPfS_(%12, %13) : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> ()
    %14 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %15 = llvm.load %5 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    llvm.call fastcc @_ZL4kernPfS_(%14, %15) : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> ()
    llvm.br ^bb3
  ^bb3:  // pred: ^bb2
    %16 = llvm.load %7 {alignment = 4 : i64} : !llvm.ptr -> i32
    %17 = llvm.add %16, %0 : i32
    llvm.store %17, %7 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.br ^bb1 {loop_annotation = #llvm.loop_annotation<mustProgress = true, startLoc = loc(fused<#llvm.di_lexical_block<scope = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = <id = distinct[0]<>, sourceLanguage = DW_LANG_C_plus_plus_11, file = <"lbm.cu" in "/home/pangoraw/Enzyme-GPU-Tests/LBM">, producer = "clang version 22.0.0git (https://github.com/llvm/llvm-project.git f7b09ad700f2d8ae9ad230f6fc85de81e3a6565b)", isOptimized = false, emissionKind = Full, nameTableKind = None>, scope = #llvm.di_file<"lbm.cu" in "/home/pangoraw/Enzyme-GPU-Tests/LBM">, name = "CUDA_LBM_kernel_loop_inner", linkageName = "_Z26CUDA_LBM_kernel_loop_inneriPfS_", file = <"lbm.cu" in "/home/pangoraw/Enzyme-GPU-Tests/LBM">, line = 115, scopeLine = 115, subprogramFlags = Definition, type = <types = #llvm.di_null_type, #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>, #llvm.di_derived_type<tag = DW_TAG_typedef, name = "LBM_Grid", baseType = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>, sizeInBits = 64>>, #llvm.di_derived_type<tag = DW_TAG_typedef, name = "LBM_Grid", baseType = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>, sizeInBits = 64>>>>, file = <"lbm.cu" in "/home/pangoraw/Enzyme-GPU-Tests/LBM">, line = 117, column = 9>>["lbm.cu":117:9]), endLoc = loc(fused<#llvm.di_lexical_block<scope = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = <id = distinct[0]<>, sourceLanguage = DW_LANG_C_plus_plus_11, file = <"lbm.cu" in "/home/pangoraw/Enzyme-GPU-Tests/LBM">, producer = "clang version 22.0.0git (https://github.com/llvm/llvm-project.git f7b09ad700f2d8ae9ad230f6fc85de81e3a6565b)", isOptimized = false, emissionKind = Full, nameTableKind = None>, scope = #llvm.di_file<"lbm.cu" in "/home/pangoraw/Enzyme-GPU-Tests/LBM">, name = "CUDA_LBM_kernel_loop_inner", linkageName = "_Z26CUDA_LBM_kernel_loop_inneriPfS_", file = <"lbm.cu" in "/home/pangoraw/Enzyme-GPU-Tests/LBM">, line = 115, scopeLine = 115, subprogramFlags = Definition, type = <types = #llvm.di_null_type, #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int", sizeInBits = 32, encoding = DW_ATE_signed>, #llvm.di_derived_type<tag = DW_TAG_typedef, name = "LBM_Grid", baseType = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>, sizeInBits = 64>>, #llvm.di_derived_type<tag = DW_TAG_typedef, name = "LBM_Grid", baseType = #llvm.di_derived_type<tag = DW_TAG_pointer_type, baseType = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "float", sizeInBits = 32, encoding = DW_ATE_float>, sizeInBits = 64>>>>, file = <"lbm.cu" in "/home/pangoraw/Enzyme-GPU-Tests/LBM">, line = 117, column = 9>>["lbm.cu":0:29])>}
  ^bb4:  // pred: ^bb1
    llvm.return
  }
  llvm.func local_unnamed_addr @_Z26__enzyme_set_checkpointingm(i64 {llvm.noundef}) attributes {approx_func_fp_math = true, frame_pointer = #llvm.framePointerKind<all>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unsafe_fp_math = true}
  llvm.func local_unnamed_addr @_ZL4kernPfS_(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) attributes {approx_func_fp_math = true, frame_pointer = #llvm.framePointerKind<all>, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unsafe_fp_math = true}
}

// CHECK:  llvm.func local_unnamed_addr @_Z26CUDA_LBM_kernel_loop_inneriPfS_(%[[UB:.+]]: i32 {llvm.noundef}, %[[ARG:.+]]: !llvm.ptr {llvm.noundef}, %[[ARG1:.+]]: !llvm.ptr {llvm.noundef}) {
// CHECK-NEXT:    %[[CST2:.+]] = arith.constant 2 : i32
// CHECK-NEXT:    %[[UB1:.+]] = arith.divsi %[[UB]], %[[CST2]] : i32
// CHECK-NEXT:    %[[UB2:.+]] = arith.index_cast %[[UB1]] : i32 to index
// CHECK-NEXT:    affine.for %[[IT:.+]] = 0 to %[[UB2:.+]] {
// CHECK-NEXT:      llvm.call fastcc @_ZL4kernPfS_(%[[ARG]], %[[ARG1]]) : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> ()
// CHECK-NEXT:      llvm.call fastcc @_ZL4kernPfS_(%[[ARG1]], %[[ARG]]) : (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> ()
// CHECK-NEXT:    } {enzyme_enable_checkpointing = true}
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
