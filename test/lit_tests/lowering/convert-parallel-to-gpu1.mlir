// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu1)" | FileCheck %s

#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "char", sizeInBits = 8, encoding = DW_ATE_signed_char>
#di_file = #llvm.di_file<"../cu2.cu" in "/home/wmoses/git/Reactant/enzyme/build">
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type, sizeInBits = 56, elements = #llvm.di_subrange<count = 7 : i64>>
#di_global_variable = #llvm.di_global_variable<file = #di_file, line = 81, type = #di_composite_type, isLocalToUnit = true, isDefined = true>
#di_global_variable_expression = #llvm.di_global_variable_expression<var = #di_global_variable, expr = <>>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 0 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>]
  llvm.mlir.global private unnamed_addr constant @".str"("%f %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dbg_exprs = [#di_global_variable_expression], dso_local, sym_visibility = "private"}
  llvm.func local_unnamed_addr @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {dso_local, passthrough = ["mustprogress", "norecurse", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.addressof @".str" : !llvm.ptr
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c4294967295_i64 = arith.constant 4294967295 : i64
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.400000e+00 : f64
    %c29_i64 = arith.constant 29 : i64
    %c32_i64 = arith.constant 32 : i64
    %c10_i32 = arith.constant 10 : i32
    %1 = llvm.mlir.zero : !llvm.ptr
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %alloca = memref.alloca() : memref<1xf64>
    %alloca_2 = memref.alloca() : memref<1xf64>
    %alloca_3 = memref.alloca() : memref<1xf64>
    %alloca_4 = memref.alloca() : memref<1xf64>
    %2 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %c1_5 = arith.constant 1 : index
    %3 = memref.load %2[%c1_5] : memref<?x!llvm.ptr>
    %4 = llvm.call @__isoc23_strtol(%3, %1, %c10_i32) {no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i64
    %5 = arith.shli %4, %c32_i64 : i64
    %6 = arith.shrsi %5, %c29_i64 {isExact} : i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = arith.divui %7, %c8 : index
    %memref = gpu.alloc  (%8) : memref<?xf64, 1>
    %9 = arith.index_cast %6 : i64 to index
    %10 = arith.divui %9, %c8 : index
    %memref_6 = gpu.alloc  (%10) : memref<?xf64, 1>
    %11 = arith.index_cast %6 : i64 to index
    %12 = arith.divui %11, %c8 : index
    %memref_7 = gpu.alloc  (%12) : memref<?xf64, 1>
    %13 = arith.index_cast %6 : i64 to index
    %14 = arith.divui %13, %c8 : index
    %memref_8 = gpu.alloc  (%14) : memref<?xf64, 1>
    %c0 = arith.constant 0 : index
    memref.store %cst_1, %alloca[%c0] : memref<1xf64>
    %c0_9 = arith.constant 0 : index
    memref.store %cst_0, %alloca_2[%c0_9] : memref<1xf64>
    %c0_10 = arith.constant 0 : index
    memref.store %cst, %alloca_3[%c0_10] : memref<1xf64>
    %c0_11 = arith.constant 0 : index
    memref.store %cst, %alloca_4[%c0_11] : memref<1xf64>
    %15 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %memref, %alloca, %15 : memref<?xf64, 1>, memref<1xf64>
    %16 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %memref_6, %alloca_2, %16 : memref<?xf64, 1>, memref<1xf64>
    %17 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %memref_7, %alloca_3, %17 : memref<?xf64, 1>, memref<1xf64>
    %18 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %memref_8, %alloca_4, %18 : memref<?xf64, 1>, memref<1xf64>
    %19 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %c100, %c1, %c1) ({
      %c0_16 = arith.constant 0 : index
      %c100_17 = arith.constant 100 : index
      %c1_18 = arith.constant 1 : index
      scf.parallel (%arg2) = (%c0_16) to (%c100_17) step (%c1_18) {
        %36 = memref.load %memref[%arg2] : memref<?xf64, 1>
        %37 = memref.load %memref_7[%arg2] : memref<?xf64, 1>
        %38 = arith.addf %36, %37 {fastmathFlags = #llvm.fastmath<contract>} : f64
        memref.store %38, %memref_7[%arg2] : memref<?xf64, 1>
        scf.reduce 
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    %20 = arith.andi %4, %c4294967295_i64 : i64
    %21 = arith.ori %20, %c4294967296_i64 {isDisjoint} : i64
    %22 = arith.trunci %21 : i64 to i32
    %23 = arith.index_cast %22 : i32 to index
    %24 = "enzymexla.gpu_wrapper"(%c1, %c1, %c1, %23, %c1, %c1) ({
      %c0_16 = arith.constant 0 : index
      %c1_17 = arith.constant 1 : index
      scf.parallel (%arg2) = (%c0_16) to (%23) step (%c1_17) {
        %36 = memref.load %memref[%arg2] : memref<?xf64, 1>
        %37 = memref.load %memref_7[%arg2] : memref<?xf64, 1>
        %38 = arith.addf %36, %37 {fastmathFlags = #llvm.fastmath<contract>} : f64
        memref.store %38, %memref_7[%arg2] : memref<?xf64, 1>
        scf.reduce 
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    %25 = llvm.call @cudaDeviceSynchronize() : () -> i32
    %26 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %alloca, %memref, %26 : memref<1xf64>, memref<?xf64, 1>
    %27 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %alloca_2, %memref_6, %27 : memref<1xf64>, memref<?xf64, 1>
    %28 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %alloca_3, %memref_7, %28 : memref<1xf64>, memref<?xf64, 1>
    %29 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %alloca_4, %memref_8, %29 : memref<1xf64>, memref<?xf64, 1>
    %c0_12 = arith.constant 0 : index
    %30 = memref.load %alloca[%c0_12] : memref<1xf64>
    %c0_13 = arith.constant 0 : index
    %31 = memref.load %alloca_3[%c0_13] : memref<1xf64>
    %32 = llvm.call @printf(%0, %30, %31) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}, f64 {llvm.noundef}) -> i32
    %c0_14 = arith.constant 0 : index
    %33 = memref.load %alloca_2[%c0_14] : memref<1xf64>
    %c0_15 = arith.constant 0 : index
    %34 = memref.load %alloca_4[%c0_15] : memref<1xf64>
    %35 = llvm.call @printf(%0, %33, %34) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}, f64 {llvm.noundef}) -> i32
    llvm.return %c0_i32 : i32
  }
  llvm.func local_unnamed_addr @cudaDeviceSynchronize() -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @__isoc23_strtol(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i64 attributes {no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
}

// CHECK: @main
