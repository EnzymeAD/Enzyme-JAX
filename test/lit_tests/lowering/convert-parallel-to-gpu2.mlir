// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-parallel-to-gpu2{backend=rocm})" | FileCheck %s

#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "char", sizeInBits = 8, encoding = DW_ATE_signed_char>
#di_file = #llvm.di_file<"../cu2.cu" in "/home/wmoses/git/Reactant/enzyme/build">
#di_derived_type = #llvm.di_derived_type<tag = DW_TAG_const_type, baseType = #di_basic_type>
#di_composite_type = #llvm.di_composite_type<tag = DW_TAG_array_type, baseType = #di_derived_type, sizeInBits = 56, elements = #llvm.di_subrange<count = 7 : i64>>
#di_global_variable = #llvm.di_global_variable<file = #di_file, line = 81, type = #di_composite_type, isLocalToUnit = true, isDefined = true>
#di_global_variable_expression = #llvm.di_global_variable_expression<var = #di_global_variable, expr = <>>

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, gpu.container_module, llvm.target_triple = "x86_64-unknown-linux-gnu"} {

  llvm.module_flags [#llvm.mlir.module_flag<max, "Dwarf Version", 5 : i32>, #llvm.mlir.module_flag<warning, "Debug Info Version", 3 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 0 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<max, "debug-info-assignment-tracking", 1 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>]

  llvm.mlir.global private unnamed_addr constant @".str"("%f %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dbg_exprs = [#di_global_variable_expression], dso_local, sym_visibility = "private"}

  llvm.func local_unnamed_addr @main(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.align = 8 : i64, llvm.dereferenceable = 16 : i64, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}) -> (i32 {llvm.noundef}) attributes {dso_local, passthrough = ["mustprogress", "norecurse", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 1.400000e+00 : f64
    %c29_i64 = arith.constant 29 : i64
    %c32_i64 = arith.constant 32 : i64
    %c10_i32 = arith.constant 10 : i32
    %1 = llvm.mlir.zero : !llvm.ptr
    %alloca = memref.alloca() : memref<1xf64>
    %alloca_3 = memref.alloca() : memref<1xf64>
    %2 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?x!llvm.ptr>
    %3 = memref.load %2[%c1] : memref<?x!llvm.ptr>
    %4 = llvm.call @__isoc23_strtol(%3, %1, %c10_i32) {no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i64
    %5 = arith.shli %4, %c32_i64 : i64
    %6 = arith.shrsi %5, %c29_i64 exact : i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = arith.divui %7, %c8 : index
    %memref = gpu.alloc (%8) : memref<?xf64, 1>
    %9 = arith.index_cast %6 : i64 to index
    %10 = arith.divui %9, %c8 : index
    %memref_6 = gpu.alloc (%10) : memref<?xf64, 1>
    memref.store %cst_1, %alloca[%c0] : memref<1xf64>
    memref.store %cst, %alloca_3[%c0] : memref<1xf64>
    %15 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy %memref, %alloca, %15 : memref<?xf64, 1>, memref<1xf64>
    %17 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy %memref_6, %alloca_3, %17 : memref<?xf64, 1>, memref<1xf64>
    %34 = "enzymexla.gpu_error"() ({
      gpu.launch_func @main_kernel::@main_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1) args(%memref : memref<?xf64, 1>, %memref_6 : memref<?xf64, 1>)
      "enzymexla.polygeist_yield"() : () -> ()
    }) : () -> index
    %23 = llvm.call @cudaDeviceSynchronize() : () -> i32
    %24 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy %alloca, %memref, %24 : memref<1xf64>, memref<?xf64, 1>
    %26 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy %alloca_3, %memref_6, %26 : memref<1xf64>, memref<?xf64, 1>
    llvm.return %c0_i32 : i32
  }

  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<?xf64, 1>, %arg1: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      %c100 = arith.constant 100 : index
      %c32 = arith.constant 32 : index
      %block_id_x = gpu.block_id x
      %thread_id_x = gpu.thread_id x
      %0 = arith.muli %block_id_x, %c32 : index
      %1 = arith.addi %0, %thread_id_x : index
      %2 = arith.cmpi ult, %1, %c100 : index
      scf.if %2 {
        %3 = memref.load %arg0[%1] : memref<?xf64, 1>
        %4 = memref.load %arg1[%1] : memref<?xf64, 1>
        %5 = arith.addf %3, %4 {fastmathFlags = #llvm.fastmath<contract>} : f64
        memref.store %5, %arg1[%1] : memref<?xf64, 1>
      }
      gpu.return
    }
  }

  llvm.func local_unnamed_addr @cudaDeviceSynchronize() -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @__isoc23_strtol(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i64 attributes {no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
}

// CHECK-LABEL: @main
// CHECK: gpu.module @main_kernel [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}