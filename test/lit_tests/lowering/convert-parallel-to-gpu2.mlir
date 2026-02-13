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
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
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
    %3 = memref.load %2[%c1] : memref<?x!llvm.ptr>
    %4 = llvm.call @__isoc23_strtol(%3, %1, %c10_i32) {no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i64
    %5 = arith.shli %4, %c32_i64 : i64
    %6 = arith.shrsi %5, %c29_i64 exact : i64
    %7 = arith.index_cast %6 : i64 to index
    %8 = arith.divui %7, %c8 : index
    %memref = gpu.alloc  (%8) : memref<?xf64, 1>
    %9 = arith.index_cast %6 : i64 to index
    %10 = arith.divui %9, %c8 : index
    %memref_5 = gpu.alloc  (%10) : memref<?xf64, 1>
    %11 = arith.index_cast %6 : i64 to index
    %12 = arith.divui %11, %c8 : index
    %memref_6 = gpu.alloc  (%12) : memref<?xf64, 1>
    %13 = arith.index_cast %6 : i64 to index
    %14 = arith.divui %13, %c8 : index
    %memref_7 = gpu.alloc  (%14) : memref<?xf64, 1>
    memref.store %cst_1, %alloca[%c0] : memref<1xf64>
    memref.store %cst_0, %alloca_2[%c0] : memref<1xf64>
    memref.store %cst, %alloca_3[%c0] : memref<1xf64>
    memref.store %cst, %alloca_4[%c0] : memref<1xf64>
    %15 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %memref, %alloca, %15 : memref<?xf64, 1>, memref<1xf64>
    %16 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %memref_5, %alloca_2, %16 : memref<?xf64, 1>, memref<1xf64>
    %17 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %memref_6, %alloca_3, %17 : memref<?xf64, 1>, memref<1xf64>
    %18 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %memref_7, %alloca_4, %18 : memref<?xf64, 1>, memref<1xf64>
    "enzymexla.alternatives"() ({
      %34 = "enzymexla.gpu_error"() ({
        gpu.launch_func  @main_kernel::@main_kernel blocks in (%c4, %c1, %c1) threads in (%c32, %c1, %c1)  args(%memref : memref<?xf64, 1>, %memref_6 : memref<?xf64, 1>)
        "enzymexla.polygeist_yield"() : () -> ()
      }) : () -> index
      "enzymexla.polygeist_yield"() : () -> ()
    }, {
      %34 = "enzymexla.gpu_error"() ({
        gpu.launch_func  @main_kernel_0::@main_kernel blocks in (%c2, %c1, %c1) threads in (%c64, %c1, %c1)  args(%memref : memref<?xf64, 1>, %memref_6 : memref<?xf64, 1>)
        "enzymexla.polygeist_yield"() : () -> ()
      }) : () -> index
      "enzymexla.polygeist_yield"() : () -> ()
    }, {
      %34 = "enzymexla.gpu_error"() ({
        gpu.launch_func  @main_kernel_1::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c100, %c1, %c1)  args(%memref : memref<?xf64, 1>, %memref_6 : memref<?xf64, 1>)
        "enzymexla.polygeist_yield"() : () -> ()
      }) : () -> index
      "enzymexla.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,", "block_size=64,", "block_size=100,"], alternatives.type = "gpu_kernel"} : () -> ()
    %19 = arith.andi %4, %c4294967295_i64 : i64
    %20 = arith.ori %19, %c4294967296_i64 {isDisjoint} : i64
    %21 = arith.trunci %20 : i64 to i32
    %22 = arith.index_cast %21 : i32 to index
    "enzymexla.alternatives"() ({
      %34 = arith.subi %22, %c1 : index
      %35 = arith.divui %34, %c32 : index
      %36 = arith.addi %35, %c1 : index
      %37 = "enzymexla.gpu_error"() ({
        %38 = arith.cmpi sge, %36, %c1 : index
        scf.if %38 {
          gpu.launch_func  @main_kernel_2::@main_kernel blocks in (%36, %c1, %c1) threads in (%c32, %c1, %c1)  args(%22 : index, %memref : memref<?xf64, 1>, %memref_6 : memref<?xf64, 1>)
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : () -> index
      "enzymexla.polygeist_yield"() : () -> ()
    }, {
      %34 = arith.subi %22, %c1 : index
      %35 = arith.divui %34, %c64 : index
      %36 = arith.addi %35, %c1 : index
      %37 = "enzymexla.gpu_error"() ({
        %38 = arith.cmpi sge, %36, %c1 : index
        scf.if %38 {
          gpu.launch_func  @main_kernel_3::@main_kernel blocks in (%36, %c1, %c1) threads in (%c64, %c1, %c1)  args(%22 : index, %memref : memref<?xf64, 1>, %memref_6 : memref<?xf64, 1>)
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : () -> index
      "enzymexla.polygeist_yield"() : () -> ()
    }, {
      %34 = arith.subi %22, %c1 : index
      %35 = arith.divui %34, %c128 : index
      %36 = arith.addi %35, %c1 : index
      %37 = "enzymexla.gpu_error"() ({
        %38 = arith.cmpi sge, %36, %c1 : index
        scf.if %38 {
          gpu.launch_func  @main_kernel_4::@main_kernel blocks in (%36, %c1, %c1) threads in (%c128, %c1, %c1)  args(%22 : index, %memref : memref<?xf64, 1>, %memref_6 : memref<?xf64, 1>)
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : () -> index
      "enzymexla.polygeist_yield"() : () -> ()
    }, {
      %34 = arith.subi %22, %c1 : index
      %35 = arith.divui %34, %c256 : index
      %36 = arith.addi %35, %c1 : index
      %37 = "enzymexla.gpu_error"() ({
        %38 = arith.cmpi sge, %36, %c1 : index
        scf.if %38 {
          gpu.launch_func  @main_kernel_5::@main_kernel blocks in (%36, %c1, %c1) threads in (%c256, %c1, %c1)  args(%22 : index, %memref : memref<?xf64, 1>, %memref_6 : memref<?xf64, 1>)
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : () -> index
      "enzymexla.polygeist_yield"() : () -> ()
    }, {
      %34 = arith.subi %22, %c1 : index
      %35 = arith.divui %34, %c512 : index
      %36 = arith.addi %35, %c1 : index
      %37 = "enzymexla.gpu_error"() ({
        %38 = arith.cmpi sge, %36, %c1 : index
        scf.if %38 {
          gpu.launch_func  @main_kernel_6::@main_kernel blocks in (%36, %c1, %c1) threads in (%c512, %c1, %c1)  args(%22 : index, %memref : memref<?xf64, 1>, %memref_6 : memref<?xf64, 1>)
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : () -> index
      "enzymexla.polygeist_yield"() : () -> ()
    }, {
      %34 = arith.subi %22, %c1 : index
      %35 = arith.divui %34, %c1024 : index
      %36 = arith.addi %35, %c1 : index
      %37 = "enzymexla.gpu_error"() ({
        %38 = arith.cmpi sge, %36, %c1 : index
        scf.if %38 {
          gpu.launch_func  @main_kernel_7::@main_kernel blocks in (%36, %c1, %c1) threads in (%c1024, %c1, %c1)  args(%22 : index, %memref : memref<?xf64, 1>, %memref_6 : memref<?xf64, 1>)
        }
        "enzymexla.polygeist_yield"() : () -> ()
      }) : () -> index
      "enzymexla.polygeist_yield"() : () -> ()
    }) {alternatives.descs = ["block_size=32,", "block_size=64,", "block_size=128,", "block_size=256,", "block_size=512,", "block_size=1024,"], alternatives.type = "gpu_kernel"} : () -> ()
    %23 = llvm.call @cudaDeviceSynchronize() : () -> i32
    %24 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %alloca, %memref, %24 : memref<1xf64>, memref<?xf64, 1>
    %25 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %alloca_2, %memref_5, %25 : memref<1xf64>, memref<?xf64, 1>
    %26 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %alloca_3, %memref_6, %26 : memref<1xf64>, memref<?xf64, 1>
    %27 = arith.index_cast %6 : i64 to index
    enzymexla.memcpy  %alloca_4, %memref_7, %27 : memref<1xf64>, memref<?xf64, 1>
    %28 = memref.load %alloca[%c0] : memref<1xf64>
    %29 = memref.load %alloca_3[%c0] : memref<1xf64>
    %30 = llvm.call @printf(%0, %28, %29) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}, f64 {llvm.noundef}) -> i32
    %31 = memref.load %alloca_2[%c0] : memref<1xf64>
    %32 = memref.load %alloca_4[%c0] : memref<1xf64>
    %33 = llvm.call @printf(%0, %31, %32) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}, f64 {llvm.noundef}) -> i32
    llvm.return %c0_i32 : i32
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<?xf64, 1>, %arg1: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>, known_grid_size = array<i32: 4, 1, 1>} {
      %c100 = arith.constant 100 : index
      %c32 = arith.constant 32 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
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
  gpu.module @main_kernel_0 {
    gpu.func @main_kernel(%arg0: memref<?xf64, 1>, %arg1: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 2, 1, 1>} {
      %c100 = arith.constant 100 : index
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c64 : index
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
  gpu.module @main_kernel_1 {
    gpu.func @main_kernel(%arg0: memref<?xf64, 1>, %arg1: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 100, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %thread_id_x = gpu.thread_id  x
      %0 = memref.load %arg0[%thread_id_x] : memref<?xf64, 1>
      %1 = memref.load %arg1[%thread_id_x] : memref<?xf64, 1>
      %2 = arith.addf %0, %1 {fastmathFlags = #llvm.fastmath<contract>} : f64
      memref.store %2, %arg1[%thread_id_x] : memref<?xf64, 1>
      gpu.return
    }
  }
  gpu.module @main_kernel_2 {
    gpu.func @main_kernel(%arg0: index, %arg1: memref<?xf64, 1>, %arg2: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 32, 1, 1>} {
      %c32 = arith.constant 32 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c32 : index
      %1 = arith.addi %0, %thread_id_x : index
      %2 = arith.cmpi ult, %1, %arg0 : index
      scf.if %2 {
        %3 = memref.load %arg1[%1] : memref<?xf64, 1>
        %4 = memref.load %arg2[%1] : memref<?xf64, 1>
        %5 = arith.addf %3, %4 {fastmathFlags = #llvm.fastmath<contract>} : f64
        memref.store %5, %arg2[%1] : memref<?xf64, 1>
      }
      gpu.return
    }
  }
  gpu.module @main_kernel_3 {
    gpu.func @main_kernel(%arg0: index, %arg1: memref<?xf64, 1>, %arg2: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c64 : index
      %1 = arith.addi %0, %thread_id_x : index
      %2 = arith.cmpi ult, %1, %arg0 : index
      scf.if %2 {
        %3 = memref.load %arg1[%1] : memref<?xf64, 1>
        %4 = memref.load %arg2[%1] : memref<?xf64, 1>
        %5 = arith.addf %3, %4 {fastmathFlags = #llvm.fastmath<contract>} : f64
        memref.store %5, %arg2[%1] : memref<?xf64, 1>
      }
      gpu.return
    }
  }
  gpu.module @main_kernel_4 {
    gpu.func @main_kernel(%arg0: index, %arg1: memref<?xf64, 1>, %arg2: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 128, 1, 1>} {
      %c128 = arith.constant 128 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c128 : index
      %1 = arith.addi %0, %thread_id_x : index
      %2 = arith.cmpi ult, %1, %arg0 : index
      scf.if %2 {
        %3 = memref.load %arg1[%1] : memref<?xf64, 1>
        %4 = memref.load %arg2[%1] : memref<?xf64, 1>
        %5 = arith.addf %3, %4 {fastmathFlags = #llvm.fastmath<contract>} : f64
        memref.store %5, %arg2[%1] : memref<?xf64, 1>
      }
      gpu.return
    }
  }
  gpu.module @main_kernel_5 {
    gpu.func @main_kernel(%arg0: index, %arg1: memref<?xf64, 1>, %arg2: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 256, 1, 1>} {
      %c256 = arith.constant 256 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c256 : index
      %1 = arith.addi %0, %thread_id_x : index
      %2 = arith.cmpi ult, %1, %arg0 : index
      scf.if %2 {
        %3 = memref.load %arg1[%1] : memref<?xf64, 1>
        %4 = memref.load %arg2[%1] : memref<?xf64, 1>
        %5 = arith.addf %3, %4 {fastmathFlags = #llvm.fastmath<contract>} : f64
        memref.store %5, %arg2[%1] : memref<?xf64, 1>
      }
      gpu.return
    }
  }
  gpu.module @main_kernel_6 {
    gpu.func @main_kernel(%arg0: index, %arg1: memref<?xf64, 1>, %arg2: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 512, 1, 1>} {
      %c512 = arith.constant 512 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c512 : index
      %1 = arith.addi %0, %thread_id_x : index
      %2 = arith.cmpi ult, %1, %arg0 : index
      scf.if %2 {
        %3 = memref.load %arg1[%1] : memref<?xf64, 1>
        %4 = memref.load %arg2[%1] : memref<?xf64, 1>
        %5 = arith.addf %3, %4 {fastmathFlags = #llvm.fastmath<contract>} : f64
        memref.store %5, %arg2[%1] : memref<?xf64, 1>
      }
      gpu.return
    }
  }
  gpu.module @main_kernel_7 {
    gpu.func @main_kernel(%arg0: index, %arg1: memref<?xf64, 1>, %arg2: memref<?xf64, 1>) kernel attributes {known_block_size = array<i32: 1024, 1, 1>} {
      %c1024 = arith.constant 1024 : index
      %block_id_x = gpu.block_id  x
      %thread_id_x = gpu.thread_id  x
      %0 = arith.muli %block_id_x, %c1024 : index
      %1 = arith.addi %0, %thread_id_x : index
      %2 = arith.cmpi ult, %1, %arg0 : index
      scf.if %2 {
        %3 = memref.load %arg1[%1] : memref<?xf64, 1>
        %4 = memref.load %arg2[%1] : memref<?xf64, 1>
        %5 = arith.addf %3, %4 {fastmathFlags = #llvm.fastmath<contract>} : f64
        memref.store %5, %arg2[%1] : memref<?xf64, 1>
      }
      gpu.return
    }
  }
  llvm.func local_unnamed_addr @cudaDeviceSynchronize() -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @__isoc23_strtol(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}) -> i64 attributes {no_unwind, passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
}

// CHECK-LABEL: @main

// CHECK: gpu.module @main_kernel [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}

// CHECK: gpu.module @main_kernel_0 [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}

// CHECK: gpu.module @main_kernel_1 [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}

// CHECK: gpu.module @main_kernel_2 [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}

// CHECK: gpu.module @main_kernel_3 [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}

// CHECK: gpu.module @main_kernel_4 [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}

// CHECK: gpu.module @main_kernel_5 [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}

// CHECK: gpu.module @main_kernel_6 [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}

// CHECK: gpu.module @main_kernel_7 [#rocdl.target<O = 3, features = "+wavefront64">] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}