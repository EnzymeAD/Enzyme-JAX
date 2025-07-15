// RUN: enzymexlamlir-opt --raise-affine-to-stablehlo --split-input-file %s | FileCheck %s


#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "double", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "_ZTS4Vec3", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc2, 8>, <#tbaa_type_desc2, 16>}>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "_ZTS8Particle", members = {<#tbaa_type_desc3, 0>, <#tbaa_type_desc3, 24>, <#tbaa_type_desc3, 48>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc2, offset = 48>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc2, offset = 56>
#tbaa_tag3 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc2, offset = 64>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 0 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>]
  llvm.mlir.global external local_unnamed_addr @enzyme_dup() {addr_space = 0 : i32, alignment = 4 : i64, sym_visibility = "private"} : i32
  llvm.mlir.global external local_unnamed_addr @enzyme_const() {addr_space = 0 : i32, alignment = 4 : i64, sym_visibility = "private"} : i32
  llvm.func @_Z26calculate_potential_energyP8Particleid(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg1: i32 {llvm.noundef}, %arg2: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {dso_local, memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>, will_return} {
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = ub.poison : f64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %cst_0 = arith.constant 4.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %1 = arith.cmpi sgt, %arg1, %c0_i32 : i32
    %2 = scf.if %1 -> (f64) {
      %3 = arith.negf %arg2 {fastmathFlags = #llvm.fastmath<none>} : f64
      %4 = arith.extui %arg1 {nonNeg} : i32 to i64
      %5 = arith.extui %arg1 {nonNeg} : i32 to i64
      %6 = arith.maxsi %5, %c1_i64 : i64
      %7 = arith.addi %6, %c1_i64 : i64
      %8:2 = scf.for %arg3 = %c1_i64 to %7 step %c1_i64 iter_args(%arg4 = %cst, %arg5 = %0) -> (f64, f64)  : i64 {
        %9 = arith.cmpi ult, %arg3, %4 : i64
        %10 = scf.if %9 -> (f64) {
          %11 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
          %12 = arith.index_cast %arg3 : i64 to index
          %13 = arith.addi %12, %c-1 : index
          %14 = arith.divui %13, %c8 : index
          %15 = memref.load %11[%14] : memref<?xf64>
          %16 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
          %17 = arith.index_cast %arg3 : i64 to index
          %18 = arith.addi %17, %c-1 : index
          %19 = arith.divui %18, %c8 : index
          %20 = arith.addi %19, %c1 : index
          %21 = memref.load %16[%20] : memref<?xf64>
          %22 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
          %23 = arith.index_cast %arg3 : i64 to index
          %24 = arith.addi %23, %c-1 : index
          %25 = arith.divui %24, %c8 : index
          %26 = arith.addi %25, %c2 : index
          %27 = memref.load %22[%26] : memref<?xf64>
          %28 = arith.extui %arg1 {nonNeg} : i32 to i64
          %29 = arith.addi %arg3, %c1_i64 : i64
          %30 = arith.maxsi %28, %29 : i64
          %31 = arith.addi %30, %c1_i64 : i64
          %32:2 = scf.for %arg6 = %29 to %31 step %c1_i64 iter_args(%arg7 = %arg4, %arg8 = %0) -> (f64, f64)  : i64 {
            %33 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
            %34 = arith.index_cast %arg3 : i64 to index
            %35 = arith.index_cast %arg6 : i64 to index
            %36 = arith.index_cast %arg3 : i64 to index
            %37 = arith.addi %36, %c1 : index
            %38 = arith.subi %35, %37 : index
            %39 = arith.addi %34, %38 : index
            %40 = arith.divui %39, %c8 : index
            %41 = memref.load %33[%40] : memref<?xf64>
            %42 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
            %43 = arith.index_cast %arg3 : i64 to index
            %44 = arith.index_cast %arg6 : i64 to index
            %45 = arith.index_cast %arg3 : i64 to index
            %46 = arith.addi %45, %c1 : index
            %47 = arith.subi %44, %46 : index
            %48 = arith.addi %43, %47 : index
            %49 = arith.divui %48, %c8 : index
            %50 = arith.addi %49, %c1 : index
            %51 = memref.load %42[%50] : memref<?xf64>
            %52 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
            %53 = arith.index_cast %arg3 : i64 to index
            %54 = arith.index_cast %arg6 : i64 to index
            %55 = arith.index_cast %arg3 : i64 to index
            %56 = arith.addi %55, %c1 : index
            %57 = arith.subi %54, %56 : index
            %58 = arith.addi %53, %57 : index
            %59 = arith.divui %58, %c8 : index
            %60 = arith.addi %59, %c2 : index
            %61 = memref.load %52[%60] : memref<?xf64>
            %62 = arith.subf %15, %41 {fastmathFlags = #llvm.fastmath<none>} : f64
            %63 = arith.subf %21, %51 {fastmathFlags = #llvm.fastmath<none>} : f64
            %64 = arith.subf %27, %61 {fastmathFlags = #llvm.fastmath<none>} : f64
            %65 = arith.divf %62, %arg2 {fastmathFlags = #llvm.fastmath<none>} : f64
            %66 = math.round %65 : f64
            %67 = llvm.intr.fmuladd(%3, %66, %62) : (f64, f64, f64) -> f64
            %68 = arith.divf %63, %arg2 {fastmathFlags = #llvm.fastmath<none>} : f64
            %69 = math.round %68 : f64
            %70 = llvm.intr.fmuladd(%3, %69, %63) : (f64, f64, f64) -> f64
            %71 = arith.divf %64, %arg2 {fastmathFlags = #llvm.fastmath<none>} : f64
            %72 = math.round %71 : f64
            %73 = llvm.intr.fmuladd(%3, %72, %64) : (f64, f64, f64) -> f64
            %74 = arith.mulf %70, %70 {fastmathFlags = #llvm.fastmath<none>} : f64
            %75 = llvm.intr.fmuladd(%67, %67, %74) : (f64, f64, f64) -> f64
            %76 = llvm.intr.fmuladd(%73, %73, %75) : (f64, f64, f64) -> f64
            %77 = arith.cmpf olt, %76, %cst_0 {fastmathFlags = #llvm.fastmath<none>} : f64
            %78 = arith.mulf %76, %76 {fastmathFlags = #llvm.fastmath<none>} : f64
            %79 = arith.mulf %76, %78 {fastmathFlags = #llvm.fastmath<none>} : f64
            %80 = arith.mulf %79, %79 {fastmathFlags = #llvm.fastmath<none>} : f64
            %81 = arith.divf %cst_1, %80 {fastmathFlags = #llvm.fastmath<none>} : f64
            %82 = arith.divf %cst_1, %79 {fastmathFlags = #llvm.fastmath<none>} : f64
            %83 = arith.subf %81, %82 {fastmathFlags = #llvm.fastmath<none>} : f64
            %84 = llvm.intr.fmuladd(%83, %cst_0, %arg7) : (f64, f64, f64) -> f64
            %85 = arith.select %77, %84, %arg7 : f64
            scf.yield %85, %85 : f64, f64
          }
          scf.yield %32#1 : f64
        } else {
          scf.yield %arg4 : f64
        }
        scf.yield %10, %10 : f64, f64
      }
      scf.yield %8#1 : f64
    } else {
      scf.yield %cst : f64
    }
    llvm.return %2 : f64
  }
  llvm.func local_unnamed_addr @_Z21host_autodiff_wrapperP8Particleiid(%arg0: !llvm.ptr {llvm.nonnull, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: f64 {llvm.noundef}) attributes {dso_local, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.addressof @_Z26calculate_potential_energyP8Particleid : !llvm.ptr
    %1 = llvm.mlir.addressof @enzyme_const : !llvm.ptr
    %c-1_i64 = arith.constant -1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c24_i64 = arith.constant 24 : i64
    %c1_i64 = arith.constant 1 : i64
    %2 = llvm.mlir.addressof @enzyme_dup : !llvm.ptr
    %3 = arith.muli %arg2, %arg1 : i32
    %4 = arith.cmpi sgt, %3, %c0_i32 : i32
    scf.if %4 {
      %10 = arith.extui %3 {nonNeg} : i32 to i64
      %11 = arith.maxsi %10, %c1_i64 : i64
      %12 = arith.addi %11, %c1_i64 : i64
      scf.for %arg4 = %c1_i64 to %12 step %c1_i64  : i64 {
        %13 = arith.addi %arg4, %c-1_i64 : i64
        %14 = llvm.getelementptr inbounds|nuw %arg0[%13, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Particle.1", (struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>)>
        "llvm.intr.memset"(%14, %c0_i8, %c24_i64) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
      }
    }
    %5 = llvm.getelementptr inbounds|nuw %arg0[48] : (!llvm.ptr) -> !llvm.ptr, i8
    %6 = "enzymexla.pointer2memref"(%2) : (!llvm.ptr) -> memref<?xi32>
    %7 = affine.load %6[0] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi32>
    %8 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi32>
    %9 = affine.load %8[0] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi32>
    llvm.call @__enzyme_autodiff(%0, %7, %arg0, %5, %9, %arg1, %9, %arg3) vararg(!llvm.func<void (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, f64 {llvm.noundef}) -> ()
    llvm.return
  }
  llvm.func local_unnamed_addr @__enzyme_autodiff(!llvm.ptr {llvm.noundef}, ...) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_Z37run_autodiff_simulation_step_launcherP8Particleiid(%arg0: !llvm.ptr {llvm.nonnull, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: f64 {llvm.noundef}) attributes {dso_local, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c-1_i64 = arith.constant -1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0_i8 = arith.constant 0 : i8
    %c24_i64 = arith.constant 24 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.mlir.addressof @enzyme_dup : !llvm.ptr
    %1 = llvm.mlir.addressof @enzyme_const : !llvm.ptr
    %2 = llvm.mlir.addressof @_Z26calculate_potential_energyP8Particleid : !llvm.ptr
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg1 : i32 to index
    %6 = arith.index_cast %arg1 : i32 to index
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg1 : i32 to index
    %9 = arith.index_cast %arg2 : i32 to index
    %10 = arith.muli %arg2, %arg1 : i32
    %11 = arith.cmpi sgt, %10, %c0_i32 : i32
    scf.if %11 {
      %22 = arith.extui %10 {nonNeg} : i32 to i64
      %23 = arith.maxsi %22, %c1_i64 : i64
      %24 = arith.addi %23, %c1_i64 : i64
      scf.for %arg4 = %c1_i64 to %24 step %c1_i64  : i64 {
        %25 = arith.addi %arg4, %c-1_i64 : i64
        %26 = llvm.getelementptr inbounds|nuw %arg0[%25, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Particle.1", (struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>, struct<"struct.Vec3.1", (f64, f64, f64)>)>
        "llvm.intr.memset"(%26, %c0_i8, %c24_i64) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
      }
    }
    %12 = llvm.getelementptr inbounds|nuw %arg0[48] : (!llvm.ptr) -> !llvm.ptr, i8
    %13 = "enzymexla.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xi32>
    %14 = affine.load %13[0] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi32>
    %15 = "enzymexla.pointer2memref"(%1) : (!llvm.ptr) -> memref<?xi32>
    %16 = affine.load %15[0] {alignment = 4 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag]} : memref<?xi32>
    llvm.call @__enzyme_autodiff(%2, %14, %arg0, %12, %16, %arg1, %16, %arg3) vararg(!llvm.func<void (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, i32 {llvm.noundef}, f64 {llvm.noundef}) -> ()
    %17 = arith.extui %arg2 : i32 to i64
    %18 = arith.ori %17, %c4294967296_i64 {isDisjoint} : i64
    %19 = arith.trunci %18 : i64 to i32
    %20 = arith.index_cast %19 : i32 to index
    %21 = "enzymexla.gpu_wrapper"(%20, %c1, %c1, %c256, %c1, %c1) ({
      %22 = arith.index_cast %arg1 : i32 to index
      affine.parallel (%arg4, %arg5) = (0, 0) to (min(symbol(%9), symbol(%20)), 256) {
        affine.for %arg6 = 0 to %22 {
          %23 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
          %24 = affine.load %23[%arg6 * 9 + (%arg4 * symbol(%5)) * 9 + 6] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64>
          %25 = arith.negf %24 {fastmathFlags = #llvm.fastmath<contract>} : f64
          %26 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
          affine.store %25, %26[%arg6 * 9 + (%arg4 * symbol(%8)) * 9 + 6] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64>
          %27 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
          %28 = affine.load %27[%arg6 * 9 + (%arg4 * symbol(%4)) * 9 + 7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag2]} : memref<?xf64>
          %29 = arith.negf %28 {fastmathFlags = #llvm.fastmath<contract>} : f64
          %30 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
          affine.store %29, %30[%arg6 * 9 + (%arg4 * symbol(%7)) * 9 + 7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag2]} : memref<?xf64>
          %31 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
          %32 = affine.load %31[%arg6 * 9 + (%arg4 * symbol(%3)) * 9 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag3]} : memref<?xf64>
          %33 = arith.negf %32 {fastmathFlags = #llvm.fastmath<contract>} : f64
          %34 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
          affine.store %33, %34[%arg6 * 9 + (%arg4 * symbol(%6)) * 9 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag3]} : memref<?xf64>
        }
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) {passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], target_features = #llvm.target_features<["+ptx80", "+sm_52"]>} : (index, index, index, index, index, index) -> index
    llvm.return
  }
  llvm.func local_unnamed_addr @__mlir_launch_kernel__Z35__device_stub__negate_forces_kernelP8Particleii(!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr, i32, i32) attributes {sym_visibility = "private"}
  llvm.func internal @_Z35__device_stub__negate_forces_kernelP8Particleii(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}) attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_52"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_52", target_features = #llvm.target_features<["+ptx80", "+sm_52"]>, will_return} {
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg1 : i32 to index
    %6 = arith.index_cast %arg1 : i32 to index
    %block_id_x = gpu.block_id  x
    %7 = arith.index_castui %block_id_x : index to i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.index_cast %7 : i32 to index
    %10 = arith.index_cast %7 : i32 to index
    %11 = arith.index_cast %7 : i32 to index
    %12 = arith.index_cast %7 : i32 to index
    %13 = arith.index_cast %7 : i32 to index
    %14 = arith.cmpi slt, %7, %arg2 : i32
    scf.if %14 {
      affine.for %arg3 = 0 to %6 {
        %15 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
        %16 = affine.load %15[%arg3 * 9 + (symbol(%2) * symbol(%10)) * 9 + 6] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64>
        %17 = arith.negf %16 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %18 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
        affine.store %17, %18[%arg3 * 9 + (symbol(%5) * symbol(%13)) * 9 + 6] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag1]} : memref<?xf64>
        %19 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
        %20 = affine.load %19[%arg3 * 9 + (symbol(%1) * symbol(%9)) * 9 + 7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag2]} : memref<?xf64>
        %21 = arith.negf %20 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %22 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
        affine.store %21, %22[%arg3 * 9 + (symbol(%4) * symbol(%12)) * 9 + 7] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag2]} : memref<?xf64>
        %23 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
        %24 = affine.load %23[%arg3 * 9 + (symbol(%0) * symbol(%8)) * 9 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag3]} : memref<?xf64>
        %25 = arith.negf %24 {fastmathFlags = #llvm.fastmath<contract>} : f64
        %26 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
        affine.store %25, %26[%arg3 * 9 + (symbol(%3) * symbol(%11)) * 9 + 8] {alignment = 8 : i64, ordering = 0 : i64, tbaa = [#tbaa_tag3]} : memref<?xf64>
      }
    }
    llvm.return
  }
}

