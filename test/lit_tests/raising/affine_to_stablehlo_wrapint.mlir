// RUN: enzymexlamlir-opt --raise-affine-to-stablehlo --split-input-file %s | FileCheck %s

// XFAIL: *

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
  llvm.func local_unnamed_addr @__enzyme_autodiff(!llvm.ptr {llvm.noundef}, ...) attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_Z37run_autodiff_simulation_step_launcherP8Particleiid(%arg0: !llvm.ptr {llvm.nonnull, llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: f64 {llvm.noundef}) attributes {dso_local, no_unwind, passthrough = ["mustprogress", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c-1_i64 = arith.constant -1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c0_i8 = arith.constant 0 : i8
    %c24_i64 = arith.constant 24 : i64
    %c1_i64 = arith.constant 1 : i64
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
}

