// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition{backend=rocm})" | FileCheck %s --check-prefix=ROCM

#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc2, access_type = #tbaa_type_desc2, offset = 0>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "p1 int", members = {<#tbaa_type_desc1, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.module_asm = [], llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 0 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>]
  llvm.mlir.global private unnamed_addr constant @".str"("res = %d\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {dso_local, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["mustprogress", "norecurse", ["approx-func-fp-math", "true"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unsafe_fp_math = true, uwtable_kind = #llvm.uwtableKind<async>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(512 : i64) : i64
    %2 = llvm.mlir.constant(true) : i1
    %3 = llvm.mlir.addressof @reactant$_Z18__device_stub__fooPi : !llvm.ptr
    %4 = llvm.mlir.constant(128 : i32) : i32
    %5 = llvm.mlir.constant(0 : i64) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.addressof @".str" : !llvm.ptr
    %9 = llvm.mlir.constant(2 : i32) : i32
    %10 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %11 = llvm.alloca %0 x !llvm.array<128 x i32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
    %12 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.intr.lifetime.start %10 : !llvm.ptr
    %13 = llvm.call @cudaMalloc(%10, %1) : (!llvm.ptr {llvm.nonnull, llvm.noundef}, i64 {llvm.noundef}) -> (i32 {llvm.noundef})
    llvm.intr.lifetime.start %11 : !llvm.ptr
    llvm.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %14 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    llvm.call @__mlir_cuda_caller_phase3(%3, %0, %0, %0, %4, %0, %0, %5, %6, %14) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    llvm.intr.lifetime.start %12 : !llvm.ptr
    llvm.store %7, %12 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    %15 = llvm.call @cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(%12, %3, %0, %5, %7) : (!llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}, i64 {llvm.noundef}, i32 {llvm.noundef}) -> (i32 {llvm.noundef})
    %16 = llvm.load %12 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
    %17 = llvm.call @printf(%8, %16) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> i32
    %18 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %19 = llvm.call @cudaMemcpy(%11, %18, %1, %9) : (!llvm.ptr {llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}, i64 {llvm.noundef}, i32 {llvm.noundef}) -> i32
    %20 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
    %21 = llvm.call @cudaFree(%20) : (!llvm.ptr {llvm.noundef}) -> i32
    llvm.intr.lifetime.end %12 : !llvm.ptr
    llvm.intr.lifetime.end %11 : !llvm.ptr
    llvm.intr.lifetime.end %10 : !llvm.ptr
    llvm.return %7 : i32
  }
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["approx-func-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaMemcpy(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i64 {llvm.noundef}, i32 {llvm.noundef}) -> i32 attributes {passthrough = [["approx-func-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaFree(!llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = [["approx-func-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaMalloc(!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> i32 attributes {passthrough = [["approx-func-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, i64 {llvm.noundef}, i32 {llvm.noundef}) -> i32 attributes {passthrough = [["approx-func-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func internal @reactant$_Z18__device_stub__fooPi(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.writeonly}) attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_infs_fp_math = true, no_inline, no_nans_fp_math = true, no_signed_zeros_fp_math = true, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["approx-func-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_120"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_120", target_features = #llvm.target_features<["+ptx88", "+sm_120"]>, unsafe_fp_math = true, will_return} {
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %1 = llvm.zext nneg %0 : i32 to i64
    %2 = llvm.getelementptr inbounds|nuw %arg0[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %0, %2 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func local_unnamed_addr @__mlir_cuda_caller_phase3(...) attributes {sym_visibility = "private"}
}


// CHECK:  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {dso_local, no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, passthrough = ["mustprogress", "norecurse", ["approx-func-fp-math", "true"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unsafe_fp_math = true, uwtable_kind = #llvm.uwtableKind<async>} {
// CHECK-NEXT:    %0 = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %1 = llvm.mlir.constant(512 : i64) : i64
// CHECK-NEXT:    %2 = llvm.mlir.constant(true) : i1
// CHECK-NEXT:    %3 = "enzymexla.gpu_kernel_address"() <{fn = @__mlir_gpu_module::@reactant$_Z18__device_stub__fooPi}> : () -> !llvm.ptr
// CHECK-NEXT:    %4 = llvm.mlir.constant(128 : i32) : i32
// CHECK-NEXT:    %5 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %6 = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %7 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %8 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:    %9 = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %10 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    %11 = llvm.alloca %0 x !llvm.array<128 x i32> {alignment = 16 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    %12 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.start %10 : !llvm.ptr
// CHECK-NEXT:    %13 = arith.index_cast %1 : i64 to index
// CHECK-NEXT:    %memref = gpu.alloc  (%13) : memref<?xi8, 1>
// CHECK-NEXT:    %14 = "enzymexla.memref2pointer"(%memref) : (memref<?xi8, 1>) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %14, %10 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %15 = llvm.mlir.zero : i32
// CHECK-NEXT:    llvm.intr.lifetime.start %11 : !llvm.ptr
// CHECK-NEXT:    llvm.cond_br %2, ^bb1, ^bb2
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    %16 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %17 = llvm.trunc %5 : i64 to i32
// CHECK-NEXT:    %18 = llvm.sext %0 : i32 to i64
// CHECK-NEXT:    %19 = llvm.sext %0 : i32 to i64
// CHECK-NEXT:    %20 = llvm.sext %0 : i32 to i64
// CHECK-NEXT:    %21 = llvm.sext %4 : i32 to i64
// CHECK-NEXT:    %22 = llvm.sext %0 : i32 to i64
// CHECK-NEXT:    %23 = llvm.sext %0 : i32 to i64
// CHECK-NEXT:    gpu.launch_func  @__mlir_gpu_module::@reactant$_Z18__device_stub__fooPi blocks in (%18, %19, %20) threads in (%21, %22, %23) : i64 dynamic_shared_memory_size %17 args(%16 : !llvm.ptr)
// CHECK-NEXT:    llvm.br ^bb2
// CHECK-NEXT:  ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:    llvm.intr.lifetime.start %12 : !llvm.ptr
// CHECK-NEXT:    llvm.store %7, %12 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : i32, !llvm.ptr
// CHECK-NEXT:    %24 = "enzymexla.gpu_occupancy"(%0, %5, %7) <{fn = @__mlir_gpu_module::@reactant$_Z18__device_stub__fooPi}> : (i32, i64, i32) -> i32
// CHECK-NEXT:    llvm.store %24, %12 : i32, !llvm.ptr
// CHECK-NEXT:    %25 = llvm.mlir.zero : i32
// CHECK-NEXT:    %26 = llvm.load %12 {alignment = 4 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> i32
// CHECK-NEXT:    %27 = llvm.call @printf(%8, %26) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, i32 {llvm.noundef}) -> i32
// CHECK-NEXT:    %28 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %29 = "enzymexla.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:    %30 = "enzymexla.pointer2memref"(%28) : (!llvm.ptr) -> memref<?xi8, 1>
// CHECK-NEXT:    %31 = arith.index_cast %1 : i64 to index
// CHECK-NEXT:    enzymexla.memcpy  %29, %30, %31 : memref<?xi8>, memref<?xi8, 1>
// CHECK-NEXT:    %32 = llvm.mlir.zero : i32
// CHECK-NEXT:    %33 = llvm.load %10 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %34 = llvm.call @cudaFree(%33) : (!llvm.ptr {llvm.noundef}) -> i32
// CHECK-NEXT:    llvm.intr.lifetime.end %12 : !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.end %11 : !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.end %10 : !llvm.ptr
// CHECK-NEXT:    llvm.return %7 : i32
// CHECK-NEXT:  }
