// RUN: enzymexlamlir-opt --affine-cfg --split-input-file --allow-unregistered-dialect %s | FileCheck %s

#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "double", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<min, "PIC Level", 0 : i32>, #llvm.mlir.module_flag<max, "PIE Level", 2 : i32>, #llvm.mlir.module_flag<max, "uwtable", 2 : i32>, #llvm.mlir.module_flag<override, "nvvm-reflect-ftz", 0 : i32>, #llvm.mlir.module_flag<max, "frame-pointer", 2 : i32>]
  llvm.mlir.global private unnamed_addr constant @".str"("CUDA Error: %s\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global private unnamed_addr constant @".str.1"("Dot product result: %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global private unnamed_addr constant @".str.2"("Expected result: %f\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global private unnamed_addr constant @".str.3"("Difference: %e\0A\00") {addr_space = 0 : i32, alignment = 1 : i64, dso_local, sym_visibility = "private"}
  llvm.mlir.global internal unnamed_addr @_ZZ18dot_product_kernelPKdS0_PdiE10shared_sum() {addr_space = 3 : i32, alignment = 8 : i64, dso_local, sym_visibility = "private"} : !llvm.array<1024 x f64> {
    %0 = llvm.mlir.undef : !llvm.array<1024 x f64>
    llvm.return %0 : !llvm.array<1024 x f64>
  }
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef, llvm.range = #llvm.constant_range<i32, -1, 1>}) attributes {dso_local, passthrough = ["mustprogress", "norecurse", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    %c8 = arith.constant 8 : index
    %c8388608 = arith.constant 8388608 : index
    %c8388608_i64 = arith.constant 8388608 : i64
    %cst = arith.constant 0.000000e+00 : f64
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8_i64 = arith.constant 8 : i64
    %c1048576_i32 = arith.constant 1048576 : i32
    %0 = llvm.mlir.addressof @".str" : !llvm.ptr
    %c-1_i32 = arith.constant -1 : i32
    %1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
    %2 = llvm.mlir.addressof @".str.2" : !llvm.ptr
    %cst_2 = arith.constant 0x4140000000000000 : f64
    %cst_3 = arith.constant 0xC140000000000000 : f64
    %3 = llvm.mlir.addressof @".str.3" : !llvm.ptr
    %4 = llvm.alloca %c1_i32 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %7 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %8 = llvm.call tail @_Znam(%c8388608_i64) : (i64 {llvm.noundef}) -> (!llvm.ptr {llvm.dereferenceable = 8388608 : i64, llvm.noalias, llvm.nonnull, llvm.noundef})
    %9 = llvm.call tail @_Znam(%c8388608_i64) : (i64 {llvm.noundef}) -> (!llvm.ptr {llvm.dereferenceable = 8388608 : i64, llvm.noalias, llvm.nonnull, llvm.noundef})
    llvm.intr.lifetime.start 8, %4 : !llvm.ptr
    llvm.store %cst, %4 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
    scf.for %arg0 = %c0_i64 to %c1048576_i64 step %c1_i64  : i64 {
      %20 = llvm.getelementptr inbounds|nuw %8[%arg0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %cst_0, %20 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
      %21 = llvm.getelementptr inbounds|nuw %9[%arg0] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      llvm.store %cst_1, %21 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
    }
    llvm.intr.lifetime.start 8, %5 : !llvm.ptr
    llvm.intr.lifetime.start 8, %6 : !llvm.ptr
    llvm.intr.lifetime.start 8, %7 : !llvm.ptr
    %memref = gpu.alloc  (%c8388608) : memref<?xi8, 1>
    %10 = "enzymexla.memref2pointer"(%memref) : (memref<?xi8, 1>) -> !llvm.ptr
    llvm.store %10, %5 : !llvm.ptr, !llvm.ptr
    %memref_4 = gpu.alloc  (%c8388608) : memref<?xi8, 1>
    %11 = "enzymexla.memref2pointer"(%memref_4) : (memref<?xi8, 1>) -> !llvm.ptr
    llvm.store %11, %6 : !llvm.ptr, !llvm.ptr
    %memref_5 = gpu.alloc  (%c8) : memref<?xi8, 1>
    %12 = "enzymexla.memref2pointer"(%memref_5) : (memref<?xi8, 1>) -> !llvm.ptr
    llvm.store %12, %7 : !llvm.ptr, !llvm.ptr
    %13 = "enzymexla.pointer2memref"(%8) : (!llvm.ptr) -> memref<?xi8>
    enzymexla.memcpy  %memref, %13, %c8388608 : memref<?xi8, 1>, memref<?xi8>
    %14 = "enzymexla.pointer2memref"(%9) : (!llvm.ptr) -> memref<?xi8>
    enzymexla.memcpy  %memref_4, %14, %c8388608 : memref<?xi8, 1>, memref<?xi8>
    %15 = llvm.call @cudaMemset(%12, %c0_i32, %c8_i64) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, i64 {llvm.noundef}) -> i32
    %16 = "enzymexla.gpu_wrapper"(%c4096, %c1, %c1, %c256, %c1, %c1) ({
      scf.parallel (%arg0) = (%c0) to (%c4096) step (%c1) {
        %alloca = memref.alloca() : memref<1024xf64>
        %20 = "enzymexla.memref2pointer"(%alloca) : (memref<1024xf64>) -> !llvm.ptr<3>
        scf.parallel (%arg1) = (%c0) to (%c256) step (%c1) {
          %21 = llvm.addrspacecast %20 : !llvm.ptr<3> to !llvm.ptr
          %22 = arith.index_castui %arg0 : index to i32
          %23 = arith.muli %22, %c256_i32 : i32
          %24 = arith.index_castui %arg1 : index to i64
          %25 = arith.index_castui %arg1 : index to i32
          %26 = arith.addi %23, %25 : i32
          %27 = llvm.getelementptr inbounds|nuw %21[0, %24] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1024 x f64>
          llvm.store %cst, %27 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
          %28 = arith.cmpi slt, %26, %c1048576_i32 : i32
          scf.if %28 {
            %31 = arith.extsi %26 : i32 to i64
            %32 = llvm.getelementptr inbounds %11[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
            %33 = llvm.getelementptr inbounds %10[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f64
            %34 = llvm.load %33 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
            %35 = llvm.load %32 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
            %36 = arith.mulf %34, %35 {fastmathFlags = #llvm.fastmath<contract>} : f64
            llvm.store %36, %27 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
          }
          "enzymexla.barrier"(%arg1, %c0, %c0) : (index, index, index) -> ()
          %29 = scf.while (%arg2 = %c256_i32) : (i32) -> i32 {
            %31 = arith.shrui %arg2, %c1_i32 : i32
            %32 = arith.cmpi ult, %25, %31 : i32
            scf.if %32 {
              %34 = arith.addi %31, %25 : i32
              %35 = arith.extui %34 {nonNeg} : i32 to i64
              %36 = llvm.getelementptr inbounds|nuw %21[0, %35] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1024 x f64>
              %37 = llvm.load %36 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
              %38 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
              %39 = arith.addf %37, %38 {fastmathFlags = #llvm.fastmath<contract>} : f64
              llvm.store %39, %27 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
            }
            %33 = arith.cmpi uge, %arg2, %c4_i32 : i32
            scf.condition(%33) %31 : i32
          } do {
          ^bb0(%arg2: i32):
            "enzymexla.barrier"(%arg1, %c0, %c0) : (index, index, index) -> ()
            scf.yield %arg2 : i32
          }
          "enzymexla.barrier"(%arg1, %c0, %c0) : (index, index, index) -> ()
          %30 = arith.cmpi eq, %25, %c0_i32 : i32
          scf.if %30 {
            %31 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
            %32 = llvm.atomicrmw fadd %12, %31 seq_cst {alignment = 8 : i64} : !llvm.ptr, f64
          }
          scf.reduce 
        }
        scf.reduce 
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    %17 = llvm.call @cudaGetLastError() : () -> i32
    %18 = arith.cmpi eq, %17, %c0_i32 : i32
    %19 = arith.select %18, %c0_i32, %c-1_i32 : i32
    scf.if %18 {
      %20 = "enzymexla.pointer2memref"(%4) : (!llvm.ptr) -> memref<?xi8>
      enzymexla.memcpy  %20, %memref_5, %c8 : memref<?xi8>, memref<?xi8, 1>
      %21 = llvm.load %4 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
      %22 = llvm.call @printf(%1, %21) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}) -> i32
      %23 = llvm.call @printf(%2, %cst_2) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}) -> i32
      %24 = llvm.load %4 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
      %25 = arith.addf %24, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
      %26 = math.absf %25 : f64
      %27 = llvm.call @printf(%3, %26) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}) -> i32
      %28 = llvm.call @cudaFree(%10) : (!llvm.ptr {llvm.noundef}) -> i32
      %29 = llvm.call @cudaFree(%11) : (!llvm.ptr {llvm.noundef}) -> i32
      %30 = llvm.call @cudaFree(%12) : (!llvm.ptr {llvm.noundef}) -> i32
      llvm.call @_ZdaPv(%8) {no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}) -> ()
      llvm.call @_ZdaPv(%9) {no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}) -> ()
    } else {
      %20 = llvm.call @cudaGetErrorString(%17) : (i32 {llvm.noundef}) -> !llvm.ptr
      %21 = llvm.call @printf(%0, %20) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
    }
    llvm.intr.lifetime.end 8, %7 : !llvm.ptr
    llvm.intr.lifetime.end 8, %6 : !llvm.ptr
    llvm.intr.lifetime.end 8, %5 : !llvm.ptr
    llvm.intr.lifetime.end 8, %4 : !llvm.ptr
    llvm.return %19 : i32
  }
  llvm.func local_unnamed_addr @_Znam(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.nonnull, llvm.noundef}) attributes {passthrough = ["nobuiltin", ["allocsize", "4294967295"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaMemcpy(!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}, i64 {llvm.noundef}, i32 {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaMemset(!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, i64 {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaGetLastError() -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @printf(!llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, ...) -> (i32 {llvm.noundef}) attributes {no_unwind, passthrough = ["nofree", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaGetErrorString(i32 {llvm.noundef}) -> !llvm.ptr attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaFree(!llvm.ptr {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @_ZdaPv(!llvm.ptr {llvm.noundef}) attributes {no_unwind, passthrough = ["nobuiltin", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @cudaMalloc(!llvm.ptr {llvm.noundef}, i64 {llvm.noundef}) -> i32 attributes {passthrough = [["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], sym_visibility = "private", target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
  llvm.func local_unnamed_addr @__mlir_launch_kernel__Z33__device_stub__dot_product_kernelPKdS0_Pdi(!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func internal @_Z33__device_stub__dot_product_kernelPKdS0_Pdi(%arg0: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.nofree, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg3: i32 {llvm.noundef}) attributes {convergent, dso_local, frame_pointer = #llvm.framePointerKind<all>, no_unwind, passthrough = ["mustprogress", "norecurse", ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "sm_90"], ["uniform-work-group-size", "true"]], sym_visibility = "private", target_cpu = "sm_90", target_features = #llvm.target_features<["+ptx82", "+sm_90"]>} {
    %0 = ub.poison : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %1 = llvm.mlir.addressof @_ZZ18dot_product_kernelPKdS0_PdiE10shared_sum : !llvm.ptr<3>
    %2 = llvm.addrspacecast %1 : !llvm.ptr<3> to !llvm.ptr
    %block_id_x = gpu.block_id  x
    %3 = arith.index_castui %block_id_x : index to i32
    %block_dim_x = gpu.block_dim  x
    %4 = arith.index_castui %block_dim_x : index to i32
    %5 = arith.muli %3, %4 : i32
    %thread_id_x = gpu.thread_id  x
    %6 = arith.index_castui %thread_id_x : index to i64
    %7 = arith.index_castui %thread_id_x : index to i32
    %8 = arith.addi %5, %7 : i32
    %9 = llvm.getelementptr inbounds|nuw %2[0, %6] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1024 x f64>
    llvm.store %cst, %9 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
    %10 = arith.cmpi slt, %8, %arg3 : i32
    scf.if %10 {
      %13 = arith.extsi %8 : i32 to i64
      %14 = llvm.getelementptr inbounds %arg1[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %15 = llvm.getelementptr inbounds %arg0[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f64
      %16 = llvm.load %15 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
      %17 = llvm.load %14 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
      %18 = arith.mulf %16, %17 {fastmathFlags = #llvm.fastmath<contract>} : f64
      llvm.store %18, %9 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
    }
    %11:2 = scf.while (%arg4 = %4, %arg5 = %c2_i32, %arg6 = %4) : (i32, i32, i32) -> (i32, i32) {
      gpu.barrier
      %13 = arith.cmpi ult, %arg4, %arg5 : i32
      %14 = arith.cmpi uge, %arg4, %arg5 : i32
      %15 = scf.if %13 -> (i32) {
        scf.yield %0 : i32
      } else {
        %16 = arith.shrui %arg6, %c1_i32 : i32
        %17 = arith.cmpi ult, %7, %16 : i32
        scf.if %17 {
          %18 = arith.addi %16, %7 : i32
          %19 = arith.extui %18 {nonNeg} : i32 to i64
          %20 = llvm.getelementptr inbounds|nuw %2[0, %19] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1024 x f64>
          %21 = llvm.load %20 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
          %22 = llvm.load %9 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
          %23 = arith.addf %21, %22 {fastmathFlags = #llvm.fastmath<contract>} : f64
          llvm.store %23, %9 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
        }
        scf.yield %16 : i32
      }
      scf.condition(%14) %arg6, %15 : i32, i32
    } do {
    ^bb0(%arg4: i32, %arg5: i32):
      scf.yield %arg4, %c4_i32, %arg5 : i32, i32, i32
    }
    %12 = arith.cmpi eq, %7, %c0_i32 : i32
    scf.if %12 {
      %13 = llvm.load %2 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
      %14 = llvm.atomicrmw fadd %arg2, %13 seq_cst {alignment = 8 : i64} : !llvm.ptr, f64
    }
    llvm.return
  }
  llvm.func @llvm.nvvm.barrier.cta.sync.aligned.all(i32) attributes {convergent, no_unwind, passthrough = ["nocallback"], sym_visibility = "private"}
}

// CHECK:  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef, llvm.range = #llvm.constant_range<i32, -1, 1>}) attributes {dso_local, passthrough = ["mustprogress", "norecurse", ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %c256_i32 = arith.constant 256 : i32
// CHECK-NEXT:    %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c256 = arith.constant 256 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c4096 = arith.constant 4096 : index
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    %c8388608 = arith.constant 8388608 : index
// CHECK-NEXT:    %c8388608_i64 = arith.constant 8388608 : i64
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %cst_0 = arith.constant 1.000000e+00 : f64
// CHECK-NEXT:    %cst_1 = arith.constant 2.000000e+00 : f64
// CHECK-NEXT:    %c8_i64 = arith.constant 8 : i64
// CHECK-NEXT:    %0 = llvm.mlir.addressof @".str" : !llvm.ptr
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %1 = llvm.mlir.addressof @".str.1" : !llvm.ptr
// CHECK-NEXT:    %2 = llvm.mlir.addressof @".str.2" : !llvm.ptr
// CHECK-NEXT:    %cst_2 = arith.constant 0x4140000000000000 : f64
// CHECK-NEXT:    %cst_3 = arith.constant 0xC140000000000000 : f64
// CHECK-NEXT:    %3 = llvm.mlir.addressof @".str.3" : !llvm.ptr
// CHECK-NEXT:    %4 = llvm.alloca %c1_i32 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    %5 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    %6 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    %7 = llvm.alloca %c1_i32 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK-NEXT:    %8 = llvm.call tail @_Znam(%c8388608_i64) : (i64 {llvm.noundef}) -> (!llvm.ptr {llvm.dereferenceable = 8388608 : i64, llvm.noalias, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:    %9 = llvm.call tail @_Znam(%c8388608_i64) : (i64 {llvm.noundef}) -> (!llvm.ptr {llvm.dereferenceable = 8388608 : i64, llvm.noalias, llvm.nonnull, llvm.noundef})
// CHECK-NEXT:    llvm.intr.lifetime.start 8, %4 : !llvm.ptr
// CHECK-NEXT:    llvm.store %cst, %4 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
// CHECK-NEXT:    affine.for %arg0 = 0 to 1048576 {
// CHECK-NEXT:      %20 = arith.index_cast %arg0 : index to i64
// CHECK-NEXT:      %21 = llvm.getelementptr inbounds|nuw %8[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:      llvm.store %cst_0, %21 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
// CHECK-NEXT:      %22 = llvm.getelementptr inbounds|nuw %9[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:      llvm.store %cst_1, %22 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm.intr.lifetime.start 8, %5 : !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.start 8, %6 : !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.start 8, %7 : !llvm.ptr
// CHECK-NEXT:    %memref = gpu.alloc  (%c8388608) : memref<?xi8, 1>
// CHECK-NEXT:    %10 = "enzymexla.memref2pointer"(%memref) : (memref<?xi8, 1>) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %10, %5 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %memref_4 = gpu.alloc  (%c8388608) : memref<?xi8, 1>
// CHECK-NEXT:    %11 = "enzymexla.memref2pointer"(%memref_4) : (memref<?xi8, 1>) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %11, %6 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %memref_5 = gpu.alloc  (%c8) : memref<?xi8, 1>
// CHECK-NEXT:    %12 = "enzymexla.memref2pointer"(%memref_5) : (memref<?xi8, 1>) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %12, %7 : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %13 = "enzymexla.pointer2memref"(%8) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:    enzymexla.memcpy  %memref, %13, %c8388608 : memref<?xi8, 1>, memref<?xi8>
// CHECK-NEXT:    %14 = "enzymexla.pointer2memref"(%9) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:    enzymexla.memcpy  %memref_4, %14, %c8388608 : memref<?xi8, 1>, memref<?xi8>
// CHECK-NEXT:    %15 = llvm.call @cudaMemset(%12, %c0_i32, %c8_i64) : (!llvm.ptr {llvm.noundef}, i32 {llvm.noundef}, i64 {llvm.noundef}) -> i32
// CHECK-NEXT:    %16 = "enzymexla.gpu_wrapper"(%c4096, %c1, %c1, %c256, %c1, %c1) ({
// CHECK-NEXT:      affine.parallel (%arg0) = (0) to (4096) {
// CHECK-NEXT:        %alloca = memref.alloca() : memref<1024xf64>
// CHECK-NEXT:        %20 = "enzymexla.memref2pointer"(%alloca) : (memref<1024xf64>) -> !llvm.ptr<3>
// CHECK-NEXT:        affine.parallel (%arg1) = (0) to (256) {
// CHECK-NEXT:          %21 = llvm.addrspacecast %20 : !llvm.ptr<3> to !llvm.ptr
// CHECK-NEXT:          %22 = arith.muli %arg0, %c256 : index
// CHECK-NEXT:          %23 = arith.index_castui %22 : index to i32
// CHECK-NEXT:          %24 = arith.index_castui %arg1 : index to i64
// CHECK-NEXT:          %25 = arith.index_castui %arg1 : index to i32
// CHECK-NEXT:          %26 = arith.addi %23, %25 : i32
// CHECK-NEXT:          %27 = llvm.getelementptr inbounds|nuw %21[0, %24] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1024 x f64>
// CHECK-NEXT:          llvm.store %cst, %27 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
// CHECK-NEXT:          %28 = arith.extsi %26 : i32 to i64
// CHECK-NEXT:          %29 = llvm.getelementptr inbounds %11[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:          %30 = llvm.getelementptr inbounds %10[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK-NEXT:          %31 = llvm.load %30 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
// CHECK-NEXT:          %32 = llvm.load %29 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
// CHECK-NEXT:          %33 = arith.mulf %31, %32 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:          llvm.store %33, %27 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
// CHECK-NEXT:          "enzymexla.barrier"(%arg1, %c0, %c0) : (index, index, index) -> ()
// CHECK-NEXT:          %34 = scf.while (%arg2 = %c256_i32) : (i32) -> i32 {
// CHECK-NEXT:            %35 = arith.shrui %arg2, %c1_i32 : i32
// CHECK-NEXT:            %36 = arith.cmpi ult, %25, %35 : i32
// CHECK-NEXT:            scf.if %36 {
// CHECK-NEXT:              %38 = arith.addi %35, %25 : i32
// CHECK-NEXT:              %39 = arith.extui %38 {nonNeg} : i32 to i64
// CHECK-NEXT:              %40 = llvm.getelementptr inbounds|nuw %21[0, %39] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1024 x f64>
// CHECK-NEXT:              %41 = llvm.load %40 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
// CHECK-NEXT:              %42 = llvm.load %27 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
// CHECK-NEXT:              %43 = arith.addf %41, %42 {fastmathFlags = #llvm.fastmath<contract>} : f64
// CHECK-NEXT:              llvm.store %43, %27 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : f64, !llvm.ptr
// CHECK-NEXT:            }
// CHECK-NEXT:            %37 = arith.cmpi uge, %arg2, %c4_i32 : i32
// CHECK-NEXT:            scf.condition(%37) %35 : i32
// CHECK-NEXT:          } do {
// CHECK-NEXT:          ^bb0(%arg2: i32):
// CHECK-NEXT:            "enzymexla.barrier"(%arg1, %c0, %c0) : (index, index, index) -> ()
// CHECK-NEXT:            scf.yield %arg2 : i32
// CHECK-NEXT:          }
// CHECK-NEXT:          "enzymexla.barrier"(%arg1, %c0, %c0) : (index, index, index) -> ()
// CHECK-NEXT:          affine.if #set(%arg1) {
// CHECK-NEXT:            %35 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
// CHECK-NEXT:            %36 = llvm.atomicrmw fadd %12, %35 seq_cst {alignment = 8 : i64} : !llvm.ptr, f64
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      "enzymexla.polygeist_yield"() : () -> ()
// CHECK-NEXT:    }) : (index, index, index, index, index, index) -> index
// CHECK-NEXT:    %17 = llvm.call @cudaGetLastError() : () -> i32
// CHECK-NEXT:    %18 = arith.cmpi eq, %17, %c0_i32 : i32
// CHECK-NEXT:    %19 = arith.select %18, %c0_i32, %c-1_i32 : i32
// CHECK-NEXT:    scf.if %18 {
// CHECK-NEXT:      %20 = "enzymexla.pointer2memref"(%4) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:      enzymexla.memcpy  %20, %memref_5, %c8 : memref<?xi8>, memref<?xi8, 1>
// CHECK-NEXT:      %21 = llvm.load %4 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
// CHECK-NEXT:      %22 = llvm.call @printf(%1, %21) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}) -> i32
// CHECK-NEXT:      %23 = llvm.call @printf(%2, %cst_2) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}) -> i32
// CHECK-NEXT:      %24 = llvm.load %4 {alignment = 8 : i64, tbaa = [#tbaa_tag]} : !llvm.ptr -> f64
// CHECK-NEXT:      %25 = arith.addf %24, %cst_3 {fastmathFlags = #llvm.fastmath<none>} : f64
// CHECK-NEXT:      %26 = math.absf %25 : f64
// CHECK-NEXT:      %27 = llvm.call @printf(%3, %26) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, f64 {llvm.noundef}) -> i32
// CHECK-NEXT:      %28 = llvm.call @cudaFree(%10) : (!llvm.ptr {llvm.noundef}) -> i32
// CHECK-NEXT:      %29 = llvm.call @cudaFree(%11) : (!llvm.ptr {llvm.noundef}) -> i32
// CHECK-NEXT:      %30 = llvm.call @cudaFree(%12) : (!llvm.ptr {llvm.noundef}) -> i32
// CHECK-NEXT:      llvm.call @_ZdaPv(%8) {no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}) -> ()
// CHECK-NEXT:      llvm.call @_ZdaPv(%9) {no_unwind} : (!llvm.ptr {llvm.nonnull, llvm.noundef}) -> ()
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %20 = llvm.call @cudaGetErrorString(%17) : (i32 {llvm.noundef}) -> !llvm.ptr
// CHECK-NEXT:      %21 = llvm.call @printf(%0, %20) vararg(!llvm.func<i32 (ptr, ...)>) {no_unwind} : (!llvm.ptr {llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !llvm.ptr {llvm.noundef}) -> i32
// CHECK-NEXT:    }
// CHECK-NEXT:    llvm.intr.lifetime.end 8, %7 : !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.end 8, %6 : !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.end 8, %5 : !llvm.ptr
// CHECK-NEXT:    llvm.intr.lifetime.end 8, %4 : !llvm.ptr
// CHECK-NEXT:    llvm.return %19 : i32
// CHECK-NEXT:  }
