// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition)" | FileCheck %s --check-prefix=GLR --implicit-check-not='{{10KokkosFlux.*\) kernel}}' --implicit-check-not='{{13KokkosScatter.*\) kernel}}' --implicit-check-not='{{10KokkosPrnt.*\) kernel}}' --implicit-check-not='llvm.call @cudaFuncGetAttributes'
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(gpu-launch-recognition,parallel-lower{wrapParallelOps=true})" | FileCheck %s --check-prefix=PL

module {
  llvm.func @__mlir_cuda_caller_phase3(...)
  llvm.func @cudaFuncGetAttributes(!llvm.ptr, !llvm.ptr) -> i32

  llvm.func internal @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosFluxPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_(%out: !llvm.ptr) attributes {passthrough = [["polygeist.host_symbol", "_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosFluxPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_"]]} {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %p = llvm.getelementptr %out[%tid] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %tid, %p : i32, !llvm.ptr
    llvm.return
  }

  llvm.func internal @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ13KokkosScatterPdPKiPKdiEUlmE_NS_11RangePolicyIJNS_4CudaEEEESA_EEEEvT_(%out: !llvm.ptr) attributes {passthrough = [["polygeist.host_symbol", "_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ13KokkosScatterPdPKiPKdiEUlmE_NS_11RangePolicyIJNS_4CudaEEEESA_EEEEvT_"]]} {
    %one = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %0 = llvm.inline_asm has_side_effects tail_call_kind = <tail> convergent asm_dialect = att "{\0A\09.reg .pred p;\0A\09isspacep.global p, $0;\0A\09@p  red.add.global.relaxed.gpu.f64 [$0],$1;\0A\09@!p red.add.relaxed.gpu.f64 [$0],$1;\0A\09}\0A\09", "l,d,~{memory}" %out, %one : (!llvm.ptr, f64) -> !llvm.void
    llvm.return
  }

  llvm.func internal @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosSyncPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_(%out: !llvm.ptr) attributes {passthrough = [["polygeist.host_symbol", "_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosSyncPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_"]]} {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %0 = llvm.inline_asm has_side_effects tail_call_kind = <tail> convergent asm_dialect = att "bar.sync 0;", "~{memory}" : () -> !llvm.void
    %p = llvm.getelementptr %out[%tid] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %tid, %p : i32, !llvm.ptr
    llvm.return
  }

  llvm.func internal @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10computeQTvPdPKdS5_iiEUlRKNS0_14CudaTeamMemberEE_NS_10TeamPolicyIJEEENS_4CudaEEEEEvT_(%out: !llvm.ptr) attributes {passthrough = [["polygeist.host_symbol", "_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10computeQTvPdPKdS5_iiEUlRKNS0_14CudaTeamMemberEE_NS_10TeamPolicyIJEEENS_4CudaEEEEEvT_"]]} {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %p = llvm.getelementptr %out[%tid] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %tid, %p : i32, !llvm.ptr
    llvm.return
  }

  llvm.func @helper(!llvm.ptr)
  llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32

  llvm.func internal @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosXtrnPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_(%out: !llvm.ptr) attributes {passthrough = [["polygeist.host_symbol", "_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosXtrnPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_"]]} {
    llvm.call @helper(%out) : (!llvm.ptr) -> ()
    llvm.return
  }

  llvm.func internal @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosPrntPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_(%out: !llvm.ptr) attributes {passthrough = [["polygeist.host_symbol", "_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosPrntPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_"]]} {
    %0 = llvm.call @vprintf(%out, %out) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.return
  }

  llvm.func internal @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosBlckPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_(%out: !llvm.ptr) attributes {passthrough = [["polygeist.host_symbol", "_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosBlckPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_"]]} {
    %one = llvm.mlir.constant(1 : i32) : i32
    %0 = llvm.atomicrmw add %out, %one syncscope("block") monotonic : !llvm.ptr, i32
    llvm.return
  }

  llvm.func internal @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosLanePdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_(%out: !llvm.ptr) attributes {passthrough = [["polygeist.host_symbol", "_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosLanePdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_"]]} {
    %lane = nvvm.read.ptx.sreg.laneid : i32
    %p = llvm.getelementptr %out[%lane] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %lane, %p : i32, !llvm.ptr
    llvm.return
  }

  llvm.func internal @plain(%out: !llvm.ptr) {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %p = llvm.getelementptr %out[%tid] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %tid, %p : i32, !llvm.ptr
    llvm.return
  }

  llvm.func @range(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %k = llvm.mlir.addressof @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosFluxPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_ : !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant(128 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    %0 = llvm.call @cudaFuncGetAttributes(%attrs, %k) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @__mlir_cuda_caller_phase3(%k, %one, %one, %one, %n, %one, %one, %shmem, %stream, %out) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @scatter(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %k = llvm.mlir.addressof @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ13KokkosScatterPdPKiPKdiEUlmE_NS_11RangePolicyIJNS_4CudaEEEESA_EEEEvT_ : !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant(128 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    %0 = llvm.call @cudaFuncGetAttributes(%attrs, %k) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @__mlir_cuda_caller_phase3(%k, %one, %one, %one, %n, %one, %one, %shmem, %stream, %out) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @sync(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %k = llvm.mlir.addressof @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosSyncPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_ : !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant(128 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    %0 = llvm.call @cudaFuncGetAttributes(%attrs, %k) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @__mlir_cuda_caller_phase3(%k, %one, %one, %one, %n, %one, %one, %shmem, %stream, %out) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @xtrn(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %k = llvm.mlir.addressof @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosXtrnPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_ : !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant(128 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    %0 = llvm.call @cudaFuncGetAttributes(%attrs, %k) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @__mlir_cuda_caller_phase3(%k, %one, %one, %one, %n, %one, %one, %shmem, %stream, %out) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @prnt(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %k = llvm.mlir.addressof @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosPrntPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_ : !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant(128 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    %0 = llvm.call @cudaFuncGetAttributes(%attrs, %k) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @__mlir_cuda_caller_phase3(%k, %one, %one, %one, %n, %one, %one, %shmem, %stream, %out) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @blck(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %k = llvm.mlir.addressof @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosBlckPdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_ : !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant(128 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    %0 = llvm.call @cudaFuncGetAttributes(%attrs, %k) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @__mlir_cuda_caller_phase3(%k, %one, %one, %one, %n, %one, %one, %shmem, %stream, %out) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @lane(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %k = llvm.mlir.addressof @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10KokkosLanePdPKdS5_iEUlmE_NS_11RangePolicyIJNS_4CudaEEEES8_EEEEvT_ : !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant(128 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    %0 = llvm.call @cudaFuncGetAttributes(%attrs, %k) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @__mlir_cuda_caller_phase3(%k, %one, %one, %one, %n, %one, %one, %shmem, %stream, %out) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @team(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %k = llvm.mlir.addressof @reactant$_ZN6Kokkos4ImplL48__device_stub__cuda_parallel_launch_local_memoryINS0_11ParallelForIZ10computeQTvPdPKdS5_iiEUlRKNS0_14CudaTeamMemberEE_NS_10TeamPolicyIJEEENS_4CudaEEEEEvT_ : !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant(128 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    %0 = llvm.call @cudaFuncGetAttributes(%attrs, %k) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @__mlir_cuda_caller_phase3(%k, %one, %one, %one, %n, %one, %one, %shmem, %stream, %out) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  llvm.func @plainhost(%attrs: !llvm.ptr, %out: !llvm.ptr) {
    %k = llvm.mlir.addressof @plain : !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant(128 : i32) : i32
    %shmem = llvm.mlir.constant(0 : i64) : i64
    %stream = llvm.mlir.zero : !llvm.ptr
    %0 = llvm.call @cudaFuncGetAttributes(%attrs, %k) : (!llvm.ptr, !llvm.ptr) -> i32
    llvm.call @__mlir_cuda_caller_phase3(%k, %one, %one, %one, %n, %one, %one, %shmem, %stream, %out) vararg(!llvm.func<void (...)>) : (!llvm.ptr, i32, i32, i32, i32, i32, i32, i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
}

// GLR-DAG:   gpu.func @reactant${{.*}}10KokkosSync{{.*}} kernel
// GLR-DAG:   gpu.func @reactant${{.*}}10computeQTv{{.*}} kernel
// GLR-DAG:   gpu.func @plain({{.*}}) kernel
// GLR-DAG:   gpu.func @reactant${{.*}}10KokkosXtrn{{.*}} kernel
// GLR-DAG:   gpu.func @reactant${{.*}}10KokkosBlck{{.*}} kernel
// GLR-DAG:   gpu.func @reactant${{.*}}10KokkosLane{{.*}} kernel
// GLR-DAG:   gpu.func @__polygeist_empty_kernel() kernel

// GLR-LABEL: llvm.func @range(
// GLR:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_ZN6Kokkos{{.*}}10KokkosFlux{{.*}}"}> : (!llvm.ptr) -> i32
// GLR:         gpu.launch blocks
// GLR-NEXT:      llvm.call @reactant${{.*}}10KokkosFlux
// GLR:         polygeist.host_symbol", "_ZN6Kokkos{{.*}}10KokkosFlux

// GLR-LABEL: llvm.func @scatter(
// GLR:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_ZN6Kokkos{{.*}}13KokkosScatter{{.*}}"}> : (!llvm.ptr) -> i32
// GLR:         gpu.launch blocks
// GLR-NEXT:      llvm.call @reactant${{.*}}13KokkosScatter
// GLR:         polygeist.host_symbol", "_ZN6Kokkos{{.*}}13KokkosScatter

// GLR-LABEL: llvm.func @sync(
// GLR:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@reactant${{.*}}10KokkosSync{{.*}}}> : (!llvm.ptr) -> i32
// GLR:         gpu.launch_func @__mlir_gpu_module::@reactant${{.*}}10KokkosSync

// GLR-LABEL: llvm.func @xtrn(
// GLR:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@reactant${{.*}}10KokkosXtrn{{.*}}}> : (!llvm.ptr) -> i32
// GLR:         gpu.launch_func @__mlir_gpu_module::@reactant${{.*}}10KokkosXtrn

// GLR-LABEL: llvm.func @prnt(
// GLR:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_ZN6Kokkos{{.*}}10KokkosPrnt{{.*}}"}> : (!llvm.ptr) -> i32
// GLR:         gpu.launch blocks
// GLR-NEXT:      llvm.call @reactant${{.*}}10KokkosPrnt

// GLR-LABEL: llvm.func @blck(
// GLR:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@reactant${{.*}}10KokkosBlck{{.*}}}> : (!llvm.ptr) -> i32
// GLR:         gpu.launch_func @__mlir_gpu_module::@reactant${{.*}}10KokkosBlck

// GLR-LABEL: llvm.func @lane(
// GLR:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@reactant${{.*}}10KokkosLane{{.*}}}> : (!llvm.ptr) -> i32
// GLR:         gpu.launch_func @__mlir_gpu_module::@reactant${{.*}}10KokkosLane

// GLR-LABEL: llvm.func @team(
// GLR:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@reactant${{.*}}10computeQTv{{.*}}}> : (!llvm.ptr) -> i32
// GLR:         gpu.launch_func @__mlir_gpu_module::@reactant${{.*}}10computeQTv

// GLR-LABEL: llvm.func @plainhost(
// GLR:         "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@plain}> : (!llvm.ptr) -> i32
// GLR:         gpu.launch_func @__mlir_gpu_module::@plain

// PL-LABEL:  llvm.func @range(
// PL:          "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_ZN6Kokkos{{.*}}10KokkosFlux{{.*}}"}>
// PL:          "enzymexla.gpu_wrapper"
// PL:          polygeist.host_symbol", "_ZN6Kokkos{{.*}}10KokkosFlux
// PL-LABEL:  llvm.func @scatter(
// PL:          "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_ZN6Kokkos{{.*}}13KokkosScatter{{.*}}"}>
// PL:          "enzymexla.gpu_wrapper"
// PL:          polygeist.host_symbol", "_ZN6Kokkos{{.*}}13KokkosScatter
// PL-LABEL:  llvm.func @sync(
// PL:          gpu.launch_func @__mlir_gpu_module::@reactant${{.*}}10KokkosSync
// PL-LABEL:  llvm.func @xtrn(
// PL:          gpu.launch_func @__mlir_gpu_module::@reactant${{.*}}10KokkosXtrn
// PL-LABEL:  llvm.func @prnt(
// PL:          "enzymexla.gpu_func_attributes"(%arg0) <{fn = @__mlir_gpu_module::@__polygeist_empty_kernel, host_symbol = "_ZN6Kokkos{{.*}}10KokkosPrnt{{.*}}"}>
// PL:          "enzymexla.gpu_wrapper"
// PL-LABEL:  llvm.func @blck(
// PL:          gpu.launch_func @__mlir_gpu_module::@reactant${{.*}}10KokkosBlck
// PL-LABEL:  llvm.func @lane(
// PL:          gpu.launch_func @__mlir_gpu_module::@reactant${{.*}}10KokkosLane
// PL-LABEL:  llvm.func @team(
// PL:          gpu.launch_func @__mlir_gpu_module::@reactant${{.*}}10computeQTv
// PL-LABEL:  llvm.func @plainhost(
// PL:          gpu.launch_func @__mlir_gpu_module::@plain
