// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-gpu})" --verify-diagnostics --split-input-file

// An xla backend cannot execute a raw device kernel: any launch that
// survived raising is a compile-time error, not a runtime crash.
module attributes {gpu.container_module} {
  func.func @foo(%c1: index, %arg: memref<?xf32, 1>) {
    // expected-error@+1 {{kernel launch survived raising; the xla-gpu backend cannot execute raw device kernels}}
    gpu.launch_func @gpumod::@gpufunc blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg : memref<?xf32, 1>)
    return
  }

  gpu.module @gpumod [#nvvm.target<O = 3, chip = "sm_120", features = "+ptx73", flags = {}>] attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>} {
    gpu.func @gpufunc(%arg0: memref<?xf32, 1>) kernel {
      %thread_id_x = gpu.thread_id x
      %ld = memref.load %arg0[%thread_id_x] : memref<?xf32, 1>
      memref.store %ld, %arg0[%thread_id_x] : memref<?xf32, 1>
      gpu.return
    }
  }
}
