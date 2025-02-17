// RUN: enzymexlamlir-opt --libdevice-funcs-raise %s | FileCheck %s

module {
  gpu.module @test_module_6 {
    llvm.func @__nv_fabsf(f32) -> f32
    llvm.func @__nv_fabs(f64) -> f64
    llvm.func @gpu_fabs(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.absf
      %0 = llvm.call @__nv_fabsf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_fabs(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_fabs(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_fabs(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_7 {
    llvm.func @__nv_cbrtf(f32) -> f32
    llvm.func @__nv_cbrt(f64) -> f64
    llvm.func @gpu_cbrt(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.cbrt
      %0 = llvm.call @__nv_cbrtf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_cbrt(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_cbrt(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_cbrt(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_8 {
    llvm.func @__nv_ceilf(f32) -> f32
    llvm.func @__nv_ceil(f64) -> f64
    llvm.func @gpu_ceil(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.ceil
      %0 = llvm.call @__nv_ceilf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_ceil(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_ceil(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_ceil(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_9 {
    llvm.func @__nv_floorf(f32) -> f32
    llvm.func @__nv_floor(f64) -> f64
    llvm.func @gpu_floor(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.floor
      %0 = llvm.call @__nv_floorf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_floor(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_floor(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_floor(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_10 {
    llvm.func @__nv_cosf(f32) -> f32
    llvm.func @__nv_cos(f64) -> f64
    llvm.func @__nv_fast_cosf(f32) -> f32
    llvm.func @gpu_cos(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64, f32)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-3: math.cos
      %0 = llvm.call @__nv_cosf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_cos(%arg1) : (f64) -> f64
      %2 = llvm.call @__nv_fast_cosf(%arg0) : (f32) -> f32
      %3 = llvm.mlir.undef : !llvm.struct<(f32, f64, f32)>
      %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(f32, f64, f32)> 
      %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(f32, f64, f32)> 
      %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(f32, f64, f32)> 
      llvm.return %6 : !llvm.struct<(f32, f64, f32)>
    }
    llvm.func @_mlir_ciface_gpu_cos(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_cos(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64, f32)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64, f32)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_11 {
    llvm.func @__nv_expf(f32) -> f32
    llvm.func @__nv_exp(f64) -> f64
    llvm.func @__nv_fast_expf(f32) -> f32
    llvm.func @gpu_exp(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64, f32)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-3: math.exp
      %0 = llvm.call @__nv_expf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_exp(%arg1) : (f64) -> f64
      %2 = llvm.call @__nv_fast_expf(%arg0) : (f32) -> f32
      %3 = llvm.mlir.undef : !llvm.struct<(f32, f64, f32)>
      %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(f32, f64, f32)> 
      %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(f32, f64, f32)> 
      %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(f32, f64, f32)> 
      llvm.return %6 : !llvm.struct<(f32, f64, f32)>
    }
    llvm.func @_mlir_ciface_gpu_exp(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_exp(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64, f32)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64, f32)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_12 {
    llvm.func @__nv_exp2f(f32) -> f32
    llvm.func @__nv_exp2(f64) -> f64
    llvm.func @gpu_exp2(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.exp2
      %0 = llvm.call @__nv_exp2f(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_exp2(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_exp2(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_exp2(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_13 {
    llvm.func @__nv_logf(f32) -> f32
    llvm.func @__nv_log(f64) -> f64
    llvm.func @__nv_fast_logf(f32) -> f32
    llvm.func @gpu_log(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64, f32)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-3: math.log
      %0 = llvm.call @__nv_logf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_log(%arg1) : (f64) -> f64
      %2 = llvm.call @__nv_fast_logf(%arg0) : (f32) -> f32
      %3 = llvm.mlir.undef : !llvm.struct<(f32, f64, f32)>
      %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(f32, f64, f32)> 
      %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(f32, f64, f32)> 
      %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(f32, f64, f32)> 
      llvm.return %6 : !llvm.struct<(f32, f64, f32)>
    }
    llvm.func @_mlir_ciface_gpu_log(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_log(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64, f32)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64, f32)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_14 {
    llvm.func @__nv_log10f(f32) -> f32
    llvm.func @__nv_log10(f64) -> f64
    llvm.func @__nv_fast_log10f(f32) -> f32
    llvm.func @gpu_log10(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64, f32)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-3: math.log10
      %0 = llvm.call @__nv_log10f(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_log10(%arg1) : (f64) -> f64
      %2 = llvm.call @__nv_fast_log10f(%arg0) : (f32) -> f32
      %3 = llvm.mlir.undef : !llvm.struct<(f32, f64, f32)>
      %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(f32, f64, f32)> 
      %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(f32, f64, f32)> 
      %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(f32, f64, f32)> 
      llvm.return %6 : !llvm.struct<(f32, f64, f32)>
    }
    llvm.func @_mlir_ciface_gpu_log10(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_log10(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64, f32)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64, f32)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_15 {
    llvm.func @__nv_log1pf(f32) -> f32
    llvm.func @__nv_log1p(f64) -> f64
    llvm.func @gpu_log1p(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.log1p
      %0 = llvm.call @__nv_log1pf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_log1p(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_log1p(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_log1p(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_16 {
    llvm.func @__nv_log2f(f32) -> f32
    llvm.func @__nv_log2(f64) -> f64
    llvm.func @__nv_fast_log2f(f32) -> f32
    llvm.func @gpu_log2(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64, f32)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-3: math.log2
      %0 = llvm.call @__nv_log2f(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_log2(%arg1) : (f64) -> f64
      %2 = llvm.call @__nv_fast_log2f(%arg0) : (f32) -> f32
      %3 = llvm.mlir.undef : !llvm.struct<(f32, f64, f32)>
      %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(f32, f64, f32)> 
      %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(f32, f64, f32)> 
      %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(f32, f64, f32)> 
      llvm.return %6 : !llvm.struct<(f32, f64, f32)>
    }
    llvm.func @_mlir_ciface_gpu_log2(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_log2(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64, f32)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64, f32)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_17 {
    llvm.func @__nv_sinf(f32) -> f32
    llvm.func @__nv_sin(f64) -> f64
    llvm.func @__nv_fast_sinf(f32) -> f32
    llvm.func @gpu_sin(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64, f32)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-3: math.sin
      %0 = llvm.call @__nv_sinf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_sin(%arg1) : (f64) -> f64
      %2 = llvm.call @__nv_fast_sinf(%arg0) : (f32) -> f32
      %3 = llvm.mlir.undef : !llvm.struct<(f32, f64, f32)>
      %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(f32, f64, f32)> 
      %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(f32, f64, f32)> 
      %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(f32, f64, f32)> 
      llvm.return %6 : !llvm.struct<(f32, f64, f32)>
    }
    llvm.func @_mlir_ciface_gpu_sin(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_sin(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64, f32)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64, f32)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_18 {
    llvm.func @__nv_tanf(f32) -> f32
    llvm.func @__nv_tan(f64) -> f64
    llvm.func @__nv_fast_tanf(f32) -> f32
    llvm.func @gpu_tan(%arg0: f16, %arg1: f32, %arg2: f64) -> !llvm.struct<(f16, f32, f64, f32)> attributes {llvm.emit_c_interface} {
      %0 = llvm.fpext %arg0 : f16 to f32
      // CHECK-COUNT-4: math.tan
      %1 = llvm.call @__nv_tanf(%0) : (f32) -> f32
      %2 = llvm.fptrunc %1 : f32 to f16
      %3 = llvm.call @__nv_tanf(%arg1) : (f32) -> f32
      %4 = llvm.call @__nv_tan(%arg2) : (f64) -> f64
      %5 = llvm.call @__nv_fast_tanf(%arg1) : (f32) -> f32
      %6 = llvm.mlir.undef : !llvm.struct<(f16, f32, f64, f32)>
      %7 = llvm.insertvalue %2, %6[0] : !llvm.struct<(f16, f32, f64, f32)> 
      %8 = llvm.insertvalue %3, %7[1] : !llvm.struct<(f16, f32, f64, f32)> 
      %9 = llvm.insertvalue %4, %8[2] : !llvm.struct<(f16, f32, f64, f32)> 
      %10 = llvm.insertvalue %5, %9[3] : !llvm.struct<(f16, f32, f64, f32)> 
      llvm.return %10 : !llvm.struct<(f16, f32, f64, f32)>
    }
    llvm.func @_mlir_ciface_gpu_tan(%arg0: !llvm.ptr, %arg1: f16, %arg2: f32, %arg3: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_tan(%arg1, %arg2, %arg3) : (f16, f32, f64) -> !llvm.struct<(f16, f32, f64, f32)>
      llvm.store %0, %arg0 : !llvm.struct<(f16, f32, f64, f32)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_19 {
    llvm.func @__nv_tanhf(f32) -> f32
    llvm.func @__nv_tanh(f64) -> f64
    llvm.func @gpu_tanh(%arg0: f16, %arg1: f32, %arg2: f64) -> !llvm.struct<(f16, f32, f64)> attributes {llvm.emit_c_interface} {
      %0 = llvm.fpext %arg0 : f16 to f32
      // CHECK-COUNT-3: math.tanh
      %1 = llvm.call @__nv_tanhf(%0) : (f32) -> f32
      %2 = llvm.fptrunc %1 : f32 to f16
      %3 = llvm.call @__nv_tanhf(%arg1) : (f32) -> f32
      %4 = llvm.call @__nv_tanh(%arg2) : (f64) -> f64
      %5 = llvm.mlir.undef : !llvm.struct<(f16, f32, f64)>
      %6 = llvm.insertvalue %2, %5[0] : !llvm.struct<(f16, f32, f64)> 
      %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(f16, f32, f64)> 
      %8 = llvm.insertvalue %4, %7[2] : !llvm.struct<(f16, f32, f64)> 
      llvm.return %8 : !llvm.struct<(f16, f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_tanh(%arg0: !llvm.ptr, %arg1: f16, %arg2: f32, %arg3: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_tanh(%arg1, %arg2, %arg3) : (f16, f32, f64) -> !llvm.struct<(f16, f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f16, f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_20 {
    llvm.func @__nv_rsqrtf(f32) -> f32
    llvm.func @__nv_rsqrt(f64) -> f64
    llvm.func @gpu_rsqrt(%arg0: f16, %arg1: f32, %arg2: f64) -> !llvm.struct<(f16, f32, f64)> attributes {llvm.emit_c_interface} {
      %0 = llvm.fpext %arg0 : f16 to f32
      // CHECK-COUNT-3: math.rsqrt
      %1 = llvm.call @__nv_rsqrtf(%0) : (f32) -> f32
      %2 = llvm.fptrunc %1 : f32 to f16
      %3 = llvm.call @__nv_rsqrtf(%arg1) : (f32) -> f32
      %4 = llvm.call @__nv_rsqrt(%arg2) : (f64) -> f64
      %5 = llvm.mlir.undef : !llvm.struct<(f16, f32, f64)>
      %6 = llvm.insertvalue %2, %5[0] : !llvm.struct<(f16, f32, f64)> 
      %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(f16, f32, f64)> 
      %8 = llvm.insertvalue %4, %7[2] : !llvm.struct<(f16, f32, f64)> 
      llvm.return %8 : !llvm.struct<(f16, f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_rsqrt(%arg0: !llvm.ptr, %arg1: f16, %arg2: f32, %arg3: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_rsqrt(%arg1, %arg2, %arg3) : (f16, f32, f64) -> !llvm.struct<(f16, f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f16, f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_21 {
    llvm.func @__nv_sqrtf(f32) -> f32
    llvm.func @__nv_sqrt(f64) -> f64
    llvm.func @gpu_sqrt(%arg0: f16, %arg1: f32, %arg2: f64) -> !llvm.struct<(f16, f32, f64)> attributes {llvm.emit_c_interface} {
      %0 = llvm.fpext %arg0 : f16 to f32
      // CHECK-COUNT-3: math.sqrt
      %1 = llvm.call @__nv_sqrtf(%0) : (f32) -> f32
      %2 = llvm.fptrunc %1 : f32 to f16
      %3 = llvm.call @__nv_sqrtf(%arg1) : (f32) -> f32
      %4 = llvm.call @__nv_sqrt(%arg2) : (f64) -> f64
      %5 = llvm.mlir.undef : !llvm.struct<(f16, f32, f64)>
      %6 = llvm.insertvalue %2, %5[0] : !llvm.struct<(f16, f32, f64)> 
      %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(f16, f32, f64)> 
      %8 = llvm.insertvalue %4, %7[2] : !llvm.struct<(f16, f32, f64)> 
      llvm.return %8 : !llvm.struct<(f16, f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_sqrt(%arg0: !llvm.ptr, %arg1: f16, %arg2: f32, %arg3: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_sqrt(%arg1, %arg2, %arg3) : (f16, f32, f64) -> !llvm.struct<(f16, f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f16, f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_22 {
    llvm.func @__nv_atanf(f32) -> f32
    llvm.func @__nv_atan(f64) -> f64
    llvm.func @gpu_atan(%arg0: f16, %arg1: f32, %arg2: f64) -> !llvm.struct<(f16, f32, f64)> attributes {llvm.emit_c_interface} {
      %0 = llvm.fpext %arg0 : f16 to f32
      // CHECK-COUNT-3: math.atan
      %1 = llvm.call @__nv_atanf(%0) : (f32) -> f32
      %2 = llvm.fptrunc %1 : f32 to f16
      %3 = llvm.call @__nv_atanf(%arg1) : (f32) -> f32
      %4 = llvm.call @__nv_atan(%arg2) : (f64) -> f64
      %5 = llvm.mlir.undef : !llvm.struct<(f16, f32, f64)>
      %6 = llvm.insertvalue %2, %5[0] : !llvm.struct<(f16, f32, f64)> 
      %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(f16, f32, f64)> 
      %8 = llvm.insertvalue %4, %7[2] : !llvm.struct<(f16, f32, f64)> 
      llvm.return %8 : !llvm.struct<(f16, f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_atan(%arg0: !llvm.ptr, %arg1: f16, %arg2: f32, %arg3: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_atan(%arg1, %arg2, %arg3) : (f16, f32, f64) -> !llvm.struct<(f16, f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f16, f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_23 {
    llvm.func @__nv_atan2f(f32, f32) -> f32
    llvm.func @__nv_atan2(f64, f64) -> f64
    llvm.func @gpu_atan2(%arg0: f16, %arg1: f32, %arg2: f64) -> !llvm.struct<(f16, f32, f64)> attributes {llvm.emit_c_interface} {
      %0 = llvm.fpext %arg0 : f16 to f32
      %1 = llvm.fpext %arg0 : f16 to f32
      // CHECK-COUNT-3: math.atan2
      %2 = llvm.call @__nv_atan2f(%0, %1) : (f32, f32) -> f32
      %3 = llvm.fptrunc %2 : f32 to f16
      %4 = llvm.call @__nv_atan2f(%arg1, %arg1) : (f32, f32) -> f32
      %5 = llvm.call @__nv_atan2(%arg2, %arg2) : (f64, f64) -> f64
      %6 = llvm.mlir.undef : !llvm.struct<(f16, f32, f64)>
      %7 = llvm.insertvalue %3, %6[0] : !llvm.struct<(f16, f32, f64)> 
      %8 = llvm.insertvalue %4, %7[1] : !llvm.struct<(f16, f32, f64)> 
      %9 = llvm.insertvalue %5, %8[2] : !llvm.struct<(f16, f32, f64)> 
      llvm.return %9 : !llvm.struct<(f16, f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_atan2(%arg0: !llvm.ptr, %arg1: f16, %arg2: f32, %arg3: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_atan2(%arg1, %arg2, %arg3) : (f16, f32, f64) -> !llvm.struct<(f16, f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f16, f32, f64)>, !llvm.ptr
      llvm.return
    }
  }

  gpu.module @test_module_25 {
    llvm.func @__nv_expm1f(f32) -> f32
    llvm.func @__nv_expm1(f64) -> f64
    llvm.func @gpu_expm1(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.expm1
      %0 = llvm.call @__nv_expm1f(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_expm1(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_expm1(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_expm1(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_26 {
    llvm.func @__nv_powf(f32, f32) -> f32
    llvm.func @__nv_pow(f64, f64) -> f64
    llvm.func @__nv_fast_powf(f32, f32) -> f32
    llvm.func @gpu_pow(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64, f32)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-3: math.powf
      %0 = llvm.call @__nv_powf(%arg0, %arg0) : (f32, f32) -> f32
      %1 = llvm.call @__nv_pow(%arg1, %arg1) : (f64, f64) -> f64
      %2 = llvm.call @__nv_fast_powf(%arg0, %arg0) : (f32, f32) -> f32
      %3 = llvm.mlir.undef : !llvm.struct<(f32, f64, f32)>
      %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(f32, f64, f32)> 
      %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(f32, f64, f32)> 
      %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(f32, f64, f32)> 
      llvm.return %6 : !llvm.struct<(f32, f64, f32)>
    }
    llvm.func @_mlir_ciface_gpu_pow(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_pow(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64, f32)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64, f32)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_31 {
    llvm.func @__nv_fmodf(f32, f32) -> f32
    llvm.func @__nv_fmod(f64, f64) -> f64
    llvm.func @gpu_fmod(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: arith.remf
      %0 = llvm.call @__nv_fmodf(%arg0, %arg0) : (f32, f32) -> f32
      %1 = llvm.call @__nv_fmod(%arg1, %arg1) : (f64, f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_fmod(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_fmod(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_32 {
    llvm.func @__nv_erff(f32) -> f32
    llvm.func @__nv_erf(f64) -> f64
    llvm.func @gpu_erf(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.erf
      %0 = llvm.call @__nv_erff(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_erf(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_erf(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_erf(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_35 {
    llvm.func @__nv_acosf(f32) -> f32
    llvm.func @__nv_acos(f64) -> f64
    llvm.func @gpu_acos(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.acos
      %0 = llvm.call @__nv_acosf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_acos(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_acos(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_acos(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_36 {
    llvm.func @__nv_acoshf(f32) -> f32
    llvm.func @__nv_acosh(f64) -> f64
    llvm.func @gpu_acosh(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.acosh
      %0 = llvm.call @__nv_acoshf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_acosh(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_acosh(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_acosh(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_37 {
    llvm.func @__nv_asinf(f32) -> f32
    llvm.func @__nv_asin(f64) -> f64
    llvm.func @gpu_asin(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.asin
      %0 = llvm.call @__nv_asinf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_asin(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_asin(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_asin(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_38 {
    llvm.func @__nv_asinhf(f32) -> f32
    llvm.func @__nv_asinh(f64) -> f64
    llvm.func @gpu_asinh(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.asinh
      %0 = llvm.call @__nv_asinhf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_asinh(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_asinh(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_asinh(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_39 {
    llvm.func @__nv_atanhf(f32) -> f32
    llvm.func @__nv_atanh(f64) -> f64
    llvm.func @gpu_atanh(%arg0: f16, %arg1: f32, %arg2: f64) -> !llvm.struct<(f16, f32, f64)> attributes {llvm.emit_c_interface} {
      %0 = llvm.fpext %arg0 : f16 to f32
      // CHECK-COUNT-3: math.atanh
      %1 = llvm.call @__nv_atanhf(%0) : (f32) -> f32
      %2 = llvm.fptrunc %1 : f32 to f16
      %3 = llvm.call @__nv_atanhf(%arg1) : (f32) -> f32
      %4 = llvm.call @__nv_atanh(%arg2) : (f64) -> f64
      %5 = llvm.mlir.undef : !llvm.struct<(f16, f32, f64)>
      %6 = llvm.insertvalue %2, %5[0] : !llvm.struct<(f16, f32, f64)> 
      %7 = llvm.insertvalue %3, %6[1] : !llvm.struct<(f16, f32, f64)> 
      %8 = llvm.insertvalue %4, %7[2] : !llvm.struct<(f16, f32, f64)> 
      llvm.return %8 : !llvm.struct<(f16, f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_atanh(%arg0: !llvm.ptr, %arg1: f16, %arg2: f32, %arg3: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_atanh(%arg1, %arg2, %arg3) : (f16, f32, f64) -> !llvm.struct<(f16, f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f16, f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_40 {
    llvm.func @__nv_copysignf(f32, f32) -> f32
    llvm.func @__nv_copysign(f64, f64) -> f64
    llvm.func @gpu_copysign(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.copysign
      %0 = llvm.call @__nv_copysignf(%arg0, %arg0) : (f32, f32) -> f32
      %1 = llvm.call @__nv_copysign(%arg1, %arg1) : (f64, f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_copysign(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_copysign(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_41 {
    llvm.func @__nv_coshf(f32) -> f32
    llvm.func @__nv_cosh(f64) -> f64
    llvm.func @gpu_cosh(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.cosh
      %0 = llvm.call @__nv_coshf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_cosh(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_cosh(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_cosh(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_42 {
    llvm.func @__nv_fmaf(f32, f32, f32) -> f32
    llvm.func @__nv_fma(f64, f64, f64) -> f64
    llvm.func @gpu_fma(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.fma
      %0 = llvm.call @__nv_fmaf(%arg0, %arg0, %arg0) : (f32, f32, f32) -> f32
      %1 = llvm.call @__nv_fma(%arg1, %arg1, %arg1) : (f64, f64, f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_fma(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_fma(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_43 {
    llvm.func @__nv_roundf(f32) -> f32
    llvm.func @__nv_round(f64) -> f64
    llvm.func @gpu_round(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.round
      %0 = llvm.call @__nv_roundf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_round(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_round(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_round(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_44 {
    llvm.func @__nv_rintf(f32) -> f32
    llvm.func @__nv_rint(f64) -> f64
    llvm.func @gpu_roundeven(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
     // CHECK-COUNT-2: math.roundeven  
      %0 = llvm.call @__nv_rintf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_rint(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_roundeven(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_roundeven(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }
  gpu.module @test_module_45 {
    llvm.func @__nv_sinhf(f32) -> f32
    llvm.func @__nv_sinh(f64) -> f64
    llvm.func @gpu_sinh(%arg0: f32, %arg1: f64) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.sinh
      %0 = llvm.call @__nv_sinhf(%arg0) : (f32) -> f32
      %1 = llvm.call @__nv_sinh(%arg1) : (f64) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_sinh(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_sinh(%arg1, %arg2) : (f32, f64) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }

  gpu.module @test_module_46 {
    llvm.func @__nv_powif(f32, i32) -> f32
    llvm.func @__nv_powi(f64, i32) -> f64
    llvm.func @gpu_powi(%arg0: f32, %arg1: f64, %arg2: i32) -> !llvm.struct<(f32, f64)> attributes {llvm.emit_c_interface} {
      // CHECK-COUNT-2: math.fpowi
      %0 = llvm.call @__nv_powif(%arg0, %arg2) : (f32, i32) -> f32
      %1 = llvm.call @__nv_powi(%arg1, %arg2) : (f64, i32) -> f64
      %2 = llvm.mlir.undef : !llvm.struct<(f32, f64)>
      %3 = llvm.insertvalue %0, %2[0] : !llvm.struct<(f32, f64)> 
      %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(f32, f64)> 
      llvm.return %4 : !llvm.struct<(f32, f64)>
    }
    llvm.func @_mlir_ciface_gpu_powi(%arg0: !llvm.ptr, %arg1: f32, %arg2: f64, %arg3: i32) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_powi(%arg1, %arg2, %arg3) : (f32, f64, i32) -> !llvm.struct<(f32, f64)>
      llvm.store %0, %arg0 : !llvm.struct<(f32, f64)>, !llvm.ptr
      llvm.return
    }
  }

  gpu.module @test_module_47 {
    llvm.func @__nv_abs(i32) -> i32
    llvm.func @__nv_llabs(i64) -> i64
    llvm.func @gpu_abs(%arg0: i32) -> i32 attributes {llvm.emit_c_interface} {
      // CHECK: math.absi
      %0 = llvm.call @__nv_abs(%arg0) : (i32) -> i32
      llvm.return %0 : i32
    }
    llvm.func @gpu_llabs(%arg0: i64) -> i64 attributes {llvm.emit_c_interface} {
      // CHECK: math.absi
      %0 = llvm.call @__nv_llabs(%arg0) : (i64) -> i64
      llvm.return %0 : i64
    }

  llvm.func local_unnamed_addr @__nv_fmax(f64, f64) -> f64
  llvm.func local_unnamed_addr @__nv_isnand(f64) -> i32
  llvm.func local_unnamed_addr @__nv_isfinited(f64) -> i32
  llvm.func local_unnamed_addr @__nv_fmin(f64, f64) -> f64
    llvm.func @gpu_fmax(%arg0: f64, %arg1 : f64) -> f64 {
      // CHECK: arith.maxnumf
      %0 = llvm.call @__nv_fmax(%arg0, %arg1) : (f64, f64) -> f64
      llvm.return %0 : f64
    }
    llvm.func @gpu_fmin(%arg0: f64, %arg1 : f64) -> f64 {
      // CHECK: arith.minnumf
      %0 = llvm.call @__nv_fmin(%arg0, %arg1) : (f64, f64) -> f64
      llvm.return %0 : f64
    }
    llvm.func @gpu_isnan(%arg0: f64) -> i32 {
      // CHECK: "llvm.intr.is.fpclass"(%arg0) <{bit = 3 : i32}> : (f64) -> i1
      %0 = llvm.call @__nv_isnand(%arg0) : (f64) -> i32
      llvm.return %0 : i32
    }
    llvm.func @gpu_isfinite(%arg0: f64) -> i32 {
      // CHECK: "llvm.intr.is.fpclass"(%arg0) <{bit = 504 : i32}> : (f64) -> i1
      %0 = llvm.call @__nv_isfinited(%arg0) : (f64) -> i32
      llvm.return %0 : i32
    }
    llvm.func @_mlir_ciface_gpu_abs(%arg0 : !llvm.ptr, %arg1: i32) attributes {llvm.emit_c_interface} {
      %0 = llvm.call @gpu_abs(%arg1) : (i32) -> i32
      llvm.store %0, %arg0 : i32, !llvm.ptr
      llvm.return
    }
  }
}

