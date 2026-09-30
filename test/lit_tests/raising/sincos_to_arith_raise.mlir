// RUN: enzymexlamlir-opt --libdevice-funcs-raise %s | FileCheck %s

gpu.module @test_module_sincos {
  llvm.func @__nv_sincosf(f32, !llvm.ptr, !llvm.ptr)
  llvm.func @__nv_sincos(f64, !llvm.ptr, !llvm.ptr)
  llvm.func @__nv_fast_sincosf(f32, !llvm.ptr, !llvm.ptr)
  llvm.func @sincos(f64, !llvm.ptr, !llvm.ptr)

  // CHECK-LABEL: @gpu_sincosf
  llvm.func @gpu_sincosf(%arg0: f32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    // CHECK: %[[S:[a-z0-9_]+]], %[[C:[a-z0-9_]+]] = math.sincos %arg0 : f32
    // CHECK: llvm.store %[[S]], %arg1 : f32, !llvm.ptr
    // CHECK: llvm.store %[[C]], %arg2 : f32, !llvm.ptr
    // CHECK-NOT: llvm.call
    llvm.call @__nv_sincosf(%arg0, %arg1, %arg2) : (f32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  // CHECK-LABEL: @gpu_sincosd
  llvm.func @gpu_sincosd(%arg0: f64, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    // CHECK: %[[S:[a-z0-9_]+]], %[[C:[a-z0-9_]+]] = math.sincos %arg0 : f64
    // CHECK: llvm.store %[[S]], %arg1 : f64, !llvm.ptr
    // CHECK: llvm.store %[[C]], %arg2 : f64, !llvm.ptr
    // CHECK-NOT: llvm.call
    llvm.call @__nv_sincos(%arg0, %arg1, %arg2) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  // CHECK-LABEL: @gpu_fast_sincosf
  llvm.func @gpu_fast_sincosf(%arg0: f32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    // CHECK: %[[S:[a-z0-9_]+]], %[[C:[a-z0-9_]+]] = math.sincos %arg0 fastmath<afn> : f32
    // CHECK: llvm.store %[[S]], %arg1 : f32, !llvm.ptr
    // CHECK: llvm.store %[[C]], %arg2 : f32, !llvm.ptr
    // CHECK-NOT: llvm.call
    llvm.call @__nv_fast_sincosf(%arg0, %arg1, %arg2) : (f32, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  // CHECK-LABEL: @libm_sincos
  llvm.func @libm_sincos(%arg0: f64, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    // CHECK: %[[S:[a-z0-9_]+]], %[[C:[a-z0-9_]+]] = math.sincos %arg0 : f64
    // CHECK: llvm.store %[[S]], %arg1 : f64, !llvm.ptr
    // CHECK: llvm.store %[[C]], %arg2 : f64, !llvm.ptr
    // CHECK-NOT: llvm.call
    llvm.call @sincos(%arg0, %arg1, %arg2) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }

  // double s, c; sincos(x, &s, &c); return s + c;
  // CHECK-LABEL: @gpu_sincos_alloca
  llvm.func @gpu_sincos_alloca(%arg0: f64) -> f64 {
    // CHECK: %[[SP:[0-9]+]] = llvm.alloca
    // CHECK: %[[CP:[0-9]+]] = llvm.alloca
    // CHECK: %[[S:[a-z0-9_]+]], %[[C:[a-z0-9_]+]] = math.sincos %arg0 : f64
    // CHECK: llvm.store %[[S]], %[[SP]] : f64, !llvm.ptr
    // CHECK: llvm.store %[[C]], %[[CP]] : f64, !llvm.ptr
    // CHECK: arith.addf
    // CHECK-NOT: llvm.call
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %0 = llvm.alloca %c1 x f64 : (i32) -> !llvm.ptr
    %1 = llvm.alloca %c1 x f64 : (i32) -> !llvm.ptr
    llvm.call @__nv_sincos(%arg0, %0, %1) : (f64, !llvm.ptr, !llvm.ptr) -> ()
    %2 = llvm.load %0 : !llvm.ptr -> f64
    %3 = llvm.load %1 : !llvm.ptr -> f64
    %4 = llvm.fadd %2, %3 : f64
    llvm.return %4 : f64
  }
}
