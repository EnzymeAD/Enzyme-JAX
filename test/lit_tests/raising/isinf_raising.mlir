// RUN: enzymexlamlir-opt --libdevice-funcs-raise %s | FileCheck %s

gpu.module @test_module_isinf {
  llvm.func @__nv_isinfd(f64) -> i32
  llvm.func @__nv_isinff(f32) -> i32

  // CHECK-LABEL: @gpu_isinfd
  llvm.func @gpu_isinfd(%arg0: f64) -> i32 {
    // CHECK: %[[INF:.+]] = math.isinf %arg0 : f64
    // CHECK-NOT: math.isfinite
    // CHECK: {{arith.extui|llvm.zext}} %[[INF]] : i1 to i32
    %0 = llvm.call @__nv_isinfd(%arg0) : (f64) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: @gpu_isinff
  llvm.func @gpu_isinff(%arg0: f32) -> i32 {
    // CHECK: %[[INF:.+]] = math.isinf %arg0 : f32
    // CHECK-NOT: math.isfinite
    // CHECK: {{arith.extui|llvm.zext}} %[[INF]] : i1 to i32
    %0 = llvm.call @__nv_isinff(%arg0) : (f32) -> i32
    llvm.return %0 : i32
  }
}
