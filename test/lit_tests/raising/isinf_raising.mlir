// RUN: enzymexlamlir-opt --libdevice-funcs-raise %s | FileCheck %s

gpu.module @test_module_isinf {
  llvm.func @__nv_isinfd(f64) -> i32
  llvm.func @__nv_isinf(f64) -> i32
  llvm.func @__nv_isinff(f32) -> i32

  // CHECK-LABEL: llvm.func @gpu_isinfd(
  // CHECK-SAME:      %arg0: f64) -> i32 {
  // CHECK-NEXT:    %0 = math.isinf %arg0 : f64
  // CHECK-NEXT:    %1 = arith.extui %0 : i1 to i32
  // CHECK-NEXT:    llvm.return %1 : i32
  // CHECK-NEXT:  }
  llvm.func @gpu_isinfd(%arg0: f64) -> i32 {
    %0 = llvm.call @__nv_isinfd(%arg0) : (f64) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @gpu_isinf(
  // CHECK-SAME:      %arg0: f64) -> i32 {
  // CHECK-NEXT:    %0 = math.isinf %arg0 : f64
  // CHECK-NEXT:    %1 = arith.extui %0 : i1 to i32
  // CHECK-NEXT:    llvm.return %1 : i32
  // CHECK-NEXT:  }
  llvm.func @gpu_isinf(%arg0: f64) -> i32 {
    %0 = llvm.call @__nv_isinf(%arg0) : (f64) -> i32
    llvm.return %0 : i32
  }

  // CHECK-LABEL: llvm.func @gpu_isinff(
  // CHECK-SAME:      %arg0: f32) -> i32 {
  // CHECK-NEXT:    %0 = math.isinf %arg0 : f32
  // CHECK-NEXT:    %1 = arith.extui %0 : i1 to i32
  // CHECK-NEXT:    llvm.return %1 : i32
  // CHECK-NEXT:  }
  llvm.func @gpu_isinff(%arg0: f32) -> i32 {
    %0 = llvm.call @__nv_isinff(%arg0) : (f32) -> i32
    llvm.return %0 : i32
  }
}
