// RUN: enzymexlamlir-opt %s -func-to-tessera | FileCheck %s

// CHECK-LABEL: tessera.define @simple_func
func.func @simple_func() {
  // CHECK: tessera.return
  func.return
}

// -----

// CHECK-LABEL: tessera.define @func_with_args
func.func @func_with_args(%arg0: i32, %arg1: f32) -> i32 {
  // CHECK: tessera.return %arg0 : i32
  func.return %arg0 : i32
}

// -----

// CHECK-LABEL: tessera.define @helper
func.func @helper() {
  func.return
}

// CHECK-LABEL: tessera.define @func_with_call
func.func @func_with_call() {
  // CHECK: tessera.call @helper() : () -> ()
  func.call @helper() : () -> ()
  // CHECK: tessera.return
  func.return
}