// RUN: enzymexlamlir-opt %s -func-to-tessera | FileCheck %s

// CHECK-LABEL: tessera.define @simple_func
func.func @simple_func() attributes {tessera.convert} {
  // CHECK: tessera.return
  func.return
}

// -----

// CHECK-LABEL: tessera.define @func_with_args
func.func @func_with_args(%arg0: i32, %arg1: f32) -> i32 attributes {tessera.convert} {
  // CHECK: tessera.return %arg0 : i32
  func.return %arg0 : i32
}

// -----

// CHECK-LABEL: tessera.define @helper
func.func @helper() attributes {tessera.convert} {
  func.return
}

// CHECK-LABEL: tessera.define @func_with_call
func.func @func_with_call() attributes {tessera.convert} {
  // CHECK: tessera.call @helper() : () -> ()
  func.call @helper() : () -> ()
  // CHECK: tessera.return
  func.return
}

// -----

// This should NOT convert (no attribute)
// CHECK-LABEL: func.func @regular_function
func.func @regular_function() {
  func.return
}