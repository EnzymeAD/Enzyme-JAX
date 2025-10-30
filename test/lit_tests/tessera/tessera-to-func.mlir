// RUN: enzymexlamlir-opt %s -tessera-to-func | FileCheck %s

// CHECK-LABEL: func.func @simple_func
tessera.define @simple_func() {
  // CHECK: func.return
  tessera.return
}

// -----

// CHECK-LABEL: func.func @func_with_args
tessera.define @func_with_args(%arg0: i32, %arg1: f32) -> i32 {
  // CHECK: func.return %arg0 : i32
  tessera.return %arg0 : i32
}

// -----

// CHECK-LABEL: func.func @helper
tessera.define @helper() {
  tessera.return
}

// CHECK-LABEL: func.func @func_with_call
tessera.define @func_with_call() {
  // CHECK: func.call @helper() : () -> ()
  tessera.call @helper() : () -> ()
  // CHECK: func.return
  tessera.return
}