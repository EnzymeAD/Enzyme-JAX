// RUN: enzymexlamlir-opt %s -tessera-to-func | FileCheck %s

tessera.define @simple_func() {
  tessera.return
}

// CHECK-LABEL: func.func @simple_func
// CHECK: return

// -----

tessera.define @func_with_args(%arg0: i32, %arg1: f32) -> i32 {
  tessera.return %arg0 : i32
}

// CHECK-LABEL: func.func @func_with_args
// CHECK: return %arg0 : i32

// -----

tessera.define @helper() {
  tessera.return
}

tessera.define @func_with_call() {
  tessera.call @helper() : () -> ()
  tessera.return
}

// CHECK-LABEL: func.func @helper
// CHECK-NEXT: return

// CHECK-LABEL: func.func @func_with_call
// CHECK: call @helper() : () -> ()
// CHECK: return