// RUN: enzymexlamlir-opt %s | FileCheck %s

tessera.define @foo() {
  tessera.return
}

// CHECK-LABEL: tessera.define @foo
// CHECK: tessera.return

// -----

tessera.define @bar() -> i32 {
  %c42_i32 = arith.constant 42 : i32
  tessera.return %c42_i32 : i32
}

// CHECK-LABEL: tessera.define @bar
// CHECK: tessera.return %{{.*}} : i32

// -----

tessera.define @caller() {
  tessera.call @foo() : () -> ()
  tessera.return
}

// CHECK-LABEL: tessera.define @caller
// CHECK: tessera.call @foo() : () -> ()
// CHECK: tessera.return

// -----

tessera.define @with_args(%arg0: i32, %arg1: f32) -> i32 {
  %0 = tessera.call @bar() : () -> i32
  tessera.return %0 : i32
}

// CHECK-LABEL: tessera.define @with_args
// CHECK: %[[V0:.*]] = tessera.call @bar() : () -> i32
// CHECK: tessera.return %[[V0]] : i32