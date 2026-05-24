// RUN: enzymexlamlir-opt %s | FileCheck %s

tessera.define @foo() attributes {tessera.convert = #tessera<convert "foo" byref = [] sizes = [] pure = true>} {
  tessera.return
}

// CHECK-LABEL: tessera.define @foo
// CHECK: tessera.return

// -----

tessera.define @bar() -> i32 attributes {tessera.convert = #tessera<convert "bar" byref = [] sizes = [] pure = true>} {
  %c42_i32 = arith.constant 42 : i32
  tessera.return %c42_i32 : i32
}

// CHECK-LABEL: tessera.define @bar
// CHECK: tessera.return %{{.*}} : i32

// -----

tessera.define @caller() attributes {tessera.convert = #tessera<convert "caller" byref = [] sizes = [] pure = true>} {
  tessera.call @foo() : () -> ()
  tessera.return
}

// CHECK-LABEL: tessera.define @caller
// CHECK: tessera.call @foo() : () -> ()
// CHECK: tessera.return

// -----

tessera.define @with_args(%arg0: i32, %arg1: f32) -> i32 attributes {tessera.convert = #tessera<convert "with_args" byref = [false, false] sizes = [4, 4] pure = true>} {
  %0 = tessera.call @bar() : () -> i32
  tessera.return %0 : i32
}

// CHECK-LABEL: tessera.define @with_args
// CHECK: %[[V0:.*]] = tessera.call @bar() : () -> i32
// CHECK: tessera.return %[[V0]] : i32
