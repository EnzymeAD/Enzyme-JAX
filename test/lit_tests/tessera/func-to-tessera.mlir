// RUN: enzymexlamlir-opt %s -func-to-tessera | FileCheck %s

func.func @simple_func() attributes {tessera.convert} {
  func.return
}

// CHECK-LABEL: tessera.define @simple_func
// CHECK-NEXT: tessera.return

// -----


func.func @func_with_args(%arg0: i32, %arg1: f32) -> i32 attributes {tessera.convert} {
  func.return %arg0 : i32
}

// CHECK-LABEL: tessera.define @func_with_args
// CHECK-NEXT: tessera.return %arg0 : i32

// -----

func.func @helper() attributes {tessera.convert} {
  func.return
}

func.func @func_with_call() attributes {tessera.convert} {
  func.call @helper() : () -> ()
  func.return
}

// CHECK-LABEL: tessera.define @helper
// CHECK-NEXT: tessera.return

// CHECK-LABEL: tessera.define @func_with_call
// CHECK-NEXT: tessera.call @helper() : () -> ()
// CHECK-NEXT: tessera.return

// -----

func.func @regular_function() {
  func.return
}

// CHECK-LABEL: func.func @regular_function
// CHECK-NEXT: return
