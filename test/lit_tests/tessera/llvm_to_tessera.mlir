// RUN: enzymexlamlir-opt %s -llvm-to-tessera | FileCheck %s

llvm.func @simple_func() attributes {tessera.convert = #tessera<convert "tessera_simple_func">} {
  llvm.return
}

// CHECK-LABEL: tessera.define @tessera_simple_func
// CHECK-NEXT: tessera.return

// -----


llvm.func @func_with_args(%arg0: i32, %arg1: f32) -> i32 attributes {tessera.convert = #tessera<convert "tessera_func_with_args">} {
  llvm.return %arg0 : i32
}

// CHECK-LABEL: tessera.define @tessera_func_with_args
// CHECK-NEXT: tessera.return %arg0 : i32

// -----

llvm.func @helper() attributes {tessera.convert = #tessera<convert "tessera_helper">} {
  llvm.return
}

llvm.func @func_with_call() attributes {tessera.convert = #tessera<convert "tessera_func_with_call">} {
  llvm.call @helper() : () -> ()
  llvm.return
}

// CHECK-LABEL: tessera.define @tessera_helper
// CHECK-NEXT: tessera.return

// CHECK-LABEL: tessera.define @tessera_func_with_call
// CHECK-NEXT: tessera.call @tessera_helper()
// CHECK-NEXT: tessera.return

// -----

llvm.func @regular_function() {
  llvm.return
}

// CHECK-LABEL: llvm.func @regular_function
// CHECK-NEXT: llvm.return
