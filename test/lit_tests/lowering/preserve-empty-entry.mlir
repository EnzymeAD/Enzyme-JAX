// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{preserve-xla-funcs=true},lower-jit{jit=false})" | FileCheck %s

// An entry point without arguments or results belongs to the XLA program and
// must not be converted to an llvm.func.

// CHECK-LABEL: func.func @main()
// CHECK-NEXT:    return
func.func @main() {
  return
}

// CHECK-LABEL: func.func @main2(
func.func @main2(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  return %arg0 : tensor<2xf32>
}

// CHECK-LABEL: llvm.func @helper()
func.func private @helper() {
  return
}
