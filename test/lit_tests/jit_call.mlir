// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-jit{jit=false backend=cpu},enzyme-hlo-opt)" | FileCheck %s

module {
  llvm.func internal unnamed_addr fastcc @throw() attributes {dso_local, no_inline, sym_visibility = "private"} {
    llvm.unreachable
  }

  // CHECK-LABEL: func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: stablehlo.custom_call @enzymexla_compile_cpu() {api_version = 3 : i32, backend_config = "\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", has_side_effect = true} : () -> ()
    enzymexla.jit_call @throw () {
        has_side_effect = true,
        backend_config = {bar = 42 : i32}
      } : () -> ()
    return %arg0 : tensor<4xf32>
  }
}
