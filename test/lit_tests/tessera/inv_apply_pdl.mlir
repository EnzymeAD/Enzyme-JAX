// RUN: enzymexlamlir-opt %s -tessera-apply-pdl | FileCheck %s

module {
  tessera.define @eigen.inv(%arg0 : f32) -> f32 {
    tessera.return %arg0 : f32
  }

  // CHECK-LABEL: llvm.func @main
  llvm.func @main(%x : f32) -> f32 {
    // CHECK: llvm.return %arg0
    %0 = tessera.call @eigen.inv(%x) : (f32) -> f32
    %1 = tessera.call @eigen.inv(%0) : (f32) -> f32
    llvm.return %1 : f32
  }

  module @patterns {
    pdl.pattern : benefit(1) {
      %0 = operand
      %1 = attribute = @eigen.inv
      %2 = type
      %3 = operation "tessera.call"(%0 : !pdl.value)  {"callee" = %1} -> (%2 : !pdl.type)
      %4 = result 0 of %3
      %5 = attribute = @eigen.inv
      %6 = type
      %7 = operation "tessera.call"(%4 : !pdl.value)  {"callee" = %5} -> (%6 : !pdl.type)
      %8 = result 0 of %7
      rewrite %7 {
        replace %7 with(%0 : !pdl.value)
      }
    }
  }
}
