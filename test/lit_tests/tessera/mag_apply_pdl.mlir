// RUN: enzymexlamlir-opt %s -tessera-apply-pdl | FileCheck %s

module {
  tessera.define @eigen.mag(%arg0 : f32, %arg1 : f32, %arg2 : f32) -> f32 {
      tessera.return %arg0 : f32
  }
  
  tessera.define @arith.negf(%arg0 : f32) -> f32 {
      tessera.return %arg0 : f32
  }
  
  // CHECK-LABEL: llvm.func @main
  llvm.func @main(%arg0 : f32, %arg1 : f32, %arg2 : f32) -> f32 {
      // CHECK: tessera.call @eigen.mag(%arg0, %arg1, %arg2)
      %0 = tessera.call @arith.negf(%arg0) : (f32) -> f32
      %1 = tessera.call @eigen.mag(%0, %arg1, %arg2) : (f32, f32, f32) -> f32
      llvm.return %1 : f32
  }

  module @patterns {
    pdl.pattern : benefit(1) {
      %0 = operand
      %1 = attribute = @arith.negf
      %2 = type
      %3 = operation "tessera.call"(%0 : !pdl.value)  {"callee" = %1} -> (%2 : !pdl.type)
      %4 = result 0 of %3
      %5 = operand
      %6 = operand
      %7 = attribute = @eigen.mag
      %8 = type
      %9 = operation "tessera.call"(%4, %5, %6 : !pdl.value, !pdl.value, !pdl.value)  {"callee" = %7} -> (%8 : !pdl.type)
      %10 = result 0 of %9
      rewrite %9 {
        %11 = attribute = @eigen.mag
        %12 = type
        %13 = operation "tessera.call"(%0, %5, %6 : !pdl.value, !pdl.value, !pdl.value)  {"callee" = %11} -> (%12 : !pdl.type)
        %14 = result 0 of %13
        replace %9 with %13
      }
    }
  }
}
