// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-gpu})" | FileCheck %s

// cudaGetLastError arrives in invoke form when exception handling is
// preserved; it cannot throw, so it becomes its (zero) result and a branch
// to the normal destination — leaving no reference to the erased
// declaration for the LLVM translator to trip over.
module {
  llvm.func @cudaGetLastError() -> i32
  llvm.func @__gxx_personality_v0(...) -> i32
  llvm.func @use(%c: i1) -> i32 attributes {personality = @__gxx_personality_v0} {
    %z = llvm.mlir.constant(0 : i32) : i32
    llvm.cond_br %c, ^call, ^done(%z : i32)
  ^call:
    %e = llvm.invoke @cudaGetLastError() to ^fwd unwind ^lp : () -> i32
  ^fwd:
    llvm.br ^done(%e : i32)
  ^done(%r: i32):
    llvm.return %r : i32
  ^lp:
    %lp = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.return %one : i32
  }
}

// CHECK-LABEL: llvm.func @use(
// CHECK-NOT: cudaGetLastError
// CHECK: %[[Z:.+]] = llvm.mlir.zero : i32
// CHECK: llvm.br ^[[FWD:.+]]{{$}}
// CHECK: ^[[FWD]]:
// CHECK: llvm.br ^{{.+}}(%[[Z]] : i32)
