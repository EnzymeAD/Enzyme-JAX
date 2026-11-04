// RUN: enzymexlamlir-opt %s -tessera-to-func -verify-diagnostics

// An argument lifted out of memory can only be put back by tessera-to-llvm.
// expected-error@+1 {{could not convert to func.func}}
tessera.define @tessera.lifted(%arg0: !llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f32}], pure = true, tessera.original_name = "lifted"} {
  tessera.return
}
