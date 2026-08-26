// RUN: enzymexlamlir-opt %s --libdevice-funcs-raise | FileCheck %s

// The enzyme entry points are compiler markers, not functions that unwind:
// raised, they become an op with no unwind edge to keep. An invoke of one
// becomes the call and the branch it meant, and the block's other exception
// handling stays as it was.

module {
  llvm.func @__gxx_personality_v0(...) -> i32
  llvm.func @__enzyme_dummy_marker(f64) -> f64
  llvm.func @maythrow()
  llvm.func @sink(f64)

  llvm.func @invoked_marker(%x: f64) attributes {personality = @__gxx_personality_v0} {
    %r = llvm.invoke @__enzyme_dummy_marker(%x) to ^ok unwind ^lp : (f64) -> f64
  ^ok:
    llvm.call @sink(%r) : (f64) -> ()
    llvm.invoke @maythrow() to ^done unwind ^lp : () -> ()
  ^done:
    llvm.return
  ^lp:
    %1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.resume %1 : !llvm.struct<(ptr, i32)>
  }
}

// CHECK-LABEL: llvm.func @invoked_marker
// CHECK:         %[[R:.+]] = llvm.call @__enzyme_dummy_marker(%{{.+}}) : (f64) -> f64
// CHECK-NEXT:    llvm.br ^[[OK:.+]]
// CHECK:       ^[[OK]]:
// CHECK-NEXT:    llvm.call @sink(%[[R]])
// A function that really unwinds keeps doing so; only the marker's edge went.
// CHECK-NEXT:    llvm.invoke @maythrow
// CHECK:         llvm.landingpad
// CHECK:         llvm.resume
