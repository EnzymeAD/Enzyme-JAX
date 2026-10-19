// RUN: enzymexlamlir-opt %s --restore-preserve-nvvm --split-input-file | FileCheck %s

// A libdevice definition as PreserveNVVM(Begin) left it: promoted to external
// linkage (external is the llvm.func default spelling), noinline, and the
// prev_* records of what it was. Restored: internal again, alwaysinline
// again, records gone -- the enzyme_math/implements markers stay.

llvm.func @__nv_sin(%arg0: f64) -> f64 attributes {dso_local, no_inline, passthrough = [["enzyme_math", "sin"], ["implements", "llvm.sin.f64"], ["implements2", "sin"], "prev_always_inline", "prev_fixup", ["prev_linkage", "7"]]} {
  llvm.return %arg0 : f64
}

// CHECK: llvm.func internal @__nv_sin(%arg0: f64) -> f64 attributes {always_inline, dso_local, passthrough = {{\[\[}}"enzyme_math", "sin"], {{\[}}"implements", "llvm.sin.f64"], {{\[}}"implements2", "sin"]]}
// CHECK-NOT: prev_fixup
// CHECK-NOT: no_inline

// -----

// A function that was noinline before the preservation stays noinline; only
// the bookkeeping goes away.

llvm.func @was_noinline(%arg0: f64) -> f64 attributes {no_inline, passthrough = ["prev_no_inline", "prev_fixup", ["prev_linkage", "0"]]} {
  llvm.return %arg0 : f64
}

// CHECK: llvm.func @was_noinline(%arg0: f64) -> f64 attributes {no_inline}
// CHECK-NOT: prev_fixup

// -----

// No prev_fixup: not PreserveNVVM's doing, left exactly alone.

llvm.func @untouched(%arg0: f64) -> f64 attributes {no_inline, passthrough = ["something_else"]} {
  llvm.return %arg0 : f64
}

// CHECK: llvm.func @untouched(%arg0: f64) -> f64 attributes {no_inline, passthrough = ["something_else"]}
