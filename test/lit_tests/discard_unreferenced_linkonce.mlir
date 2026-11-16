// RUN: enzymexlamlir-opt %s --discard-unreferenced-linkonce | FileCheck %s

// Unreferenced linkonce definitions go; a comdat selector is not a
// reference. A weak definition stays: an explicit template instantiation is
// weak_odr and the only definition its `extern template` users have. So do
// external definitions and declarations.

llvm.comdat @__llvm_global_comdat {
  llvm.comdat_selector @dead_odr any
  llvm.comdat_selector @live_odr any
}

llvm.func linkonce_odr @dead_odr() comdat(@__llvm_global_comdat::@dead_odr) {
  llvm.return
}

llvm.func weak_odr @instantiation() {
  llvm.return
}

llvm.func linkonce_odr @live_odr() comdat(@__llvm_global_comdat::@live_odr) {
  llvm.return
}

llvm.func @decl()

llvm.func @external_unreferenced() {
  llvm.return
}

llvm.func @main() {
  llvm.call @live_odr() : () -> ()
  llvm.call @decl() : () -> ()
  llvm.return
}

// CHECK:    llvm.comdat @__llvm_global_comdat {
// CHECK-NEXT:    llvm.comdat_selector @dead_odr any
// CHECK-NEXT:    llvm.comdat_selector @live_odr any
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func weak_odr @instantiation() {
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func linkonce_odr @live_odr() comdat(@__llvm_global_comdat::@live_odr) {
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @decl()
// CHECK-NEXT:  llvm.func @external_unreferenced() {
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
// CHECK-NEXT:  llvm.func @main() {
// CHECK-NEXT:    llvm.call @live_odr() : () -> ()
// CHECK-NEXT:    llvm.call @decl() : () -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
