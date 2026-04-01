// RUN: enzymexlamlir-opt %s -func-attr-to-tessera-attr | FileCheck %s

module {
  llvm.func @inverse() -> i32 attributes {tessera_op = "eigen.inv"} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}
// CHECK: llvm.func @inverse
// CHECK-SAME: tessera.convert = #tessera<convert eigen.inv>
// CHECK-NOT: tessera_op
