// RUN: enzymexlamlir-opt %s -func-attr-to-tessera-attr | FileCheck %s

module {
  llvm.func @inverse() -> i32 attributes {tessera_op = "eigen.inv(x):4"} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
  llvm.func @pure_inverse() -> i32 attributes {pure_tessera_op = "eigen.inv(x):4"} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %0 : i32
  }
}
// CHECK: llvm.func @inverse
// CHECK-SAME: tessera.convert = #tessera<convert eigen.inv byref = [false] sizes = [4] pure = false>
// CHECK-NOT: tessera_op
// CHECK: llvm.func @pure_inverse
// CHECK-SAME: tessera.convert = #tessera<convert eigen.inv byref = [false] sizes = [4] pure = true>
// CHECK-NOT: pure_tessera_op
