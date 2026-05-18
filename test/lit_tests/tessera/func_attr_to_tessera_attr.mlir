// RUN: enzymexlamlir-opt %s -func-attr-to-tessera-attr | FileCheck %s

llvm.func @foo() -> i32 attributes {tessera_op = "tessera.foo(x):4"} {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}

// CHECK: llvm.func
// CHECK-SAME: tessera.convert = #tessera<convert tessera.foo byref = [false] sizes = [4] pure = false>
// CHECK-NOT: tessera_op
// CHECK-NEXT: %0 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.return %0 : i32

// -----


llvm.func @foo_pure() -> i32 attributes {pure_tessera_op = "tessera.foo(x):4"} {  
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}

// CHECK: llvm.func @foo_pure
// CHECK-SAME: tessera.convert = #tessera<convert tessera.foo byref = [false] sizes = [4] pure = true>
// CHECK-NOT: pure_tessera_op
// CHECK-NEXT: %0 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.return %0 : i32
  
// -----


llvm.func @bar() -> i32 attributes {pure_tessera_op = "tessera.bar(x:byref, y:byval):64,4"} {  
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}

// CHECK: llvm.func @bar
// CHECK-SAME: tessera.convert = #tessera<convert tessera.bar byref = [true, false] sizes = [64, 4] pure = true>
// CHECK-NOT: pure_tessera_op
// CHECK-NEXT: %0 = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.return %0 : i32
