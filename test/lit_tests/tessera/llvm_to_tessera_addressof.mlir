// RUN: enzymexlamlir-opt %s -llvm-to-tessera | FileCheck %s
// RUN: enzymexlamlir-opt %s -llvm-to-tessera -tessera-to-llvm | FileCheck %s --check-prefix=ROUNDTRIP

// llvm.mlir.addressof may only name an LLVM symbol, so taking the address of a
// tessera op -- directly, or by putting it in llvm.compiler.used, which is how
// `__attribute__((used))` shows up -- used to make the module fail to verify
// once the function became a tessera.define.
//
// While the function is a tessera.define, taken addresses refer to a stub
// declared in its place instead; calls follow it to its tessera name as usual.

module {
  llvm.mlir.global appending @llvm.compiler.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x ptr> {
    %0 = llvm.mlir.addressof @inverse : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr>
    llvm.return %2 : !llvm.array<1 x ptr>
  }

  llvm.func @inverse(%arg0: f32) -> f32 attributes {pure_tessera_op = "eigen.inv(x)"} {
    llvm.return %arg0 : f32
  }

  llvm.func @main(%x: f32) -> f32 {
    %fp = llvm.mlir.addressof @inverse : !llvm.ptr
    %0 = llvm.call @inverse(%x) : (f32) -> f32
    %1 = llvm.call %fp(%0) : !llvm.ptr, (f32) -> f32
    llvm.return %1 : f32
  }
}

// CHECK: llvm.mlir.global appending @llvm.compiler.used()
// CHECK-NEXT: llvm.mlir.addressof @inverse.tessera_address : !llvm.ptr
// CHECK: llvm.func @inverse.tessera_address(f32) -> f32 attributes {tessera.address_stub_for = "inverse"}
// CHECK: tessera.define @eigen.inv(
// CHECK-LABEL: llvm.func @main
// CHECK-NEXT: %[[FP:.*]] = llvm.mlir.addressof @inverse.tessera_address : !llvm.ptr
// CHECK-NEXT: %[[R:.*]] = tessera.call @eigen.inv(%arg0)
// CHECK-NEXT: llvm.call %[[FP]](%[[R]])

// Leaving the tessera dialect points them back at the function, under its
// original name, and drops the stub.
// ROUNDTRIP-NOT: tessera_address
// ROUNDTRIP: llvm.mlir.addressof @inverse : !llvm.ptr
// ROUNDTRIP: llvm.func @inverse(
// ROUNDTRIP-LABEL: llvm.func @main
// ROUNDTRIP-NEXT: %[[FP:.*]] = llvm.mlir.addressof @inverse : !llvm.ptr
// ROUNDTRIP-NEXT: %[[R:.*]] = llvm.call @inverse(%arg0)
// ROUNDTRIP-NEXT: llvm.call %[[FP]](%[[R]])
// ROUNDTRIP-NOT: tessera_address
