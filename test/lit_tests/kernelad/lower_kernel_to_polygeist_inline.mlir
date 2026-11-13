// RUN: enzymexlamlir-opt %s --split-input-file --verify-diagnostics --lower-kernel-to-polygeist="gpu-wrapper=false" | FileCheck %s

// The functions the kernel calls are inlined, so that the thread position,
// barriers and shared memory they use are rewritten with the kernel's.

module {
  llvm.mlir.global internal @tile() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<4 x f32>

  llvm.func internal @index() -> i64 {
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %1 = llvm.zext %0 : i32 to i64
    llvm.return %1 : i64
  }

  // Several blocks, and a call of its own.
  llvm.func internal @exchange(%v: f32, %flag: i1) -> f32 {
    %i = llvm.call @index() : () -> i64
    %t = llvm.mlir.addressof @tile : !llvm.ptr<3>
    %p = llvm.getelementptr inbounds %t[%i] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %v, %p : f32, !llvm.ptr<3>
    nvvm.barrier
    llvm.cond_br %flag, ^first, ^own
  ^first:
    %f = llvm.load %t : !llvm.ptr<3> -> f32
    llvm.return %f : f32
  ^own:
    llvm.return %v : f32
  }

  llvm.func private @kernel(%arg0: !llvm.ptr<1>) {
    %i = llvm.call @index() : () -> i64
    %0 = llvm.getelementptr inbounds %arg0[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %1 = llvm.load %0 : !llvm.ptr<1> -> f32
    %true = llvm.mlir.constant(true) : i1
    %2 = llvm.call @exchange(%1, %true) : (f32, i1) -> f32
    llvm.store %2, %0 : f32, !llvm.ptr<1>
    llvm.return
  }

  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = enzymexla.kernel_call @kernel blocks in(%c2, %c1, %c1) threads in(%c4, %c1, %c1) shmem = %c0 (%arg0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// CHECK-LABEL: llvm.func private @kernel_host(
// CHECK:         affine.parallel
// CHECK-NEXT:      %[[TILE:.+]] = memref.alloca() alignment = 4 : memref<4xf32>
// CHECK-NEXT:      affine.parallel (%[[TX:.+]], %[[TY:.+]], %[[TZ:.+]]) =
// CHECK-NOT:         llvm.call
// CHECK-NOT:         nvvm.
// CHECK:             arith.index_castui %[[TX]] : index to i32
// CHECK:             "enzymexla.memref2pointer"(%[[TILE]])
// CHECK:             "enzymexla.barrier"(%[[TX]], %[[TY]], %[[TZ]])
// CHECK-NOT:         llvm.call
// CHECK-NOT:         nvvm.
// CHECK:         llvm.return

// -----

// A function that cannot be inlined may not use the launch context.

module {
  llvm.func internal @index() -> i64 attributes {no_inline} {
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %1 = llvm.zext %0 : i32 to i64
    llvm.return %1 : i64
  }

  llvm.func private @kernel(%arg0: !llvm.ptr<1>) {
    %i = llvm.call @index() : () -> i64
    %0 = llvm.getelementptr inbounds %arg0[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %1 = llvm.load %0 : !llvm.ptr<1> -> f32
    llvm.store %1, %0 : f32, !llvm.ptr<1>
    llvm.return
  }

  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    // expected-error @+1 {{cannot inline index, which uses the position of the thread, the launch dimensions, barriers or shared memory}}
    %0 = enzymexla.kernel_call @kernel blocks in(%c1, %c1, %c1) threads in(%c4, %c1, %c1) shmem = %c0 (%arg0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// -----

// Recursion cannot be inlined.

module {
  llvm.func internal @loop(%n: i64) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.icmp "eq" %n, %0 : i64
    llvm.cond_br %1, ^done, ^again
  ^again:
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.sub %n, %2 : i64
    llvm.call @loop(%3) : (i64) -> ()
    llvm.br ^done
  ^done:
    llvm.return
  }

  llvm.func private @kernel(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(3 : i64) : i64
    llvm.call @loop(%0) : (i64) -> ()
    llvm.return
  }

  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    // expected-error @+1 {{the kernel calls loop recursively, which cannot be inlined}}
    %0 = enzymexla.kernel_call @kernel blocks in(%c1, %c1, %c1) threads in(%c4, %c1, %c1) shmem = %c0 (%arg0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
