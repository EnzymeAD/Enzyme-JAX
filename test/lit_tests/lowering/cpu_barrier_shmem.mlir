// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-kernel{backend=cpu},canonicalize)" --split-input-file | FileCheck %s --implicit-check-not=ptx_kernelcc --implicit-check-not=amdgpu_kernelcc --implicit-check-not="addr_space = 3"

// External ptx_kernelcc kernel with nvvm.barrier, gpu.barrier, rocdl.s.barrier,
// and addr_space == 3 static shared memory lowers to per-block stack alloca
// and enzymexla.barrier over thread IVs, and erases the dead device symbols.

module {
  llvm.mlir.global internal @shmem() {addr_space = 3 : i32, alignment = 8 : i64} : !llvm.array<2 x f32>
  llvm.func ptx_kernelcc @barrier_kern(%ptr: !llvm.ptr<1>) {
    %tid = nvvm.read.ptx.sreg.tid.x : i32
    %tid64 = llvm.sext %tid : i32 to i64
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %sh = llvm.mlir.addressof @shmem : !llvm.ptr<3>
    %gep_sh = llvm.getelementptr %sh[%c0, %tid64] : (!llvm.ptr<3>, i64, i64) -> !llvm.ptr<3>, !llvm.array<2 x f32>
    %gep_in = llvm.getelementptr %ptr[%tid64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %v = llvm.load %gep_in : !llvm.ptr<1> -> f32
    llvm.store %v, %gep_sh : f32, !llvm.ptr<3>
    %z = llvm.mlir.constant(0 : i32) : i32
    nvvm.barrier id = %z
    %u = llvm.load %gep_sh : !llvm.ptr<3> -> f32
    llvm.store %u, %gep_sh : f32, !llvm.ptr<3>
    gpu.barrier
    %q = llvm.load %gep_sh : !llvm.ptr<3> -> f32
    llvm.store %q, %gep_sh : f32, !llvm.ptr<3>
    rocdl.s.barrier
    %w = llvm.load %gep_sh : !llvm.ptr<3> -> f32
    llvm.store %w, %gep_in : f32, !llvm.ptr<1>
    llvm.return
  }
  func.func @main(%x: tensor<6xf32>) -> tensor<6xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %c3 = stablehlo.constant dense<3> : tensor<i64>
    %res = enzymexla.kernel_call @barrier_kern
      blocks in(%c3, %c1, %c1) threads in(%c2, %c1, %c1)
      shmem = %c0 (%x)
      <{output_operand_aliases = [#stablehlo.output_operand_alias<
        output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}>
      : (tensor<6xf32>) -> tensor<6xf32>
    return %res : tensor<6xf32>
  }
}

// CHECK-LABEL: func.func private @barrier_kern$par0(
// CHECK: llvm.alloca {{.*}} x !llvm.array<3 x array<2 x f32>>
// CHECK: affine.parallel (%[[BX:.*]], %[[BY:.*]], %[[BZ:.*]], %[[TX:.*]], %[[TY:.*]], %[[TZ:.*]]) = (0, 0, 0, 0, 0, 0) to (3, 1, 1, 2, 1, 1)
// CHECK: "enzymexla.barrier"(%[[TX]], %[[TY]], %[[TZ]])
// CHECK: "enzymexla.barrier"(%[[TX]], %[[TY]], %[[TZ]])
// CHECK: "enzymexla.barrier"(%[[TX]], %[[TY]], %[[TZ]])
// CHECK-LABEL: func.func @main(
// CHECK: enzymexla.jit_call @barrier_kern$par0

// -----

// External zero-length addr_space == 3 dynamic shared memory lowers using the
// kernel_call shmem byte size aligned per block, and rocdl.barrier lowers to
// enzymexla.barrier.

module {
  llvm.mlir.global external @dyn_shmem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x f32>
  llvm.func amdgpu_kernelcc @dyn_barrier_kern(%ptr: !llvm.ptr<1>) {
    %tid = gpu.thread_id x
    %tid64 = arith.index_castui %tid : index to i64
    %sh = llvm.mlir.addressof @dyn_shmem : !llvm.ptr<3>
    %gep_sh = llvm.getelementptr %sh[0, %tid64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, !llvm.array<0 x f32>
    %gep_in = llvm.getelementptr %ptr[%tid64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %v = llvm.load %gep_in : !llvm.ptr<1> -> f32
    llvm.store %v, %gep_sh : f32, !llvm.ptr<3>
    rocdl.barrier
    %w = llvm.load %gep_sh : !llvm.ptr<3> -> f32
    llvm.store %w, %gep_in : f32, !llvm.ptr<1>
    llvm.return
  }
  func.func @main_dyn(%x: tensor<8xf32>) -> tensor<8xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %c16 = stablehlo.constant dense<16> : tensor<i64>
    %res = enzymexla.kernel_call @dyn_barrier_kern
      blocks in(%c2, %c1, %c1) threads in(%c4, %c1, %c1)
      shmem = %c16 (%x)
      <{output_operand_aliases = [#stablehlo.output_operand_alias<
        output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}>
      : (tensor<8xf32>) -> tensor<8xf32>
    return %res : tensor<8xf32>
  }
}

// CHECK-LABEL: func.func private @dyn_barrier_kern$par{{[0-9]+}}(
// CHECK: llvm.alloca {{.*}} x !llvm.array<2 x array<16 x i8>>
// CHECK: affine.parallel (%[[DBX:.*]], %[[DBY:.*]], %[[DBZ:.*]], %[[DTX:.*]], %[[DTY:.*]], %[[DTZ:.*]]) = (0, 0, 0, 0, 0, 0) to (2, 1, 1, 4, 1, 1)
// CHECK: "enzymexla.barrier"(%[[DTX]], %[[DTY]], %[[DTZ]])
// CHECK-LABEL: func.func @main_dyn(
// CHECK: enzymexla.jit_call @dyn_barrier_kern$par{{[0-9]+}}
