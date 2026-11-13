// RUN: enzymexlamlir-opt %s --lower-kernel-to-polygeist="gpu-wrapper=false" | FileCheck %s

// Each block has its own copy of the static shared memory, allocated in the
// loop over blocks, and barriers synchronize the threads of a block.

module {
  llvm.mlir.global internal @tile() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<4 x f32>

  llvm.func private @kernel(%arg0: !llvm.ptr<1>) {
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %1 = llvm.zext %0 : i32 to i64
    %2 = llvm.getelementptr inbounds %arg0[%1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %3 = llvm.load %2 : !llvm.ptr<1> -> f32
    %4 = llvm.mlir.addressof @tile : !llvm.ptr<3>
    %5 = llvm.getelementptr inbounds %4[%1] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
    llvm.store %3, %5 : f32, !llvm.ptr<3>
    nvvm.barrier
    %6 = llvm.mlir.addressof @tile : !llvm.ptr<3>
    %7 = llvm.load %6 : !llvm.ptr<3> -> f32
    llvm.store %7, %2 : f32, !llvm.ptr<1>
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
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK:         affine.parallel (%{{.+}}, %{{.+}}, %{{.+}}) = (0, 0, 0) to (symbol(%[[C2]]), symbol(%{{.+}}), symbol(%{{.+}})) {
// CHECK-NEXT:      %[[TILE:.+]] = memref.alloca() alignment = 4 : memref<4xf32>
// CHECK-NEXT:      affine.parallel (%[[TX:.+]], %[[TY:.+]], %[[TZ:.+]]) = (0, 0, 0) to (symbol(%[[C4]]), symbol(%{{.+}}), symbol(%{{.+}})) {
// CHECK:             %[[P0:.+]] = "enzymexla.memref2pointer"(%[[TILE]]) : (memref<4xf32>) -> !llvm.ptr<3>
// CHECK:             llvm.getelementptr inbounds %[[P0]]
// CHECK:             "enzymexla.barrier"(%[[TX]], %[[TY]], %[[TZ]])
// CHECK:             %[[P1:.+]] = "enzymexla.memref2pointer"(%[[TILE]]) : (memref<4xf32>) -> !llvm.ptr<3>
// CHECK:             llvm.load %[[P1]]
// CHECK-NOT:     llvm.mlir.addressof
// CHECK:         llvm.return
