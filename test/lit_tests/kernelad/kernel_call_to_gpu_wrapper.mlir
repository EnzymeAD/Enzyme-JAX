// RUN: enzymexlamlir-opt %s --kernel-call-to-gpu-wrapper | FileCheck %s

module {
  llvm.func private @kernel(%arg0: !llvm.ptr<1>) {
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    %1 = llvm.zext %0 : i32 to i64
    %2 = llvm.getelementptr inbounds %arg0[%1] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %3 = llvm.load %2 : !llvm.ptr<1> -> f32
    %4 = llvm.fmul %3, %3 : f32
    llvm.store %4, %2 : f32, !llvm.ptr<1>
    llvm.return
  }

  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c4 = stablehlo.constant dense<4> : tensor<i64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = enzymexla.kernel_call @kernel blocks in(%c1, %c1, %c1) threads in(%c4, %c1, %c1) shmem = %c0 (%arg0) <{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}> : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @main(
// CHECK-SAME:      %[[A:.+]]: tensor<4xf32>)
// CHECK:         %[[R:.+]] = enzymexla.jit_call @kernel_gpu_wrapper (%[[A]]) <output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]> : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:    return %[[R]]

// CHECK-LABEL: llvm.func private @kernel_gpu_wrapper(
// CHECK-SAME:      %[[P:.+]]: !llvm.ptr<1>) {
// CHECK:         "enzymexla.gpu_wrapper"
// CHECK:           affine.parallel (%{{.+}}, %{{.+}}, %{{.+}}, %[[TX:.+]], %{{.+}}, %{{.+}}) =
// CHECK:             scf.execute_region {
// CHECK:               %[[T:.+]] = arith.index_castui %[[TX]] : index to i32
// CHECK:               llvm.getelementptr inbounds %[[P]]
// CHECK:               scf.yield
// CHECK:         "enzymexla.polygeist_yield"
// CHECK:         llvm.return
// CHECK-NEXT:  }
