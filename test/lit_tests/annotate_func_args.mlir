// RUN: enzymexlamlir-opt %s --propagate-constant-bounds | FileCheck %s


// CHECK-LABEL: foo
// CHECK: llvm.align = 128 : i32, llvm.dereferenceable = 16 : i32, llvm.noalias
llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {llvm.nocapture, llvm.nofree}) {
  llvm.return
}

// CHECK-LABEL: bar
// CHECK: llvm.align = 128 : i32, llvm.dereferenceable = 64 : i32, llvm.noalias
llvm.func ptx_kernelcc @bar(%arg0: !llvm.ptr<1> {llvm.nocapture, llvm.nofree}) {
  llvm.return
}

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<8xf64>) {
  %c_4 = stablehlo.constant dense<1> : tensor<i64>
  %c_5 = stablehlo.constant dense<2> : tensor<i64>
  %c_6 = stablehlo.constant dense<3> : tensor<i64>
  %c_8 = stablehlo.constant dense<4> : tensor<i64>
  enzymexla.kernel_call @foo blocks in(%c_5, %c_8, %c_8) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg0) {} : (tensor<4xf32>) -> ()
  enzymexla.kernel_call @bar blocks in(%c_5, %c_8, %c_8) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg1) {} : (tensor<8xf64>) -> ()
  return
}