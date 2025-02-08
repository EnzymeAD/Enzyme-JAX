// RUN: enzymexlamlir-opt %s --propagate-constant-bounds --split-input-file | FileCheck %s

// CHECK-LABEL: ptx_kernelcc @foo
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 16 : i64, llvm.noalias
llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {llvm.nocapture, llvm.nofree}) {
  llvm.return
}

// CHECK-LABEL: ptx_kernelcc @bar
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 64 : i64, llvm.noalias
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

// -----

// CHECK-LABEL: ptx_kernelcc @bar
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 4 : i64, llvm.noalias, llvm.nocapture, llvm.nofree
llvm.func ptx_kernelcc @bar(%arg0: !llvm.ptr<1> {llvm.nocapture, llvm.nofree}) {
  llvm.return
}

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<f32>) {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_3 = stablehlo.constant dense<3> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_5 = stablehlo.constant dense<5> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  %c_8 = stablehlo.constant dense<8> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_5, %c_8, %c_8) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg0) {} : (tensor<4xf32>) -> ()
  enzymexla.kernel_call @bar blocks in(%c_1, %c_2, %c_3) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg1) {} : (tensor<f32>) -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc @bar
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 4 : i64, llvm.noalias, llvm.nocapture, llvm.nofree
llvm.func ptx_kernelcc @bar(%arg0: !llvm.ptr<1> {llvm.nocapture, llvm.nofree}) {
  llvm.return
}

func.func @main(%arg0: tensor<f32>, %arg1: tensor<4xf32>) {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_3 = stablehlo.constant dense<3> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_5 = stablehlo.constant dense<5> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  %c_8 = stablehlo.constant dense<8> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_5, %c_8, %c_8) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg0) {} : (tensor<f32>) -> ()
  enzymexla.kernel_call @bar blocks in(%c_1, %c_2, %c_3) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg1) {} : (tensor<4xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc @bar
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 4 : i64, llvm.noalias, llvm.nocapture, llvm.nofree
llvm.func ptx_kernelcc @bar(%arg0: !llvm.ptr<1> {llvm.noalias, llvm.nocapture, llvm.nofree}) {
  llvm.return
}

func.func @main(%arg0: tensor<f32>, %arg1: tensor<4xf32>) {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_3 = stablehlo.constant dense<3> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_5 = stablehlo.constant dense<5> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  %c_8 = stablehlo.constant dense<8> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_5, %c_8, %c_8) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg0) {} : (tensor<f32>) -> ()
  enzymexla.kernel_call @bar blocks in(%c_1, %c_2, %c_3) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg1) {} : (tensor<4xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc @foo
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 16 : i64, llvm.noalias
llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {llvm.align = 32, llvm.nocapture, llvm.nofree}) {
  llvm.return
}

// CHECK-LABEL: ptx_kernelcc @bar
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 64 : i64, llvm.noalias
llvm.func ptx_kernelcc @bar(%arg0: !llvm.ptr<1> {llvm.dereferenceable = 1, llvm.nocapture, llvm.nofree}) {
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

// -----

// CHECK-LABEL: ptx_kernelcc @foo
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 8 : i64, llvm.noalias
llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {llvm.align = 32, llvm.nocapture, llvm.nofree}) {
  llvm.return
}

func.func @main(%arg0: tensor<complex<i32>>) {
  %c_4 = stablehlo.constant dense<1> : tensor<i64>
  %c_5 = stablehlo.constant dense<2> : tensor<i64>
  %c_6 = stablehlo.constant dense<3> : tensor<i64>
  %c_8 = stablehlo.constant dense<4> : tensor<i64>
  enzymexla.kernel_call @foo blocks in(%c_5, %c_8, %c_8) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg0) {} : (tensor<complex<i32>>) -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc @foo
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 40 : i64, llvm.noalias
llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {llvm.align = 32, llvm.nocapture, llvm.nofree}) {
  llvm.return
}

func.func @main(%arg0: tensor<5xcomplex<i32>>) {
  %c_4 = stablehlo.constant dense<1> : tensor<i64>
  %c_5 = stablehlo.constant dense<2> : tensor<i64>
  %c_6 = stablehlo.constant dense<3> : tensor<i64>
  %c_8 = stablehlo.constant dense<4> : tensor<i64>
  enzymexla.kernel_call @foo blocks in(%c_5, %c_8, %c_8) threads in(%c_4, %c_8, %c_8) shmem = %c_6 (%arg0) {} : (tensor<5xcomplex<i32>>) -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc @foo
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 512 : i64, llvm.noalias
llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {llvm.align = 32, llvm.nocapture, llvm.nofree}) {
  llvm.return
}

func.func @main(%arg0: tensor<64xi64>) -> tensor<64xi64> {
  %0 = enzymexla.jit_call @foo (%arg0) {
      output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], 
      operand_index = 0, operand_tuple_indices = []>]
    } : (tensor<64xi64>) -> tensor<64xi64>
  return %0 : tensor<64xi64>
}

// -----

// CHECK-LABEL: ptx_kernelcc @foo
// CHECK-SAME: llvm.align = 128 : i32, llvm.dereferenceable = 256 : i64, llvm.noalias
llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {llvm.align = 32, llvm.nocapture, llvm.nofree}) {
  llvm.return
}

func.func @main(%arg0: tensor<64xi64>, %arg1: tensor<32xi64>) -> (tensor<64xi64>, tensor<32xi64>) {
  %0 = enzymexla.jit_call @foo (%arg0) {
      output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], 
      operand_index = 0, operand_tuple_indices = []>]
    } : (tensor<64xi64>) -> tensor<64xi64>
  %1 = enzymexla.jit_call @foo (%arg1) {
      output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], 
      operand_index = 0, operand_tuple_indices = []>]
    } : (tensor<32xi64>) -> tensor<32xi64>
  return %0, %1 : tensor<64xi64>, tensor<32xi64>
}