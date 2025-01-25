// RUN: enzymexlamlir-opt %s --propagate-constant-bounds | FileCheck %s

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @"##foo#3846"() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 2> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK-NEXT: nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 1> : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    llvm.return
}

func.func @main() {
  %c_4 = stablehlo.constant dense<1> : tensor<i64>
  %c_5 = stablehlo.constant dense<2> : tensor<i64>
  %c_6 = stablehlo.constant dense<3> : tensor<i64>
  %c_8 = stablehlo.constant dense<4> : tensor<i64>
  enzymexla.kernel_call @"##foo#3846" blocks in(%c_5, %c_8, %c_8) threads in(%c_4, %c_8, %c_8) shmem = %c_6 () {} : () -> ()
  return
}