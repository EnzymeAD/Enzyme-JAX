// RUN: enzymexlamlir-opt %s --propagate-constant-bounds --split-input-file | FileCheck %s

llvm.func @foo(%arg0: i32) -> i32 {
  llvm.return %arg0 : i32
}

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 1> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK-NEXT: nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 2> : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK-NEXT: %[[CST:.+]] = llvm.mlir.constant(1 : i32) : i32
    %2 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: %{{.+}} = llvm.call @foo(%[[CST]]) : (i32) -> i32
    llvm.call @foo(%2) : (i32) -> i32
    llvm.return
}

func.func @main() {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_4) threads in(%c_1, %c_4, %c_4) shmem = %c_6 () {} : () -> ()
  return
}

// -----

llvm.func @foo(%arg0: i32) -> i32 {
  llvm.return %arg0 : i32
}

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 4> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK-NEXT: nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 4> : i32
    %1 = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK-NEXT: %[[CST:.+]] = llvm.mlir.constant(4 : i32) : i32
    %2 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: %{{.+}} = llvm.call @foo(%[[CST]]) : (i32) -> i32
    llvm.call @foo(%2) : (i32) -> i32
    llvm.return
}

func.func @main() {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_4, %c_4, %c_4) threads in(%c_4, %c_4, %c_4) shmem = %c_6 () {} : () -> ()
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_4) threads in(%c_1, %c_4, %c_4) shmem = %c_6 () {} : () -> ()
  return
}