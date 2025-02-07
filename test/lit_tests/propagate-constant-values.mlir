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
    // CHECK: %[[DIM:.+]] = nvvm.read.ptx.sreg.ntid.x range <i32, 0, 4> : i32
    %2 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: %{{.+}} = llvm.call @foo(%[[DIM]]) : (i32) -> i32
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

// -----

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 6> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: nvvm.read.ptx.sreg.tid.y range <i32, 0, 4> : i32
    %1 = nvvm.read.ptx.sreg.tid.y : i32
    // CHECK: nvvm.read.ptx.sreg.tid.z range <i32, 0, 2> : i32
    %2 = nvvm.read.ptx.sreg.tid.z : i32

    // CHECK: nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 2> : i32
    %3 = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.y range <i32, 0, 4> : i32
    %4 = nvvm.read.ptx.sreg.ctaid.y : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.z range <i32, 0, 6> : i32
    %5 = nvvm.read.ptx.sreg.ctaid.z : i32
    
    // CHECK: llvm.mlir.constant(6 : i32) : i32
    %6 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: llvm.mlir.constant(4 : i32) : i32
    %7 = nvvm.read.ptx.sreg.ntid.y : i32
    // CHECK: llvm.mlir.constant(2 : i32) : i32
    %8 = nvvm.read.ptx.sreg.ntid.z : i32

    // CHECK: llvm.mlir.constant(2 : i32) : i32
    %9 = nvvm.read.ptx.sreg.nctaid.x : i32
    // CHECK: llvm.mlir.constant(4 : i32) : i32
    %10 = nvvm.read.ptx.sreg.nctaid.y : i32
    // CHECK: llvm.mlir.constant(6 : i32) : i32
    %11 = nvvm.read.ptx.sreg.nctaid.z : i32

    llvm.return
}

func.func @main() {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_6) threads in(%c_6, %c_4, %c_2) shmem = %c_6 () {} : () -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 6> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: nvvm.read.ptx.sreg.tid.y range <i32, 0, 4> : i32
    %1 = nvvm.read.ptx.sreg.tid.y : i32
    // CHECK: nvvm.read.ptx.sreg.tid.z range <i32, 0, 6> : i32
    %2 = nvvm.read.ptx.sreg.tid.z : i32

    // CHECK: nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 6> : i32
    %3 = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.y range <i32, 0, 4> : i32
    %4 = nvvm.read.ptx.sreg.ctaid.y : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.z range <i32, 0, 6> : i32
    %5 = nvvm.read.ptx.sreg.ctaid.z : i32
    
    // CHECK: nvvm.read.ptx.sreg.ntid.x range <i32, 0, 6> : i32
    %6 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: nvvm.read.ptx.sreg.ntid.y range <i32, 0, 4> : i32
    %7 = nvvm.read.ptx.sreg.ntid.y : i32
    // CHECK: nvvm.read.ptx.sreg.ntid.z range <i32, 0, 6> : i32
    %8 = nvvm.read.ptx.sreg.ntid.z : i32

    // CHECK: nvvm.read.ptx.sreg.nctaid.x range <i32, 0, 6> : i32
    %9 = nvvm.read.ptx.sreg.nctaid.x : i32
    // CHECK: nvvm.read.ptx.sreg.nctaid.y range <i32, 0, 4> : i32
    %10 = nvvm.read.ptx.sreg.nctaid.y : i32
    // CHECK: nvvm.read.ptx.sreg.nctaid.z range <i32, 0, 6> : i32
    %11 = nvvm.read.ptx.sreg.nctaid.z : i32

    llvm.return
}

func.func @main() {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_6) threads in(%c_6, %c_4, %c_2) shmem = %c_6 () {} : () -> ()
  enzymexla.kernel_call @bar blocks in(%c_6, %c_4, %c_2) threads in(%c_2, %c_4, %c_6) shmem = %c_6 () {} : () -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
    // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 6> : i32
    %0 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: nvvm.read.ptx.sreg.tid.y range <i32, 0, 4> : i32
    %1 = nvvm.read.ptx.sreg.tid.y : i32
    // CHECK: nvvm.read.ptx.sreg.tid.z : i32
    // CHECK-NOT: range <i32, 0, 2> : i32
    %2 = nvvm.read.ptx.sreg.tid.z : i32

    // CHECK: nvvm.read.ptx.sreg.ctaid.x
    // CHECK-NOT: range <i32, 0, 2> : i32
    %3 = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.y range <i32, 0, 4> : i32
    %4 = nvvm.read.ptx.sreg.ctaid.y : i32
    // CHECK: nvvm.read.ptx.sreg.ctaid.z range <i32, 0, 6> : i32
    %5 = nvvm.read.ptx.sreg.ctaid.z : i32
    
    // CHECK: llvm.mlir.constant(6 : i32) : i32
    %6 = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: llvm.mlir.constant(4 : i32) : i32
    %7 = nvvm.read.ptx.sreg.ntid.y : i32
    // CHECK: nvvm.read.ptx.sreg.ntid.z : i32
    %8 = nvvm.read.ptx.sreg.ntid.z : i32

    // CHECK: nvvm.read.ptx.sreg.nctaid.x : i32
    %9 = nvvm.read.ptx.sreg.nctaid.x : i32
    // CHECK: llvm.mlir.constant(4 : i32) : i32
    %10 = nvvm.read.ptx.sreg.nctaid.y : i32
    // CHECK: llvm.mlir.constant(6 : i32) : i32
    %11 = nvvm.read.ptx.sreg.nctaid.z : i32

    llvm.return
}

func.func @main(%c_2 : tensor<i64>) {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_6) threads in(%c_6, %c_4, %c_2) shmem = %c_6 () {} : () -> ()
  return
}

// -----

// CHECK-LABEL: ptx_kernelcc
llvm.func ptx_kernelcc @bar() {
  // CHECK: nvvm.read.ptx.sreg.tid.x range <i32, 0, 6> : i32
  %0 = nvvm.read.ptx.sreg.tid.x range <i32, 1, 2000> : i32
  llvm.return
}

func.func @main() {
  %c_1 = stablehlo.constant dense<1> : tensor<i64>
  %c_2 = stablehlo.constant dense<2> : tensor<i64>
  %c_4 = stablehlo.constant dense<4> : tensor<i64>
  %c_6 = stablehlo.constant dense<6> : tensor<i64>
  enzymexla.kernel_call @bar blocks in(%c_2, %c_4, %c_6) threads in(%c_6, %c_4, %c_2) shmem = %c_6 () {} : () -> ()
  return
}