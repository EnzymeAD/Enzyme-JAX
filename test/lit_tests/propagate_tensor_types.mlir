// RUN: enzymexlamlir-opt %s --propagate-constant-bounds --split-input-file | FileCheck %s

llvm.func @use(%arg0: !llvm.ptr<1>)

// CHECK:  func.func @foo(%arg0: memref<?xi64, 1> {llvm.align = 128 : i32, llvm.dereferenceable = 40 : i64, llvm.noalias, llvm.nocapture, llvm.nofree}) {
// CHECK-NEXT:    %0 = "enzymexla.memref2pointer"(%arg0) : (memref<?xi64, 1>) -> !llvm.ptr<1>
// CHECK-NEXT:    llvm.call @use(%0) : (!llvm.ptr<1>) -> ()
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }

llvm.func ptx_kernelcc @foo(%arg0: !llvm.ptr<1> {llvm.align = 32, llvm.nocapture, llvm.nofree}) {
  llvm.call @use(%arg0) : (!llvm.ptr<1>) -> ()
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
