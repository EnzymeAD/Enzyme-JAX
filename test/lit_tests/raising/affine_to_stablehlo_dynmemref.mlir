// RUN: enzymexlamlir-opt --raise-affine-to-stablehlo --split-input-file %s | FileCheck %s

module {
  func.func @h(%62: !llvm.ptr) -> () {
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c39063 = arith.constant 39063 : index
    %cst_8 = arith.constant 0.000000e+00 : f32
    %100 = "enzymexla.gpu_wrapper"(%c39063, %c1, %c1, %c256, %c1, %c1) ({
      affine.parallel (%arg0) = (0) to (10000000) {
        %234 = "enzymexla.pointer2memref"(%62) : (!llvm.ptr) -> memref<?xf32>
        affine.store %cst_8, %234[%arg0] : memref<?xf32>
      }
      "enzymexla.polygeist_yield"() : () -> ()
    }) : (index, index, index, index, index, index) -> index
    func.return
  }
}

// CHECK:  func.func @h(%arg0: !llvm.ptr) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c256 = arith.constant 256 : index
// CHECK-NEXT:    %c39063 = arith.constant 39063 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NEXT:    enzymexla.xla_wrapper @rxla$raised_0 (%0) : (memref<?xf32>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK:  func.func private @rxla$raised_0(%arg0: tensor<?xf32>) -> tensor<?xf32> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 0 : tensor<10000000xi64>
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<10000000xi64>
// CHECK-NEXT:    %1 = stablehlo.add %0, %c : tensor<10000000xi64>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<10000000xi64>
// CHECK-NEXT:    %2 = stablehlo.multiply %1, %c_0 : tensor<10000000xi64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<10000000xf32>
// CHECK-NEXT:    %4 = stablehlo.dynamic_update_slice %arg0, %3, %c_1 : (tensor<?xf32>, tensor<10000000xf32>, tensor<i64>) -> tensor<?xf32>
// CHECK-NEXT:    return %4 : tensor<?xf32>
// CHECK-NEXT:  }
