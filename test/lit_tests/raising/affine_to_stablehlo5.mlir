// RUN: enzymexlamlir-opt %s --raise-affine-to-stablehlo --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: memref<180xf32>, %arg1: memref<180xf32>, %arg2: memref<180xf32>) {
  	%c90 = arith.constant 90 : index
  	affine.parallel (%i) = (0) to (180) {
  	  %1 = affine.load %arg0[%i] : memref<180xf32>
  	  %2 = affine.load %arg1[%i] : memref<180xf32>
  	  %cond = arith.cmpi sgt, %i, %c90 : index
  	  %3 = arith.select %cond, %1, %2 : f32
  	  affine.store %3, %arg2[%i] : memref<180xf32>
  	}
  	return
  }
}

// CHECK:  func.func private @main_raised(%arg0: tensor<180xf32>, %arg1: tensor<180xf32>, %arg2: tensor<180xf32>) -> (tensor<180xf32>, tensor<180xf32>, tensor<180xf32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<"0x0000000000000000000000F8FFFFFFFFFFFFFFFFFFFF0F"> : tensor<180xi1>
// CHECK-NEXT:    %0 = stablehlo.select %c, %arg0, %arg1 : tensor<180xi1>, tensor<180xf32>
// CHECK-NEXT:    return %arg0, %arg1, %0 : tensor<180xf32>, tensor<180xf32>, tensor<180xf32>
// CHECK-NEXT:  }
