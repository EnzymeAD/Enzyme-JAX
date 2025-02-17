// RUN: enzymexlamlir-opt %s --enzyme-batch -enzyme-wrap="infn=main retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --canonicalize --enzyme-hlo-opt | FileCheck %s

module {
  func.func @f(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }

  func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
    %0 = enzyme.batch @f(%arg0) { batch_shape = array<i64: 10> } : (tensor<10xf32>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:    %0 = call @batched_f(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %1 = call @diffebatched_f(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    return %1 : tensor<10xf32>
// CHECK-NEXT:  }

// CHECK:  func.func private @diffebatched_f(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
// CHECK-NEXT:    %0 = stablehlo.multiply %arg1, %arg0 : tensor<10xf32>
// CHECK-NEXT:    %1 = stablehlo.add %0, %0 : tensor<10xf32>
// CHECK-NEXT:    return %1 : tensor<10xf32>
// CHECK-NEXT:  }
