// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_rotate" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

module {  
  func.func @transpose_rotate_test(%arg0: tensor<10x20x30xf32>) -> tensor<30x10x20xf32> {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 5 : si32, dimension = 1 : si32}> : (tensor<10x20x30xf32>) -> tensor<10x20x30xf32>
    %1 = stablehlo.transpose %0, dims = [2, 0, 1] : (tensor<10x20x30xf32>) -> tensor<30x10x20xf32>
    return %1 : tensor<30x10x20xf32>
  }
}

// CHECK-LABEL: func.func @transpose_rotate_test
// CHECK-SAME: %[[ARG1:.*]]: tensor<10x20x30xf32>
// CHECK: %[[TRANSPOSE2:.*]] = stablehlo.transpose %[[ARG1]], dims = [2, 0, 1] : (tensor<10x20x30xf32>) -> tensor<30x10x20xf32>
// CHECK: %[[ROTATE2:.*]] = "enzymexla.rotate"(%[[TRANSPOSE2]]) <{amount = 5 : si32, dimension = 2 : si32}> : (tensor<30x10x20xf32>) -> tensor<30x10x20xf32>
// CHECK: return %[[ROTATE2]] : tensor<30x10x20xf32>
