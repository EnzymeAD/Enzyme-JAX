// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<3056x20x1536xf32>) -> tensor<20x1536x1xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<3056x20x1536xf32>) -> tensor<3056x1x1x20x1536x1xf32>
    %1 = stablehlo.slice %0 [3054:3055, 0:1, 0:1, 0:20, 0:1536, 0:1] : (tensor<3056x1x1x20x1536x1xf32>) -> tensor<1x1x1x20x1536x1xf32>
    %2 = stablehlo.slice %0 [3055:3056, 0:1, 0:1, 0:20, 0:1536, 0:1] : (tensor<3056x1x1x20x1536x1xf32>) -> tensor<1x1x1x20x1536x1xf32>
    %3 = stablehlo.reshape %1 : (tensor<1x1x1x20x1536x1xf32>) -> tensor<20x1536x1xf32>
    %4 = stablehlo.reshape %2 : (tensor<1x1x1x20x1536x1xf32>) -> tensor<20x1536x1xf32>
    %5 = stablehlo.subtract %3, %4 : (tensor<20x1536x1xf32>, tensor<20x1536x1xf32>) -> tensor<20x1536x1xf32>
    return %5 : tensor<20x1536x1xf32>
}

// CHECK: func.func @main1(%arg0: tensor<3056x20x1536xf32>) -> tensor<20x1536x1xf32> {
// CHECK-NEXT:     %0 = stablehlo.slice %arg0 [3054:3055, 0:20, 0:1536] : (tensor<3056x20x1536xf32>) -> tensor<1x20x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %arg0 [3055:3056, 0:20, 0:1536] : (tensor<3056x20x1536xf32>) -> tensor<1x20x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.reshape %0 : (tensor<1x20x1536xf32>) -> tensor<20x1536x1xf32>
// CHECK-NEXT:     %3 = stablehlo.reshape %1 : (tensor<1x20x1536xf32>) -> tensor<20x1536x1xf32>
// CHECK-NEXT:     %4 = stablehlo.subtract %2, %3 : tensor<20x1536x1xf32>
// CHECK-NEXT:     return %4 : tensor<20x1536x1xf32>
// CHECK-NEXT: }
