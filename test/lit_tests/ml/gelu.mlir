func.func @apply_gelu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.ml.gelu %arg0, approximation = TANH : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

func.func @apply_gelu2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.ml.gelu %arg0, approximation = SIGMOID : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}

func.func @apply_gelu3(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = enzymexla.ml.gelu %arg0, approximation = NONE : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
