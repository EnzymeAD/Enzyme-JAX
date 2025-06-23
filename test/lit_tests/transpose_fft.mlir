func.func @main1(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2, 4] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x6x7x5xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  FFT, length = [6, 7, 5] : (tensor<8x9x6x7x5xcomplex<f32>>) -> tensor<8x9x6x7x5xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [1, 0, 3, 2, 4] : (tensor<8x9x6x7x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>>
    return %2 : tensor<9x8x7x6x5xcomplex<f32>>
}

func.func @main1_2(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2, 4] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x6x7x5xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  FFT, length = [6, 7, 5] : (tensor<8x9x6x7x5xcomplex<f32>>) -> tensor<8x9x6x7x5xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [0, 1, 3, 2, 4] : (tensor<8x9x6x7x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>>
    return %2 : tensor<8x9x7x6x5xcomplex<f32>>
}

func.func @main1_3(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 2, 4] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<6x9x8x7x5xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  FFT, length = [8, 7, 5] : (tensor<6x9x8x7x5xcomplex<f32>>) -> tensor<6x9x8x7x5xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [1, 2, 3, 0, 4] : (tensor<6x9x8x7x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>>
    return %2 : tensor<9x8x7x6x5xcomplex<f32>>
}

// func.func @main2(%arg0: tensor<9x8x7x6x5xf32>) -> tensor<9x8x4x6x5xcomplex<f32>> {
//     %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 4, 2] : (tensor<9x8x7x6x5xf32>) -> tensor<6x9x8x5x7xf32>
//     %1 = stablehlo.fft %0, type =  RFFT, length = [8, 5, 7] : (tensor<6x9x8x5x7xf32>) -> tensor<6x9x8x5x4xcomplex<f32>>
//     %2 = stablehlo.transpose %1, dims = [1, 2, 4, 0, 3] : (tensor<6x9x8x5x4xcomplex<f32>>) -> tensor<9x8x4x6x5xcomplex<f32>>
//     return %2 : tensor<9x8x4x6x5xcomplex<f32>>
// }

// func.func @main3(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>> {
//     %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 4, 2] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<6x9x8x5x7xcomplex<f32>>
//     %1 = stablehlo.fft %0, type =  IFFT, length = [8, 5, 7] : (tensor<6x9x8x5x7xcomplex<f32>>) -> tensor<6x9x8x5x7xcomplex<f32>>
//     %2 = stablehlo.transpose %1, dims = [1, 2, 4, 0, 3] : (tensor<6x9x8x5x7xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>>
//     return %2 : tensor<9x8x7x6x5xcomplex<f32>>
// }

// func.func @main4(%arg0: tensor<9x8x4x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xf32> {
//     %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 4, 2] : (tensor<9x8x4x6x5xcomplex<f32>>) -> tensor<6x9x8x5x4xcomplex<f32>>
//     %1 = stablehlo.fft %0, type =  IRFFT, length = [8, 5, 7] : (tensor<6x9x8x5x4xcomplex<f32>>) -> tensor<6x9x8x5x7xf32>
//     %2 = stablehlo.transpose %1, dims = [1, 2, 4, 0, 3] : (tensor<6x9x8x5x7xf32>) -> tensor<9x8x7x6x5xf32>
//     return %2 : tensor<9x8x7x6x5xf32>
// }

// func.func @main5(%arg0: tensor<9x8x4x6x5xcomplex<f32>>) -> tensor<9x8x6x6x5xf32> {
//     %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 4, 2] : (tensor<9x8x4x6x5xcomplex<f32>>) -> tensor<6x9x8x5x4xcomplex<f32>>
//     %1 = stablehlo.fft %0, type =  IRFFT, length = [8, 5, 6] : (tensor<6x9x8x5x4xcomplex<f32>>) -> tensor<6x9x8x5x6xf32>
//     %2 = stablehlo.transpose %1, dims = [1, 2, 4, 0, 3] : (tensor<6x9x8x5x6xf32>) -> tensor<9x8x6x6x5xf32>
//     return %2 : tensor<9x8x6x6x5xf32>
// }
