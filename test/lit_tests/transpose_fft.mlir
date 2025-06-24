// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=transpose_fft},transform-interpreter,enzyme-hlo-remove-transform,enzyme-hlo-opt)" | FileCheck %s

func.func @main1(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2, 4] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x6x7x5xcomplex<f32>>
    %1 = stablehlo.fft %0, type = FFT, length = [6, 7, 5] : (tensor<8x9x6x7x5xcomplex<f32>>) -> tensor<8x9x6x7x5xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [1, 0, 3, 2, 4] : (tensor<8x9x6x7x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>>
    return %2 : tensor<9x8x7x6x5xcomplex<f32>>
}

// CHECK: func.func @main1(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>> {
// CHECK-NEXT:     %0 = stablehlo.fft %arg0, type =  FFT, length = [7, 6, 5] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>>
// CHECK-NEXT:     return %0 : tensor<9x8x7x6x5xcomplex<f32>>
// CHECK-NEXT:   }

func.func @main1_2(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2, 4] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x6x7x5xcomplex<f32>>
    %1 = stablehlo.fft %0, type = FFT, length = [6, 7, 5] : (tensor<8x9x6x7x5xcomplex<f32>>) -> tensor<8x9x6x7x5xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [0, 1, 3, 2, 4] : (tensor<8x9x6x7x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>>
    return %2 : tensor<8x9x7x6x5xcomplex<f32>>
}

// CHECK:   func.func @main1_2(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0, 2, 3, 4] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>>
// CHECK-NEXT:     %1 = stablehlo.fft %0, type =  FFT, length = [7, 6, 5] : (tensor<8x9x7x6x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>>
// CHECK-NEXT:     return %1 : tensor<8x9x7x6x5xcomplex<f32>>
// CHECK-NEXT:   }

func.func @main1_3(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 2, 4] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<6x9x8x7x5xcomplex<f32>>
    %1 = stablehlo.fft %0, type = FFT, length = [8, 7, 5] : (tensor<6x9x8x7x5xcomplex<f32>>) -> tensor<6x9x8x7x5xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [1, 2, 3, 0, 4] : (tensor<6x9x8x7x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>>
    return %2 : tensor<9x8x7x6x5xcomplex<f32>>
}


// CHECK:   func.func @main1_3(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 2, 4] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<6x9x8x7x5xcomplex<f32>>
// CHECK-NEXT:     %1 = stablehlo.fft %0, type =  FFT, length = [8, 7, 5] : (tensor<6x9x8x7x5xcomplex<f32>>) -> tensor<6x9x8x7x5xcomplex<f32>>
// CHECK-NEXT:     %2 = stablehlo.transpose %1, dims = [1, 2, 3, 0, 4] : (tensor<6x9x8x7x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>>
// CHECK-NEXT:     return %2 : tensor<9x8x7x6x5xcomplex<f32>>
// CHECK-NEXT:   }

func.func @main2(%arg0: tensor<9x8x7x6x5xf32>) -> tensor<9x8x7x6x3xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2, 4] : (tensor<9x8x7x6x5xf32>) -> tensor<8x9x6x7x5xf32>
    %1 = stablehlo.fft %0, type = RFFT, length = [6, 7, 5] : (tensor<8x9x6x7x5xf32>) -> tensor<8x9x6x7x3xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [1, 0, 3, 2, 4] : (tensor<8x9x6x7x3xcomplex<f32>>) -> tensor<9x8x7x6x3xcomplex<f32>>
    return %2 : tensor<9x8x7x6x3xcomplex<f32>>
}

// CHECK:   func.func @main2(%arg0: tensor<9x8x7x6x5xf32>) -> tensor<9x8x7x6x3xcomplex<f32>> {
// CHECK-NEXT:     %0 = stablehlo.fft %arg0, type =  RFFT, length = [7, 6, 5] : (tensor<9x8x7x6x5xf32>) -> tensor<9x8x7x6x3xcomplex<f32>>
// CHECK-NEXT:     return %0 : tensor<9x8x7x6x3xcomplex<f32>>
// CHECK-NEXT:   }

func.func @main2_2(%arg0: tensor<9x8x7x6x5xf32>) -> tensor<8x9x7x6x3xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2, 4] : (tensor<9x8x7x6x5xf32>) -> tensor<8x9x6x7x5xf32>
    %1 = stablehlo.fft %0, type = RFFT, length = [6, 7, 5] : (tensor<8x9x6x7x5xf32>) -> tensor<8x9x6x7x3xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [0, 1, 3, 2, 4] : (tensor<8x9x6x7x3xcomplex<f32>>) -> tensor<8x9x7x6x3xcomplex<f32>>
    return %2 : tensor<8x9x7x6x3xcomplex<f32>>
}

// CHECK:   func.func @main2_2(%arg0: tensor<9x8x7x6x5xf32>) -> tensor<8x9x7x6x3xcomplex<f32>> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0, 2, 3, 4] : (tensor<9x8x7x6x5xf32>) -> tensor<8x9x7x6x5xf32>
// CHECK-NEXT:     %1 = stablehlo.fft %0, type =  RFFT, length = [7, 6, 5] : (tensor<8x9x7x6x5xf32>) -> tensor<8x9x7x6x3xcomplex<f32>>
// CHECK-NEXT:     return %1 : tensor<8x9x7x6x3xcomplex<f32>>
// CHECK-NEXT:   }

func.func @main3(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 2, 4] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x6x7x5xcomplex<f32>>
    %1 = stablehlo.fft %0, type = IFFT, length = [6, 7, 5] : (tensor<8x9x6x7x5xcomplex<f32>>) -> tensor<8x9x6x7x5xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [0, 1, 3, 2, 4] : (tensor<8x9x6x7x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>>
    return %2 : tensor<8x9x7x6x5xcomplex<f32>>
}

// CHECK:   func.func @main3(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0, 2, 3, 4] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>>
// CHECK-NEXT:     %1 = stablehlo.fft %0, type =  IFFT, length = [7, 6, 5] : (tensor<8x9x7x6x5xcomplex<f32>>) -> tensor<8x9x7x6x5xcomplex<f32>>
// CHECK-NEXT:     return %1 : tensor<8x9x7x6x5xcomplex<f32>>
// CHECK-NEXT:   }

func.func @main3_2(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>> {
    %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 4, 2] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<6x9x8x5x7xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  IFFT, length = [8, 5, 7] : (tensor<6x9x8x5x7xcomplex<f32>>) -> tensor<6x9x8x5x7xcomplex<f32>>
    %2 = stablehlo.transpose %1, dims = [1, 2, 4, 0, 3] : (tensor<6x9x8x5x7xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>>
    return %2 : tensor<9x8x7x6x5xcomplex<f32>>
}

// CHECK:   func.func @main3_2(%arg0: tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 4, 2] : (tensor<9x8x7x6x5xcomplex<f32>>) -> tensor<6x9x8x5x7xcomplex<f32>>
// CHECK-NEXT:     %1 = stablehlo.fft %0, type =  IFFT, length = [8, 5, 7] : (tensor<6x9x8x5x7xcomplex<f32>>) -> tensor<6x9x8x5x7xcomplex<f32>>
// CHECK-NEXT:     %2 = stablehlo.transpose %1, dims = [1, 2, 4, 0, 3] : (tensor<6x9x8x5x7xcomplex<f32>>) -> tensor<9x8x7x6x5xcomplex<f32>>
// CHECK-NEXT:     return %2 : tensor<9x8x7x6x5xcomplex<f32>>
// CHECK-NEXT:   }

func.func @main4(%arg0: tensor<9x8x4x6x5xcomplex<f32>>) -> tensor<8x9x5x6x7xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0, 3, 4, 2] : (tensor<9x8x4x6x5xcomplex<f32>>) -> tensor<8x9x6x5x4xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  IRFFT, length = [6, 5, 7] : (tensor<8x9x6x5x4xcomplex<f32>>) -> tensor<8x9x6x5x7xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1, 3, 2, 4] : (tensor<8x9x6x5x7xf32>) -> tensor<8x9x5x6x7xf32>
    return %2 : tensor<8x9x5x6x7xf32>
}


// CHECK:   func.func @main4(%arg0: tensor<9x8x4x6x5xcomplex<f32>>) -> tensor<8x9x5x6x7xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [1, 0, 4, 3, 2] : (tensor<9x8x4x6x5xcomplex<f32>>) -> tensor<8x9x5x6x4xcomplex<f32>>
// CHECK-NEXT:     %1 = stablehlo.fft %0, type =  IRFFT, length = [5, 6, 7] : (tensor<8x9x5x6x4xcomplex<f32>>) -> tensor<8x9x5x6x7xf32>
// CHECK-NEXT:     return %1 : tensor<8x9x5x6x7xf32>
// CHECK-NEXT:   }

func.func @main4_2(%arg0: tensor<9x8x3x6x7xcomplex<f32>>) -> tensor<9x6x7x8x5xf32> {
    %0 = stablehlo.transpose %arg0, dims = [3, 0, 1, 4, 2] : (tensor<9x8x3x6x7xcomplex<f32>>) -> tensor<6x9x8x7x3xcomplex<f32>>
    %1 = stablehlo.fft %0, type =  IRFFT, length = [8, 7, 5] : (tensor<6x9x8x7x3xcomplex<f32>>) -> tensor<6x9x8x7x5xf32>
    %2 = stablehlo.transpose %1, dims = [1, 0, 3, 2, 4] : (tensor<6x9x8x7x5xf32>) -> tensor<9x6x7x8x5xf32>
    return %2 : tensor<9x6x7x8x5xf32>
}

// CHECK:   func.func @main4_2(%arg0: tensor<9x8x3x6x7xcomplex<f32>>) -> tensor<9x6x7x8x5xf32> {
// CHECK-NEXT:     %0 = stablehlo.transpose %arg0, dims = [0, 3, 4, 1, 2] : (tensor<9x8x3x6x7xcomplex<f32>>) -> tensor<9x6x7x8x3xcomplex<f32>>
// CHECK-NEXT:     %1 = stablehlo.fft %0, type =  IRFFT, length = [7, 8, 5] : (tensor<9x6x7x8x3xcomplex<f32>>) -> tensor<9x6x7x8x5xf32>
// CHECK-NEXT:     return %1 : tensor<9x6x7x8x5xf32>
// CHECK-NEXT:   }
