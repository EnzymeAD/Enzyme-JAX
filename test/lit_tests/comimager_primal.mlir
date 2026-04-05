// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<274x2xf64> {enzymexla.memory_effects = []}, %arg1: tensor<274x2x10xf64> {enzymexla.memory_effects = []}, %arg2: tensor<274x2x10xi64> {enzymexla.memory_effects = []}, %arg3: tensor<4096xi64> {enzymexla.memory_effects = []}, %arg4: tensor<4096xf64> {enzymexla.memory_effects = []}, %arg5: tensor<64x64xcomplex<f64>> {enzymexla.memory_effects = []}) -> tensor<64x64xcomplex<f64>> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<-0.000000e+00> : tensor<64x64xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<274x128x128xf64>
    %cst_1 = stablehlo.constant dense<-0.000000e+00> : tensor<274xf64>
    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<274xf64>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<128x128xf64>
    %c = stablehlo.constant dense<1> : tensor<274x100x2xi64>
    %c_5 = stablehlo.constant dense<128> : tensor<4096x2xi64>
    %c_6 = stablehlo.constant dense<128> : tensor<4096x1xi64>
    %c_7 = stablehlo.constant dense<1> : tensor<4096x1xi64>
    %cst_8 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<274xf64>
    %0 = stablehlo.reshape %arg3 : (tensor<4096xi64>) -> tensor<4096x1xi64>
    %1 = stablehlo.reshape %arg5 : (tensor<64x64xcomplex<f64>>) -> tensor<4096xcomplex<f64>>
    %2 = stablehlo.convert %arg4 : (tensor<4096xf64>) -> tensor<4096xcomplex<f64>>
    %3 = stablehlo.reshape %2 : (tensor<4096xcomplex<f64>>) -> tensor<64x64xcomplex<f64>>
    %4 = stablehlo.multiply %1, %2 : tensor<4096xcomplex<f64>>
    %5 = stablehlo.subtract %0, %c_7 : tensor<4096x1xi64>
    %6 = stablehlo.divide %5, %c_6 : tensor<4096x1xi64>
    %7 = stablehlo.concatenate %5, %6, dim = 1 : (tensor<4096x1xi64>, tensor<4096x1xi64>) -> tensor<4096x2xi64>
    %8 = stablehlo.remainder %7, %c_5 : tensor<4096x2xi64>
    %9 = stablehlo.real %4 : (tensor<4096xcomplex<f64>>) -> tensor<4096xf64>
    %10 = stablehlo.imag %4 : (tensor<4096xcomplex<f64>>) -> tensor<4096xf64>
    %11 = "stablehlo.scatter"(%cst_4, %8, %9) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg6: tensor<f64>, %arg7: tensor<f64>):
      stablehlo.return %arg7 : tensor<f64>
    }) : (tensor<128x128xf64>, tensor<4096x2xi64>, tensor<4096xf64>) -> tensor<128x128xf64>
    %12 = "stablehlo.scatter"(%cst_4, %8, %10) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg6: tensor<f64>, %arg7: tensor<f64>):
      stablehlo.return %arg7 : tensor<f64>
    }) : (tensor<128x128xf64>, tensor<4096x2xi64>, tensor<4096xf64>) -> tensor<128x128xf64>
    %13 = stablehlo.complex %11, %12 : tensor<128x128xcomplex<f64>>
    %14 = stablehlo.fft %13, type =  FFT, length = [128, 128] : (tensor<128x128xcomplex<f64>>) -> tensor<128x128xcomplex<f64>>
    %15 = stablehlo.slice %arg1 [0:274, 1:2, 0:10] : (tensor<274x2x10xf64>) -> tensor<274x1x10xf64>
    %16 = stablehlo.slice %arg1 [0:274, 0:1, 0:10] : (tensor<274x2x10xf64>) -> tensor<274x1x10xf64>
    %17 = stablehlo.slice %arg2 [0:274, 1:2, 0:10] : (tensor<274x2x10xi64>) -> tensor<274x1x10xi64>
    %18 = stablehlo.reshape %17 : (tensor<274x1x10xi64>) -> tensor<274x10xi64>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<274x10xi64>) -> tensor<274x10x10xi64>
    %20 = stablehlo.slice %arg2 [0:274, 0:1, 0:10] : (tensor<274x2x10xi64>) -> tensor<274x1x10xi64>
    %21 = stablehlo.reshape %20 : (tensor<274x1x10xi64>) -> tensor<274x10xi64>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 2] : (tensor<274x10xi64>) -> tensor<274x10x10xi64>
    %23 = stablehlo.reshape %22 : (tensor<274x10x10xi64>) -> tensor<274x100x1xi64>
    %24 = stablehlo.reshape %19 : (tensor<274x10x10xi64>) -> tensor<274x100x1xi64>
    %25 = stablehlo.concatenate %23, %24, dim = 2 : (tensor<274x100x1xi64>, tensor<274x100x1xi64>) -> tensor<274x100x2xi64>
    %26 = stablehlo.subtract %25, %c : tensor<274x100x2xi64>
    %27 = stablehlo.broadcast_in_dim %14, dims = [1, 2] : (tensor<128x128xcomplex<f64>>) -> tensor<274x128x128xcomplex<f64>>
    %28 = stablehlo.real %27 : (tensor<274x128x128xcomplex<f64>>) -> tensor<274x128x128xf64>
    %29 = stablehlo.imag %27 : (tensor<274x128x128xcomplex<f64>>) -> tensor<274x128x128xf64>
    %30 = stablehlo.reshape %28 : (tensor<274x128x128xf64>) -> tensor<274x128x128x1xf64>
    %31 = stablehlo.reshape %29 : (tensor<274x128x128xf64>) -> tensor<274x128x128x1xf64>
    %32 = stablehlo.concatenate %30, %31, dim = 3 : (tensor<274x128x128x1xf64>, tensor<274x128x128x1xf64>) -> tensor<274x128x128x2xf64>
    %33 = "stablehlo.gather"(%32, %26) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [1, 2], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 1, 2>}> : (tensor<274x128x128x2xf64>, tensor<274x100x2xi64>) -> tensor<274x100x2xf64>
    %34 = stablehlo.slice %33 [0:274, 0:100, 0:1] : (tensor<274x100x2xf64>) -> tensor<274x100x1xf64>
    %35 = stablehlo.slice %33 [0:274, 0:100, 1:2] : (tensor<274x100x2xf64>) -> tensor<274x100x1xf64>
    %36 = stablehlo.reshape %34 : (tensor<274x100x1xf64>) -> tensor<274x100xf64>
    %37 = stablehlo.reshape %35 : (tensor<274x100x1xf64>) -> tensor<274x100xf64>
    %38 = stablehlo.complex %36, %37 : tensor<274x100xcomplex<f64>>
    %39 = stablehlo.reshape %38 : (tensor<274x100xcomplex<f64>>) -> tensor<274x10x10xcomplex<f64>>
    %40 = stablehlo.transpose %39, dims = [0, 2, 1] : (tensor<274x10x10xcomplex<f64>>) -> tensor<274x10x10xcomplex<f64>>
    %41 = stablehlo.convert %16 : (tensor<274x1x10xf64>) -> tensor<274x1x10xcomplex<f64>>
    %42 = stablehlo.reshape %41 : (tensor<274x1x10xcomplex<f64>>) -> tensor<274x10xcomplex<f64>>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1] : (tensor<274x10xcomplex<f64>>) -> tensor<274x10x10xcomplex<f64>>
    %44 = stablehlo.broadcast_in_dim %42, dims = [0, 2] : (tensor<274x10xcomplex<f64>>) -> tensor<274x10x10xcomplex<f64>>
    %45 = stablehlo.convert %15 : (tensor<274x1x10xf64>) -> tensor<274x1x10xcomplex<f64>>
    %46 = stablehlo.reshape %45 : (tensor<274x1x10xcomplex<f64>>) -> tensor<274x10xcomplex<f64>>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 2] : (tensor<274x10xcomplex<f64>>) -> tensor<274x10x10xcomplex<f64>>
    %48 = stablehlo.broadcast_in_dim %46, dims = [0, 1] : (tensor<274x10xcomplex<f64>>) -> tensor<274x10x10xcomplex<f64>>
    %49 = stablehlo.multiply %47, %40 : tensor<274x10x10xcomplex<f64>>
    %50 = stablehlo.multiply %43, %49 : tensor<274x10x10xcomplex<f64>>
    %51 = stablehlo.reduce(%50 init: %cst_8) applies stablehlo.add across dimensions = [1, 2] : (tensor<274x10x10xcomplex<f64>>, tensor<complex<f64>>) -> tensor<274xcomplex<f64>>
    %52 = stablehlo.real %51 : (tensor<274xcomplex<f64>>) -> tensor<274xf64>
    %53 = stablehlo.imag %51 : (tensor<274xcomplex<f64>>) -> tensor<274xf64>
    %54 = stablehlo.multiply %53, %cst_2 : tensor<274xf64>
    %55 = stablehlo.multiply %52, %cst_2 : tensor<274xf64>
    %56 = stablehlo.complex %cst_9, %54 : tensor<274xcomplex<f64>>
    %57 = stablehlo.complex %55, %cst_1 : tensor<274xcomplex<f64>>
    %58 = stablehlo.add %56, %57 : tensor<274xcomplex<f64>>
    %59 = chlo.conj %58 : tensor<274xcomplex<f64>> -> tensor<274xcomplex<f64>>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0] : (tensor<274xcomplex<f64>>) -> tensor<274x10x10xcomplex<f64>>
    %61 = stablehlo.multiply %60, %44 : tensor<274x10x10xcomplex<f64>>
    %62 = stablehlo.multiply %61, %48 : tensor<274x10x10xcomplex<f64>>
    %63 = stablehlo.reshape %62 : (tensor<274x10x10xcomplex<f64>>) -> tensor<274x100xcomplex<f64>>
    %64 = stablehlo.real %63 : (tensor<274x100xcomplex<f64>>) -> tensor<274x100xf64>
    %65 = stablehlo.imag %63 : (tensor<274x100xcomplex<f64>>) -> tensor<274x100xf64>
    %66 = "stablehlo.scatter"(%cst_0, %26, %64) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1, 2], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg6: tensor<f64>, %arg7: tensor<f64>):
      %93 = stablehlo.add %arg6, %arg7 : tensor<f64>
      stablehlo.return %93 : tensor<f64>
    }) : (tensor<274x128x128xf64>, tensor<274x100x2xi64>, tensor<274x100xf64>) -> tensor<274x128x128xf64>
    %67 = "stablehlo.scatter"(%cst_0, %26, %65) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1, 2], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg6: tensor<f64>, %arg7: tensor<f64>):
      %93 = stablehlo.add %arg6, %arg7 : tensor<f64>
      stablehlo.return %93 : tensor<f64>
    }) : (tensor<274x128x128xf64>, tensor<274x100x2xi64>, tensor<274x100xf64>) -> tensor<274x128x128xf64>
    %68 = stablehlo.complex %66, %67 : tensor<274x128x128xcomplex<f64>>
    %69 = chlo.conj %68 : tensor<274x128x128xcomplex<f64>> -> tensor<274x128x128xcomplex<f64>>
    %70 = stablehlo.reduce(%69 init: %cst_8) applies stablehlo.add across dimensions = [0] : (tensor<274x128x128xcomplex<f64>>, tensor<complex<f64>>) -> tensor<128x128xcomplex<f64>>
    %71 = chlo.conj %70 : tensor<128x128xcomplex<f64>> -> tensor<128x128xcomplex<f64>>
    %72 = stablehlo.fft %71, type =  FFT, length = [128, 128] : (tensor<128x128xcomplex<f64>>) -> tensor<128x128xcomplex<f64>>
    %73 = stablehlo.real %72 : (tensor<128x128xcomplex<f64>>) -> tensor<128x128xf64>
    %74 = stablehlo.imag %72 : (tensor<128x128xcomplex<f64>>) -> tensor<128x128xf64>
    %75 = stablehlo.reshape %73 : (tensor<128x128xf64>) -> tensor<128x128x1xf64>
    %76 = stablehlo.reshape %74 : (tensor<128x128xf64>) -> tensor<128x128x1xf64>
    %77 = stablehlo.concatenate %75, %76, dim = 2 : (tensor<128x128x1xf64>, tensor<128x128x1xf64>) -> tensor<128x128x2xf64>
    %78 = "stablehlo.gather"(%77, %8) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1, 2>}> : (tensor<128x128x2xf64>, tensor<4096x2xi64>) -> tensor<4096x2xf64>
    %79 = stablehlo.slice %78 [0:4096, 0:1] : (tensor<4096x2xf64>) -> tensor<4096x1xf64>
    %80 = stablehlo.slice %78 [0:4096, 1:2] : (tensor<4096x2xf64>) -> tensor<4096x1xf64>
    %81 = stablehlo.reshape %79 : (tensor<4096x1xf64>) -> tensor<4096xf64>
    %82 = stablehlo.reshape %80 : (tensor<4096x1xf64>) -> tensor<4096xf64>
    %83 = stablehlo.complex %81, %82 : tensor<4096xcomplex<f64>>
    %84 = stablehlo.reshape %83 : (tensor<4096xcomplex<f64>>) -> tensor<64x64xcomplex<f64>>
    %85 = stablehlo.imag %84 : (tensor<64x64xcomplex<f64>>) -> tensor<64x64xf64>
    %86 = stablehlo.real %84 : (tensor<64x64xcomplex<f64>>) -> tensor<64x64xf64>
    %87 = stablehlo.complex %cst_3, %85 : tensor<64x64xcomplex<f64>>
    %88 = stablehlo.complex %86, %cst : tensor<64x64xcomplex<f64>>
    %89 = stablehlo.add %87, %88 : tensor<64x64xcomplex<f64>>
    %90 = chlo.conj %89 : tensor<64x64xcomplex<f64>> -> tensor<64x64xcomplex<f64>>
    %91 = stablehlo.multiply %90, %3 : tensor<64x64xcomplex<f64>>
    %92 = chlo.conj %91 : tensor<64x64xcomplex<f64>> -> tensor<64x64xcomplex<f64>>
    return %92 : tensor<64x64xcomplex<f64>>
  }
}

// We are just checking that we don't hit an infinite compilation here
// CHECK-LABEL: @main
