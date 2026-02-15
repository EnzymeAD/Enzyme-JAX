// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_auto_batching_passes=true" %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<274xcomplex<f64>>, %arg1: tensor<274xf64>, %arg2: tensor<274xf64>, %arg3: tensor<274xf64>, %arg4: tensor<274xf64>, %arg5: tensor<274xi32>, %arg6: tensor<274xi32>, %arg7: tensor<274xi32>, %arg8: tensor<274xi32>, %arg9: tensor<126xf64>, %arg10: tensor<126xf64>) -> tensor<274xcomplex<f64>> {
    %c = stablehlo.constant dense<1> : tensor<274xi32>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %c_3 = stablehlo.constant dense<274> : tensor<i32>
    %8:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0) : tensor<i32>, tensor<274xcomplex<f64>>
    cond {
      %9 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %9 : tensor<i1>
    } do {
      %9 = stablehlo.add %c_2, %iterArg {enzymexla.bounds = [[1, 274]]} : tensor<i32>
      %11 = stablehlo.subtract %9, %c_0 {enzymexla.bounds = [[0, 273]]} : tensor<i32>
      %12 = stablehlo.dynamic_slice %arg0, %11, sizes = [1] : (tensor<274xcomplex<f64>>, tensor<i32>) -> tensor<1xcomplex<f64>>
      %13 = stablehlo.dynamic_slice %arg5, %iterArg, sizes = [1] : (tensor<274xi32>, tensor<i32>) -> tensor<1xi32>
      %14 = stablehlo.reshape %13 : (tensor<1xi32>) -> tensor<i32>
      %15 = stablehlo.dynamic_slice %arg9, %14, sizes = [1] : (tensor<126xf64>, tensor<i32>) -> tensor<1xf64>
      %16 = stablehlo.dynamic_slice %arg7, %iterArg, sizes = [1] : (tensor<274xi32>, tensor<i32>) -> tensor<1xi32>
      %17 = stablehlo.reshape %16 : (tensor<1xi32>) -> tensor<i32>
      %18 = stablehlo.dynamic_slice %arg10, %17, sizes = [1] : (tensor<126xf64>, tensor<i32>) -> tensor<1xf64>
      %19 = stablehlo.complex %15, %18 : tensor<1xcomplex<f64>>
      %20 = stablehlo.exponential %19 : tensor<1xcomplex<f64>>
      %21 = stablehlo.dynamic_slice %arg6, %iterArg, sizes = [1] : (tensor<274xi32>, tensor<i32>) -> tensor<1xi32>
      %22 = stablehlo.reshape %21 : (tensor<1xi32>) -> tensor<i32>
      %23 = stablehlo.dynamic_slice %arg9, %22, sizes = [1] : (tensor<126xf64>, tensor<i32>) -> tensor<1xf64>
      %24 = stablehlo.dynamic_slice %arg8, %iterArg, sizes = [1] : (tensor<274xi32>, tensor<i32>) -> tensor<1xi32>
      %25 = stablehlo.reshape %24 : (tensor<1xi32>) -> tensor<i32>
      %26 = stablehlo.dynamic_slice %arg10, %25, sizes = [1] : (tensor<126xf64>, tensor<i32>) -> tensor<1xf64>
      %27 = stablehlo.complex %23, %26 : tensor<1xcomplex<f64>>
      %28 = stablehlo.exponential %27 : tensor<1xcomplex<f64>>
      %29 = chlo.conj %28 : tensor<1xcomplex<f64>> -> tensor<1xcomplex<f64>>
      %30 = stablehlo.multiply %20, %12 : tensor<1xcomplex<f64>>
      %31 = stablehlo.multiply %30, %29 : tensor<1xcomplex<f64>>
      %32 = stablehlo.dynamic_update_slice %iterArg_4, %31, %11 : (tensor<274xcomplex<f64>>, tensor<1xcomplex<f64>>, tensor<i32>) -> tensor<274xcomplex<f64>>
      stablehlo.return %9, %32 : tensor<i32>, tensor<274xcomplex<f64>>
    }
    return %8#1 : tensor<274xcomplex<f64>>
  }
}

// CHECK: func.func @main(%arg0: tensor<274xcomplex<f64>>, %arg1: tensor<274xf64>, %arg2: tensor<274xf64>, %arg3: tensor<274xf64>, %arg4: tensor<274xf64>, %arg5: tensor<274xi32>, %arg6: tensor<274xi32>, %arg7: tensor<274xi32>, %arg8: tensor<274xi32>, %arg9: tensor<126xf64>, %arg10: tensor<126xf64>) -> tensor<274xcomplex<f64>> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg8 : (tensor<274xi32>) -> tensor<274x1xi32>
// CHECK-NEXT:   %1 = "stablehlo.gather"(%arg10, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<126xf64>, tensor<274x1xi32>) -> tensor<274xf64>
// CHECK-NEXT:   %2 = stablehlo.reshape %arg6 : (tensor<274xi32>) -> tensor<274x1xi32>
// CHECK-NEXT:   %3 = "stablehlo.gather"(%arg9, %2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<126xf64>, tensor<274x1xi32>) -> tensor<274xf64>
// CHECK-NEXT:   %4 = stablehlo.reshape %arg7 : (tensor<274xi32>) -> tensor<274x1xi32>
// CHECK-NEXT:   %5 = "stablehlo.gather"(%arg10, %4) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<126xf64>, tensor<274x1xi32>) -> tensor<274xf64>
// CHECK-NEXT:   %6 = stablehlo.reshape %arg5 : (tensor<274xi32>) -> tensor<274x1xi32>
// CHECK-NEXT:   %7 = "stablehlo.gather"(%arg9, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<126xf64>, tensor<274x1xi32>) -> tensor<274xf64>
// CHECK-NEXT:   %8 = stablehlo.complex %7, %5 {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<274xcomplex<f64>>
// CHECK-NEXT:   %9 = stablehlo.complex %3, %1 {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<274xcomplex<f64>>
// CHECK-NEXT:   %10 = stablehlo.exponential %9 {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<274xcomplex<f64>>
// CHECK-NEXT:   %11 = chlo.conj %10 : tensor<274xcomplex<f64>> -> tensor<274xcomplex<f64>>
// CHECK-NEXT:   %12 = stablehlo.exponential %8 {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<274xcomplex<f64>>
// CHECK-NEXT:   %13 = stablehlo.multiply %12, %arg0 {enzymexla.complex_is_purely_real = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<274xcomplex<f64>>
// CHECK-NEXT:   %14 = stablehlo.multiply %13, %11 : tensor<274xcomplex<f64>>
// CHECK-NEXT:   return %14 : tensor<274xcomplex<f64>>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<18x3x10xf32>, %arg1: tensor<5xi32>, %arg2: tensor<5xi32>) -> tensor<5x3xf32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<5xi32>
    %c_1 = stablehlo.constant dense<5> : tensor<i32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<5x3xf32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<i32>
    %c_4 = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<18x3x10xf32>) -> tensor<10x3x18xf32>
    %5 = stablehlo.subtract %arg2, %c_0 : tensor<5xi32>
    %6 = stablehlo.subtract %arg1, %c_0 : tensor<5xi32>
    %7:2 = stablehlo.while(%iterArg = %c_3, %iterArg_5 = %cst) : tensor<i32>, tensor<5x3xf32>
    cond {
      %8 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %8 : tensor<i1>
    } do {
      %8 = stablehlo.add %c_4, %iterArg {enzymexla.bounds = [[1, 5]]} : tensor<i32>
      %9 = stablehlo.dynamic_slice %6, %iterArg, sizes = [1] : (tensor<5xi32>, tensor<i32>) -> tensor<1xi32>
      %10 = stablehlo.reshape %9 : (tensor<1xi32>) -> tensor<i32>
      %11 = stablehlo.dynamic_slice %5, %iterArg, sizes = [1] : (tensor<5xi32>, tensor<i32>) -> tensor<1xi32>
      %12 = stablehlo.reshape %11 : (tensor<1xi32>) -> tensor<i32>
      %13 = stablehlo.dynamic_slice %0, %10, %c, %12, sizes = [1, 3, 1] : (tensor<10x3x18xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x3x1xf32>
      %15 = stablehlo.subtract %8, %c_2 {enzymexla.bounds = [[0, 4]]} : tensor<i32>
      %16 = stablehlo.reshape %13 : (tensor<1x3x1xf32>) -> tensor<1x3xf32>
      %17 = stablehlo.dynamic_update_slice %iterArg_5, %16, %15, %c : (tensor<5x3xf32>, tensor<1x3xf32>, tensor<i32>, tensor<i32>) -> tensor<5x3xf32>
      stablehlo.return %8, %17 : tensor<i32>, tensor<5x3xf32>
    }
    return %7#1 : tensor<5x3xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<18x3x10xf32>, %arg1: tensor<5xi32>, %arg2: tensor<5xi32>) -> tensor<5x3xf32> {
// CHECK-NEXT:   %c = stablehlo.constant dense<1> : tensor<5x2xi32>
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<18x3x10xf32>) -> tensor<10x3x18xf32>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg1 : (tensor<5xi32>) -> tensor<5x1xi32>
// CHECK-NEXT:   %2 = stablehlo.reshape %arg2 : (tensor<5xi32>) -> tensor<5x1xi32>
// CHECK-NEXT:   %3 = stablehlo.concatenate %1, %2, dim = 1 : (tensor<5x1xi32>, tensor<5x1xi32>) -> tensor<5x2xi32>
// CHECK-NEXT:   %4 = stablehlo.subtract %3, %c : tensor<5x2xi32>
// CHECK-NEXT:   %5 = "stablehlo.gather"(%0, %4) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 2], start_index_map = [0, 2], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3, 1>}> : (tensor<10x3x18xf32>, tensor<5x2xi32>) -> tensor<5x3xf32>
// CHECK-NEXT:   return %5 : tensor<5x3xf32>
// CHECK-NEXT: }
