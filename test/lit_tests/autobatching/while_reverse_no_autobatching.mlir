// RUN: enzymexlamlir-opt --enzyme-hlo-opt=enable_auto_batching_passes=true %s | FileCheck %s

module {
  func.func @grad_loss(%arg0: tensor<64x64xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg1: tensor<64x64xf64> {enzymexla.memory_effects = []}, %arg2: tensor<2xf64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}, %arg3: tensor<2xf64> {enzymexla.memory_effects = []}) -> (tensor<2xf64>, tensor<f64>, tensor<64x64xf64>, tensor<64x64xf64>, tensor<2xf64>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = stablehlo.constant {enzymexla.non_negative = [#enzymexla.guaranteed<GUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<GUARANTEED>]} dense<0.000000e+00> : tensor<64x64xf64>
    %c = stablehlo.constant dense<228> : tensor<i64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<229x64x64xf64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_2 = stablehlo.constant dense<229> : tensor<i64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %cst_4 = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %cst_5 = stablehlo.constant dense<1.000000e-03> : tensor<1xf64>
    %cst_6 = stablehlo.constant {enzymexla.non_negative = [#enzymexla.guaranteed<GUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<GUARANTEED>]} dense<5.000000e-01> : tensor<64x64xf64>
    %cst_7 = stablehlo.constant {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<GUARANTEED>]} dense<1.000000e-01> : tensor<64x64xf64>
    %cst_8 = stablehlo.constant dense<-1.000000e-03> : tensor<64x64xf64>
    %cst_9 = stablehlo.constant {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<GUARANTEED>]} dense<1.000000e+00> : tensor<64x64xf64>
    %cst_10 = stablehlo.constant dense<-4.000000e+00> : tensor<64x64xf64>
    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<1xf64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<64x64xf64>) -> tensor<64x64xf64>
    %1 = stablehlo.slice %arg2 [0:1] {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<2xf64>) -> tensor<1xf64>
    %2 = stablehlo.slice %arg2 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
    %3 = stablehlo.reshape %2 : (tensor<1xf64>) -> tensor<f64>
    %4 = stablehlo.broadcast_in_dim %1, dims = [0] {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<1xf64>) -> tensor<64x64xf64>
    %5 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<1xf64>) -> tensor<64x64xf64>
    %6 = stablehlo.tanh %0 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
    %7 = stablehlo.tanh %arg0 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
    %8 = stablehlo.multiply %cst_6, %6 : tensor<64x64xf64>
    %9 = stablehlo.add %cst_6, %8 : tensor<64x64xf64>
    %10 = stablehlo.multiply %4, %9 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
    %11 = stablehlo.transpose %10, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
    %12 = stablehlo.multiply %cst_4, %3 : tensor<f64>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f64>) -> tensor<64x64xf64>
    %14:3 = stablehlo.while(%iterArg = %c_3, %iterArg_13 = %0, %iterArg_14 = %cst_0) : tensor<i64>, tensor<64x64xf64>, tensor<229x64x64xf64> attributes {enzymexla.non_negative = [#enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<NOTGUARANTEED>, #enzymexla.guaranteed<UNKNOWN>]}
    cond {
      %43 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %43 : tensor<i1>
    } do {
      %43 = stablehlo.reshape %iterArg_13 : (tensor<64x64xf64>) -> tensor<1x64x64xf64>
      %44 = stablehlo.dynamic_update_slice %iterArg_14, %43, %iterArg, %c_3, %c_3 : (tensor<229x64x64xf64>, tensor<1x64x64xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<229x64x64xf64>
      %45 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = [[1, 229]]} : tensor<i64>
      %46 = stablehlo.multiply %cst_7, %iterArg_13 : tensor<64x64xf64>
      %47 = stablehlo.add %cst_9, %46 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %48 = stablehlo.multiply %10, %47 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %49 = stablehlo.slice %iterArg_13 [1:64, 0:64] : (tensor<64x64xf64>) -> tensor<63x64xf64>
      %50 = stablehlo.slice %iterArg_13 [0:1, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %51 = stablehlo.concatenate %49, %50, dim = 0 : (tensor<63x64xf64>, tensor<1x64xf64>) -> tensor<64x64xf64>
      %52 = stablehlo.compare GT, %48, %cst : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
      %53 = stablehlo.multiply %48, %iterArg_13 : tensor<64x64xf64>
      %54 = stablehlo.multiply %48, %51 : tensor<64x64xf64>
      %55 = stablehlo.select %52, %53, %54 : tensor<64x64xi1>, tensor<64x64xf64>
      %56 = stablehlo.slice %iterArg_13 [0:64, 1:64] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %57 = stablehlo.slice %iterArg_13 [0:64, 0:1] : (tensor<64x64xf64>) -> tensor<64x1xf64>
      %58 = stablehlo.concatenate %56, %57, dim = 1 : (tensor<64x63xf64>, tensor<64x1xf64>) -> tensor<64x64xf64>
      %59 = stablehlo.multiply %48, %58 : tensor<64x64xf64>
      %60 = stablehlo.select %52, %53, %59 : tensor<64x64xi1>, tensor<64x64xf64>
      %61 = stablehlo.slice %55 [63:64, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %62 = stablehlo.slice %55 [0:63, 0:64] : (tensor<64x64xf64>) -> tensor<63x64xf64>
      %63 = stablehlo.concatenate %61, %62, dim = 0 : (tensor<1x64xf64>, tensor<63x64xf64>) -> tensor<64x64xf64>
      %64 = stablehlo.slice %60 [0:64, 63:64] : (tensor<64x64xf64>) -> tensor<64x1xf64>
      %65 = stablehlo.slice %60 [0:64, 0:63] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %66 = stablehlo.concatenate %64, %65, dim = 1 : (tensor<64x1xf64>, tensor<64x63xf64>) -> tensor<64x64xf64>
      %67 = stablehlo.subtract %55, %63 : tensor<64x64xf64>
      %68 = stablehlo.subtract %60, %66 : tensor<64x64xf64>
      %69 = stablehlo.add %67, %68 : tensor<64x64xf64>
      %70 = stablehlo.slice %iterArg_13 [63:64, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %71 = stablehlo.slice %iterArg_13 [0:63, 0:64] : (tensor<64x64xf64>) -> tensor<63x64xf64>
      %72 = stablehlo.concatenate %70, %71, dim = 0 : (tensor<1x64xf64>, tensor<63x64xf64>) -> tensor<64x64xf64>
      %73 = stablehlo.slice %iterArg_13 [0:64, 63:64] : (tensor<64x64xf64>) -> tensor<64x1xf64>
      %74 = stablehlo.slice %iterArg_13 [0:64, 0:63] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %75 = stablehlo.concatenate %73, %74, dim = 1 : (tensor<64x1xf64>, tensor<64x63xf64>) -> tensor<64x64xf64>
      %76 = stablehlo.add %51, %72 : tensor<64x64xf64>
      %77 = stablehlo.add %76, %58 : tensor<64x64xf64>
      %78 = stablehlo.add %77, %75 : tensor<64x64xf64>
      %79 = stablehlo.multiply %cst_10, %iterArg_13 : tensor<64x64xf64>
      %80 = stablehlo.add %78, %79 : tensor<64x64xf64>
      %81 = stablehlo.multiply %cst_8, %69 : tensor<64x64xf64>
      %82 = stablehlo.add %iterArg_13, %81 : tensor<64x64xf64>
      %83 = stablehlo.multiply %13, %80 : tensor<64x64xf64>
      %84 = stablehlo.add %82, %83 : tensor<64x64xf64>
      stablehlo.return %45, %84, %44 : tensor<i64>, tensor<64x64xf64>, tensor<229x64x64xf64>
    }
    %15 = stablehlo.dot_general %14#1, %14#1, contracting_dims = [0, 1] x [0, 1] : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<f64>
    %16 = stablehlo.transpose %14#1, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
    %17 = stablehlo.add %16, %16 : tensor<64x64xf64>
    %18:4 = stablehlo.while(%iterArg = %c_3, %iterArg_13 = %17, %iterArg_14 = %cst, %iterArg_15 = %cst) : tensor<i64>, tensor<64x64xf64>, tensor<64x64xf64>, tensor<64x64xf64> attributes {enzymexla.non_negative = [#enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<NOTGUARANTEED>, #enzymexla.guaranteed<UNKNOWN>, #enzymexla.guaranteed<NOTGUARANTEED>]}
    cond {
      %43 = stablehlo.compare LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %43 : tensor<i1>
    } do {
      %43 = stablehlo.transpose %iterArg_13, dims = [1, 0] {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<64x64xf64>) -> tensor<64x64xf64>
      %44 = stablehlo.subtract %c, %iterArg {enzymexla.bounds = [[0, 228]]} : tensor<i64>
      %45 = stablehlo.dynamic_slice %14#2, %44, %c_3, %c_3, sizes = [1, 64, 64] {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<229x64x64xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x64x64xf64>
      %46 = stablehlo.reshape %45 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<1x64x64xf64>) -> tensor<64x64xf64>
      %47 = stablehlo.multiply %cst_7, %46 : tensor<64x64xf64>
      %48 = stablehlo.add %cst_9, %47 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %49 = stablehlo.multiply %10, %48 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %50 = stablehlo.slice %46 [1:64, 0:64] : (tensor<64x64xf64>) -> tensor<63x64xf64>
      %51 = stablehlo.slice %46 [0:1, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %52 = stablehlo.concatenate %50, %51, dim = 0 : (tensor<63x64xf64>, tensor<1x64xf64>) -> tensor<64x64xf64>
      %53 = stablehlo.compare GT, %49, %cst : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
      %54 = stablehlo.slice %46 [0:64, 1:64] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %55 = stablehlo.slice %46 [0:64, 0:1] : (tensor<64x64xf64>) -> tensor<64x1xf64>
      %56 = stablehlo.concatenate %54, %55, dim = 1 : (tensor<64x63xf64>, tensor<64x1xf64>) -> tensor<64x64xf64>
      %57 = stablehlo.slice %46 [63:64, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %58 = stablehlo.slice %46 [0:63, 0:64] : (tensor<64x64xf64>) -> tensor<63x64xf64>
      %59 = stablehlo.concatenate %57, %58, dim = 0 : (tensor<1x64xf64>, tensor<63x64xf64>) -> tensor<64x64xf64>
      %60 = stablehlo.slice %46 [0:64, 63:64] : (tensor<64x64xf64>) -> tensor<64x1xf64>
      %61 = stablehlo.slice %46 [0:64, 0:63] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %62 = stablehlo.concatenate %60, %61, dim = 1 : (tensor<64x1xf64>, tensor<64x63xf64>) -> tensor<64x64xf64>
      %63 = stablehlo.add %52, %59 : tensor<64x64xf64>
      %64 = stablehlo.add %63, %56 : tensor<64x64xf64>
      %65 = stablehlo.add %64, %62 : tensor<64x64xf64>
      %66 = stablehlo.multiply %cst_10, %46 : tensor<64x64xf64>
      %67 = stablehlo.add %65, %66 : tensor<64x64xf64>
      %68 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = [[1, 229]]} : tensor<i64>
      %69 = stablehlo.compare EQ, %43, %cst {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
      %70 = stablehlo.multiply %43, %67 : tensor<64x64xf64>
      %71 = stablehlo.select %69, %cst, %70 : tensor<64x64xi1>, tensor<64x64xf64>
      %72 = stablehlo.add %iterArg_14, %71 : tensor<64x64xf64>
      %73 = stablehlo.multiply %43, %13 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %74 = stablehlo.select %69, %cst, %73 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
      %75 = stablehlo.multiply %43, %cst_8 : tensor<64x64xf64>
      %76 = stablehlo.select %69, %cst, %75 : tensor<64x64xi1>, tensor<64x64xf64>
      %77 = stablehlo.compare EQ, %74, %cst : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
      %78 = stablehlo.multiply %74, %cst_10 : tensor<64x64xf64>
      %79 = stablehlo.select %77, %cst, %78 : tensor<64x64xi1>, tensor<64x64xf64>
      %80 = stablehlo.add %43, %79 : tensor<64x64xf64>
      %81 = stablehlo.slice %74 [0:64, 0:1] : (tensor<64x64xf64>) -> tensor<64x1xf64>
      %82 = stablehlo.slice %74 [0:64, 1:64] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %83 = stablehlo.pad %82, %cst_11, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<64x63xf64>, tensor<f64>) -> tensor<64x64xf64>
      %84 = stablehlo.add %80, %83 : tensor<64x64xf64>
      %85 = stablehlo.slice %84 [0:64, 0:63] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %86 = stablehlo.slice %84 [0:64, 63:64] : (tensor<64x64xf64>) -> tensor<64x1xf64>
      %87 = stablehlo.add %86, %81 : tensor<64x1xf64>
      %88 = stablehlo.concatenate %85, %87, dim = 1 : (tensor<64x63xf64>, tensor<64x1xf64>) -> tensor<64x64xf64>
      %89 = stablehlo.slice %74 [0:1, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %90 = stablehlo.slice %74 [1:64, 0:64] : (tensor<64x64xf64>) -> tensor<63x64xf64>
      %91 = stablehlo.pad %90, %cst_11, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<63x64xf64>, tensor<f64>) -> tensor<64x64xf64>
      %92 = stablehlo.add %88, %91 : tensor<64x64xf64>
      %93 = stablehlo.slice %92 [63:64, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %94 = stablehlo.add %93, %89 : tensor<1x64xf64>
      %95 = stablehlo.negate %76 : tensor<64x64xf64>
      %96 = stablehlo.slice %95 [0:64, 0:1] : (tensor<64x64xf64>) -> tensor<64x1xf64>
      %97 = stablehlo.slice %95 [0:64, 1:64] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %98 = stablehlo.pad %97, %cst_11, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<64x63xf64>, tensor<f64>) -> tensor<64x64xf64>
      %99 = stablehlo.add %76, %98 : tensor<64x64xf64>
      %100 = stablehlo.slice %99 [0:64, 0:63] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %101 = stablehlo.slice %99 [0:64, 63:64] : (tensor<64x64xf64>) -> tensor<64x1xf64>
      %102 = stablehlo.add %101, %96 : tensor<64x1xf64>
      %103 = stablehlo.concatenate %100, %102, dim = 1 : (tensor<64x63xf64>, tensor<64x1xf64>) -> tensor<64x64xf64>
      %104 = stablehlo.slice %95 [0:1, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %105 = stablehlo.slice %95 [1:64, 0:64] : (tensor<64x64xf64>) -> tensor<63x64xf64>
      %106 = stablehlo.pad %105, %cst_11, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<63x64xf64>, tensor<f64>) -> tensor<64x64xf64>
      %107 = stablehlo.add %76, %106 : tensor<64x64xf64>
      %108 = stablehlo.slice %107 [0:63, 0:64] : (tensor<64x64xf64>) -> tensor<63x64xf64>
      %109 = stablehlo.slice %107 [63:64, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %110 = stablehlo.add %109, %104 : tensor<1x64xf64>
      %111 = stablehlo.concatenate %108, %110, dim = 0 : (tensor<63x64xf64>, tensor<1x64xf64>) -> tensor<64x64xf64>
      %112 = stablehlo.select %53, %103, %cst {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
      %113 = stablehlo.select %53, %cst, %103 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
      %114 = stablehlo.compare EQ, %113, %cst : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
      %115 = stablehlo.multiply %113, %56 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %116 = stablehlo.select %114, %cst, %115 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
      %117 = stablehlo.multiply %113, %49 : tensor<64x64xf64>
      %118 = stablehlo.select %114, %cst, %117 : tensor<64x64xi1>, tensor<64x64xf64>
      %119 = stablehlo.add %74, %118 : tensor<64x64xf64>
      %120 = stablehlo.slice %119 [0:64, 0:63] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %121 = stablehlo.slice %119 [0:64, 63:64] : (tensor<64x64xf64>) -> tensor<64x1xf64>
      %122 = stablehlo.slice %92 [0:63, 0:1] : (tensor<64x64xf64>) -> tensor<63x1xf64>
      %123 = stablehlo.slice %94 [0:1, 0:1] : (tensor<1x64xf64>) -> tensor<1x1xf64>
      %124 = stablehlo.concatenate %122, %123, dim = 0 : (tensor<63x1xf64>, tensor<1x1xf64>) -> tensor<64x1xf64>
      %125 = stablehlo.add %124, %121 : tensor<64x1xf64>
      %126 = stablehlo.slice %92 [0:63, 1:64] : (tensor<64x64xf64>) -> tensor<63x63xf64>
      %127 = stablehlo.slice %94 [0:1, 1:64] : (tensor<1x64xf64>) -> tensor<1x63xf64>
      %128 = stablehlo.concatenate %126, %127, dim = 0 : (tensor<63x63xf64>, tensor<1x63xf64>) -> tensor<64x63xf64>
      %129 = stablehlo.concatenate %125, %128, dim = 1 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<64x1xf64>, tensor<64x63xf64>) -> tensor<64x64xf64>
      %130 = stablehlo.pad %120, %cst_11, low = [0, 1], high = [0, 0], interior = [0, 0] : (tensor<64x63xf64>, tensor<f64>) -> tensor<64x64xf64>
      %131 = stablehlo.add %129, %130 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %132 = stablehlo.select %53, %111, %cst : tensor<64x64xi1>, tensor<64x64xf64>
      %133 = stablehlo.add %112, %132 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %134 = stablehlo.select %53, %cst, %111 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
      %135 = stablehlo.compare EQ, %134, %cst : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
      %136 = stablehlo.multiply %134, %52 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %137 = stablehlo.select %135, %cst, %136 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
      %138 = stablehlo.add %116, %137 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %139 = stablehlo.multiply %134, %49 : tensor<64x64xf64>
      %140 = stablehlo.select %135, %cst, %139 : tensor<64x64xi1>, tensor<64x64xf64>
      %141 = stablehlo.add %74, %140 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %142 = stablehlo.compare EQ, %133, %cst : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
      %143 = stablehlo.multiply %133, %46 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %144 = stablehlo.select %142, %cst, %143 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
      %145 = stablehlo.add %138, %144 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %146 = stablehlo.transpose %145, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
      %147 = stablehlo.multiply %133, %49 : tensor<64x64xf64>
      %148 = stablehlo.select %142, %cst, %147 : tensor<64x64xi1>, tensor<64x64xf64>
      %149 = stablehlo.add %131, %148 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %150 = stablehlo.slice %141 [63:64, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %151 = stablehlo.reshape %150 : (tensor<1x64xf64>) -> tensor<64x1xf64>
      %152 = stablehlo.slice %149 [0:1, 0:64] : (tensor<64x64xf64>) -> tensor<1x64xf64>
      %153 = stablehlo.reshape %152 : (tensor<1x64xf64>) -> tensor<64x1xf64>
      %154 = stablehlo.add %153, %151 : tensor<64x1xf64>
      %155 = stablehlo.transpose %149, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
      %156 = stablehlo.slice %155 [0:64, 1:64] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %157 = stablehlo.concatenate %154, %156, dim = 1 : (tensor<64x1xf64>, tensor<64x63xf64>) -> tensor<64x64xf64>
      %158 = stablehlo.transpose %141, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
      %159 = stablehlo.slice %158 [0:64, 0:63] : (tensor<64x64xf64>) -> tensor<64x63xf64>
      %160 = stablehlo.pad %159, %cst_11, low = [0, 1], high = [0, 0], interior = [0, 0] : (tensor<64x63xf64>, tensor<f64>) -> tensor<64x64xf64>
      %161 = stablehlo.add %157, %160 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %162 = stablehlo.compare EQ, %145, %cst {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
      %163 = stablehlo.multiply %145, %48 : tensor<64x64xf64>
      %164 = stablehlo.select %162, %cst, %163 : tensor<64x64xi1>, tensor<64x64xf64>
      %165 = stablehlo.add %iterArg_15, %164 : tensor<64x64xf64>
      %166 = stablehlo.multiply %146, %11 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %167 = stablehlo.transpose %162, dims = [1, 0] : (tensor<64x64xi1>) -> tensor<64x64xi1>
      %168 = stablehlo.select %167, %cst, %166 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
      %169 = stablehlo.compare EQ, %168, %cst {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
      %170 = stablehlo.multiply %168, %cst_7 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      %171 = stablehlo.select %169, %cst, %170 : tensor<64x64xi1>, tensor<64x64xf64>
      %172 = stablehlo.add %161, %171 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
      stablehlo.return %68, %172, %72, %165 : tensor<i64>, tensor<64x64xf64>, tensor<64x64xf64>, tensor<64x64xf64>
    }
    %19 = stablehlo.transpose %18#3, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
    %20 = stablehlo.reduce(%18#2 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<64x64xf64>, tensor<f64>) -> tensor<f64>
    %21 = stablehlo.reshape %20 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<f64>) -> tensor<1xf64>
    %22 = stablehlo.compare EQ, %21, %cst_12 : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xi1>
    %23 = stablehlo.multiply %21, %cst_5 : tensor<1xf64>
    %24 = stablehlo.select %22, %cst_12, %23 : tensor<1xi1>, tensor<1xf64>
    %25 = stablehlo.compare EQ, %18#3, %cst {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
    %26 = stablehlo.multiply %18#3, %9 : tensor<64x64xf64>
    %27 = stablehlo.select %25, %cst, %26 : tensor<64x64xi1>, tensor<64x64xf64>
    %28 = stablehlo.multiply %19, %5 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
    %29 = stablehlo.transpose %25, dims = [1, 0] : (tensor<64x64xi1>) -> tensor<64x64xi1>
    %30 = stablehlo.select %29, %cst, %28 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
    %31 = stablehlo.compare EQ, %30, %cst {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
    %32 = stablehlo.multiply %30, %cst_6 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>], enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
    %33 = stablehlo.select %31, %cst, %32 {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
    %34 = stablehlo.multiply %7, %7 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
    %35 = stablehlo.subtract %cst_9, %34 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
    %36 = stablehlo.compare EQ, %33, %cst {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xi1>
    %37 = stablehlo.multiply %33, %35 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
    %38 = stablehlo.select %36, %cst, %37 : tensor<64x64xi1>, tensor<64x64xf64>
    %39 = stablehlo.add %18#1, %38 {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : tensor<64x64xf64>
    %40 = stablehlo.reduce(%27 init: %cst_11) applies stablehlo.add across dimensions = [0, 1] : (tensor<64x64xf64>, tensor<f64>) -> tensor<f64>
    %41 = stablehlo.reshape %40 : (tensor<f64>) -> tensor<1xf64>
    %42 = stablehlo.concatenate %41, %24, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
    return %42, %15, %16, %39, %arg2 : tensor<2xf64>, tensor<f64>, tensor<64x64xf64>, tensor<64x64xf64>, tensor<2xf64>
  }
}

// CHECK: %[[WHILE_FWD:.+]]:3 = stablehlo.while(%iterArg = %{{.*}}, %iterArg_13 = %{{.*}}, %iterArg_14 = %{{.*}}) : tensor<i64>, tensor<64x64xf64>, tensor<229x64x64xf64>
// CHECK: %[[WHILE_REV:.+]]:4 = stablehlo.while(%iterArg = %{{.*}}, %iterArg_13 = %{{.*}}, %iterArg_14 = %{{.*}}, %iterArg_15 = %{{.*}}) : tensor<i64>, tensor<64x64xf64>, tensor<64x64xf64>, tensor<64x64xf64>
// CHECK:      %[[CKPT:.+]] = stablehlo.dynamic_slice %[[WHILE_FWD]]#2, %{{.*}}, %{{.*}}, %{{.*}}, sizes = [1, 64, 64] {enzymexla.non_negative = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<229x64x64xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x64x64xf64>
// CHECK-NOT:  stablehlo.dynamic_slice
