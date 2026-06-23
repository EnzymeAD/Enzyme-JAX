module @reactant_stencil attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<6x6xf64> {enzymexla.memory_effects = []}, %arg1: tensor<f64> {enzymexla.memory_effects = []}) -> tensor<f64> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<[true, true, true, true, true, false]> : tensor<6xi1>
    %c_0 = stablehlo.constant dense<[false, true, true, true, true, true]> : tensor<6xi1>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %cst = stablehlo.constant dense<4.000000e+00> : tensor<f64>
    %c_2 = stablehlo.constant dense<0> : tensor<i64>
    %c_3 = stablehlo.constant dense<6> : tensor<i64>
    %c_4 = stablehlo.constant dense<1> : tensor<i64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %1 = stablehlo.add %cst, %arg1 : tensor<f64>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f64>) -> tensor<6x6xf64>
    %3 = stablehlo.multiply %2, %arg0 : tensor<6x6xf64>
    %4:2 = stablehlo.while(%iterArg = %c_2, %iterArg_6 = %cst_5) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.symmetric_matrix = [#enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed NOTGUARANTEED>, #enzymexla<guaranteed UNKNOWN>, #enzymexla<guaranteed UNKNOWN>]}
    cond {
      %5 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    } do {
      %5 = stablehlo.add %c_4, %iterArg {enzymexla.bounds = [[1, 6]], enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<i64>
      %6 = stablehlo.convert %5 {enzymexla.bounds = [[1, 6]]} : (tensor<i64>) -> tensor<i32>
      %7 = stablehlo.dynamic_slice %c_0, %iterArg, sizes = [1] : (tensor<6xi1>, tensor<i64>) -> tensor<1xi1>
      %8 = stablehlo.reshape %7 : (tensor<1xi1>) -> tensor<i1>
      %9 = stablehlo.dynamic_slice %c, %iterArg, sizes = [1] : (tensor<6xi1>, tensor<i64>) -> tensor<1xi1>
      %10 = stablehlo.reshape %9 : (tensor<1xi1>) -> tensor<i1>
      %11 = stablehlo.subtract %6, %c_1 {enzymexla.bounds = [[0, 5]]} : tensor<i32>
      %12:2 = stablehlo.while(%iterArg_7 = %c_2, %iterArg_8 = %iterArg_6) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
      cond {
        %13 = stablehlo.compare LT, %iterArg_7, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %13 : tensor<i1>
      } do {
        %13 = stablehlo.add %c_4, %iterArg_7 {enzymexla.bounds = [[1, 6]], enzymexla.non_negative = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<i64>
        %14 = stablehlo.convert %13 {enzymexla.bounds = [[1, 6]]} : (tensor<i64>) -> tensor<i32>
        %15 = stablehlo.subtract %14, %c_1 {enzymexla.bounds = [[0, 5]]} : tensor<i32>
        %16 = stablehlo.dynamic_slice %3, %iterArg, %iterArg_7, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
        %17 = stablehlo.reshape %16 : (tensor<1x1xf64>) -> tensor<f64>
        %18 = stablehlo.dynamic_slice %c, %iterArg_7, sizes = [1] : (tensor<6xi1>, tensor<i64>) -> tensor<1xi1>
        %19 = stablehlo.reshape %18 : (tensor<1xi1>) -> tensor<i1>
        %20 = "stablehlo.if"(%19) ({
          %30 = stablehlo.add %13, %c_4 {enzymexla.bounds = [[2, 7]]} : tensor<i64>
          %31 = stablehlo.convert %30 {enzymexla.bounds = [[2, 7]]} : (tensor<i64>) -> tensor<i32>
          %32 = stablehlo.subtract %31, %c_1 {enzymexla.bounds = [[1, 6]]} : tensor<i32>
          %33 = stablehlo.dynamic_slice %0, %32, %11, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
          %34 = stablehlo.reshape %33 : (tensor<1x1xf64>) -> tensor<f64>
          %35 = stablehlo.subtract %17, %34 : tensor<f64>
          stablehlo.return %35 : tensor<f64>
        }, {
          stablehlo.return %17 : tensor<f64>
        }) : (tensor<i1>) -> tensor<f64>
        %21 = "stablehlo.if"(%10) ({
          %30 = stablehlo.add %5, %c_4 {enzymexla.bounds = [[2, 7]]} : tensor<i64>
          %31 = stablehlo.convert %30 {enzymexla.bounds = [[2, 7]]} : (tensor<i64>) -> tensor<i32>
          %32 = stablehlo.subtract %31, %c_1 {enzymexla.bounds = [[1, 6]]} : tensor<i32>
          %33 = stablehlo.dynamic_slice %0, %15, %32, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
          %34 = stablehlo.reshape %33 : (tensor<1x1xf64>) -> tensor<f64>
          %35 = stablehlo.subtract %20, %34 : tensor<f64>
          stablehlo.return %35 : tensor<f64>
        }, {
          stablehlo.return %20 : tensor<f64>
        }) : (tensor<i1>) -> tensor<f64>
        %22 = stablehlo.dynamic_slice %c_0, %iterArg_7, sizes = [1] : (tensor<6xi1>, tensor<i64>) -> tensor<1xi1>
        %23 = stablehlo.reshape %22 : (tensor<1xi1>) -> tensor<i1>
        %24 = "stablehlo.if"(%23) ({
          %30 = stablehlo.convert %iterArg_7 {enzymexla.bounds = [[0, 5]]} : (tensor<i64>) -> tensor<i32>
          %31 = stablehlo.subtract %30, %c_1 {enzymexla.bounds = [[-1, 4]]} : tensor<i32>
          %32 = stablehlo.dynamic_slice %0, %31, %11, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
          %33 = stablehlo.reshape %32 : (tensor<1x1xf64>) -> tensor<f64>
          %34 = stablehlo.subtract %21, %33 : tensor<f64>
          stablehlo.return %34 : tensor<f64>
        }, {
          stablehlo.return %21 : tensor<f64>
        }) : (tensor<i1>) -> tensor<f64>
        %25 = "stablehlo.if"(%8) ({
          %30 = stablehlo.convert %iterArg {enzymexla.bounds = [[0, 5]]} : (tensor<i64>) -> tensor<i32>
          %31 = stablehlo.subtract %30, %c_1 {enzymexla.bounds = [[-1, 4]]} : tensor<i32>
          %32 = stablehlo.dynamic_slice %0, %15, %31, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
          %33 = stablehlo.reshape %32 : (tensor<1x1xf64>) -> tensor<f64>
          %34 = stablehlo.subtract %24, %33 : tensor<f64>
          stablehlo.return %34 : tensor<f64>
        }, {
          stablehlo.return %24 : tensor<f64>
        }) : (tensor<i1>) -> tensor<f64>
        %26 = stablehlo.dynamic_slice %0, %15, %11, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
        %27 = stablehlo.reshape %26 : (tensor<1x1xf64>) -> tensor<f64>
        %28 = stablehlo.multiply %25, %27 : tensor<f64>
        %29 = stablehlo.add %iterArg_8, %28 : tensor<f64>
        stablehlo.return %13, %29 : tensor<i64>, tensor<f64>
      }
      stablehlo.return %5, %12#1 : tensor<i64>, tensor<f64>
    }
    return %4#1 : tensor<f64>
  }
}
