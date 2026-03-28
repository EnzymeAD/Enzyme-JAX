module {
  func.func @main(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_2 = stablehlo.constant dense<2> : tensor<i64>
    %c_3 = stablehlo.constant dense<127> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_0, %iterArg_4 = %arg0) : tensor<i64>, tensor<128xf32>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %c_2, %iterArg {enzymexla.bounds = [[2, 128]]} : tensor<i64>
      %2 = stablehlo.add %iterArg, %c_1 {enzymexla.bounds = [[1, 127]]} : tensor<i64>
      %3 = stablehlo.convert %1 {enzymexla.bounds = [[2, 128]]} : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c {enzymexla.bounds = [[1, 127]]} : tensor<i32>
      %5 = stablehlo.dynamic_slice %arg0, %4, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %6 = stablehlo.subtract %1, %c_1 {enzymexla.bounds = [[1, 127]]} : tensor<i64>
      %7 = stablehlo.convert %6 {enzymexla.bounds = [[1, 127]]} : (tensor<i64>) -> tensor<i32>
      %8 = stablehlo.subtract %7, %c {enzymexla.bounds = [[0, 126]]} : tensor<i32>
      %9 = stablehlo.dynamic_slice %iterArg_4, %8, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %10 = stablehlo.add %5, %9 : tensor<1xf32>
      %11 = stablehlo.dynamic_update_slice %iterArg_4, %10, %4 : (tensor<128xf32>, tensor<1xf32>, tensor<i32>) -> tensor<128xf32>
      stablehlo.return %2, %11 : tensor<i64>, tensor<128xf32>
    }
    return %0#1 : tensor<128xf32>
  }
}

module @reactant_looped_... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<128xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 0 : i32}) -> tensor<128xf32> attributes {enzymexla.memory_effects = []} {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<128> : tensor<i64>
    %c_3 = stablehlo.constant dense<127> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_4 = %arg0) : tensor<i64>, tensor<128xf32> attributes {enzyme.disable_mincut}
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.subtract %c_2, %iterArg {enzymexla.bounds = [[2, 128]]} : tensor<i64>
      %2 = stablehlo.add %iterArg, %c_0 {enzymexla.bounds = [[1, 127]]} : tensor<i64>
      %3 = stablehlo.convert %1 {enzymexla.bounds = [[2, 128]]} : (tensor<i64>) -> tensor<i32>
      %4 = stablehlo.subtract %3, %c {enzymexla.bounds = [[1, 127]]} : tensor<i32>
      %5 = stablehlo.dynamic_slice %arg0, %4, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %6 = stablehlo.add %1, %c_0 {enzymexla.bounds = [[3, 129]]} : tensor<i64>
      %7 = stablehlo.convert %6 {enzymexla.bounds = [[3, 129]]} : (tensor<i64>) -> tensor<i32>
      %8 = stablehlo.subtract %7, %c {enzymexla.bounds = [[2, 128]]} : tensor<i32>
      %9 = stablehlo.dynamic_slice %iterArg_4, %8, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %10 = stablehlo.add %5, %9 : tensor<1xf32>
      %11 = stablehlo.dynamic_update_slice %iterArg_4, %10, %4 : (tensor<128xf32>, tensor<1xf32>, tensor<i32>) -> tensor<128xf32>
      stablehlo.return %2, %11 : tensor<i64>, tensor<128xf32>
    }
    return %0#1 : tensor<128xf32>
  }
}
