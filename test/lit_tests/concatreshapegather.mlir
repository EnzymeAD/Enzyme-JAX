module @reactant_batched... attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64} {
  func.func @main(%arg0: tensor<5x4x3xf64>, %arg1: tensor<7xi64>) -> tensor<5x4x7xf64> {
    %c = stablehlo.constant dense<3> : tensor<i64>
    %c_0 = stablehlo.constant dense<2> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<7xi64>
    %c_3 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<5x4x3xf64>) -> tensor<3x4x5xf64>
    %1 = stablehlo.subtract %arg1, %c_2 : tensor<7xi64>
    %2 = stablehlo.reshape %1 : (tensor<7xi64>) -> tensor<7x1xi64>
    %idx = stablehlo.pad %2, %c_3, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
    %3 = "stablehlo.gather"(%0, %idx) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 5>}> : (tensor<3x4x5xf64>, tensor<7x2xi64>) -> tensor<7x5xf64>
    %4 = stablehlo.reshape %3 : (tensor<7x5xf64>) -> tensor<7x1x5xf64>
    %5 = stablehlo.pad %2, %c_1, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
    %6 = "stablehlo.gather"(%0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 5>}> : (tensor<3x4x5xf64>, tensor<7x2xi64>) -> tensor<7x5xf64>
    %7 = stablehlo.reshape %6 : (tensor<7x5xf64>) -> tensor<7x1x5xf64>
    %8 = stablehlo.pad %2, %c_0, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
    %9 = "stablehlo.gather"(%0, %8) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 5>}> : (tensor<3x4x5xf64>, tensor<7x2xi64>) -> tensor<7x5xf64>
    %10 = stablehlo.reshape %9 : (tensor<7x5xf64>) -> tensor<7x1x5xf64>
    %11 = stablehlo.pad %2, %c, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<7x1xi64>, tensor<i64>) -> tensor<7x2xi64>
    %12 = "stablehlo.gather"(%0, %11) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 5>}> : (tensor<3x4x5xf64>, tensor<7x2xi64>) -> tensor<7x5xf64>
    %13 = stablehlo.reshape %12 : (tensor<7x5xf64>) -> tensor<7x1x5xf64>
    %14 = stablehlo.concatenate %4, %7, %10, %13, dim = 1 : (tensor<7x1x5xf64>, tensor<7x1x5xf64>, tensor<7x1x5xf64>, tensor<7x1x5xf64>) -> tensor<7x4x5xf64>
    %15 = stablehlo.transpose %14, dims = [2, 1, 0] : (tensor<7x4x5xf64>) -> tensor<5x4x7xf64>
    return %15 : tensor<5x4x7xf64>
  }
}
