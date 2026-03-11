func.func @main1(%arg0: tensor<100x100xf64>) -> tensor<100x100xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<100x100xf64>
    %1 = stablehlo.iota dim = 1 : tensor<100x100xi64>
    %2 = stablehlo.iota dim = 0 : tensor<100x100xi64>
    %3 = stablehlo.compare GE, %1, %2 : (tensor<100x100xi64>, tensor<100x100xi64>) -> tensor<100x100xi1>
    %4 = stablehlo.select %3, %cst, %arg0 : tensor<100x100xi1>, tensor<100x100xf64>
    %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<100x100xf64>) -> tensor<100x100xf64>
    %6 = stablehlo.transpose %5, dims = [1, 0] : (tensor<100x100xf64>) -> tensor<100x100xf64>
    %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<100x100xf64>) -> tensor<100x100xf64>
    %8 = stablehlo.add %4, %6 : tensor<100x100xf64>
    %11 = stablehlo.transpose %8, dims = [1, 0] : (tensor<100x100xf64>) -> tensor<100x100xf64>
    %12:4 = enzymexla.linalg.lu %11 : (tensor<100x100xf64>) -> (tensor<100x100xf64>, tensor<100xi32>, tensor<100xi32>, tensor<i32>)
    return %12 : tensor<100x100xf64>
}

// CHECK:  func.func @main1(%arg0: tensor<100x100xf64>) -> tensor<100x100xf64> {
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<100x100xf64>
// CHECK-NEXT:    %0 = stablehlo.iota dim = 1 : tensor<100x100xi64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<100x100xi64>
// CHECK-NEXT:    %2 = stablehlo.compare  GE, %0, %1 : (tensor<100x100xi64>, tensor<100x100xi64>) -> tensor<100x100xi1>
// CHECK-NEXT:    %3 = stablehlo.select %2, %cst, %arg0 {enzymexla.lower_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<100x100xi1>, tensor<100x100xf64>
// CHECK-NEXT:    %4 = stablehlo.transpose %3, dims = [1, 0] {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<100x100xf64>) -> tensor<100x100xf64>
// CHECK-NEXT:    %5 = stablehlo.transpose %4, dims = [1, 0] {enzymexla.lower_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<100x100xf64>) -> tensor<100x100xf64>
// CHECK-NEXT:    %6 = stablehlo.add %3, %5 {enzymexla.lower_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<100x100xf64>
// CHECK-NEXT:    %7 = stablehlo.transpose %6, dims = [1, 0] {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<100x100xf64>) -> tensor<100x100xf64>
// CHECK-NEXT:    return %7 : tensor<100x100xf64>
// CHECK-NEXT:  }

func.func @main2(%arg0: tensor<64x64xf64> {enzymexla.memory_effects = []}, %arg1: tensor<64x64xf64> {enzymexla.memory_effects = []}) -> tensor<64x64xf64> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<64x64xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
    %1 = stablehlo.iota dim = 0 : tensor<64x64xi64>
    %2 = stablehlo.iota dim = 1 : tensor<64x64xi64>
    %3 = stablehlo.compare  NE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %4 = stablehlo.compare  LE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %5 = stablehlo.select %4, %0, %cst_0 : tensor<64x64xi1>, tensor<64x64xf64>
    %6 = stablehlo.select %3, %5, %cst : tensor<64x64xi1>, tensor<64x64xf64>
    %7:4 = enzymexla.linalg.lu %6 : (tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
    return %7 : tensor<64x64xf64>
}

// CHECK:  func.func @main(%arg0: tensor<64x64xf64> {enzymexla.memory_effects = []}, %arg1: tensor<64x64xf64> {enzymexla.memory_effects = []}) -> tensor<64x64xf64> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<64x64xf64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf64>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<64x64xi64>
// CHECK-NEXT:    %2 = stablehlo.iota dim = 1 : tensor<64x64xi64>
// CHECK-NEXT:    %3 = stablehlo.compare  NE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
// CHECK-NEXT:    %4 = stablehlo.compare  LE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
// CHECK-NEXT:    %5 = stablehlo.select %4, %0, %cst_0 {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
// CHECK-NEXT:    %6 = stablehlo.select %3, %5, %cst {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>], enzymexla.upper_unit_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
// CHECK-NEXT:    return %6 : tensor<64x64xf64>
// CHECK-NEXT:  }

func.func @main3(%arg0: tensor<64x64xf64> {enzymexla.memory_effects = []}, %arg1: tensor<64x64xf64> {enzymexla.memory_effects = []}) -> tensor<64x64xf64> attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<64x64xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
    %1 = stablehlo.iota dim = 0 : tensor<64x64xi64>
    %2 = stablehlo.iota dim = 1 : tensor<64x64xi64>
    %3 = stablehlo.compare  NE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %4 = stablehlo.compare  LE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
    %5 = stablehlo.select %4, %0, %cst_0 : tensor<64x64xi1>, tensor<64x64xf64>
    %6 = stablehlo.select %3, %5, %cst : tensor<64x64xi1>, tensor<64x64xf64>
    %7 = stablehlo.dot_general %6, %6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xf64>
    %8 = stablehlo.add %6, %7 : tensor<64x64xf64>
    %9:4 = enzymexla.linalg.lu %7 : (tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
    %10:4 = enzymexla.linalg.lu %8 : (tensor<64x64xf64>) -> (tensor<64x64xf64>, tensor<64xi32>, tensor<64xi32>, tensor<i32>)
    return %10 : tensor<64x64xf64>
}

// CHECK:  func.func @main3(%arg0: tensor<64x64xf64> {enzymexla.memory_effects = []}, %arg1: tensor<64x64xf64> {enzymexla.memory_effects = []}) -> tensor<64x64xf64> attributes {enzymexla.memory_effects = []} {
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<64x64xf64>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<64x64xf64>
// CHECK-NEXT:    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<64x64xf64>) -> tensor<64x64xf64>
// CHECK-NEXT:    %1 = stablehlo.iota dim = 0 : tensor<64x64xi64>
// CHECK-NEXT:    %2 = stablehlo.iota dim = 1 : tensor<64x64xi64>
// CHECK-NEXT:    %3 = stablehlo.compare  NE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
// CHECK-NEXT:    %4 = stablehlo.compare  LE, %1, %2 : (tensor<64x64xi64>, tensor<64x64xi64>) -> tensor<64x64xi1>
// CHECK-NEXT:    %5 = stablehlo.select %4, %0, %cst_0 {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
// CHECK-NEXT:    %6 = stablehlo.select %3, %5, %cst {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>], enzymexla.upper_unit_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<64x64xi1>, tensor<64x64xf64>
// CHECK-NEXT:    %7 = stablehlo.dot_general %6, %6, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : (tensor<64x64xf64>, tensor<64x64xf64>) -> tensor<64x64xf64>
// CHECK-NEXT:    %8 = stablehlo.add %6, %7 {enzymexla.upper_tri_matrix = [#enzymexla<guaranteed GUARANTEED>]} : tensor<64x64xf64>
// CHECK-NEXT:    return %8 : tensor<64x64xf64>
// CHECK-NEXT:  }