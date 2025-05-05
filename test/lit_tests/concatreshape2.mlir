// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=concat_reshape},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=concat_reshape},transform-interpreter,enzyme-hlo-remove-transform)" %s |  stablehlo-translate - --interpret

module {
    func.func @main() -> tensor<2x4x2xf32> {
        %0 = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]> : tensor<2x4xf32>
        %1 = stablehlo.reshape %0: (tensor<2x4xf32>) -> tensor<4x2xf32>
        %2 = stablehlo.concatenate %1, %1, dim = 0 : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>

        check.expect_eq_const %2, dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<8x2xf32>

        %3 = stablehlo.reshape %2: (tensor<8x2xf32>) -> tensor<2x4x2xf32>
        return %3 : tensor<2x4x2xf32>
    }
}

// CHECK: func.func @main() -> tensor<2x4x2xf32> {
// CHECK-DAG{LITERAL}:    %cst = stablehlo.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]]> : tensor<2x4xf32>
// CHECK-DAG:    %[[cat:.*]] = stablehlo.concatenate %cst, %cst, dim = 0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:    %[[reshape:.*]] = stablehlo.reshape %[[cat]] : (tensor<4x4xf32>) -> tensor<8x2xf32>
// CHECK-DAG:    %[[reshape2:.*]] = stablehlo.reshape %[[reshape]] : (tensor<8x2xf32>) -> tensor<2x4x2xf32>
// CHECK-DAG:    return %[[reshape2]] : tensor<2x4x2xf32>
// CHECK-DAG:  }
