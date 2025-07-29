// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @main1(%arg0: tensor<50xf64> {tf.aliasing_output = 0 : i32}, %arg1: tensor<50x8000xf64>, %arg2: tensor<8000xf64> {tf.aliasing_output = 1 : i32}, %arg3: tensor<8000xf64>, %arg4: tensor<50x8000xf64>, %arg5: tensor<8000xf64>, %arg6: tensor<8000xf64> {tf.aliasing_output = 2 : i32}) -> (tensor<50xf64>, tensor<8000xf64>, tensor<8000xf64>) {
    %cst = stablehlo.constant dense<-1.000000e+00> : tensor<8000xf64>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<8000x8000xf64>
    %0 = stablehlo.reshape %arg5 : (tensor<8000xf64>) -> tensor<8000x1xf64>
    %1 = stablehlo.dot_general %arg1, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<50x8000xf64>, tensor<8000x1xf64>) -> tensor<50x1xf64>
    %2 = stablehlo.reshape %1 : (tensor<50x1xf64>) -> tensor<50xf64>
    %3 = stablehlo.dot_general %arg4, %1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<50x8000xf64>, tensor<50x1xf64>) -> tensor<8000x1xf64>
    %4 = stablehlo.reshape %3 : (tensor<8000x1xf64>) -> tensor<8000xf64>
    %5 = stablehlo.multiply %4, %cst : tensor<8000xf64>
    %6 = stablehlo.add %5, %arg5 : tensor<8000xf64>
    %7 = stablehlo.iota dim = 0 : tensor<8000x2xi64>
    %8 = "stablehlo.scatter"(%cst_0, %7, %arg3) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>}> ({
    ^bb0(%arg7: tensor<f64>, %arg8: tensor<f64>):
      stablehlo.return %arg8 : tensor<f64>
    }) : (tensor<8000x8000xf64>, tensor<8000x2xi64>, tensor<8000xf64>) -> tensor<8000x8000xf64>
    %9 = stablehlo.reshape %6 : (tensor<8000xf64>) -> tensor<8000x1xf64>
    %10 = stablehlo.dot_general %8, %9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8000x8000xf64>, tensor<8000x1xf64>) -> tensor<8000x1xf64>
    %11 = stablehlo.reshape %10 : (tensor<8000x1xf64>) -> tensor<8000xf64>
    return %2, %6, %11 : tensor<50xf64>, tensor<8000xf64>, tensor<8000xf64>
}


