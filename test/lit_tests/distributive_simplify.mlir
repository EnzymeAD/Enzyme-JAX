// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main1(%arg0: tensor<2x4xf64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>, %arg3: tensor<2xf64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xf64>, %arg8: tensor<2xf64>, %arg9: tensor<2xf64>, %arg10: tensor<2xf64>) -> tensor<4xf64> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
    %1 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
    %2 = stablehlo.add %0, %1 : tensor<4xf64>
    %3 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
    %4 = stablehlo.add %2, %3 : tensor<4xf64>
    %5 = stablehlo.dot_general %arg0, %arg4, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
    %6 = stablehlo.add %4, %5 : tensor<4xf64>
    %7 = stablehlo.dot_general %arg0, %arg5, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
    %8 = stablehlo.add %6, %7 : tensor<4xf64>
    %9 = stablehlo.dot_general %arg0, %arg6, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
    %10 = stablehlo.add %8, %9 : tensor<4xf64>
    %11 = stablehlo.dot_general %arg0, %arg7, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
    %12 = stablehlo.add %10, %11 : tensor<4xf64>
    %13 = stablehlo.dot_general %arg0, %arg8, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
    %14 = stablehlo.add %12, %13 : tensor<4xf64>
    %15 = stablehlo.dot_general %arg0, %arg9, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
    %16 = stablehlo.add %14, %15 : tensor<4xf64>
    %17 = stablehlo.dot_general %arg0, %arg10, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
    %18 = stablehlo.add %16, %17 : tensor<4xf64>
    return %18 : tensor<4xf64>
}

// CHECK: func.func @main1(%arg0: tensor<2x4xf64>, %arg1: tensor<2xf64>, %arg2: tensor<2xf64>, %arg3: tensor<2xf64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xf64>, %arg8: tensor<2xf64>, %arg9: tensor<2xf64>, %arg10: tensor<2xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.add %arg1, %arg2 : tensor<2xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %arg3 : tensor<2xf64>
// CHECK-NEXT:     %2 = stablehlo.add %1, %arg4 : tensor<2xf64>
// CHECK-NEXT:     %3 = stablehlo.add %2, %arg5 : tensor<2xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %arg6 : tensor<2xf64>
// CHECK-NEXT:     %5 = stablehlo.add %4, %arg7 : tensor<2xf64>
// CHECK-NEXT:     %6 = stablehlo.add %5, %arg8 : tensor<2xf64>
// CHECK-NEXT:     %7 = stablehlo.add %6, %arg9 : tensor<2xf64>
// CHECK-NEXT:     %8 = stablehlo.add %7, %arg10 : tensor<2xf64>
// CHECK-NEXT:     %9 = stablehlo.dot_general %arg0, %8, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x4xf64>, tensor<2xf64>) -> tensor<4xf64>
// CHECK-NEXT:     return %9 : tensor<4xf64>
// CHECK-NEXT: }
