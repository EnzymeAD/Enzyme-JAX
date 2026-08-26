// RUN: enzymexlamlir-opt --enzyme-hlo-opt="enable_convert_to_convolution=true" %s | FileCheck %s

func.func @main1(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : i32, dimension = 1 : i32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.subtract %arg0, %0 : tensor<1520x3056xf64>
    return %1 : tensor<1520x3056xf64>
}

// CHECK: func.func @main1(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[[[1.000000e+00, -1.000000e+00]]]]> : tensor<1x1x1x2xf64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 3 : i64, lhs = 0 : i64, rhs = 235 : i64}> : (tensor<1x1x1520x3056xf64>) -> tensor<1x1x1520x3291xf64>
// CHECK-NEXT:    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {rhs_dilate = [1, 235]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x1520x3291xf64>, tensor<1x1x1x2xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1x1520x3056xf64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:    return %3 : tensor<1520x3056xf64>
// CHECK-NEXT:  }

func.func @main2(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : i32, dimension = 1 : i32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.subtract %0, %arg0 : tensor<1520x3056xf64>
    return %1 : tensor<1520x3056xf64>
}

// CHECK: func.func @main2(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[[[-1.000000e+00, 1.000000e+00]]]]> : tensor<1x1x1x2xf64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 3 : i64, lhs = 0 : i64, rhs = 235 : i64}> : (tensor<1x1x1520x3056xf64>) -> tensor<1x1x1520x3291xf64>
// CHECK-NEXT:    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {rhs_dilate = [1, 235]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x1520x3291xf64>, tensor<1x1x1x2xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1x1520x3056xf64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:    return %3 : tensor<1520x3056xf64>
// CHECK-NEXT:  }

func.func @main3(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<1520x3056xf64>
    %cst_2 = stablehlo.constant dense<-2.000000e+00> : tensor<1520x3056xf64>
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : i32, dimension = 1 : i32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.multiply %0, %cst_2 : tensor<1520x3056xf64>
    %2 = stablehlo.multiply %cst_1, %arg0 : tensor<1520x3056xf64>
    %3 = stablehlo.subtract %2, %1 : tensor<1520x3056xf64>
    return %3 : tensor<1520x3056xf64>
}

// CHECK: func.func @main3(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[[[5.000000e+00, 2.000000e+00]]]]> : tensor<1x1x1x2xf64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 3 : i64, lhs = 0 : i64, rhs = 235 : i64}> : (tensor<1x1x1520x3056xf64>) -> tensor<1x1x1520x3291xf64>
// CHECK-NEXT:    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {rhs_dilate = [1, 235]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x1520x3291xf64>, tensor<1x1x1x2xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1x1520x3056xf64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:    return %3 : tensor<1520x3056xf64>
// CHECK-NEXT:  }

func.func @main4(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<1520x3056xf64>
    %cst_2 = stablehlo.constant dense<-2.000000e+00> : tensor<1520x3056xf64>
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : i32, dimension = 1 : i32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.multiply %0, %cst_2 : tensor<1520x3056xf64>
    %2 = stablehlo.multiply %cst_1, %arg0 : tensor<1520x3056xf64>
    %3 = stablehlo.subtract %1, %2 : tensor<1520x3056xf64>
    return %3 : tensor<1520x3056xf64>
}

// CHECK: func.func @main4(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[[[-5.000000e+00, -2.000000e+00]]]]> : tensor<1x1x1x2xf64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 3 : i64, lhs = 0 : i64, rhs = 235 : i64}> : (tensor<1x1x1520x3056xf64>) -> tensor<1x1x1520x3291xf64>
// CHECK-NEXT:    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {rhs_dilate = [1, 235]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x1520x3291xf64>, tensor<1x1x1x2xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1x1520x3056xf64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:    return %3 : tensor<1520x3056xf64>
// CHECK-NEXT:  }

func.func @main5(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
    %cst_1 = stablehlo.constant dense<5.000000e+00> : tensor<1520x3056xf64>
    %0 = "enzymexla.rotate"(%arg0) <{amount = 235 : i32, dimension = 1 : i32}> : (tensor<1520x3056xf64>) -> tensor<1520x3056xf64>
    %1 = stablehlo.multiply %cst_1, %arg0 : tensor<1520x3056xf64>
    %2 = stablehlo.subtract %0, %1 : tensor<1520x3056xf64>
    return %2 : tensor<1520x3056xf64>
}

// CHECK: func.func @main5(%arg0: tensor<1520x3056xf64>) -> tensor<1520x3056xf64> {
// CHECK-NEXT{LITERAL}:    %cst = stablehlo.constant dense<[[[[-5.000000e+00, 1.000000e+00]]]]> : tensor<1x1x1x2xf64>
// CHECK-NEXT:    %0 = stablehlo.reshape %arg0 : (tensor<1520x3056xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:    %1 = "enzymexla.wrap"(%0) <{dimension = 3 : i64, lhs = 0 : i64, rhs = 235 : i64}> : (tensor<1x1x1520x3056xf64>) -> tensor<1x1x1520x3291xf64>
// CHECK-NEXT:    %2 = stablehlo.convolution(%1, %cst) dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1], window = {rhs_dilate = [1, 235]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x1520x3291xf64>, tensor<1x1x1x2xf64>) -> tensor<1x1x1520x3056xf64>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1x1520x3056xf64>) -> tensor<1520x3056xf64>
// CHECK-NEXT:    return %3 : tensor<1520x3056xf64>
// CHECK-NEXT:  }
