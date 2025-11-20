// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

func.func @gather_constprop1(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<6x4xf64>) -> tensor<6x4xf64> {
    %c = stablehlo.constant dense<1> : tensor<24x2xi64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1024x1024xf64>
    %0 = stablehlo.concatenate %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, dim = 0 : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<24xi64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<6xi64>) -> tensor<6x4xi64>
    %2 = stablehlo.reshape %0 : (tensor<24xi64>) -> tensor<24x1xi64>
    %3 = stablehlo.reshape %1 : (tensor<6x4xi64>) -> tensor<24x1xi64>
    %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<24x1xi64>, tensor<24x1xi64>) -> tensor<24x2xi64>
    %5 = stablehlo.subtract %4, %c : tensor<24x2xi64>
    %6 = "stablehlo.gather"(%cst, %5) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<1024x1024xf64>, tensor<24x2xi64>) -> tensor<24xf64>
    %7 = stablehlo.reshape %6 : (tensor<24xf64>) -> tensor<6x4xf64>
    %8 = stablehlo.multiply %7, %arg2 : tensor<6x4xf64>
    return %8 : tensor<6x4xf64>
}

// CHECK: func.func @gather_constprop1(%arg0: tensor<4xi64>, %arg1: tensor<6xi64>, %arg2: tensor<6x4xf64>) -> tensor<6x4xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant {enzymexla.guaranteed_finite = [#enzymexla<guaranteed GUARANTEED>]} dense<0.000000e+00> : tensor<6x4xf64>
// CHECK-NEXT:     %0 = stablehlo.multiply %cst, %arg2 {enzymexla.guaranteed_no_nan = [#enzymexla<guaranteed NOTGUARANTEED>]} : tensor<6x4xf64>
// CHECK-NEXT:     return %0 : tensor<6x4xf64>
// CHECK-NEXT: }

func.func @gather_constprop2(%arg0: tensor<6x4xf64>) -> tensor<6x4xf64> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<1024x1024xf64>
    %c = stablehlo.constant dense<[[0, 0], [3, 0], [5, 0], [11, 0], [0, 31], [3, 31], [5, 31], [11, 31], [0, 32], [3, 32], [5, 32], [11, 32], [0, 34], [3, 34], [5, 34], [11, 34], [0, 11], [3, 11], [5, 11], [11, 11], [0, 110], [3, 110], [5, 110], [11, 110]]> : tensor<24x2xi64>
    %0 = "stablehlo.gather"(%cst, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<1024x1024xf64>, tensor<24x2xi64>) -> tensor<24xf64>
    %1 = stablehlo.reshape %0 : (tensor<24xf64>) -> tensor<6x4xf64>
    return %1 : tensor<6x4xf64>
}

// CHECK: func.func @gather_constprop2(%arg0: tensor<6x4xf64>) -> tensor<6x4xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<6x4xf64>
// CHECK-NEXT:     return %cst : tensor<6x4xf64>
// CHECK-NEXT: }

func.func @gather_constprop3(%arg0: tensor<6x4xf64>) -> tensor<6x4xf64> {
    %cst = stablehlo.constant dense<"0x0000803F000050410000C841000014420000444200007442000092420000AA420000C2420000DA420000F2420000054300000040000060410000D041000018420000484200007842000094420000AC420000C4420000DC420000F4420000064300004040000070410000D84100001C4200004C4200007C42000096420000AE420000C6420000DE420000F6420000074300008040000080410000E041000020420000504200008042000098420000B0420000C8420000E0420000F842000008430000A040000088410000E84100002442000054420000824200009A420000B2420000CA420000E2420000FA42000009430000C040000090410000F04100002842000058420000844200009C420000B4420000CC420000E4420000FC4200000A430000E040000098410000F84100002C4200005C420000864200009E420000B6420000CE420000E6420000FE4200000B43000000410000A041000000420000304200006042000088420000A0420000B8420000D0420000E8420000004300000C43000010410000A84100000442000034420000644200008A420000A2420000BA420000D2420000EA420000014300000D43000020410000B04100000842000038420000684200008C420000A4420000BC420000D4420000EC420000024300000E43000030410000B84100000C4200003C4200006C4200008E420000A6420000BE420000D6420000EE420000034300000F43000040410000C041000010420000404200007042000090420000A8420000C0420000D8420000F0420000044300001043"> : tensor<12x12xf32>
    %c = stablehlo.constant dense<[[0, 0], [3, 0], [5, 0], [11, 0], [0, 1], [3, 1], [5, 1], [11, 1], [0, 2], [3, 2], [5, 2], [11, 2], [0, 4], [3, 4], [5, 4], [11, 4], [0, 11], [3, 11], [5, 11], [11, 11], [0, 10], [3, 10], [5, 10], [11, 10]]> : tensor<24x2xi64>
    %0 = "stablehlo.gather"(%cst, %c) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1>}> : (tensor<12x12xf32>, tensor<24x2xi64>) -> tensor<24xf32>
    %1 = stablehlo.convert %0 : (tensor<24xf32>) -> tensor<24xf64>
    %2 = stablehlo.reshape %1 : (tensor<24xf64>) -> tensor<6x4xf64>
    %3 = stablehlo.multiply %2, %arg0 : tensor<6x4xf64>
    return %3 : tensor<6x4xf64>
}

// CHECK: func.func @gather_constprop3(%arg0: tensor<6x4xf64>) -> tensor<6x4xf64> {
// CHECK-NEXT{LITERAL}:     %cst = stablehlo.constant dense<[[1.000000e+00, 4.000000e+00, 6.000000e+00, 1.200000e+01], [1.300000e+01, 1.600000e+01, 1.800000e+01, 2.400000e+01], [2.500000e+01, 2.800000e+01, 3.000000e+01, 3.600000e+01], [4.900000e+01, 5.200000e+01, 5.400000e+01, 6.000000e+01], [1.330000e+02, 1.360000e+02, 1.380000e+02, 1.440000e+02], [1.210000e+02, 1.240000e+02, 1.260000e+02, 1.320000e+02]]> : tensor<6x4xf64>
// CHECK-NEXT:     %0 = stablehlo.multiply %cst, %arg0 : tensor<6x4xf64>
// CHECK-NEXT:     return %0 : tensor<6x4xf64>
// CHECK-NEXT: }
