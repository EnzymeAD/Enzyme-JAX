// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt='passses=1049088' | FileCheck %s --check-prefix=LOWER

// CHECK-LABEL:   func.func @dus_dus_pad_pad(
// CHECK-SAME:                   %[[ARG0:.*]]: tensor<20x6144x12272xf64>) -> (tensor<20x6144x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>) {
// CHECK:           %[[VAL_0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_1:.*]] = stablehlo.broadcast %[[VAL_0]], sizes = [20, 6144, 12272] : (tensor<f64>) -> tensor<20x6144x12272xf64>
// CHECK:           %[[VAL_2:.*]] = "enzymexla.piecewise_select"(%[[VAL_1]], %[[ARG0]]) <{boxes = {{\[\[}}8, 12, 6136, 6137, 0, 12272], [7, 13, 8, 9, 0, 12272], [12, 13, 9, 6136, 0, 12272], [7, 8, 9, 6136, 0, 12272]]}> : (tensor<20x6144x12272xf64>, tensor<20x6144x12272xf64>) -> tensor<20x6144x12272xf64>
// CHECK:           %[[VAL_3:.*]] = stablehlo.slice %[[VAL_2]] [8:12, 7:6135, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6128x12272xf64>
// CHECK:           %[[VAL_4:.*]] = stablehlo.slice %[[VAL_2]] [8:12, 10:6138, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6128x12272xf64>
// CHECK:           %[[VAL_5:.*]] = stablehlo.slice %[[VAL_2]] [8:12, 6:6135, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
// CHECK:           %[[VAL_6:.*]] = stablehlo.slice %[[VAL_2]] [8:12, 7:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
// CHECK:           %[[VAL_7:.*]] = stablehlo.slice %[[VAL_2]] [8:12, 9:6138, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
// CHECK:           return %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %[[VAL_7]] : tensor<20x6144x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>

// LOWER-LABEL:   func.func @dus_dus_pad_pad(
// LOWER-SAME:                   %[[ARG0:.*]]: tensor<20x6144x12272xf64>) -> (tensor<20x6144x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>) {
// LOWER:           %[[VAL_0:.*]] = stablehlo.constant dense<true> : tensor<20x6127x12272xi1>
// LOWER:           %[[VAL_1:.*]] = stablehlo.constant dense<true> : tensor<1x6144x12272xi1>
// LOWER:           %[[VAL_2:.*]] = stablehlo.constant dense<true> : tensor<6x6144x12272xi1>
// LOWER:           %[[VAL_3:.*]] = stablehlo.constant dense<true> : tensor<20x1x12272xi1>
// LOWER:           %[[VAL_4:.*]] = stablehlo.constant dense<false> : tensor<i1>
// LOWER:           %[[VAL_5:.*]] = stablehlo.constant dense<true> : tensor<4x6144x12272xi1>
// LOWER:           %[[VAL_6:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// LOWER:           %[[VAL_7:.*]] = stablehlo.broadcast %[[VAL_6]], sizes = [20, 6144, 12272] : (tensor<f64>) -> tensor<20x6144x12272xf64>
// LOWER:           %[[VAL_8:.*]] = stablehlo.pad %[[VAL_5]], %[[VAL_4]], low = [8, 0, 0], high = [8, 0, 0], interior = [0, 0, 0] : (tensor<4x6144x12272xi1>, tensor<i1>) -> tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_9:.*]] = stablehlo.pad %[[VAL_3]], %[[VAL_4]], low = [0, 6136, 0], high = [0, 7, 0], interior = [0, 0, 0] : (tensor<20x1x12272xi1>, tensor<i1>) -> tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_10:.*]] = stablehlo.and %[[VAL_8]], %[[VAL_9]] : tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_11:.*]] = stablehlo.pad %[[VAL_2]], %[[VAL_4]], low = [7, 0, 0], high = [7, 0, 0], interior = [0, 0, 0] : (tensor<6x6144x12272xi1>, tensor<i1>) -> tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_12:.*]] = stablehlo.pad %[[VAL_3]], %[[VAL_4]], low = [0, 8, 0], high = [0, 6135, 0], interior = [0, 0, 0] : (tensor<20x1x12272xi1>, tensor<i1>) -> tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_13:.*]] = stablehlo.and %[[VAL_11]], %[[VAL_12]] : tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_14:.*]] = stablehlo.or %[[VAL_10]], %[[VAL_13]] : tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_15:.*]] = stablehlo.pad %[[VAL_1]], %[[VAL_4]], low = [12, 0, 0], high = [7, 0, 0], interior = [0, 0, 0] : (tensor<1x6144x12272xi1>, tensor<i1>) -> tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_16:.*]] = stablehlo.pad %[[VAL_0]], %[[VAL_4]], low = [0, 9, 0], high = [0, 8, 0], interior = [0, 0, 0] : (tensor<20x6127x12272xi1>, tensor<i1>) -> tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_17:.*]] = stablehlo.and %[[VAL_15]], %[[VAL_16]] : tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_18:.*]] = stablehlo.or %[[VAL_14]], %[[VAL_17]] : tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_19:.*]] = stablehlo.pad %[[VAL_1]], %[[VAL_4]], low = [7, 0, 0], high = [12, 0, 0], interior = [0, 0, 0] : (tensor<1x6144x12272xi1>, tensor<i1>) -> tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_20:.*]] = stablehlo.and %[[VAL_19]], %[[VAL_16]] : tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_21:.*]] = stablehlo.or %[[VAL_18]], %[[VAL_20]] : tensor<20x6144x12272xi1>
// LOWER:           %[[VAL_22:.*]] = stablehlo.select %[[VAL_21]], %[[VAL_7]], %[[ARG0]] : tensor<20x6144x12272xi1>, tensor<20x6144x12272xf64>
// LOWER:           %[[VAL_23:.*]] = stablehlo.slice %[[VAL_22]] [8:12, 7:6135, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6128x12272xf64>
// LOWER:           %[[VAL_24:.*]] = stablehlo.slice %[[VAL_22]] [8:12, 10:6138, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6128x12272xf64>
// LOWER:           %[[VAL_25:.*]] = stablehlo.slice %[[VAL_22]] [8:12, 6:6135, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
// LOWER:           %[[VAL_26:.*]] = stablehlo.slice %[[VAL_22]] [8:12, 7:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
// LOWER:           %[[VAL_27:.*]] = stablehlo.slice %[[VAL_22]] [8:12, 9:6138, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
// LOWER:           return %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_27]] : tensor<20x6144x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>, tensor<4x6129x12272xf64>
func.func @dus_dus_pad_pad(%iterArg_177: tensor<20x6144x12272xf64>) -> (tensor<20x6144x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6128x12272xf64>,  tensor<4x6129x12272xf64>,  tensor<4x6129x12272xf64>,  tensor<4x6129x12272xf64>){
    %cst_161 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c_169 = stablehlo.constant dense<7> : tensor<i32>
    %c_171 = stablehlo.constant dense<8> : tensor<i32>
    %c_172 = stablehlo.constant dense<0> : tensor<i32>
    %503 = stablehlo.slice %iterArg_177 [8:12, 9:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6127x12272xf64>
    %504 = stablehlo.pad %503, %cst_161, low = [0, 1, 0], high = [0, 1, 0], interior = [0, 0, 0] : (tensor<4x6127x12272xf64>, tensor<f64>) -> tensor<4x6129x12272xf64>
    %505 = stablehlo.dynamic_update_slice %iterArg_177, %504, %c_171, %c_171, %c_172 : (tensor<20x6144x12272xf64>, tensor<4x6129x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
    %506 = stablehlo.slice %505 [8:12, 7:6135, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6128x12272xf64>
    %509 = stablehlo.slice %505 [8:12, 10:6138, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6128x12272xf64>
    %2533 = stablehlo.slice %505 [8:12, 6:6135, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
    %2535 = stablehlo.slice %505 [8:12, 7:6136, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
    %2539 = stablehlo.slice %505 [8:12, 9:6138, 0:12272] : (tensor<20x6144x12272xf64>) -> tensor<4x6129x12272xf64>
    %513 = stablehlo.pad %503, %cst_161, low = [1, 1, 0], high = [1, 0, 0], interior = [0, 0, 0] : (tensor<4x6127x12272xf64>, tensor<f64>) -> tensor<6x6128x12272xf64>
    %514 = stablehlo.dynamic_update_slice %505, %513, %c_169, %c_171, %c_172 : (tensor<20x6144x12272xf64>, tensor<6x6128x12272xf64>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x6144x12272xf64>
    return %514, %506, %509, %2533, %2535, %2539 : tensor<20x6144x12272xf64>, tensor<4x6128x12272xf64>, tensor<4x6128x12272xf64>,  tensor<4x6129x12272xf64>,  tensor<4x6129x12272xf64>,  tensor<4x6129x12272xf64>
}
