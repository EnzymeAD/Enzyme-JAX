// RUN: enzymexlamlir-opt %s --enzyme-batch --enzyme-hlo-opt | FileCheck %s

func.func @custom_call(%arg0: tensor<64x64xf32>) -> (tensor<64x64xf32>) {
    %c = stablehlo.constant dense<1> : tensor<64xi32>
    %0:3 = stablehlo.custom_call @cusolver_getrf_ffi(%arg0) {api_version = 4 : i32, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>, dense<[]> : tensor<0xindex>]} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>)
    return %0#0 : tensor<64x64xf32>
}

func.func @main(%arg0: tensor<2x3x64x64xf32>) -> (tensor<2x3x64x64xf32>) {
    %0 = enzyme.batch @custom_call(%arg0) {batch_shape = array<i64: 2, 3>} : (tensor<2x3x64x64xf32>) -> tensor<2x3x64x64xf32>
    return %0 : tensor<2x3x64x64xf32>
}

// CHECK: func.func private @batched_custom_call(%arg0: tensor<2x3x64x64xf32>) -> tensor<2x3x64x64xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : tensor<2x3x64x64xf32>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_3 = %cst) : tensor<i64>, tensor<2x3x64x64xf32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %1 = stablehlo.compare  LT, %iterArg, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.remainder %iterArg, %c : tensor<i64>
// CHECK-NEXT:       %3 = stablehlo.divide %iterArg, %c : tensor<i64>
// CHECK-NEXT:       %4 = stablehlo.dynamic_slice %arg0, %2, %3, %c_2, %c_2, sizes = [1, 1, 64, 64] : (tensor<2x3x64x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x64x64xf32>
// CHECK-NEXT:       %5 = stablehlo.reshape %4 : (tensor<1x1x64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:       %6:3 = stablehlo.custom_call @cusolver_getrf_ffi(%5) {api_version = 4 : i32, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>)
// CHECK-NEXT:       %7 = stablehlo.reshape %6#0 : (tensor<64x64xf32>) -> tensor<1x1x64x64xf32>
// CHECK-NEXT:       %8 = stablehlo.dynamic_update_slice %iterArg_3, %7, %2, %3, %c_2, %c_2 : (tensor<2x3x64x64xf32>, tensor<1x1x64x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x3x64x64xf32>
// CHECK-NEXT:       stablehlo.return %1, %8 : tensor<i64>, tensor<2x3x64x64xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<2x3x64x64xf32>
// CHECK-NEXT:   }
