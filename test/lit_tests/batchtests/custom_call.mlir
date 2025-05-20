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
// CHECK-NEXT:     %c = stablehlo.constant dense<3> : tensor<i64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<2> : tensor<i64>
// CHECK-NEXT:     %c_1 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<6> : tensor<i64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %cst = arith.constant dense<0.000000e+00> : tensor<2x3x64x64xf32>
// CHECK-NEXT:     %0:2 = stablehlo.while(%iterArg = %c_3, %iterArg_4 = %cst) : tensor<i64>, tensor<2x3x64x64xf32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:       %1 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %1 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %1 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// CHECK-NEXT:       %2 = stablehlo.remainder %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:       %3 = stablehlo.divide %iterArg, %c_0 : tensor<i64>
// CHECK-NEXT:       %4 = stablehlo.remainder %3, %c : tensor<i64>
// CHECK-NEXT:       %5 = stablehlo.dynamic_slice %arg0, %2, %4, %c_3, %c_3, sizes = [1, 1, 64, 64] : (tensor<2x3x64x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x64x64xf32>
// CHECK-NEXT:       %6 = stablehlo.reshape %5 : (tensor<1x1x64x64xf32>) -> tensor<64x64xf32>
// CHECK-NEXT:       %7:3 = stablehlo.custom_call @cusolver_getrf_ffi(%6) {api_version = 4 : i32, operand_layouts = [dense<[1, 0]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>]} : (tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64xi32>, tensor<i32>)
// CHECK-NEXT:       %8 = stablehlo.reshape %7#0 : (tensor<64x64xf32>) -> tensor<1x1x64x64xf32>
// CHECK-NEXT:       %9 = stablehlo.dynamic_update_slice %iterArg_4, %8, %2, %4, %c_3, %c_3 : (tensor<2x3x64x64xf32>, tensor<1x1x64x64xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<2x3x64x64xf32>
// CHECK-NEXT:       stablehlo.return %1, %9 : tensor<i64>, tensor<2x3x64x64xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : tensor<2x3x64x64xf32>
// CHECK-NEXT: }
