// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s --match-full-lines --implicit-check-not=scf.for

// Clone a public function. Insert two bound buffers before the data buffer.
// The two specialized scalars stay at the end and have no results. Move each
// input and result attribute with its value. Keep other wrapper attributes.
module {
  func.func @dynamic_attributes(%limit: i64, %step: i64,
                               %data: memref<i32>, %scale: i32, %bias: i32) {
    %zero = arith.constant 0 : i64
    scf.for %iv = %zero to %limit step %step : i64 {
      enzymexla.xla_wrapper @affine_update(%data, %scale, %bias)
          <{num_specialized = 2 : i64,
            arg_attrs = [{test.input = "data"}, {test.input = "scale"}, {test.input = "bias"}],
            res_attrs = [{test.output = "data"}]}>
          {test.wrapper = "keep"} : (memref<i32>, i32, i32) -> ()
    }
    return
  }

  func.func @affine_update(%data: tensor<i32> {test.input = "data"},
                          %scale: tensor<i32> {test.input = "scale"},
                          %bias: tensor<i32> {test.input = "bias"})
      -> (tensor<i32> {test.output = "data"}) attributes {test.function = "keep"} {
    %scaled = stablehlo.multiply %data, %scale : tensor<i32>
    %updated = stablehlo.add %scaled, %bias : tensor<i32>
    return %updated : tensor<i32>
  }
}

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @dynamic_attributes(%[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64, %[[VAL_2:.*]]: memref<i32>, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 8 : index
// CHECK-NEXT:     %[[ALLOCA_0:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT:     %[[ALLOCA_1:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT:     memref.store %[[VAL_0]], %[[ALLOCA_1]][] : memref<i64>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_0:.*]] = enzymexla.get_global_temp @rxla$megakernel_0_bound_1 : memref<i64, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_0]], %[[ALLOCA_1]], %[[CONSTANT_0]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT:     memref.store %[[VAL_1]], %[[ALLOCA_0]][] : memref<i64>
// CHECK-NEXT:     %[[GET_GLOBAL_TEMP_1:.*]] = enzymexla.get_global_temp @rxla$megakernel_0_bound_2 : memref<i64, 1>
// CHECK-NEXT:     enzymexla.memcpy  %[[GET_GLOBAL_TEMP_1]], %[[ALLOCA_0]], %[[CONSTANT_0]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT:     enzymexla.xla_wrapper @rxla$megakernel_0 (%[[GET_GLOBAL_TEMP_0]], %[[GET_GLOBAL_TEMP_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]]) <arg_attrs = [{}, {}, {test.input = "data"}, {test.input = "scale"}, {test.input = "bias"}], res_attrs = [{}, {}, {test.output = "data"}], num_specialized = 2> {test.wrapper = "keep"} : (memref<i64, 1>, memref<i64, 1>, memref<i32>, i32, i32) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @affine_update(%[[VAL_5:.*]]: tensor<i32> {test.input = "data"}, %[[VAL_6:.*]]: tensor<i32> {test.input = "scale"}, %[[VAL_7:.*]]: tensor<i32> {test.input = "bias"}) -> (tensor<i32> {test.output = "data"}) attributes {test.function = "keep"} {
// CHECK-NEXT:     %[[MULTIPLY_0:.*]] = stablehlo.multiply %[[VAL_5]], %[[VAL_6]] : tensor<i32>
// CHECK-NEXT:     %[[ADD_0:.*]] = stablehlo.add %[[MULTIPLY_0]], %[[VAL_7]] : tensor<i32>
// CHECK-NEXT:     return %[[ADD_0]] : tensor<i32>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @rxla$megakernel_0(%[[VAL_8:.*]]: tensor<i64>, %[[VAL_9:.*]]: tensor<i64>, %[[VAL_10:.*]]: tensor<i32> {test.input = "data"}, %[[VAL_11:.*]]: tensor<i32> {test.input = "scale"}, %[[VAL_12:.*]]: tensor<i32> {test.input = "bias"}) -> (tensor<i64>, tensor<i64>, tensor<i32> {test.output = "data"}) attributes {test.function = "keep"} {
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %[[WHILE_0:.*]]:2 = stablehlo.while(%[[VAL_13:.*]] = %[[CONSTANT_1]], %[[VAL_14:.*]] = %[[VAL_10]]) : tensor<i64>, tensor<i32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_13]], %[[VAL_8]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_1:.*]] = stablehlo.add %[[VAL_13]], %[[VAL_9]] : tensor<i64>
// CHECK-NEXT:       %[[MULTIPLY_1:.*]] = stablehlo.multiply %[[VAL_14]], %[[VAL_11]] : tensor<i32>
// CHECK-NEXT:       %[[ADD_2:.*]] = stablehlo.add %[[MULTIPLY_1]], %[[VAL_12]] : tensor<i32>
// CHECK-NEXT:       stablehlo.return %[[ADD_1]], %[[ADD_2]] : tensor<i64>, tensor<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_8]], %[[VAL_9]], %[[WHILE_0]]#1 : tensor<i64>, tensor<i64>, tensor<i32>
// CHECK-NEXT:   }
// CHECK-NEXT:   enzymexla.temp_alloc "private" @rxla$megakernel_0_bound_1 : memref<i64, 1>
// CHECK-NEXT:   enzymexla.temp_alloc "private" @rxla$megakernel_0_bound_2 : memref<i64, 1>
// CHECK-NEXT: }

// -----

// Reuse a private function with constant bounds. No new inputs or results
// are needed, so all attributes keep their original positions.
module {
  func.func @constant_attributes(%data: memref<i32>, %scale: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %four = arith.constant 4 : i32
    scf.for %iv = %zero to %four step %one : i32 {
      enzymexla.xla_wrapper @static_update(%data, %scale)
          <{num_specialized = 1 : i64,
            arg_attrs = [{test.input = "data"}, {test.input = "scale"}],
            res_attrs = [{test.output = "data"}]}>
          {test.wrapper = "keep"} : (memref<i32>, i32) -> ()
    }
    return
  }
  func.func private @static_update(%data: tensor<i32> {test.input = "data"},
                                   %scale: tensor<i32> {test.input = "scale"})
      -> (tensor<i32> {test.output = "data"}) {
    %updated = stablehlo.multiply %data, %scale : tensor<i32>
    return %updated : tensor<i32>
  }
}

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @constant_attributes(%[[VAL_0:.*]]: memref<i32>, %[[VAL_1:.*]]: i32) {
// CHECK-NEXT:     enzymexla.xla_wrapper @static_update (%[[VAL_0]], %[[VAL_1]]) <arg_attrs = [{test.input = "data"}, {test.input = "scale"}], res_attrs = [{test.output = "data"}], num_specialized = 1> {test.wrapper = "keep"} : (memref<i32>, i32) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @static_update(%[[VAL_2:.*]]: tensor<i32> {test.input = "data"}, %[[VAL_3:.*]]: tensor<i32> {test.input = "scale"}) -> (tensor<i32> {test.output = "data"}) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = stablehlo.constant dense<4> : tensor<i32>
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:     %[[WHILE_0:.*]]:2 = stablehlo.while(%[[VAL_4:.*]] = %[[CONSTANT_0]], %[[VAL_5:.*]] = %[[VAL_2]]) : tensor<i32>, tensor<i32>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %[[COMPARE_0:.*]] = stablehlo.compare LT, %[[VAL_4]], %[[CONSTANT_1]], SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %[[COMPARE_0]] : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[ADD_0:.*]] = stablehlo.add %[[VAL_4]], %[[CONSTANT_2]] : tensor<i32>
// CHECK-NEXT:       %[[MULTIPLY_0:.*]] = stablehlo.multiply %[[VAL_5]], %[[VAL_3]] : tensor<i32>
// CHECK-NEXT:       stablehlo.return %[[ADD_0]], %[[MULTIPLY_0]] : tensor<i32>, tensor<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[WHILE_0]]#1 : tensor<i32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
