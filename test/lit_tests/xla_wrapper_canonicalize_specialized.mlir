// RUN: enzymexlamlir-opt %s --split-input-file --canonicalize --symbol-dce | FileCheck %s

// Remove the unused buffer and the first and last specialized scalars.
// Keep data and increment, in that order. Their attributes must follow them.
// Both wrappers can use the reduced private function, although their attributes
// differ. The function has one result because increment is a scalar input.
// Before: update(spare, data, unused, increment, unused_last) -> (spare, data + increment)
// After:  update(data, increment) -> data + increment
module {
  func.func @remove_unused_specialized(%spare: memref<i32>, %data: memref<i32>,
                                       %unused: i32, %increment: i32,
                                       %unused_last: i32) {
    enzymexla.xla_wrapper @update (%spare, %data, %unused, %increment, %unused_last)
        <{arg_attrs = [{test.drop}, {test.data}, {test.drop}, {test.increment}, {test.drop}],
          res_attrs = [{test.drop}, {test.result}], num_specialized = 3 : i64}> :
        (memref<i32>, memref<i32>, i32, i32, i32) -> ()
    enzymexla.xla_wrapper @update (%spare, %data, %unused, %increment, %unused_last)
        <{arg_attrs = [{test.drop}, {test.other_data}, {test.drop}, {test.other_increment}, {test.drop}],
          res_attrs = [{test.drop}, {test.other_result}], num_specialized = 3 : i64}> :
        (memref<i32>, memref<i32>, i32, i32, i32) -> ()
    return
  }
  func.func private @update(%spare: tensor<i32> {test.drop},
                            %data: tensor<i32> {test.function_data},
                            %unused: tensor<i32> {test.drop},
                            %increment: tensor<i32> {test.function_increment},
                            %unused_last: tensor<i32> {test.drop})
      -> (tensor<i32> {test.drop}, tensor<i32> {test.function_result}) {
    %sum = stablehlo.add %data, %increment : tensor<i32>
    return %spare, %sum : tensor<i32>, tensor<i32>
  }
}

// CHECK-LABEL: func.func @remove_unused_specialized(
// CHECK-SAME: %[[SPARE:[^:]+]]: memref<i32>, %[[DATA:[^:]+]]: memref<i32>, %[[UNUSED:[^:]+]]: i32, %[[INC:[^:]+]]: i32,
// CHECK-NEXT: enzymexla.xla_wrapper @update (%[[DATA]], %[[INC]]) <arg_attrs = [{test.data}, {test.increment}], res_attrs = [{test.result}], num_specialized = 1> : (memref<i32>, i32) -> ()
// CHECK-NEXT: enzymexla.xla_wrapper @update (%[[DATA]], %[[INC]]) <arg_attrs = [{test.other_data}, {test.other_increment}], res_attrs = [{test.other_result}], num_specialized = 1> : (memref<i32>, i32) -> ()
// CHECK-LABEL: func.func private @update(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<i32> {test.function_data}, %[[INC:[^:]+]]: tensor<i32> {test.function_increment}) -> (tensor<i32> {test.function_result}) {
// CHECK-NEXT: %[[SUM:[^ ]+]] = stablehlo.add %[[DATA]], %[[INC]] : tensor<i32>
// CHECK-NEXT: return %[[SUM]] : tensor<i32>

// -----

// Remove a specialized scalar without removing a buffer. The public function
// can have unknown callers, so keep it and give the wrapper a reduced clone.
module {
  func.func @remove_only_scalar(%data: memref<i32>, %unused: i32) {
    enzymexla.xla_wrapper @negate (%data, %unused)
        <{num_specialized = 1 : i64}> : (memref<i32>, i32) -> ()
    return
  }
  func.func @negate(%data: tensor<i32>, %unused: tensor<i32>) -> tensor<i32> {
    %negative = stablehlo.negate %data : tensor<i32>
    return %negative : tensor<i32>
  }
}

// CHECK-LABEL: func.func @remove_only_scalar(
// CHECK-SAME: %[[DATA:[^:]+]]: memref<i32>,
// CHECK-NEXT: enzymexla.xla_wrapper @negate_without_unused (%[[DATA]]) : (memref<i32>) -> ()
// CHECK-LABEL: func.func @negate(
// CHECK-SAME: tensor<i32>, {{.*}}: tensor<i32>) -> tensor<i32>
// CHECK-LABEL: func.func private @negate_without_unused(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<i32>) -> tensor<i32> {
// CHECK-NEXT: %[[NEG:[^ ]+]] = stablehlo.negate %[[DATA]] : tensor<i32>
// CHECK-NEXT: return %[[NEG]] : tensor<i32>

// -----

// Both buffer positions write the same specialized scalar to the same buffer.
// Remove the second buffer and its attributes. Keep both scalar positions and
// their attributes, even though the wrapper passes the same scalar to both.
// The first scalar supplies the result; the side effect reads the second.
// Before: write(data, data, value, value) -> (value, value)
// After:  write_without_duplicates(data, value, value) -> value
module {
  func.func @deduplicate_with_specialized(%data: memref<i32>, %value: i32) {
    enzymexla.xla_wrapper @write (%data, %data, %value, %value)
        <{arg_attrs = [{test.buffer}, {test.buffer}, {test.value}, {test.effect}],
          res_attrs = [{test.result}, {test.result}], num_specialized = 2 : i64}> :
        (memref<i32>, memref<i32>, i32, i32) -> ()
    return
  }
  func.func private @write(%first: tensor<i32> {test.function_buffer},
                           %second: tensor<i32> {test.function_buffer},
                           %value: tensor<i32> {test.function_value},
                           %effect: tensor<i32> {test.function_effect})
      -> (tensor<i32> {test.function_result}, tensor<i32> {test.function_result}) {
    stablehlo.custom_call @record(%effect) {has_side_effect = true} : (tensor<i32>) -> ()
    return %value, %value : tensor<i32>, tensor<i32>
  }
}

// CHECK-LABEL: func.func @deduplicate_with_specialized(
// CHECK-SAME: %[[DATA:[^:]+]]: memref<i32>, %[[VALUE:[^:]+]]: i32) {
// CHECK-NEXT: enzymexla.xla_wrapper @write_without_duplicates (%[[DATA]], %[[VALUE]], %[[VALUE]]) <arg_attrs = [{test.buffer}, {test.value}, {test.effect}], res_attrs = [{test.result}], num_specialized = 2> : (memref<i32>, i32, i32) -> ()
// CHECK-LABEL: func.func private @write_without_duplicates(
// CHECK-SAME: %[[DATA:[^:]+]]: tensor<i32> {test.function_buffer}, %[[VALUE:[^:]+]]: tensor<i32> {test.function_value}, %[[EFFECT:[^:]+]]: tensor<i32> {test.function_effect}) -> (tensor<i32> {test.function_result}) {
// CHECK-NEXT: stablehlo.custom_call @record(%[[EFFECT]]) {has_side_effect = true} : (tensor<i32>) -> ()
// CHECK-NEXT: return %[[VALUE]] : tensor<i32>

// -----

// Keep both buffer positions when their attributes differ. Do not choose one
// position's attributes and discard the other position's attributes.
module {
  func.func @keep_different_attributes(%data: memref<i32>) {
    enzymexla.xla_wrapper @pair (%data, %data)
        <{arg_attrs = [{test.first}, {test.second}]}> : (memref<i32>, memref<i32>) -> ()
    enzymexla.xla_wrapper @pair (%data, %data)
        <{res_attrs = [{test.first}, {test.second}]}> : (memref<i32>, memref<i32>) -> ()
    enzymexla.xla_wrapper @arg_attributes (%data, %data) : (memref<i32>, memref<i32>) -> ()
    enzymexla.xla_wrapper @result_attributes (%data, %data) : (memref<i32>, memref<i32>) -> ()
    return
  }
  func.func private @pair(%a: tensor<i32>, %b: tensor<i32>)
      -> (tensor<i32>, tensor<i32>) {
    %sum = stablehlo.add %a, %b : tensor<i32>
    return %sum, %sum : tensor<i32>, tensor<i32>
  }
  func.func private @arg_attributes(%a: tensor<i32> {test.first},
                                    %b: tensor<i32> {test.second})
      -> (tensor<i32>, tensor<i32>) {
    %sum = stablehlo.add %a, %b : tensor<i32>
    return %sum, %sum : tensor<i32>, tensor<i32>
  }
  func.func private @result_attributes(%a: tensor<i32>, %b: tensor<i32>)
      -> (tensor<i32> {test.first}, tensor<i32> {test.second}) {
    %sum = stablehlo.add %a, %b : tensor<i32>
    return %sum, %sum : tensor<i32>, tensor<i32>
  }
}

// CHECK-LABEL: func.func @keep_different_attributes(
// CHECK-SAME: %[[DATA:[^:]+]]: memref<i32>) {
// CHECK-NEXT: enzymexla.xla_wrapper @pair (%[[DATA]], %[[DATA]]) <arg_attrs = [{test.first}, {test.second}]> : (memref<i32>, memref<i32>) -> ()
// CHECK-NEXT: enzymexla.xla_wrapper @pair (%[[DATA]], %[[DATA]]) <res_attrs = [{test.first}, {test.second}]> : (memref<i32>, memref<i32>) -> ()
// CHECK-NEXT: enzymexla.xla_wrapper @arg_attributes (%[[DATA]], %[[DATA]]) : (memref<i32>, memref<i32>) -> ()
// CHECK-NEXT: enzymexla.xla_wrapper @result_attributes (%[[DATA]], %[[DATA]]) : (memref<i32>, memref<i32>) -> ()
// CHECK-NEXT: return
