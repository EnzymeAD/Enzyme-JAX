// RUN: enzymexlamlir-opt --fuse-jit="dag=true" %s | FileCheck %s

// Inputs computed between the calls move above the group, while a call reading
// a value computed from a result of the group stays out of it. The others still
// fuse across it when XLA may reorder it, but not when it may have effects.

module {
  llvm.func private @k(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) {
    %0 = llvm.load %arg1 : !llvm.ptr<1> -> f32
    llvm.store %0, %arg0 : f32, !llvm.ptr<1>
    llvm.return
  }
  func.func @reorderable(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>, %z: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
    %0 = enzymexla.jit_call @k(%x, %y) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ], xla_side_effect_free
    > : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %t = stablehlo.transpose %y, dims = [1, 0] : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = enzymexla.jit_call @k(%0, %t) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ], xla_side_effect_free
    > : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %n = stablehlo.negate %1 : tensor<4x4xf32>
    %2 = enzymexla.jit_call @k(%n, %y) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ], xla_side_effect_free
    > : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = enzymexla.jit_call @k(%z, %1) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ], xla_side_effect_free
    > : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %2, %3 : tensor<4x4xf32>, tensor<4x4xf32>
  }
  func.func @effectful(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>, %z: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
    %0 = enzymexla.jit_call @k(%x, %y) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ], xla_side_effect_free
    > : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = enzymexla.jit_call @k(%0, %y) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ], xla_side_effect_free
    > : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %n = stablehlo.negate %1 : tensor<4x4xf32>
    %2 = enzymexla.jit_call @k(%n, %y) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ]
    > : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = enzymexla.jit_call @k(%z, %1) <
      output_operand_aliases = [
        #stablehlo.output_operand_alias<output_tuple_indices = [],
                                        operand_index = 0,
                                        operand_tuple_indices = []>
      ], xla_side_effect_free
    > : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %2, %3 : tensor<4x4xf32>, tensor<4x4xf32>
  }
}

// CHECK-LABEL: func.func @reorderable(
// CHECK-SAME:      %[[X:[^:]+]]: tensor<4x4xf32>, %[[Y:[^:]+]]: tensor<4x4xf32>, %[[Z:[^:]+]]: tensor<4x4xf32>)
// CHECK-NEXT:    %[[T:.+]] = stablehlo.transpose %[[Y]], dims = [1, 0]
// CHECK-NEXT:    %[[F:.+]]:2 = enzymexla.jit_call @fused__k_k_k (%[[X]], %[[Y]], %[[T]], %[[Z]])
// CHECK-NEXT:    %[[N:.+]] = stablehlo.negate %[[F]]#0
// CHECK-NEXT:    %[[R:.+]] = enzymexla.jit_call @k (%[[N]], %[[Y]])
// CHECK-NEXT:    return %[[R]], %[[F]]#1

// CHECK-LABEL: func.func @effectful(
// CHECK-SAME:      %[[X:[^:]+]]: tensor<4x4xf32>, %[[Y:[^:]+]]: tensor<4x4xf32>, %[[Z:[^:]+]]: tensor<4x4xf32>)
// CHECK-NEXT:    %[[F:.+]] = enzymexla.jit_call @fused__k_k (%[[X]], %[[Y]])
// CHECK-NEXT:    %[[N:.+]] = stablehlo.negate %[[F]]
// CHECK-NEXT:    %[[R:.+]] = enzymexla.jit_call @k (%[[N]], %[[Y]])
// CHECK-NEXT:    %[[S:.+]] = enzymexla.jit_call @k (%[[Z]], %[[F]])
// CHECK-NEXT:    return %[[R]], %[[S]]
