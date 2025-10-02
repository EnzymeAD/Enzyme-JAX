// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(mark-func-memory-effects{assume_no_memory_effects=true})" %s | FileCheck %s --check-prefix=ASSUME
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(mark-func-memory-effects{assume_no_memory_effects=false})" %s | FileCheck %s --check-prefix=NOASSUME

module {
    func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
      %0 = stablehlo.sine %arg0 : tensor<f64>
      return %0 : tensor<f64>
    }

    // ASSUME: @main(%arg0: tensor<f64> {enzymexla.memory_effects = []}) -> tensor<f64> attributes {enzymexla.memory_effects = []} {
    // NOASSUME: @main(%arg0: tensor<f64> {enzymexla.memory_effects = []}) -> tensor<f64> attributes {enzymexla.memory_effects = []} {

  func.func private @kern$call$1(%arg0: !llvm.ptr<1>)

  func.func @main2(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %0 = enzymexla.jit_call @kern$call$1 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], xla_side_effect_free} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }

    // ASSUME: @main2(%arg0: tensor<64xi64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<64xi64> attributes {enzymexla.memory_effects = []} {
    // NOASSUME: @main2(%arg0: tensor<64xi64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<64xi64> attributes {enzymexla.memory_effects = []} {


  func.func @main3(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %0 = enzymexla.jit_call @kern$call$1 (%arg0) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }

    // ASSUME: @main3(%arg0: tensor<64xi64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<64xi64> attributes {enzymexla.memory_effects = []} {
    // NOASSUME: @main3(%arg0: tensor<64xi64> {enzymexla.memory_effects = ["read", "write", "allocate", "free"]}) -> tensor<64xi64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {

  func.func @main4(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %0 = stablehlo.custom_call @mycall1(%arg0) {has_side_effect = true} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }

    // ASSUME: @main4(%arg0: tensor<64xi64> {enzymexla.memory_effects = []}) -> tensor<64xi64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    // NOASSUME: func.func @main4(%arg0: tensor<64xi64> {enzymexla.memory_effects = []}) -> tensor<64xi64> attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {

  func.func @main5(%arg0: tensor<64xi64>) -> tensor<64xi64> {
    %0 = stablehlo.custom_call @mycall1(%arg0) {has_side_effect = false} : (tensor<64xi64>) -> tensor<64xi64>
    return %0 : tensor<64xi64>
  }

    // ASSUME: @main5(%arg0: tensor<64xi64> {enzymexla.memory_effects = []}) -> tensor<64xi64> attributes {enzymexla.memory_effects = []} {
    // NOASSUME: func.func @main5(%arg0: tensor<64xi64> {enzymexla.memory_effects = []}) -> tensor<64xi64> attributes {enzymexla.memory_effects = []} {


}



