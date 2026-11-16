// RUN: enzymexlamlir-opt --split-input-file --verify-diagnostics %s

func.func @valid(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %r:2 = "enzymexla.jit_region"(%a, %b) <{
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = []>,
      #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 0, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%ma: memref<4xf32, 1>, %mb: memref<4xf32>):
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  return %r#0, %r#1 : tensor<4xf32>, tensor<4xf32>
}

// -----

func.func @block_argument_count(%a: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{expects one block argument per input, got 0 for 1 inputs}}
  %r = "enzymexla.jit_region"(%a) <{
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
    ]
  }> ({
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

func.func @block_argument_shape(%a: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{block argument #0 of type 'memref<3xf32>' cannot hold input of type 'tensor<4xf32>'}}
  %r = "enzymexla.jit_region"(%a) <{
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%m: memref<3xf32>):
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

func.func @input_aliased_twice(%a: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // expected-error @+1 {{input #0 is aliased by more than one result}}
  %r:2 = "enzymexla.jit_region"(%a) <{
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>,
      #stablehlo.output_operand_alias<output_tuple_indices = [1], operand_index = 0, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%m: memref<4xf32>):
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  return %r#0, %r#1 : tensor<4xf32>, tensor<4xf32>
}

// -----

func.func @operand_attrs_count(%a: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{expects one operand_attrs entry per input, got 2 for 1 inputs}}
  %r = "enzymexla.jit_region"(%a) <{
    operand_attrs = [{}, {}],
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%m: memref<4xf32>):
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----

func.func @alias_out_of_range(%a: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{output_operand_alias refers to operand 1, but there are only 1 inputs}}
  %r = "enzymexla.jit_region"(%a) <{
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 1, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%m: memref<4xf32>):
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
