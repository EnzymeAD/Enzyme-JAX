// RUN: enzymexlamlir-opt %s | FileCheck %s

// The body of a jit_region is an affine scope, so values defined at its top
// level, such as the size of an argument, are valid affine symbols.

// CHECK-LABEL: func.func @dim_symbol(
// CHECK:         "enzymexla.jit_region"
// CHECK:           %[[N:.+]] = memref.dim
// CHECK:           affine.for %{{.+}} = 0 to %[[N]] {
func.func @dim_symbol(%a: tensor<?xf32>) -> tensor<?xf32> {
  %r = "enzymexla.jit_region"(%a) <{
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%m: memref<?xf32>):
    %c0 = arith.constant 0 : index
    %n = memref.dim %m, %c0 : memref<?xf32>
    %cst = arith.constant 0.0 : f32
    affine.for %i = 0 to %n {
      affine.store %cst, %m[%i] : memref<?xf32>
    }
    enzymexla.jit_region_yield
  }) : (tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// A value loaded at the top level of the body is a valid symbol too.

// CHECK-LABEL: func.func @loaded_symbol(
// CHECK:         "enzymexla.jit_region"
// CHECK:           %[[N:.+]] = arith.index_cast
// CHECK:           affine.for %{{.+}} = 0 to %[[N]] {
func.func @loaded_symbol(%a: tensor<4xf32>, %b: tensor<i64>) -> tensor<4xf32> {
  %r = "enzymexla.jit_region"(%a, %b) <{
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>
    ]
  }> ({
  ^bb0(%m: memref<4xf32>, %bound: memref<i64>):
    %v = memref.load %bound[] : memref<i64>
    %n = arith.index_cast %v : i64 to index
    %cst = arith.constant 0.0 : f32
    affine.for %i = 0 to %n {
      affine.store %cst, %m[%i] : memref<4xf32>
    }
    enzymexla.jit_region_yield
  }) : (tensor<4xf32>, tensor<i64>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
