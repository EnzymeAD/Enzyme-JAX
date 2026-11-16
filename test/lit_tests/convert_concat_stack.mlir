// RUN: enzymexlamlir-opt --split-input-file --pass-pipeline="any(enzyme-hlo-generate-td{patterns=convert_concat<1>;concat_insert_dim_elementwise;reorder_elementwise_and_shape_op<16>},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --pass-pipeline="any(enzyme-hlo-generate-td{patterns=convert_concat<1>;concat_insert_dim_elementwise;elementwise_reshape_like},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s
// RUN: enzymexlamlir-opt --split-input-file --pass-pipeline="any(enzyme-hlo-generate-td{patterns=convert_concat<1>},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=ALONE

// A convert of a stack (a concatenate of reshapes that insert the
// concatenated dimension) used to be split into one convert per operand.
// concat_insert_dim_elementwise batches the per-operand converts back into a
// single convert of the stack, and the reshape reordering patterns move each
// split convert below its reshape, so the three rewrites cycled forever. This
// is the shape a batched, unrolled loop produces, e.g. the i64 pivots of a
// batched LU converted to i32.

// CHECK-LABEL: func.func @stack
// CHECK-SAME:    %[[A:.+]]: tensor<4xi64>, %[[B:.+]]: tensor<4xi64>, %[[C:.+]]: tensor<4xi64>
// CHECK:         %[[CA:.+]] = stablehlo.custom_call @opaque(%[[A]])
// CHECK:         %[[CB:.+]] = stablehlo.custom_call @opaque(%[[B]])
// CHECK:         %[[CC:.+]] = stablehlo.custom_call @opaque(%[[C]])
// CHECK:         %[[RA:.+]] = stablehlo.reshape %[[CA]] : (tensor<4xi64>) -> tensor<1x4xi64>
// CHECK:         %[[RB:.+]] = stablehlo.reshape %[[CB]] : (tensor<4xi64>) -> tensor<1x4xi64>
// CHECK:         %[[RC:.+]] = stablehlo.reshape %[[CC]] : (tensor<4xi64>) -> tensor<1x4xi64>
// CHECK:         %[[CAT:.+]] = stablehlo.concatenate %[[RA]], %[[RB]], %[[RC]], dim = 0 : (tensor<1x4xi64>, tensor<1x4xi64>, tensor<1x4xi64>) -> tensor<3x4xi64>
// CHECK:         %[[CVT:.+]] = stablehlo.convert %[[CAT]] : (tensor<3x4xi64>) -> tensor<3x4xi32>
// CHECK:         %[[T:.+]] = stablehlo.transpose %[[CVT]], dims = [0, 1]
// CHECK:         return %[[T]]

// ALONE-LABEL: func.func @stack
// ALONE:         stablehlo.convert %{{.+}} : (tensor<4xi64>) -> tensor<4xi32>
// ALONE:         stablehlo.convert %{{.+}} : (tensor<4xi64>) -> tensor<4xi32>
// ALONE:         stablehlo.convert %{{.+}} : (tensor<4xi64>) -> tensor<4xi32>
// ALONE:         %[[CAT:.+]] = stablehlo.concatenate %{{.+}}, %{{.+}}, %{{.+}}, dim = 0 : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<3x4xi32>
// ALONE-NOT:     stablehlo.convert
// ALONE:         return %[[CAT]]
func.func @stack(%a: tensor<4xi64>, %b: tensor<4xi64>, %c: tensor<4xi64>) -> tensor<3x4xi32> {
  %0 = stablehlo.custom_call @opaque(%a) : (tensor<4xi64>) -> tensor<4xi64>
  %1 = stablehlo.custom_call @opaque(%b) : (tensor<4xi64>) -> tensor<4xi64>
  %2 = stablehlo.custom_call @opaque(%c) : (tensor<4xi64>) -> tensor<4xi64>
  %3 = stablehlo.convert %0 : (tensor<4xi64>) -> tensor<4xi32>
  %4 = stablehlo.convert %1 : (tensor<4xi64>) -> tensor<4xi32>
  %5 = stablehlo.convert %2 : (tensor<4xi64>) -> tensor<4xi32>
  %6 = stablehlo.reshape %3 : (tensor<4xi32>) -> tensor<1x4xi32>
  %7 = stablehlo.reshape %4 : (tensor<4xi32>) -> tensor<1x4xi32>
  %8 = stablehlo.reshape %5 : (tensor<4xi32>) -> tensor<1x4xi32>
  %9 = stablehlo.concatenate %6, %7, %8, dim = 0 : (tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<3x4xi32>
  return %9 : tensor<3x4xi32>
}

// -----

// A convert of a concatenate that is not a stack still splits.

// ALONE-LABEL: func.func @not_a_stack
// ALONE:         %[[CA:.+]] = stablehlo.convert %{{.+}} : (tensor<2x4xi64>) -> tensor<2x4xi32>
// ALONE:         %[[CB:.+]] = stablehlo.convert %{{.+}} : (tensor<1x4xi64>) -> tensor<1x4xi32>
// ALONE:         %[[CAT:.+]] = stablehlo.concatenate %[[CA]], %[[CB]], dim = 0
// ALONE:         return %[[CAT]]
// CHECK-LABEL: func.func @not_a_stack
// CHECK:         stablehlo.concatenate %{{.+}}, %{{.+}}, dim = 0 : (tensor<2x4xi32>, tensor<1x4xi32>) -> tensor<3x4xi32>
func.func @not_a_stack(%a: tensor<2x4xi64>, %b: tensor<4xi64>) -> tensor<3x4xi32> {
  %0 = stablehlo.reshape %b : (tensor<4xi64>) -> tensor<1x4xi64>
  %1 = stablehlo.concatenate %a, %0, dim = 0 : (tensor<2x4xi64>, tensor<1x4xi64>) -> tensor<3x4xi64>
  %2 = stablehlo.convert %1 : (tensor<3x4xi64>) -> tensor<3x4xi32>
  return %2 : tensor<3x4xi32>
}

// -----

// Reshapes that insert a dimension other than the concatenated one are not a
// stack along that dimension; the convert still splits.

// ALONE-LABEL: func.func @insert_other_dim
// ALONE:         stablehlo.convert %{{.+}} : (tensor<1x4xi64>) -> tensor<1x4xi32>
// ALONE:         stablehlo.convert %{{.+}} : (tensor<1x4xi64>) -> tensor<1x4xi32>
// ALONE:         stablehlo.concatenate %{{.+}}, %{{.+}}, dim = 1 : (tensor<1x4xi32>, tensor<1x4xi32>) -> tensor<1x8xi32>
func.func @insert_other_dim(%a: tensor<4xi64>, %b: tensor<4xi64>) -> tensor<1x8xi32> {
  %0 = stablehlo.reshape %a : (tensor<4xi64>) -> tensor<1x4xi64>
  %1 = stablehlo.reshape %b : (tensor<4xi64>) -> tensor<1x4xi64>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<1x4xi64>, tensor<1x4xi64>) -> tensor<1x8xi64>
  %3 = stablehlo.convert %2 : (tensor<1x8xi64>) -> tensor<1x8xi32>
  return %3 : tensor<1x8xi32>
}
