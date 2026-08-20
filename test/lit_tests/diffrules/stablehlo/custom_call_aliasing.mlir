// RUN: not enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" 2>&1 | FileCheck %s

// A result that aliases an operand may have been overwritten in place by the
// time the reverse pass runs, which would make the cached primal result -- and
// therefore the gradient -- meaningless. Refuse rather than cache it.

func.func @scale_rev(%x: tensor<4xf32>, %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> tensor<4xf32> {
  %three = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
  %res = stablehlo.multiply %dy, %three : tensor<4xf32>
  func.return %res : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>,
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [],
                                      operand_index = 0,
                                      operand_tuple_indices = []>
    ]
  } : (tensor<4xf32>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: cannot differentiate a custom call with output_operand_aliasing
