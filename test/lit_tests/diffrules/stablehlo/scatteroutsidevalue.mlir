// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func @main(%operand: tensor<126xf64>, %indices: tensor<25x1xi64>, %fixed: tensor<f64>) -> tensor<126xf64> {
    %updates = stablehlo.constant dense<0.000000e+00> : tensor<25xf64>
    %r = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%old: tensor<f64>, %new: tensor<f64>):
      stablehlo.return %fixed : tensor<f64>
    }) : (tensor<126xf64>, tensor<25x1xi64>, tensor<25xf64>) -> tensor<126xf64>
    return %r : tensor<126xf64>
  }
}

// REVERSE-LABEL: func.func @main(
// REVERSE-SAME:      %[[OPERAND:.+]]: tensor<126xf64>, %[[INDICES:.+]]: tensor<25x1xi64>, %[[FIXED:.+]]: tensor<f64>, %[[SEED:.+]]: tensor<126xf64>) -> tensor<126xf64> {
// REVERSE-NEXT:    %[[ZSCALAR:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// REVERSE-NEXT:    %[[ZUPD:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<25xf64>
// REVERSE-NEXT:    %[[DOP:.+]] = "stablehlo.scatter"(%[[SEED]], %[[INDICES]], %[[ZUPD]]) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
// REVERSE-NEXT:    ^bb0(%[[A:.+]]: tensor<f64>, %[[B:.+]]: tensor<f64>):
// REVERSE-NEXT:      stablehlo.return %[[ZSCALAR]] : tensor<f64>
// REVERSE-NEXT:    }) : (tensor<126xf64>, tensor<25x1xi64>, tensor<25xf64>) -> tensor<126xf64>
// REVERSE-NEXT:    return %[[DOP]] : tensor<126xf64>
// REVERSE-NEXT:  }
// REVERSE-NOT:   stablehlo.gather
