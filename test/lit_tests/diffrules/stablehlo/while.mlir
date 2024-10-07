// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// TODO: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func public @main(%arg0: tensor<3xf64>) -> (tensor<3xf64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %c_10 = stablehlo.constant dense<10> : tensor<i64>
    %c_1 = stablehlo.constant dense<1> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0) : tensor<i64>, tensor<3xf64>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_10,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %2 = stablehlo.add %iterArg, %c_1 : tensor<i64>
      stablehlo.return %2, %iterArg_0 : tensor<i64>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
// FORWARD-NEXT:    %c_0 = stablehlo.constant dense<10> : tensor<i64>
// FORWARD-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<i64>
// FORWARD-NEXT:    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %arg0, %iterArg_3 = %arg1) : tensor<i64>, tensor<3xf64>, tensor<3xf64>
// FORWARD-NEXT:     cond {
// FORWARD-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %c_0,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// FORWARD-NEXT:      stablehlo.return %1 : tensor<i1>
// FORWARD-NEXT:    } do {
// FORWARD-NEXT:      %1 = stablehlo.add %iterArg, %c_1 : tensor<i64>
// FORWARD-NEXT:      stablehlo.return %1, %iterArg_2, %iterArg_3 : tensor<i64>, tensor<3xf64>, tensor<3xf64>
// FORWARD-NEXT:    }
// FORWARD-NEXT:    return %0#1, %0#2 : tensor<3xf64>, tensor<3xf64>
// FORWARD-NEXT:  }

// FORWARD:  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<5xf32>, %arg3: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<[[3], [2], [1]]> : tensor<3x1xi32>
// FORWARD-NEXT:    %0:2 = "stablehlo.scatter"(%arg2, %arg3, %c, %arg0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// FORWARD-NEXT:    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
// FORWARD-NEXT:      stablehlo.return %arg6, %arg7 : tensor<f32>, tensor<f32>
// FORWARD-NEXT:    }) : (tensor<5xf32>, tensor<5xf32>, tensor<3x1xi32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<5xf32>, tensor<5xf32>)
// FORWARD-NEXT:    return %0#0, %0#1 : tensor<5xf32>, tensor<5xf32>
// FORWARD-NEXT:  }
