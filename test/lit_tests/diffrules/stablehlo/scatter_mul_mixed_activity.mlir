// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=mul_scatter_1 outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=ACT_CONST
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=mul_scatter_1 outfn= retTys=enzyme_active argTys=enzyme_const,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=CONST_ACT
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=mul_scatter_variadic outfn= retTys=enzyme_active,enzyme_active argTys=enzyme_active,enzyme_const,enzyme_const,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=VARIADIC
// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

module {
  func.func @mul_scatter_1(%inp: tensor<4xf64>, %upd: tensor<2xf64>) -> tensor<4xf64> {
    %idx = stablehlo.constant dense<[[1], [3]]> : tensor<2x1xi32>
    %0 = "stablehlo.scatter"(%inp, %idx, %upd) <{
      indices_are_sorted = true,
      scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>,
      unique_indices = true
    }> ({
    ^bb0(%a: tensor<f64>, %b: tensor<f64>):
      %m = stablehlo.multiply %a, %b : tensor<f64>
      stablehlo.return %m : tensor<f64>
    }) : (tensor<4xf64>, tensor<2x1xi32>, tensor<2xf64>) -> tensor<4xf64>
    return %0 : tensor<4xf64>
  }

  func.func @mul_scatter_variadic(
      %inp0: tensor<4xf64>, %inp1: tensor<4xf64>,
      %upd0: tensor<2xf64>, %upd1: tensor<2xf64>
  ) -> (tensor<4xf64>, tensor<4xf64>) {
    %idx = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi32>
    %0:2 = "stablehlo.scatter"(%inp0, %inp1, %idx, %upd0, %upd1) <{
      indices_are_sorted = true,
      scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>,
      unique_indices = true
    }> ({
    ^bb0(%a0: tensor<f64>, %a1: tensor<f64>, %b0: tensor<f64>, %b1: tensor<f64>):
      %m0 = stablehlo.multiply %a0, %b0 : tensor<f64>
      %m1 = stablehlo.multiply %a1, %b1 : tensor<f64>
      stablehlo.return %m0, %m1 : tensor<f64>, tensor<f64>
    }) : (tensor<4xf64>, tensor<4xf64>, tensor<2x1xi32>, tensor<2xf64>, tensor<2xf64>) -> (tensor<4xf64>, tensor<4xf64>)
    return %0#0, %0#1 : tensor<4xf64>, tensor<4xf64>
  }

  func.func @main() {
    %inp0 = stablehlo.constant dense<[2.0, 3.0, 5.0, 7.0]> : tensor<4xf64>
    %inp1 = stablehlo.constant dense<[11.0, 13.0, 17.0, 19.0]> : tensor<4xf64>
    %upd0 = stablehlo.constant dense<[4.0, 6.0]> : tensor<2xf64>
    %upd1 = stablehlo.constant dense<[8.0, 9.0]> : tensor<2xf64>
    %dy0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>
    %dy1 = stablehlo.constant dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf64>

    // Single-input active operand, const update: dinp0 = [1*1, 2*4, 3*1, 4*6]? idx=[1,3] -> [1.0, 8.0, 3.0, 24.0]
    %dinp = enzyme.autodiff @mul_scatter_1(%inp0, %upd0, %dy0) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<4xf64>, tensor<2xf64>, tensor<4xf64>) -> tensor<4xf64>
    check.expect_almost_eq_const %dinp, dense<[1.0, 8.0, 3.0, 24.0]> : tensor<4xf64>

    // Single-input const operand, active update: dupd = [dy0[1]*inp0[1], dy0[3]*inp0[3]] = [2*3, 4*7] = [6.0, 28.0]
    %dupd = enzyme.autodiff @mul_scatter_1(%inp0, %upd0, %dy0) <{
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<4xf64>, tensor<2xf64>, tensor<4xf64>) -> tensor<2xf64>
    check.expect_almost_eq_const %dupd, dense<[6.0, 28.0]> : tensor<2xf64>

    // Variadic 2-input mixed activity:
    // inp0 active, inp1 const, upd0 const, upd1 active (idx = [0, 2])
    // dinp0 = [dy0[0]*upd0[0], dy0[1], dy0[2]*upd0[1], dy0[3]] = [1*4, 2, 3*6, 4] = [4.0, 2.0, 18.0, 4.0]
    // dupd1 = [dy1[0]*inp1[0], dy1[2]*inp1[2]] = [10*11, 30*17] = [110.0, 510.0]
    %vgrads:2 = enzyme.autodiff @mul_scatter_variadic(%inp0, %inp1, %upd0, %upd1, %dy0, %dy1) <{
      activity = [
        #enzyme.activity<enzyme_active>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_const>,
        #enzyme.activity<enzyme_active>
      ],
      ret_activity = [
        #enzyme.activity<enzyme_activenoneed>,
        #enzyme.activity<enzyme_activenoneed>
      ]
    }> : (tensor<4xf64>, tensor<4xf64>, tensor<2xf64>, tensor<2xf64>, tensor<4xf64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<2xf64>)
    check.expect_almost_eq_const %vgrads#0, dense<[4.0, 2.0, 18.0, 4.0]> : tensor<4xf64>
    check.expect_almost_eq_const %vgrads#1, dense<[110.0, 510.0]> : tensor<2xf64>
    return
  }
}

// ACT_CONST-LABEL: func.func @mul_scatter_1
// ACT_CONST: "stablehlo.scatter"(%arg2, {{%.*}}, %arg1)
// ACT_CONST-NOT: stablehlo.gather

// CONST_ACT-LABEL: func.func @mul_scatter_1
// CONST_ACT: %[[MUL:.*]] = stablehlo.multiply %arg2, %arg0 : tensor<4xf64>
// CONST_ACT: "stablehlo.gather"(%[[MUL]], {{%.*}})

// VARIADIC-LABEL: func.func @mul_scatter_variadic
// VARIADIC: "stablehlo.scatter"(%arg4, {{%.*}}, %arg2)
// VARIADIC: %[[VMUL:.*]] = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
// VARIADIC: "stablehlo.gather"(%[[VMUL]], {{%.*}})
