// RUN: enzymexlamlir-opt %s --enzyme --enzyme-batch-to-stablehlo --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --enzyme-batch-to-stablehlo --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --inline --chlo-legalize-to-stablehlo --enzyme-hlo-opt --verify-each=0 | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func private @concat_mixed(%theta: tensor<2xf64>, %p: tensor<1xf64>) -> tensor<2xf64> {
  %s0 = stablehlo.slice %theta [0:1] : (tensor<2xf64>) -> tensor<1xf64>
  %s1 = stablehlo.slice %theta [1:2] : (tensor<2xf64>) -> tensor<1xf64>
  %sin0 = stablehlo.sine %s0 : tensor<1xf64>
  %prod = stablehlo.multiply %sin0, %s1 : tensor<1xf64>
  %c1 = stablehlo.concatenate %prod, %s0, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  %c2 = stablehlo.concatenate %p, %s1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  %res = stablehlo.add %c1, %c2 : tensor<2xf64>
  return %res : tensor<2xf64>
}

func.func private @pad_active_padval(%x: tensor<2xf64>) -> tensor<4xf64> {
  %s0 = stablehlo.slice %x [0:1] : (tensor<2xf64>) -> tensor<1xf64>
  %s1 = stablehlo.slice %x [1:2] : (tensor<2xf64>) -> tensor<1xf64>
  %pv = stablehlo.reshape %s0 : (tensor<1xf64>) -> tensor<f64>
  %in = stablehlo.concatenate %s1, %s1, dim = 0 : (tensor<1xf64>, tensor<1xf64>) -> tensor<2xf64>
  %padded = stablehlo.pad %in, %pv, low = [1], high = [0], interior = [1] : (tensor<2xf64>, tensor<f64>) -> tensor<4xf64>
  return %padded : tensor<4xf64>
}

func.func private @pad_const_padval(%x: tensor<2xf64>, %p: tensor<f64>) -> tensor<3xf64> {
  %padded = stablehlo.pad %x, %p, low = [1], high = [0], interior = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<3xf64>
  return %padded : tensor<3xf64>
}

func.func private @dus_const_operand(%base: tensor<3xf64>, %upd: tensor<1xf64>) -> tensor<3xf64> {
  %idx = stablehlo.constant dense<1> : tensor<i32>
  %res = stablehlo.dynamic_update_slice %base, %upd, %idx : (tensor<3xf64>, tensor<1xf64>, tensor<i32>) -> tensor<3xf64>
  return %res : tensor<3xf64>
}

// CHECK-LABEL: func.func @main
func.func @main() {
  %theta = stablehlo.constant dense<[0.0, 3.0]> : tensor<2xf64>
  %dtheta = stablehlo.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
  %p = stablehlo.constant dense<[5.0]> : tensor<1xf64>

  // 1. Batched ConcatenateOp with mixed constant and active operands (Reactant #3296)
  // At theta = [0, 3], p = [5]:
  // res = [sin(theta[0])*theta[1] + p[0], theta[0] + theta[1]] = [5.0, 3.0]
  // d/dtheta[0] = [cos(0)*3, 1] = [3.0, 1.0]
  // d/dtheta[1] = [sin(0), 1]   = [0.0, 1.0]
  %c_res:2 = enzyme.fwddiff @concat_mixed(%theta, %dtheta, %p) <{
    activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
    ret_activity = [#enzyme.activity<enzyme_dup>],
    width = 2 : i64
  }> : (tensor<2xf64>, tensor<2x2xf64>, tensor<1xf64>) -> (tensor<2xf64>, tensor<2x2xf64>)

  check.expect_almost_eq_const %c_res#0, dense<[5.0, 3.0]> : tensor<2xf64>
  check.expect_almost_eq_const %c_res#1, dense<[[3.0, 1.0], [0.0, 1.0]]> : tensor<2x2xf64>

  // 2. Batched PadOp with active 0D padding_value + interior padding (Reactant #3295)
  // pad([x1, x1], x0, low=[1], high=[0], interior=[1]) = [x0, x1, x0, x1]
  // lane 0 (dx = [1, 0]): [1, 0, 1, 0]
  // lane 1 (dx = [0, 1]): [0, 1, 0, 1]
  %p_act:2 = enzyme.fwddiff @pad_active_padval(%theta, %dtheta) <{
    activity = [#enzyme.activity<enzyme_dup>],
    ret_activity = [#enzyme.activity<enzyme_dup>],
    width = 2 : i64
  }> : (tensor<2xf64>, tensor<2x2xf64>) -> (tensor<4xf64>, tensor<2x4xf64>)

  check.expect_almost_eq_const %p_act#0, dense<[0.0, 3.0, 0.0, 3.0]> : tensor<4xf64>
  check.expect_almost_eq_const %p_act#1, dense<[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]> : tensor<2x4xf64>

  // 3. Width-1 PadOp with active padding_value
  %dx1 = stablehlo.constant dense<[2.0, 7.0]> : tensor<2xf64>
  %p_act_w1:2 = enzyme.fwddiff @pad_active_padval(%theta, %dx1) <{
    activity = [#enzyme.activity<enzyme_dup>],
    ret_activity = [#enzyme.activity<enzyme_dup>],
    width = 1 : i64
  }> : (tensor<2xf64>, tensor<2xf64>) -> (tensor<4xf64>, tensor<4xf64>)

  check.expect_almost_eq_const %p_act_w1#1, dense<[2.0, 7.0, 2.0, 7.0]> : tensor<4xf64>

  // 4. Batched PadOp with inactive padding_value
  %pv_cst = stablehlo.constant dense<9.0> : tensor<f64>
  %p_cst:2 = enzyme.fwddiff @pad_const_padval(%theta, %dtheta, %pv_cst) <{
    activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
    ret_activity = [#enzyme.activity<enzyme_dup>],
    width = 2 : i64
  }> : (tensor<2xf64>, tensor<2x2xf64>, tensor<f64>) -> (tensor<3xf64>, tensor<2x3xf64>)

  check.expect_almost_eq_const %p_cst#0, dense<[9.0, 0.0, 3.0]> : tensor<3xf64>
  check.expect_almost_eq_const %p_cst#1, dense<[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<2x3xf64>

  // 5. Batched DynamicUpdateSliceOp with inactive operand and active update
  %base = stablehlo.constant dense<[10.0, 20.0, 30.0]> : tensor<3xf64>
  %upd = stablehlo.constant dense<[4.0]> : tensor<1xf64>
  %dupd = stablehlo.constant dense<[[2.0], [5.0]]> : tensor<2x1xf64>
  %dus_res:2 = enzyme.fwddiff @dus_const_operand(%base, %upd, %dupd) <{
    activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>],
    ret_activity = [#enzyme.activity<enzyme_dup>],
    width = 2 : i64
  }> : (tensor<3xf64>, tensor<1xf64>, tensor<2x1xf64>) -> (tensor<3xf64>, tensor<2x3xf64>)

  check.expect_almost_eq_const %dus_res#0, dense<[10.0, 4.0, 30.0]> : tensor<3xf64>
  check.expect_almost_eq_const %dus_res#1, dense<[[0.0, 2.0, 0.0], [0.0, 5.0, 0.0]]> : tensor<2x3xf64>

  return
}
