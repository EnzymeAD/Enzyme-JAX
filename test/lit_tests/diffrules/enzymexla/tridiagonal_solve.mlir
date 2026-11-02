// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=solve outfn=solve_fwd retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup,enzyme_dup,enzyme_dup mode=ForwardMode" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=solve outfn=solve_rev retTys=enzyme_active argTys=enzyme_active,enzyme_active,enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REV
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --lower-enzymexla-linalg --lower-enzymexla-lapack="backend=tpu" --inline --canonicalize --arith-raise | stablehlo-translate - --interpret

module {
  func.func @solve(%dl: tensor<2x4xf64>, %d: tensor<2x4xf64>, %du: tensor<2x4xf64>, %b: tensor<2x4x3xf64>) -> tensor<2x4x3xf64> {
    %x, %info = enzymexla.linalg.tridiagonal_solve %dl, %d, %du, %b : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>) -> (tensor<2x4x3xf64>, tensor<2xi32>)
    return %x : tensor<2x4x3xf64>
  }

  func.func @dot2(%a: tensor<2x4xf64>, %b: tensor<2x4xf64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %prod = stablehlo.multiply %a, %b : tensor<2x4xf64>
    %sum = stablehlo.reduce(%prod init: %zero) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x4xf64>, tensor<f64>) -> tensor<f64>
    return %sum : tensor<f64>
  }

  func.func @dot3(%a: tensor<2x4x3xf64>, %b: tensor<2x4x3xf64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %prod = stablehlo.multiply %a, %b : tensor<2x4x3xf64>
    %sum = stablehlo.reduce(%prod init: %zero) applies stablehlo.add across dimensions = [0, 1, 2] : (tensor<2x4x3xf64>, tensor<f64>) -> tensor<f64>
    return %sum : tensor<f64>
  }

  func.func @stack2(%a: tensor<2x4xf64>, %b: tensor<2x4xf64>) -> tensor<2x2x4xf64> {
    %a1 = stablehlo.reshape %a : (tensor<2x4xf64>) -> tensor<1x2x4xf64>
    %b1 = stablehlo.reshape %b : (tensor<2x4xf64>) -> tensor<1x2x4xf64>
    %r = stablehlo.concatenate %a1, %b1, dim = 0 : (tensor<1x2x4xf64>, tensor<1x2x4xf64>) -> tensor<2x2x4xf64>
    return %r : tensor<2x2x4xf64>
  }

  func.func @stack3(%a: tensor<2x4x3xf64>, %b: tensor<2x4x3xf64>) -> tensor<2x2x4x3xf64> {
    %a1 = stablehlo.reshape %a : (tensor<2x4x3xf64>) -> tensor<1x2x4x3xf64>
    %b1 = stablehlo.reshape %b : (tensor<2x4x3xf64>) -> tensor<1x2x4x3xf64>
    %r = stablehlo.concatenate %a1, %b1, dim = 0 : (tensor<1x2x4x3xf64>, tensor<1x2x4x3xf64>) -> tensor<2x2x4x3xf64>
    return %r : tensor<2x2x4x3xf64>
  }

  func.func @main() {
    %dl = stablehlo.constant dense<[[0.0, 0.5, -0.3, 0.7], [0.0, -0.4, 0.6, 0.2]]> : tensor<2x4xf64>
    %d = stablehlo.constant dense<[[4.0, 3.5, 5.0, 4.5], [3.0, 4.2, 3.8, 5.1]]> : tensor<2x4xf64>
    %du = stablehlo.constant dense<[[0.6, -0.2, 0.9, 0.0], [0.3, 0.8, -0.5, 0.0]]> : tensor<2x4xf64>
    %b = stablehlo.constant dense<[[[1.0, -2.0, 0.5], [0.3, 1.2, -0.7], [2.1, 0.4, 1.5], [-1.1, 0.9, 0.2]], [[0.8, 1.6, -0.4], [-0.6, 0.2, 1.3], [1.4, -1.0, 0.6], [0.5, 0.7, -1.2]]]> : tensor<2x4x3xf64>

    // dl[0] and du[m-1] are outside of the matrix, so their tangents must not
    // contribute to the result.
    %tdl = stablehlo.constant dense<[[0.9, 0.2, -0.5, 0.3], [-0.7, 0.6, 0.1, -0.4]]> : tensor<2x4xf64>
    %td = stablehlo.constant dense<[[0.4, -0.3, 0.2, 0.6], [0.1, 0.5, -0.2, 0.3]]> : tensor<2x4xf64>
    %tdu = stablehlo.constant dense<[[-0.2, 0.7, 0.4, 0.8], [0.5, -0.1, 0.3, -0.6]]> : tensor<2x4xf64>
    %tb = stablehlo.constant dense<[[[0.2, 0.1, -0.3], [0.5, -0.4, 0.6], [-0.1, 0.3, 0.2], [0.4, 0.0, -0.5]], [[-0.3, 0.6, 0.1], [0.2, -0.5, 0.4], [0.7, 0.1, -0.2], [-0.4, 0.3, 0.5]]]> : tensor<2x4x3xf64>

    %tdl_b = stablehlo.constant dense<[[0.1, -0.6, 0.3, 0.2], [0.4, 0.2, -0.3, 0.5]]> : tensor<2x4xf64>
    %td_b = stablehlo.constant dense<[[-0.5, 0.2, 0.4, -0.1], [0.3, -0.2, 0.6, 0.1]]> : tensor<2x4xf64>
    %tdu_b = stablehlo.constant dense<[[0.3, 0.1, -0.6, 0.4], [-0.2, 0.4, 0.5, 0.3]]> : tensor<2x4xf64>
    %tb_b = stablehlo.constant dense<[[[-0.4, 0.3, 0.2], [0.1, 0.6, -0.2], [0.5, -0.3, 0.4], [-0.2, 0.1, 0.3]], [[0.6, -0.1, 0.2], [-0.3, 0.4, 0.1], [0.2, 0.5, -0.6], [0.1, -0.2, 0.4]]]> : tensor<2x4x3xf64>

    %dx = stablehlo.constant dense<[[[0.7, -0.2, 0.4], [0.1, 0.5, -0.3], [-0.6, 0.2, 0.8], [0.3, -0.4, 0.1]], [[0.2, 0.9, -0.5], [0.4, -0.1, 0.3], [-0.2, 0.6, 0.1], [0.5, 0.3, -0.7]]]> : tensor<2x4x3xf64>
    %dx_b = stablehlo.constant dense<[[[-0.3, 0.5, 0.1], [0.6, -0.2, 0.4], [0.2, 0.3, -0.5], [-0.1, 0.7, 0.2]], [[0.4, -0.6, 0.3], [-0.5, 0.2, 0.1], [0.3, -0.4, 0.6], [0.1, 0.2, -0.3]]]> : tensor<2x4x3xf64>

    // 1. Forward mode against central finite differences.
    %fwd = enzyme.fwddiff @solve(%dl, %tdl, %d, %td, %du, %tdu, %b, %tb) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dupnoneed>]
    }> : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<2x4x3xf64>

    %h2 = stablehlo.constant dense<1.0e-6> : tensor<2x4xf64>
    %h3 = stablehlo.constant dense<1.0e-6> : tensor<2x4x3xf64>
    %inv2h = stablehlo.constant dense<5.0e5> : tensor<2x4x3xf64>
    %sdl = stablehlo.multiply %h2, %tdl : tensor<2x4xf64>
    %sd = stablehlo.multiply %h2, %td : tensor<2x4xf64>
    %sdu = stablehlo.multiply %h2, %tdu : tensor<2x4xf64>
    %sb = stablehlo.multiply %h3, %tb : tensor<2x4x3xf64>
    %dl_p = stablehlo.add %dl, %sdl : tensor<2x4xf64>
    %d_p = stablehlo.add %d, %sd : tensor<2x4xf64>
    %du_p = stablehlo.add %du, %sdu : tensor<2x4xf64>
    %b_p = stablehlo.add %b, %sb : tensor<2x4x3xf64>
    %dl_m = stablehlo.subtract %dl, %sdl : tensor<2x4xf64>
    %d_m = stablehlo.subtract %d, %sd : tensor<2x4xf64>
    %du_m = stablehlo.subtract %du, %sdu : tensor<2x4xf64>
    %b_m = stablehlo.subtract %b, %sb : tensor<2x4x3xf64>
    %x_p = func.call @solve(%dl_p, %d_p, %du_p, %b_p) : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>) -> tensor<2x4x3xf64>
    %x_m = func.call @solve(%dl_m, %d_m, %du_m, %b_m) : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>) -> tensor<2x4x3xf64>
    %x_diff = stablehlo.subtract %x_p, %x_m : tensor<2x4x3xf64>
    %fd = stablehlo.multiply %x_diff, %inv2h : tensor<2x4x3xf64>
    check.expect_almost_eq %fwd, %fd : tensor<2x4x3xf64>

    // 2. Reverse mode against forward mode: <dx, J t> == <J^T dx, t>.
    %rev:4 = enzyme.autodiff @solve(%dl, %d, %du, %b, %dx) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>)

    %lhs = func.call @dot3(%dx, %fwd) : (tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<f64>
    %r0 = func.call @dot2(%rev#0, %tdl) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<f64>
    %r1 = func.call @dot2(%rev#1, %td) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<f64>
    %r2 = func.call @dot2(%rev#2, %tdu) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<f64>
    %r3 = func.call @dot3(%rev#3, %tb) : (tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<f64>
    %r01 = stablehlo.add %r0, %r1 : tensor<f64>
    %r012 = stablehlo.add %r01, %r2 : tensor<f64>
    %rhs = stablehlo.add %r012, %r3 : tensor<f64>
    check.expect_almost_eq %lhs, %rhs : tensor<f64>

    // The adjoints of dl[0] and du[m-1] are zero.
    %zero2 = stablehlo.constant dense<0.0> : tensor<2x1xf64>
    %rdl0 = stablehlo.slice %rev#0 [0:2, 0:1] : (tensor<2x4xf64>) -> tensor<2x1xf64>
    %rdu3 = stablehlo.slice %rev#2 [0:2, 3:4] : (tensor<2x4xf64>) -> tensor<2x1xf64>
    check.expect_almost_eq %rdl0, %zero2 : tensor<2x1xf64>
    check.expect_almost_eq %rdu3, %zero2 : tensor<2x1xf64>

    // 3. Reverse mode with partial activity matches the fully active adjoints.
    %rev_d:1 = enzyme.autodiff @solve(%dl, %d, %du, %b, %dx) <{
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<2x4xf64>
    check.expect_almost_eq %rev_d#0, %rev#1 : tensor<2x4xf64>

    %rev_b:1 = enzyme.autodiff @solve(%dl, %d, %du, %b, %dx) <{
      activity = [#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<2x4x3xf64>
    check.expect_almost_eq %rev_b#0, %rev#3 : tensor<2x4x3xf64>

    // 4. Width 2 matches two width 1 evaluations.
    %fwd_b = enzyme.fwddiff @solve(%dl, %tdl_b, %d, %td_b, %du, %tdu_b, %b, %tb_b) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dupnoneed>]
    }> : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<2x4x3xf64>

    %tdl_w2 = func.call @stack2(%tdl, %tdl_b) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x2x4xf64>
    %td_w2 = func.call @stack2(%td, %td_b) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x2x4xf64>
    %tdu_w2 = func.call @stack2(%tdu, %tdu_b) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x2x4xf64>
    %tb_w2 = func.call @stack3(%tb, %tb_b) : (tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<2x2x4x3xf64>
    %fwd_w2 = enzyme.fwddiff @solve(%dl, %tdl_w2, %d, %td_w2, %du, %tdu_w2, %b, %tb_w2) <{
      activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
      ret_activity = [#enzyme.activity<enzyme_dupnoneed>],
      width = 2 : i64
    }> : (tensor<2x4xf64>, tensor<2x2x4xf64>, tensor<2x4xf64>, tensor<2x2x4xf64>, tensor<2x4xf64>, tensor<2x2x4xf64>, tensor<2x4x3xf64>, tensor<2x2x4x3xf64>) -> tensor<2x2x4x3xf64>
    %fwd_w2_ref = func.call @stack3(%fwd, %fwd_b) : (tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<2x2x4x3xf64>
    check.expect_almost_eq %fwd_w2, %fwd_w2_ref : tensor<2x2x4x3xf64>

    %rev_b2:4 = enzyme.autodiff @solve(%dl, %d, %du, %b, %dx_b) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>]
    }> : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>)

    %dx_w2 = func.call @stack3(%dx, %dx_b) : (tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<2x2x4x3xf64>
    %rev_w2:4 = enzyme.autodiff @solve(%dl, %d, %du, %b, %dx_w2) <{
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity = [#enzyme.activity<enzyme_activenoneed>],
      width = 2 : i64
    }> : (tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4xf64>, tensor<2x4x3xf64>, tensor<2x2x4x3xf64>) -> (tensor<2x2x4xf64>, tensor<2x2x4xf64>, tensor<2x2x4xf64>, tensor<2x2x4x3xf64>)
    %rdl_w2_ref = func.call @stack2(%rev#0, %rev_b2#0) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x2x4xf64>
    %rd_w2_ref = func.call @stack2(%rev#1, %rev_b2#1) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x2x4xf64>
    %rdu_w2_ref = func.call @stack2(%rev#2, %rev_b2#2) : (tensor<2x4xf64>, tensor<2x4xf64>) -> tensor<2x2x4xf64>
    %rb_w2_ref = func.call @stack3(%rev#3, %rev_b2#3) : (tensor<2x4x3xf64>, tensor<2x4x3xf64>) -> tensor<2x2x4x3xf64>
    check.expect_almost_eq %rev_w2#0, %rdl_w2_ref : tensor<2x2x4xf64>
    check.expect_almost_eq %rev_w2#1, %rd_w2_ref : tensor<2x2x4xf64>
    check.expect_almost_eq %rev_w2#2, %rdu_w2_ref : tensor<2x2x4xf64>
    check.expect_almost_eq %rev_w2#3, %rb_w2_ref : tensor<2x2x4x3xf64>

    return
  }
}

// FWD-LABEL: func.func private @solve_fwd
// FWD-SAME: (%[[DL:.+]]: tensor<2x4xf64>, %[[TDL:.+]]: tensor<2x4xf64>, %[[D:.+]]: tensor<2x4xf64>, %[[TD:.+]]: tensor<2x4xf64>, %[[DU:.+]]: tensor<2x4xf64>, %[[TDU:.+]]: tensor<2x4xf64>, %[[B:.+]]: tensor<2x4x3xf64>, %[[TB:.+]]: tensor<2x4x3xf64>)
// FWD: %[[X:.+]], %{{.+}} = enzymexla.linalg.tridiagonal_solve %[[DL]], %[[D]], %[[DU]], %[[B]]
// FWD: %[[TDLB:.+]] = stablehlo.broadcast_in_dim %[[TDL]], dims = [0, 1] : (tensor<2x4xf64>) -> tensor<2x4x3xf64>
// FWD: %[[XU:.+]] = stablehlo.slice %[[X]] [0:2, 0:3, 0:3]
// FWD: %[[XD:.+]] = stablehlo.pad %[[XU]], %{{.+}}, low = [0, 1, 0], high = [0, 0, 0]
// FWD: %[[P0:.+]] = stablehlo.multiply %[[TDLB]], %[[XD]]
// FWD: %[[R0:.+]] = stablehlo.subtract %[[TB]], %[[P0]]
// FWD: %[[TDB:.+]] = stablehlo.broadcast_in_dim %[[TD]], dims = [0, 1]
// FWD: %[[P1:.+]] = stablehlo.multiply %[[TDB]], %[[X]]
// FWD: %[[R1:.+]] = stablehlo.subtract %[[R0]], %[[P1]]
// FWD: %[[TDUB:.+]] = stablehlo.broadcast_in_dim %[[TDU]], dims = [0, 1]
// FWD: %[[XS:.+]] = stablehlo.slice %[[X]] [0:2, 1:4, 0:3]
// FWD: %[[XUP:.+]] = stablehlo.pad %[[XS]], %{{.+}}, low = [0, 0, 0], high = [0, 1, 0]
// FWD: %[[P2:.+]] = stablehlo.multiply %[[TDUB]], %[[XUP]]
// FWD: %[[R2:.+]] = stablehlo.subtract %[[R1]], %[[P2]]
// FWD: %[[DX:.+]], %{{.+}} = enzymexla.linalg.tridiagonal_solve %[[DL]], %[[D]], %[[DU]], %[[R2]]
// FWD: return %[[X]], %[[DX]]

// REV-LABEL: func.func private @solve_rev
// REV-SAME: (%[[DL:.+]]: tensor<2x4xf64>, %[[D:.+]]: tensor<2x4xf64>, %[[DU:.+]]: tensor<2x4xf64>, %[[B:.+]]: tensor<2x4x3xf64>, %[[DX:.+]]: tensor<2x4x3xf64>)
// REV: %[[X:.+]], %{{.+}} = enzymexla.linalg.tridiagonal_solve %[[DL]], %[[D]], %[[DU]], %[[B]]
// REV: %[[DUS:.+]] = stablehlo.slice %[[DU]] [0:2, 0:3]
// REV: %[[DLT:.+]] = stablehlo.pad %[[DUS]], %{{.+}}, low = [0, 1], high = [0, 0]
// REV: %[[DLS:.+]] = stablehlo.slice %[[DL]] [0:2, 1:4]
// REV: %[[DUT:.+]] = stablehlo.pad %[[DLS]], %{{.+}}, low = [0, 0], high = [0, 1]
// REV: %[[DB:.+]], %{{.+}} = enzymexla.linalg.tridiagonal_solve %[[DLT]], %[[D]], %[[DUT]], %[[DX]]
// REV: %[[XS0:.+]] = stablehlo.slice %[[X]] [0:2, 0:3, 0:3]
// REV: %[[XD:.+]] = stablehlo.pad %[[XS0]], %{{.+}}, low = [0, 1, 0], high = [0, 0, 0]
// REV: %[[M0:.+]] = stablehlo.multiply %[[DB]], %[[XD]]
// REV: %[[S0:.+]] = stablehlo.reduce(%[[M0]] init: %{{.+}}) applies stablehlo.add across dimensions = [2]
// REV: %[[GDL:.+]] = stablehlo.negate %[[S0]]
// REV: %[[M1:.+]] = stablehlo.multiply %[[DB]], %[[X]]
// REV: %[[S1:.+]] = stablehlo.reduce(%[[M1]] init: %{{.+}}) applies stablehlo.add across dimensions = [2]
// REV: %[[GD:.+]] = stablehlo.negate %[[S1]]
// REV: %[[XS1:.+]] = stablehlo.slice %[[X]] [0:2, 1:4, 0:3]
// REV: %[[XU:.+]] = stablehlo.pad %[[XS1]], %{{.+}}, low = [0, 0, 0], high = [0, 1, 0]
// REV: %[[M2:.+]] = stablehlo.multiply %[[DB]], %[[XU]]
// REV: %[[S2:.+]] = stablehlo.reduce(%[[M2]] init: %{{.+}}) applies stablehlo.add across dimensions = [2]
// REV: %[[GDU:.+]] = stablehlo.negate %[[S2]]
// REV: return %[[GDL]], %[[GD]], %[[GDU]], %[[DB]]
