// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --arith-raise --canonicalize | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-batch --inline --enzyme-hlo-opt --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-ml --inline --enzyme-hlo-opt --drop-unsupported-attributes --symbol-dce | stablehlo-translate --interpret

// A loop counting in tensor<i32>, as JAX's do, checkpointed with each
// schedule: the step uses its induction variable, and the gradient must be
// that of the same loop counting in tensor<i64> without checkpointing, as is
// that of the tensor<i32> loop without checkpointing, whose caches are indexed
// by its induction variable.

module {
  func.func @plain(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<13> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i64>, tensor<3xf64> attributes {enzyme.disable_mincut}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i64>
      %2 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<f64>
      %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst = stablehlo.constant dense<0.01> : tensor<3xf64>
      %cst_0 = stablehlo.constant dense<0.1> : tensor<3xf64>
      %4 = stablehlo.multiply %3, %cst : tensor<3xf64>
      %5 = stablehlo.add %4, %cst_0 : tensor<3xf64>
      %6 = stablehlo.sine %iterArg_2 : tensor<3xf64>
      %7 = stablehlo.multiply %6, %5 : tensor<3xf64>
      %8 = stablehlo.add %iterArg_2, %7 : tensor<3xf64>
      stablehlo.return %1, %8 : tensor<i64>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }

  func.func @plain_i32(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<13> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i32>, tensor<3xf64>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i32>
      %2 = stablehlo.convert %iterArg : (tensor<i32>) -> tensor<f64>
      %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst = stablehlo.constant dense<0.01> : tensor<3xf64>
      %cst_0 = stablehlo.constant dense<0.1> : tensor<3xf64>
      %4 = stablehlo.multiply %3, %cst : tensor<3xf64>
      %5 = stablehlo.add %4, %cst_0 : tensor<3xf64>
      %6 = stablehlo.sine %iterArg_2 : tensor<3xf64>
      %7 = stablehlo.multiply %6, %5 : tensor<3xf64>
      %8 = stablehlo.add %iterArg_2, %7 : tensor<3xf64>
      stablehlo.return %1, %8 : tensor<i32>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }

  func.func @periodic(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<13> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i32>, tensor<3xf64> attributes {enzyme.enable_checkpointing = true, enzyme.checkpoint_period = 4 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i32>
      %2 = stablehlo.convert %iterArg : (tensor<i32>) -> tensor<f64>
      %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst = stablehlo.constant dense<0.01> : tensor<3xf64>
      %cst_0 = stablehlo.constant dense<0.1> : tensor<3xf64>
      %4 = stablehlo.multiply %3, %cst : tensor<3xf64>
      %5 = stablehlo.add %4, %cst_0 : tensor<3xf64>
      %6 = stablehlo.sine %iterArg_2 : tensor<3xf64>
      %7 = stablehlo.multiply %6, %5 : tensor<3xf64>
      %8 = stablehlo.add %iterArg_2, %7 : tensor<3xf64>
      stablehlo.return %1, %8 : tensor<i32>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }

  func.func @periodic_default(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<13> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i32>, tensor<3xf64> attributes {enzyme.enable_checkpointing = true}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i32>
      %2 = stablehlo.convert %iterArg : (tensor<i32>) -> tensor<f64>
      %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst = stablehlo.constant dense<0.01> : tensor<3xf64>
      %cst_0 = stablehlo.constant dense<0.1> : tensor<3xf64>
      %4 = stablehlo.multiply %3, %cst : tensor<3xf64>
      %5 = stablehlo.add %4, %cst_0 : tensor<3xf64>
      %6 = stablehlo.sine %iterArg_2 : tensor<3xf64>
      %7 = stablehlo.multiply %6, %5 : tensor<3xf64>
      %8 = stablehlo.add %iterArg_2, %7 : tensor<3xf64>
      stablehlo.return %1, %8 : tensor<i32>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }

  func.func @binomial(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<13> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i32>, tensor<3xf64> attributes {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing, enzyme.checkpoint_period = 3 : i64}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i32>
      %2 = stablehlo.convert %iterArg : (tensor<i32>) -> tensor<f64>
      %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst = stablehlo.constant dense<0.01> : tensor<3xf64>
      %cst_0 = stablehlo.constant dense<0.1> : tensor<3xf64>
      %4 = stablehlo.multiply %3, %cst : tensor<3xf64>
      %5 = stablehlo.add %4, %cst_0 : tensor<3xf64>
      %6 = stablehlo.sine %iterArg_2 : tensor<3xf64>
      %7 = stablehlo.multiply %6, %5 : tensor<3xf64>
      %8 = stablehlo.add %iterArg_2, %7 : tensor<3xf64>
      stablehlo.return %1, %8 : tensor<i32>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }

  func.func @binomial_default(%arg0: tensor<3xf64>) -> tensor<3xf64> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<13> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %0:2 = stablehlo.while(%iterArg = %c_1, %iterArg_2 = %arg0) : tensor<i32>, tensor<3xf64> attributes {enzyme.enable_checkpointing = true, enzyme.binomial_checkpointing}
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i32>
      %2 = stablehlo.convert %iterArg : (tensor<i32>) -> tensor<f64>
      %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f64>) -> tensor<3xf64>
      %cst = stablehlo.constant dense<0.01> : tensor<3xf64>
      %cst_0 = stablehlo.constant dense<0.1> : tensor<3xf64>
      %4 = stablehlo.multiply %3, %cst : tensor<3xf64>
      %5 = stablehlo.add %4, %cst_0 : tensor<3xf64>
      %6 = stablehlo.sine %iterArg_2 : tensor<3xf64>
      %7 = stablehlo.multiply %6, %5 : tensor<3xf64>
      %8 = stablehlo.add %iterArg_2, %7 : tensor<3xf64>
      stablehlo.return %1, %8 : tensor<i32>, tensor<3xf64>
    }
    return %0#1 : tensor<3xf64>
  }

  func.func @main() {
    %input = stablehlo.constant dense<[0.3, 0.5, 0.7]> : tensor<3xf64>
    %diffe = stablehlo.constant dense<1.0> : tensor<3xf64>
    %d_plain:2 = enzyme.autodiff @plain(%input, %diffe) <{
      activity=[#enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_active>]
    }> : (tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)
    %d_plain_i32:2 = enzyme.autodiff @plain_i32(%input, %diffe) <{
      activity=[#enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_active>]
    }> : (tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)
    %d_periodic:2 = enzyme.autodiff @periodic(%input, %diffe) <{
      activity=[#enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_active>]
    }> : (tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)
    %d_periodic_default:2 = enzyme.autodiff @periodic_default(%input, %diffe) <{
      activity=[#enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_active>]
    }> : (tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)
    %d_binomial:2 = enzyme.autodiff @binomial(%input, %diffe) <{
      activity=[#enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_active>]
    }> : (tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)
    check.expect_almost_eq %d_plain_i32#0, %d_plain#0 : tensor<3xf64>
    check.expect_almost_eq %d_plain_i32#1, %d_plain#1 : tensor<3xf64>
    %d_binomial_default:2 = enzyme.autodiff @binomial_default(%input, %diffe) <{
      activity=[#enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_active>]
    }> : (tensor<3xf64>, tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>)
    check.expect_almost_eq %d_periodic#0, %d_plain#0 : tensor<3xf64>
    check.expect_almost_eq %d_periodic#1, %d_plain#1 : tensor<3xf64>
    check.expect_almost_eq %d_periodic_default#0, %d_plain#0 : tensor<3xf64>
    check.expect_almost_eq %d_periodic_default#1, %d_plain#1 : tensor<3xf64>
    check.expect_almost_eq %d_binomial#0, %d_plain#0 : tensor<3xf64>
    check.expect_almost_eq %d_binomial#1, %d_plain#1 : tensor<3xf64>
    check.expect_almost_eq %d_binomial_default#0, %d_plain#0 : tensor<3xf64>
    check.expect_almost_eq %d_binomial_default#1, %d_plain#1 : tensor<3xf64>
    return
  }
}

// The checkpointed loops' scaffolds count in tensor<i64>; the step's copies of
// the body still see the induction variable as tensor<i32>.
// CHECK-LABEL: func.func private @diffeperiodic(
// CHECK:         stablehlo.convert %{{.*}} : (tensor<i64>) -> tensor<i32>
// CHECK-LABEL: func.func private @diffebinomial(
// CHECK:         stablehlo.convert %{{.*}} : (tensor<i64>) -> tensor<i32>
// Without a budget, the default's floor(sqrt(13)) = 3 slots.
// CHECK-LABEL: func.func private @diffebinomial_default(
// CHECK:         stablehlo.constant dense<0.000000e+00> : tensor<3x3xf64>
