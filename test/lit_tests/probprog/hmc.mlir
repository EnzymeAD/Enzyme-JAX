// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(arith-raise,lower-probprog-to-stablehlo{backend=cpu},canonicalize,outline-enzyme-regions,enzyme,canonicalize,inline,remove-unnecessary-enzyme-ops,canonicalize,enzyme-simplify-math,cse)" | FileCheck %s --check-prefix=CPU

module {
  func.func private @normal(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %arg0, %arg1 : tensor<2xui64>, tensor<f64>
  }

  func.func private @logpdf(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<f64> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
    return %cst : tensor<f64>
  }

  func.func @test(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    %0:2 = enzyme.sample @normal(%arg0, %arg1, %arg2) {logpdf = @logpdf, name = "s", symbol = #enzyme.symbol<1>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %1:2 = enzyme.sample @normal(%0#0, %0#1, %arg2) {logpdf = @logpdf, name = "t", symbol = #enzyme.symbol<2>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    return %1#0, %1#1 : tensor<2xui64>, tensor<f64>
  }

  func.func @hmc(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
    %cst = arith.constant dense<5.000000e-02> : tensor<f64>
    %cst_0 = arith.constant dense<1> : tensor<i64>
    %cst_1 = arith.constant dense<0> : tensor<i64>
    %cst_2 = arith.constant dense<5.000000e-01> : tensor<f64>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<f64>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_5 = arith.constant dense<10> : tensor<i64>
    %cst_6 = arith.constant dense<1.000000e-01> : tensor<f64>
    %cst_7 = arith.constant dense<[[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]]> : tensor<2x2xf64>
    %0 = enzyme.initTrace : !enzyme.Trace
    %1 = enzyme.getFlattenedSamplesFromTrace %0 {selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]]} : tensor<2xf64>
    %2 = enzyme.getWeightFromTrace %0 : tensor<f64>
    %3 = arith.negf %2 : tensor<f64>
    %4 = enzyme.cholesky_solve %cst_7, %cst_7 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
    %output_rng_state, %result = enzyme.random %arg0, %cst_4, %cst_3 {rng_distribution = #enzyme<rng_distribution NORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
    %5 = enzyme.dot %4, %result {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %6 = enzyme.dot %cst_7, %5 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %7 = enzyme.dot %5, %6 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %8 = arith.mulf %7, %cst_2 : tensor<f64>
    %9 = arith.addf %3, %8 : tensor<f64>
    %10:2 = enzyme.autodiff_region(%1, %cst_3) {
    ^bb0(%arg3: tensor<2xf64>):
      %25:3 = func.call @test.update(%0, %arg3, %output_rng_state, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
      %26 = arith.negf %25#1 : tensor<f64>
      enzyme.yield %26, %25#2 : tensor<f64>, tensor<2xui64>
    } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
    %11 = "enzyme.broadcast"(%cst_6) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
    %12 = "enzyme.broadcast"(%cst) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
    %13:4 = enzyme.for_loop(%cst_1 : tensor<i64>) to(%cst_5 : tensor<i64>) step(%cst_0 : tensor<i64>) iter_args(%1, %5, %10#1, %10#0 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64> {
    ^bb0(%arg3: tensor<i64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xui64>):
      %25 = arith.mulf %12, %arg6 : tensor<2xf64>
      %26 = arith.subf %arg5, %25 : tensor<2xf64>
      %27 = enzyme.dot %cst_7, %26 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
      %28 = arith.mulf %11, %27 : tensor<2xf64>
      %29 = arith.addf %arg4, %28 : tensor<2xf64>
      %30:2 = enzyme.autodiff_region(%29, %cst_3) {
      ^bb0(%arg8: tensor<2xf64>):
        %33:3 = func.call @test.update(%0, %arg8, %arg7, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
        %34 = arith.negf %33#1 : tensor<f64>
        enzyme.yield %34, %33#2 : tensor<f64>, tensor<2xui64>
      } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
      %31 = arith.mulf %12, %30#1 : tensor<2xf64>
      %32 = arith.subf %26, %31 : tensor<2xf64>
      enzyme.yield %29, %32, %30#1, %30#0 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64>
    }
    %14:3 = call @test.update(%0, %13#0, %13#3, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
    %15 = arith.negf %14#1 : tensor<f64>
    %16 = enzyme.dot %cst_7, %13#1 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 1>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %17 = enzyme.dot %13#1, %16 {lhs_batching_dimensions = array<i64>, lhs_contracting_dimensions = array<i64: 0>, rhs_batching_dimensions = array<i64>, rhs_contracting_dimensions = array<i64: 0>} : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %18 = arith.mulf %17, %cst_2 : tensor<f64>
    %19 = arith.addf %15, %18 : tensor<f64>
    %20 = arith.subf %9, %19 : tensor<f64>
    %21 = math.exp %20 : tensor<f64>
    %22 = arith.minimumf %21, %cst_3 : tensor<f64>
    %output_rng_state_8, %result_9 = enzyme.random %14#2, %cst_4, %cst_3 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %23 = arith.cmpf olt, %result_9, %22 : tensor<f64>
    %24 = enzyme.selectTrace %23, %14#0, %0 : tensor<i1>
    return %24, %23, %output_rng_state_8 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
  }

  func.func @test.update(%arg0: !enzyme.Trace, %arg1: tensor<2xf64>, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
    %0 = enzyme.initTrace : !enzyme.Trace
    %1 = enzyme.unflatten_slice %arg1[0] : tensor<2xf64> -> tensor<f64>
    %2 = call @logpdf(%1, %arg3, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %3 = arith.addf %2, %cst : tensor<f64>
    %4 = enzyme.addSampleToTrace(%1 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
    %5 = enzyme.unflatten_slice %arg1[1] : tensor<2xf64> -> tensor<f64>
    %6 = call @logpdf(%5, %1, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
    %7 = arith.addf %3, %6 : tensor<f64>
    %8 = enzyme.addSampleToTrace(%5 : tensor<f64>) into %4 {symbol = #enzyme.symbol<2>}
    %9 = enzyme.addWeightToTrace(%7 : tensor<f64>) into %8
    %10 = enzyme.addRetvalToTrace(%5 : tensor<f64>) into %9
    return %10, %7, %arg2 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
  }
}

// CPU:  func.func @hmc(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<i1>, tensor<2xui64>) {
// CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %c = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
// CPU-NEXT:    %c_1 = stablehlo.constant dense<12> : tensor<ui64>
// CPU-NEXT:    %cst_2 = stablehlo.constant dense<1.4142135623730951> : tensor<2xf64>
// CPU-NEXT:    %cst_3 = stablehlo.constant dense<2.000000e+00> : tensor<2xf64>
// CPU-NEXT:    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<2xf64>
// CPU-NEXT:    %c_5 = stablehlo.constant dense<4607182418800017408> : tensor<2xui64>
// CPU-NEXT:    %c_6 = stablehlo.constant dense<12> : tensor<2xui64>
// CPU-NEXT:    %cst_7 = stablehlo.constant dense<5.000000e-02> : tensor<f64>
// CPU-NEXT:    %c_8 = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:    %c_9 = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT:    %cst_10 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CPU-NEXT:    %cst_11 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CPU-NEXT:    %c_12 = stablehlo.constant dense<10> : tensor<i64>
// CPU-NEXT:    %cst_13 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
// CPU-NEXT:    %cst_14 = stablehlo.constant dense<{{\[}}[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]{{\]}}> : tensor<2x2xf64>
// CPU-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CPU-NEXT:    %1 = enzyme.getFlattenedSamplesFromTrace %0 {selection = {{\[}}[#enzyme.symbol<1>], [#enzyme.symbol<2>]{{\]}}} : tensor<2xf64>
// CPU-NEXT:    %2 = enzyme.getWeightFromTrace %0 : tensor<f64>
// CPU-NEXT:    %3 = stablehlo.negate %2 : tensor<f64>
// CPU-NEXT:    %4 = stablehlo.cholesky %cst_14, lower = true : tensor<2x2xf64>
// CPU-NEXT:    %5 = "stablehlo.triangular_solve"(%4, %cst_14) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
// CPU-NEXT:    %6 = "stablehlo.triangular_solve"(%4, %5) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
// CPU-NEXT:    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CPU-NEXT:    %7 = stablehlo.shift_right_logical %output, %c_6 : tensor<2xui64>
// CPU-NEXT:    %8 = stablehlo.or %7, %c_5 : tensor<2xui64>
// CPU-NEXT:    %9 = stablehlo.bitcast_convert %8 : (tensor<2xui64>) -> tensor<2xf64>
// CPU-NEXT:    %10 = stablehlo.subtract %9, %cst_4 : tensor<2xf64>
// CPU-NEXT:    %11 = stablehlo.multiply %10, %cst_3 : tensor<2xf64>
// CPU-NEXT:    %12 = stablehlo.subtract %11, %cst_4 : tensor<2xf64>
// CPU-NEXT:    %13 = chlo.erf_inv %12 : tensor<2xf64> -> tensor<2xf64>
// CPU-NEXT:    %14 = stablehlo.multiply %13, %cst_2 : tensor<2xf64>
// CPU-NEXT:    %15 = stablehlo.dot_general %6, %14, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CPU-NEXT:    %16 = stablehlo.dot_general %cst_14, %15, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CPU-NEXT:    %17 = stablehlo.dot_general %15, %16, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CPU-NEXT:    %18 = stablehlo.multiply %17, %cst_10 : tensor<f64>
// CPU-NEXT:    %19 = stablehlo.add %3, %18 : tensor<f64>
// CPU-NEXT:    %20 = stablehlo.reshape %cst_0 : (tensor<f64>) -> tensor<1xf64>
// CPU-NEXT:    %21 = stablehlo.pad %20, %cst_0, low = [1], high = [0], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %22 = stablehlo.pad %20, %cst_0, low = [0], high = [1], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %23 = arith.addf %21, %22 : tensor<2xf64>
// CPU-NEXT:    %24 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %25 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %26:5 = stablehlo.while(%iterArg = %c_9, %iterArg_17 = %1, %iterArg_18 = %15, %iterArg_19 = %23, %iterArg_20 = %output_state) : tensor<i64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64>
// CPU-NEXT:    cond {
// CPU-NEXT:      %50 = stablehlo.compare  LT, %iterArg, %c_12 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CPU-NEXT:      stablehlo.return %50 : tensor<i1>
// CPU-NEXT:    } do {
// CPU-NEXT:      %50 = stablehlo.multiply %25, %iterArg_19 : tensor<2xf64>
// CPU-NEXT:      %51 = stablehlo.subtract %iterArg_18, %50 : tensor<2xf64>
// CPU-NEXT:      %52 = stablehlo.dot_general %cst_14, %51, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CPU-NEXT:      %53 = stablehlo.multiply %24, %52 : tensor<2xf64>
// CPU-NEXT:      %54 = stablehlo.add %iterArg_17, %53 : tensor<2xf64>
// CPU-NEXT:      %55 = stablehlo.multiply %25, %23 : tensor<2xf64>
// CPU-NEXT:      %56 = stablehlo.subtract %51, %55 : tensor<2xf64>
// CPU-NEXT:      %57 = stablehlo.add %iterArg, %c_8 : tensor<i64>
// CPU-NEXT:      stablehlo.return %57, %54, %56, %23, %iterArg_20 : tensor<i64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64>
// CPU-NEXT:    }
// CPU-NEXT:    %27 = enzyme.initTrace : !enzyme.Trace
// CPU-NEXT:    %28 = stablehlo.slice %26#1 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %29 = stablehlo.reshape %28 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %30 = enzyme.addSampleToTrace(%29 : tensor<f64>) into %27 {symbol = #enzyme.symbol<1>}
// CPU-NEXT:    %31 = stablehlo.slice %26#1 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %32 = stablehlo.reshape %31 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %33 = enzyme.addSampleToTrace(%32 : tensor<f64>) into %30 {symbol = #enzyme.symbol<2>}
// CPU-NEXT:    %34 = enzyme.addWeightToTrace(%cst : tensor<f64>) into %33
// CPU-NEXT:    %35 = enzyme.addRetvalToTrace(%32 : tensor<f64>) into %34
// CPU-NEXT:    %36 = stablehlo.negate %cst : tensor<f64>
// CPU-NEXT:    %37 = stablehlo.dot_general %cst_14, %26#2, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CPU-NEXT:    %38 = stablehlo.dot_general %26#2, %37, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CPU-NEXT:    %39 = stablehlo.multiply %38, %cst_10 : tensor<f64>
// CPU-NEXT:    %40 = stablehlo.add %36, %39 : tensor<f64>
// CPU-NEXT:    %41 = stablehlo.subtract %19, %40 : tensor<f64>
// CPU-NEXT:    %42 = stablehlo.exponential %41 : tensor<f64>
// CPU-NEXT:    %43 = stablehlo.minimum %42, %cst_11 : tensor<f64>
// CPU-NEXT:    %output_state_15, %output_16 = stablehlo.rng_bit_generator %26#4, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
// CPU-NEXT:    %44 = stablehlo.shift_right_logical %output_16, %c_1 : tensor<ui64>
// CPU-NEXT:    %45 = stablehlo.or %44, %c : tensor<ui64>
// CPU-NEXT:    %46 = stablehlo.bitcast_convert %45 : (tensor<ui64>) -> tensor<f64>
// CPU-NEXT:    %47 = stablehlo.subtract %46, %cst_11 : tensor<f64>
// CPU-NEXT:    %48 = stablehlo.compare  LT, %47, %43,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CPU-NEXT:    %49 = enzyme.selectTrace %48, %35, %0 : tensor<i1>
// CPU-NEXT:    return %49, %48, %output_state_15 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
// CPU-NEXT:  }
