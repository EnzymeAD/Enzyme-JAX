// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(arith-raise,lower-probprog-to-stablehlo{backend=cpu},canonicalize,outline-enzyme-regions,enzyme,canonicalize,remove-unnecessary-enzyme-ops,canonicalize,enzyme-simplify-math,cse)" --mlir-print-ir-after=cse | FileCheck %s --check-prefix=CPU --dump-input=always

module {
  // Placeholder
  func.func private @normal(%arg0: tensor<2xui64>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> (tensor<2xui64>, tensor<f64>) {
    return %arg0, %arg1 : tensor<2xui64>, tensor<f64>
  }

  // Placeholder
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
    %cst = arith.constant dense<5.000000e-02> : tensor<2xf64>
    %cst_0 = arith.constant dense<1.000000e-01> : tensor<2xf64>
    %cst_1 = arith.constant dense<1> : tensor<i64>
    %cst_2 = arith.constant dense<0> : tensor<i64>
    %cst_3 = arith.constant dense<5.000000e-01> : tensor<f64>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<f64>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<f64>
    %cst_6 = arith.constant dense<10> : tensor<i64>
    %cst_7 = arith.constant dense<[[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]]> : tensor<2x2xf64>
    %0 = enzyme.initTrace : !enzyme.Trace
    %1 = enzyme.getFlattenedSamplesFromTrace %0 {selection = [[#enzyme.symbol<1>], [#enzyme.symbol<2>]]} : tensor<2xf64>
    %2 = enzyme.getWeightFromTrace %0 : tensor<f64>
    %3 = arith.negf %2 : tensor<f64>
    %output_rng_state, %result = enzyme.random %arg0, %cst_5, %cst_7 {rng_distribution = #enzyme<rng_distribution MULTINORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<2x2xf64>) -> (tensor<2xui64>, tensor<2xf64>)
    %4 = enzyme.cholesky_solve %cst_7, %result : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %5 = enzyme.dot %result, %4 : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %6 = arith.mulf %5, %cst_3 : tensor<f64>
    %7 = arith.addf %3, %6 : tensor<f64>
    %8:3 = enzyme.autodiff_region(%1, %cst_4) {
    ^bb0(%arg3: tensor<2xf64>):
      %21:3 = func.call @test.update_0(%0, %arg3, %output_rng_state, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
      %22 = arith.negf %21#1 : tensor<f64>
      enzyme.yield %22, %21#0, %21#2 : tensor<f64>, !enzyme.Trace, tensor<2xui64>
    } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (!enzyme.Trace, tensor<2xui64>, tensor<2xf64>)
    %9:5 = enzyme.loop(%cst_2 : tensor<i64>) to(%cst_6 : tensor<i64>) step(%cst_1 : tensor<i64>) iter_args(%1, %result, %8#2, %0, %8#1 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, !enzyme.Trace, tensor<2xui64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, !enzyme.Trace, tensor<2xui64> {
    ^bb0(%arg3: tensor<i64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: !enzyme.Trace, %arg8: tensor<2xui64>):
      %21 = arith.mulf %arg6, %cst : tensor<2xf64>
      %22 = arith.addf %arg5, %21 : tensor<2xf64>
      %23 = enzyme.cholesky_solve %cst_7, %22 : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
      %24 = arith.mulf %23, %cst_0 : tensor<2xf64>
      %25 = arith.addf %arg4, %24 : tensor<2xf64>
      %26:3 = enzyme.autodiff_region(%25, %cst_4) {
      ^bb0(%arg9: tensor<2xf64>):
        %29:3 = func.call @test.update(%arg7, %arg9, %arg8, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
        %30 = arith.negf %29#1 : tensor<f64>
        enzyme.yield %30, %29#0, %29#2 : tensor<f64>, !enzyme.Trace, tensor<2xui64>
      } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (!enzyme.Trace, tensor<2xui64>, tensor<2xf64>)
      %27 = arith.mulf %26#2, %cst : tensor<2xf64>
      %28 = arith.addf %22, %27 : tensor<2xf64>
      enzyme.yield %25, %28, %26#2, %26#0, %26#1 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, !enzyme.Trace, tensor<2xui64>
    }
    %10 = enzyme.getWeightFromTrace %9#3 : tensor<f64>
    %11 = arith.negf %10 : tensor<f64>
    %12 = enzyme.cholesky_solve %cst_7, %9#1 : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %13 = enzyme.dot %9#1, %12 : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %14 = arith.mulf %13, %cst_3 : tensor<f64>
    %15 = arith.addf %11, %14 : tensor<f64>
    %16 = arith.subf %7, %15 : tensor<f64>
    %17 = math.exp %16 : tensor<f64>
    %18 = arith.minimumf %17, %cst_4 : tensor<f64>
    %output_rng_state_8, %result_9 = enzyme.random %9#4, %cst_5, %cst_4 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %19 = arith.cmpf olt, %result_9, %18 : tensor<f64>
    %20 = enzyme.selectTrace %19, %9#3, %0 : tensor<i1>
    return %20, %19, %output_rng_state_8 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
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
  func.func @test.update_0(%arg0: !enzyme.Trace, %arg1: tensor<2xf64>, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
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
// CPU-NEXT:    %c = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<12> : tensor<ui64>
// CPU-NEXT:    %cst = stablehlo.constant dense<1.4142135623730951> : tensor<2xf64>
// CPU-NEXT:    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<2xf64>
// CPU-NEXT:    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<2xf64>
// CPU-NEXT:    %c_3 = stablehlo.constant dense<4607182418800017408> : tensor<2xui64>
// CPU-NEXT:    %c_4 = stablehlo.constant dense<12> : tensor<2xui64>
// CPU-NEXT:    %cst_5 = stablehlo.constant dense<5.000000e-02> : tensor<2xf64>
// CPU-NEXT:    %cst_6 = stablehlo.constant dense<1.000000e-01> : tensor<2xf64>
// CPU-NEXT:    %c_7 = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT:    %cst_9 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CPU-NEXT:    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CPU-NEXT:    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %c_12 = stablehlo.constant dense<10> : tensor<i64>
// CPU-NEXT:    %cst_13 = stablehlo.constant dense<{{\[}}{{\[}}1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]{{\]}}> : tensor<2x2xf64>
// CPU-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CPU-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : !enzyme.Trace to tensor<ui64>
// CPU-NEXT:    %2 = enzyme.getFlattenedSamplesFromTrace %0 {selection = {{\[}}{{\[}}#enzyme.symbol<1>], [#enzyme.symbol<2>]{{\]}}} : tensor<2xf64>
// CPU-NEXT:    %3 = enzyme.getWeightFromTrace %0 : tensor<f64>
// CPU-NEXT:    %4 = stablehlo.negate %3 : tensor<f64>
// CPU-NEXT:    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CPU-NEXT:    %5 = stablehlo.shift_right_logical %output, %c_4 : tensor<2xui64>
// CPU-NEXT:    %6 = stablehlo.or %5, %c_3 : tensor<2xui64>
// CPU-NEXT:    %7 = stablehlo.bitcast_convert %6 : (tensor<2xui64>) -> tensor<2xf64>
// CPU-NEXT:    %8 = stablehlo.subtract %7, %cst_2 : tensor<2xf64>
// CPU-NEXT:    %9 = stablehlo.multiply %8, %cst_1 : tensor<2xf64>
// CPU-NEXT:    %10 = stablehlo.subtract %9, %cst_2 : tensor<2xf64>
// CPU-NEXT:    %11 = chlo.erf_inv %10 : tensor<2xf64> -> tensor<2xf64>
// CPU-NEXT:    %12 = stablehlo.multiply %11, %cst : tensor<2xf64>
// CPU-NEXT:    %13 = stablehlo.cholesky %cst_13, lower = true : tensor<2x2xf64>
// CPU-NEXT:    %14 = stablehlo.dot_general %13, %12, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CPU-NEXT:    %15 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %16 = stablehlo.add %15, %14 : tensor<2xf64>
// CPU-NEXT:    %17 = stablehlo.reshape %16 : (tensor<2xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %18 = "stablehlo.triangular_solve"(%13, %17) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %19 = "stablehlo.triangular_solve"(%13, %18) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %20 = stablehlo.reshape %19 : (tensor<2x1xf64>) -> tensor<2xf64>
// CPU-NEXT:    %21 = stablehlo.dot_general %16, %20, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CPU-NEXT:    %22 = stablehlo.multiply %21, %cst_9 : tensor<f64>
// CPU-NEXT:    %23 = stablehlo.add %4, %22 : tensor<f64>
// CPU-NEXT:    %24:2 = call @diffehmc_to_diff0(%2, %0, %output_state, %arg1, %arg2, %cst_10) : (tensor<2xf64>, !enzyme.Trace, tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
// CPU-NEXT:    %25:6 = stablehlo.while(%iterArg = %c_8, %iterArg_16 = %2, %iterArg_17 = %16, %iterArg_18 = %24#1, %iterArg_19 = %1, %iterArg_20 = %24#0) : tensor<i64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<ui64>, tensor<2xui64>
// CPU-NEXT:    cond {
// CPU-NEXT:      %45 = stablehlo.compare  LT, %iterArg, %c_12 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CPU-NEXT:      stablehlo.return %45 : tensor<i1>
// CPU-NEXT:    } do {
// CPU-NEXT:      %45 = builtin.unrealized_conversion_cast %iterArg_19 : tensor<ui64> to !enzyme.Trace
// CPU-NEXT:      %46 = stablehlo.multiply %iterArg_18, %cst_5 : tensor<2xf64>
// CPU-NEXT:      %47 = stablehlo.add %iterArg_17, %46 : tensor<2xf64>
// CPU-NEXT:      %48 = stablehlo.reshape %47 : (tensor<2xf64>) -> tensor<2x1xf64>
// CPU-NEXT:      %49 = "stablehlo.triangular_solve"(%13, %48) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:      %50 = "stablehlo.triangular_solve"(%13, %49) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:      %51 = stablehlo.reshape %50 : (tensor<2x1xf64>) -> tensor<2xf64>
// CPU-NEXT:      %52 = stablehlo.multiply %51, %cst_6 : tensor<2xf64>
// CPU-NEXT:      %53 = stablehlo.add %iterArg_16, %52 : tensor<2xf64>
// CPU-NEXT:      %54:3 = func.call @diffehmc_to_diff1(%53, %45, %iterArg_20, %arg1, %arg2, %cst_10) : (tensor<2xf64>, !enzyme.Trace, tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<2xui64>, tensor<2xf64>)
// CPU-NEXT:      %55 = builtin.unrealized_conversion_cast %54#0 : !enzyme.Trace to tensor<ui64>
// CPU-NEXT:      %56 = stablehlo.multiply %54#2, %cst_5 : tensor<2xf64>
// CPU-NEXT:      %57 = stablehlo.add %47, %56 : tensor<2xf64>
// CPU-NEXT:      %58 = stablehlo.add %iterArg, %c_7 : tensor<i64>
// CPU-NEXT:      stablehlo.return %58, %53, %57, %54#2, %55, %54#1 : tensor<i64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<ui64>, tensor<2xui64>
// CPU-NEXT:    }
// CPU-NEXT:    %26 = builtin.unrealized_conversion_cast %25#4 : tensor<ui64> to !enzyme.Trace
// CPU-NEXT:    %27 = enzyme.getWeightFromTrace %26 : tensor<f64>
// CPU-NEXT:    %28 = stablehlo.negate %27 : tensor<f64>
// CPU-NEXT:    %29 = stablehlo.reshape %25#2 : (tensor<2xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %30 = "stablehlo.triangular_solve"(%13, %29) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %31 = "stablehlo.triangular_solve"(%13, %30) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %32 = stablehlo.reshape %31 : (tensor<2x1xf64>) -> tensor<2xf64>
// CPU-NEXT:    %33 = stablehlo.dot_general %25#2, %32, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CPU-NEXT:    %34 = stablehlo.multiply %33, %cst_9 : tensor<f64>
// CPU-NEXT:    %35 = stablehlo.add %28, %34 : tensor<f64>
// CPU-NEXT:    %36 = stablehlo.subtract %23, %35 : tensor<f64>
// CPU-NEXT:    %37 = stablehlo.exponential %36 : tensor<f64>
// CPU-NEXT:    %38 = stablehlo.minimum %37, %cst_10 : tensor<f64>
// CPU-NEXT:    %output_state_14, %output_15 = stablehlo.rng_bit_generator %25#5, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
// CPU-NEXT:    %39 = stablehlo.shift_right_logical %output_15, %c_0 : tensor<ui64>
// CPU-NEXT:    %40 = stablehlo.or %39, %c : tensor<ui64>
// CPU-NEXT:    %41 = stablehlo.bitcast_convert %40 : (tensor<ui64>) -> tensor<f64>
// CPU-NEXT:    %42 = stablehlo.subtract %41, %cst_10 : tensor<f64>
// CPU-NEXT:    %43 = stablehlo.compare  LT, %42, %38,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CPU-NEXT:    %44 = enzyme.selectTrace %43, %26, %0 : tensor<i1>
// CPU-NEXT:    return %44, %43, %output_state_14 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
// CPU-NEXT:  }

// CPU:  func.func @hmc_to_diff1(%arg0: tensor<2xf64>, %arg1: !enzyme.Trace, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (tensor<f64>, !enzyme.Trace, tensor<2xui64>) {
// CPU-NEXT:    %0:3 = call @test.update(%arg1, %arg0, %arg2, %arg3, %arg4) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CPU-NEXT:    %1 = stablehlo.negate %0#1 : tensor<f64>
// CPU-NEXT:    return %1, %0#0, %0#2 : tensor<f64>, !enzyme.Trace, tensor<2xui64>
// CPU-NEXT:  }

// CPU:  func.func @hmc_to_diff0(%arg0: tensor<2xf64>, %arg1: !enzyme.Trace, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (tensor<f64>, !enzyme.Trace, tensor<2xui64>) {
// CPU-NEXT:    %0:3 = call @test.update_0(%arg1, %arg0, %arg2, %arg3, %arg4) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CPU-NEXT:    %1 = stablehlo.negate %0#1 : tensor<f64>
// CPU-NEXT:    return %1, %0#0, %0#2 : tensor<f64>, !enzyme.Trace, tensor<2xui64>
// CPU-NEXT:  }

// CPU:  func.func @test.update(%arg0: !enzyme.Trace, %arg1: tensor<2xf64>, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
// CPU-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CPU-NEXT:    %1 = stablehlo.slice %arg1 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %3 = call @logpdf(%2, %arg3, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %4 = enzyme.addSampleToTrace(%2 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
// CPU-NEXT:    %5 = stablehlo.slice %arg1 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %7 = call @logpdf(%6, %2, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %8 = stablehlo.add %3, %7 : tensor<f64>
// CPU-NEXT:    %9 = enzyme.addSampleToTrace(%6 : tensor<f64>) into %4 {symbol = #enzyme.symbol<2>}
// CPU-NEXT:    %10 = enzyme.addWeightToTrace(%8 : tensor<f64>) into %9
// CPU-NEXT:    %11 = enzyme.addRetvalToTrace(%6 : tensor<f64>) into %10
// CPU-NEXT:    return %11, %8, %arg2 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
// CPU-NEXT:  }

// CPU:  func.func @test.update_0(%arg0: !enzyme.Trace, %arg1: tensor<2xf64>, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>) {
// CPU-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CPU-NEXT:    %1 = stablehlo.slice %arg1 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %2 = stablehlo.reshape %1 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %3 = call @logpdf(%2, %arg3, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %4 = enzyme.addSampleToTrace(%2 : tensor<f64>) into %0 {symbol = #enzyme.symbol<1>}
// CPU-NEXT:    %5 = stablehlo.slice %arg1 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %6 = stablehlo.reshape %5 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %7 = call @logpdf(%6, %2, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %8 = stablehlo.add %3, %7 : tensor<f64>
// CPU-NEXT:    %9 = enzyme.addSampleToTrace(%6 : tensor<f64>) into %4 {symbol = #enzyme.symbol<2>}
// CPU-NEXT:    %10 = enzyme.addWeightToTrace(%8 : tensor<f64>) into %9
// CPU-NEXT:    %11 = enzyme.addRetvalToTrace(%6 : tensor<f64>) into %10
// CPU-NEXT:    return %11, %8, %arg2 : !enzyme.Trace, tensor<f64>, tensor<2xui64>
// CPU-NEXT:  }

// CPU:  func.func private @diffehmc_to_diff0(%arg0: tensor<2xf64>, %arg1: !enzyme.Trace, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>, %arg5: tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>) {
// CPU-NEXT:    %0:3 = call @test.update_0(%arg1, %arg0, %arg2, %arg3, %arg4) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CPU-NEXT:    %1 = stablehlo.negate %arg5 : tensor<f64>
// CPU-NEXT:    %2 = call @diffetest.update_0(%arg1, %arg0, %arg2, %arg3, %arg4, %1) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    return %0#2, %2 : tensor<2xui64>, tensor<2xf64>
// CPU-NEXT:  }

// CPU:  func.func private @diffetest.update_0(%arg0: !enzyme.Trace, %arg1: tensor<2xf64>, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>, %arg5: tensor<f64>) -> tensor<2xf64> {
// CPU-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %2 = call @logpdf(%1, %arg3, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %3 = stablehlo.slice %arg1 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %5 = call @logpdf(%4, %1, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %6:2 = call @diffelogpdf(%4, %1, %arg4, %arg5) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
// CPU-NEXT:    %7 = stablehlo.reshape %6#0 : (tensor<f64>) -> tensor<1xf64>
// CPU-NEXT:    %8 = stablehlo.pad %7, %cst, low = [1], high = [0], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %9 = call @diffelogpdf_0(%1, %arg3, %arg4, %arg5) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %10 = arith.addf %6#1, %9 : tensor<f64>
// CPU-NEXT:    %11 = stablehlo.reshape %10 : (tensor<f64>) -> tensor<1xf64>
// CPU-NEXT:    %12 = stablehlo.pad %11, %cst, low = [0], high = [1], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %13 = arith.addf %8, %12 : tensor<2xf64>
// CPU-NEXT:    return %13 : tensor<2xf64>
// CPU-NEXT:  }

// CPU:  func.func private @diffelogpdf(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
// CPU-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    return %cst, %cst : tensor<f64>, tensor<f64>
// CPU-NEXT:  }

// CPU:  func.func private @diffelogpdf_0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> tensor<f64> {
// CPU-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    return %cst : tensor<f64>
// CPU-NEXT:  }

// CPU:  func.func private @diffehmc_to_diff1(%arg0: tensor<2xf64>, %arg1: !enzyme.Trace, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>, %arg5: tensor<f64>) -> (!enzyme.Trace, tensor<2xui64>, tensor<2xf64>) {
// CPU-NEXT:    %0:3 = call @test.update(%arg1, %arg0, %arg2, %arg3, %arg4) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
// CPU-NEXT:    %1 = stablehlo.negate %arg5 : tensor<f64>
// CPU-NEXT:    %2 = call @diffetest.update(%arg1, %arg0, %arg2, %arg3, %arg4, %1) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    return %0#0, %0#2, %2 : !enzyme.Trace, tensor<2xui64>, tensor<2xf64>
// CPU-NEXT:  }

// CPU:  func.func private @diffetest.update(%arg0: !enzyme.Trace, %arg1: tensor<2xf64>, %arg2: tensor<2xui64>, %arg3: tensor<f64>, %arg4: tensor<f64>, %arg5: tensor<f64>) -> tensor<2xf64> {
// CPU-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %0 = stablehlo.slice %arg1 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %2 = call @logpdf(%1, %arg3, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %3 = stablehlo.slice %arg1 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %4 = stablehlo.reshape %3 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %5 = call @logpdf(%4, %1, %arg4) : (tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %6:2 = call @diffelogpdf(%4, %1, %arg4, %arg5) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
// CPU-NEXT:    %7 = stablehlo.reshape %6#0 : (tensor<f64>) -> tensor<1xf64>
// CPU-NEXT:    %8 = stablehlo.pad %7, %cst, low = [1], high = [0], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %9 = call @diffelogpdf_0(%1, %arg3, %arg4, %arg5) : (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>) -> tensor<f64>
// CPU-NEXT:    %10 = arith.addf %6#1, %9 : tensor<f64>
// CPU-NEXT:    %11 = stablehlo.reshape %10 : (tensor<f64>) -> tensor<1xf64>
// CPU-NEXT:    %12 = stablehlo.pad %11, %cst, low = [0], high = [1], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %13 = arith.addf %8, %12 : tensor<2xf64>
// CPU-NEXT:    return %13 : tensor<2xf64>
// CPU-NEXT:  }