// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(arith-raise,lower-probprog-to-stablehlo{backend=cpu},canonicalize,outline-enzyme-regions,enzyme,canonicalize,remove-unnecessary-enzyme-ops,canonicalize,enzyme-simplify-math,cse,inline,cse)" | FileCheck %s --check-prefix=CPU --dump-input=always

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
    %output_rng_state, %result = enzyme.random %arg0, %cst_4, %cst_7 {rng_distribution = #enzyme<rng_distribution MULTINORMAL>} : (tensor<2xui64>, tensor<f64>, tensor<2x2xf64>) -> (tensor<2xui64>, tensor<2xf64>)
    %4 = enzyme.cholesky_solve %cst_7, %result : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %5 = enzyme.dot %result, %4 : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %6 = arith.mulf %5, %cst_2 : tensor<f64>
    %7 = arith.addf %3, %6 : tensor<f64>
    %8:2 = enzyme.autodiff_region(%1, %cst_3) {
    ^bb0(%arg3: tensor<2xf64>):
      %23:3 = func.call @test.update(%0, %arg3, %output_rng_state, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
      %24 = arith.negf %23#1 : tensor<f64>
      enzyme.yield %24, %23#2 : tensor<f64>, tensor<2xui64>
    } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
    %9 = "enzyme.broadcast"(%cst_6) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
    %10 = "enzyme.broadcast"(%cst) <{shape = array<i64: 2>}> : (tensor<f64>) -> tensor<2xf64>
    %11:4 = enzyme.for_loop(%cst_1 : tensor<i64>) to(%cst_5 : tensor<i64>) step(%cst_0 : tensor<i64>) iter_args(%1, %result, %8#1, %8#0 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64>) -> tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64> {
    ^bb0(%arg3: tensor<i64>, %arg4: tensor<2xf64>, %arg5: tensor<2xf64>, %arg6: tensor<2xf64>, %arg7: tensor<2xui64>):
      %23 = arith.mulf %10, %arg6 : tensor<2xf64>
      %24 = arith.subf %arg5, %23 : tensor<2xf64>
      %25 = enzyme.cholesky_solve %cst_7, %24 : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
      %26 = arith.mulf %9, %25 : tensor<2xf64>
      %27 = arith.addf %arg4, %26 : tensor<2xf64>
      %28:2 = enzyme.autodiff_region(%27, %cst_3) {
      ^bb0(%arg8: tensor<2xf64>):
        %31:3 = func.call @test.update(%0, %arg8, %arg7, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
        %32 = arith.negf %31#1 : tensor<f64>
        enzyme.yield %32, %31#2 : tensor<f64>, tensor<2xui64>
      } attributes {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]} : (tensor<2xf64>, tensor<f64>) -> (tensor<2xui64>, tensor<2xf64>)
      %29 = arith.mulf %10, %28#1 : tensor<2xf64>
      %30 = arith.subf %24, %29 : tensor<2xf64>
      enzyme.yield %27, %30, %28#1, %28#0 : tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64>
    }
    %12:3 = call @test.update(%0, %11#0, %11#3, %arg1, %arg2) : (!enzyme.Trace, tensor<2xf64>, tensor<2xui64>, tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>, tensor<2xui64>)
    %13 = arith.negf %12#1 : tensor<f64>
    %14 = enzyme.cholesky_solve %cst_7, %11#1 : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
    %15 = enzyme.dot %11#1, %14 : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
    %16 = arith.mulf %15, %cst_2 : tensor<f64>
    %17 = arith.addf %13, %16 : tensor<f64>
    %18 = arith.subf %7, %17 : tensor<f64>
    %19 = math.exp %18 : tensor<f64>
    %20 = arith.minimumf %19, %cst_3 : tensor<f64>
    %output_rng_state_8, %result_9 = enzyme.random %12#2, %cst_4, %cst_3 {rng_distribution = #enzyme<rng_distribution UNIFORM>} : (tensor<2xui64>, tensor<f64>, tensor<f64>) -> (tensor<2xui64>, tensor<f64>)
    %21 = arith.cmpf olt, %result_9, %20 : tensor<f64>
    %22 = enzyme.selectTrace %21, %12#0, %0 : tensor<i1>
    return %22, %21, %output_rng_state_8 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
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
// CPU-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %c = stablehlo.constant dense<4607182418800017408> : tensor<ui64>
// CPU-NEXT:    %c_0 = stablehlo.constant dense<12> : tensor<ui64>
// CPU-NEXT:    %cst_1 = stablehlo.constant dense<1.4142135623730951> : tensor<2xf64>
// CPU-NEXT:    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<2xf64>
// CPU-NEXT:    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<2xf64>
// CPU-NEXT:    %c_4 = stablehlo.constant dense<4607182418800017408> : tensor<2xui64>
// CPU-NEXT:    %c_5 = stablehlo.constant dense<12> : tensor<2xui64>
// CPU-NEXT:    %cst_6 = stablehlo.constant dense<5.000000e-02> : tensor<f64>
// CPU-NEXT:    %c_7 = stablehlo.constant dense<1> : tensor<i64>
// CPU-NEXT:    %c_8 = stablehlo.constant dense<0> : tensor<i64>
// CPU-NEXT:    %cst_9 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CPU-NEXT:    %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
// CPU-NEXT:    %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CPU-NEXT:    %c_12 = stablehlo.constant dense<10> : tensor<i64>
// CPU-NEXT:    %cst_13 = stablehlo.constant dense<1.000000e-01> : tensor<f64>
// CPU-NEXT:    %cst_14 = stablehlo.constant dense<{{\[}}[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]{{\]}}> : tensor<2x2xf64>
// CPU-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
// CPU-NEXT:    %1 = enzyme.getFlattenedSamplesFromTrace %0 {selection = {{\[}}[#enzyme.symbol<1>], [#enzyme.symbol<2>]{{\]}}} : tensor<2xf64>
// CPU-NEXT:    %2 = enzyme.getWeightFromTrace %0 : tensor<f64>
// CPU-NEXT:    %3 = stablehlo.negate %2 : tensor<f64>
// CPU-NEXT:    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui64>)
// CPU-NEXT:    %4 = stablehlo.shift_right_logical %output, %c_5 : tensor<2xui64>
// CPU-NEXT:    %5 = stablehlo.or %4, %c_4 : tensor<2xui64>
// CPU-NEXT:    %6 = stablehlo.bitcast_convert %5 : (tensor<2xui64>) -> tensor<2xf64>
// CPU-NEXT:    %7 = stablehlo.subtract %6, %cst_3 : tensor<2xf64>
// CPU-NEXT:    %8 = stablehlo.multiply %7, %cst_2 : tensor<2xf64>
// CPU-NEXT:    %9 = stablehlo.subtract %8, %cst_3 : tensor<2xf64>
// CPU-NEXT:    %10 = chlo.erf_inv %9 : tensor<2xf64> -> tensor<2xf64>
// CPU-NEXT:    %11 = stablehlo.multiply %10, %cst_1 : tensor<2xf64>
// CPU-NEXT:    %12 = stablehlo.cholesky %cst_14, lower = true : tensor<2x2xf64>
// CPU-NEXT:    %13 = stablehlo.dot_general %12, %11, contracting_dims = [1] x [0] : (tensor<2x2xf64>, tensor<2xf64>) -> tensor<2xf64>
// CPU-NEXT:    %14 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %15 = stablehlo.add %14, %13 : tensor<2xf64>
// CPU-NEXT:    %16 = stablehlo.reshape %15 : (tensor<2xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %17 = "stablehlo.triangular_solve"(%12, %16) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %18 = "stablehlo.triangular_solve"(%12, %17) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %19 = stablehlo.reshape %18 : (tensor<2x1xf64>) -> tensor<2xf64>
// CPU-NEXT:    %20 = stablehlo.dot_general %15, %19, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CPU-NEXT:    %21 = stablehlo.multiply %20, %cst_9 : tensor<f64>
// CPU-NEXT:    %22 = stablehlo.add %3, %21 : tensor<f64>
// CPU-NEXT:    %23 = stablehlo.reshape %cst : (tensor<f64>) -> tensor<1xf64>
// CPU-NEXT:    %24 = stablehlo.pad %23, %cst, low = [1], high = [0], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %25 = stablehlo.pad %23, %cst, low = [0], high = [1], interior = [0] : (tensor<1xf64>, tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %26 = arith.addf %24, %25 : tensor<2xf64>
// CPU-NEXT:    %27 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %28 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f64>) -> tensor<2xf64>
// CPU-NEXT:    %29:5 = stablehlo.while(%iterArg = %c_8, %iterArg_17 = %1, %iterArg_18 = %15, %iterArg_19 = %26, %iterArg_20 = %output_state) : tensor<i64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64>
// CPU-NEXT:    cond {
// CPU-NEXT:      %57 = stablehlo.compare  LT, %iterArg, %c_12 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CPU-NEXT:      stablehlo.return %57 : tensor<i1>
// CPU-NEXT:    } do {
// CPU-NEXT:      %57 = stablehlo.multiply %28, %iterArg_19 : tensor<2xf64>
// CPU-NEXT:      %58 = stablehlo.subtract %iterArg_18, %57 : tensor<2xf64>
// CPU-NEXT:      %59 = stablehlo.reshape %58 : (tensor<2xf64>) -> tensor<2x1xf64>
// CPU-NEXT:      %60 = "stablehlo.triangular_solve"(%12, %59) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:      %61 = "stablehlo.triangular_solve"(%12, %60) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:      %62 = stablehlo.reshape %61 : (tensor<2x1xf64>) -> tensor<2xf64>
// CPU-NEXT:      %63 = stablehlo.multiply %27, %62 : tensor<2xf64>
// CPU-NEXT:      %64 = stablehlo.add %iterArg_17, %63 : tensor<2xf64>
// CPU-NEXT:      %65 = stablehlo.multiply %28, %26 : tensor<2xf64>
// CPU-NEXT:      %66 = stablehlo.subtract %58, %65 : tensor<2xf64>
// CPU-NEXT:      %67 = stablehlo.add %iterArg, %c_7 : tensor<i64>
// CPU-NEXT:      stablehlo.return %67, %64, %66, %26, %iterArg_20 : tensor<i64>, tensor<2xf64>, tensor<2xf64>, tensor<2xf64>, tensor<2xui64>
// CPU-NEXT:    }
// CPU-NEXT:    %30 = enzyme.initTrace : !enzyme.Trace
// CPU-NEXT:    %31 = stablehlo.slice %29#1 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %32 = stablehlo.reshape %31 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %33 = enzyme.addSampleToTrace(%32 : tensor<f64>) into %30 {symbol = #enzyme.symbol<1>}
// CPU-NEXT:    %34 = stablehlo.slice %29#1 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
// CPU-NEXT:    %35 = stablehlo.reshape %34 : (tensor<1xf64>) -> tensor<f64>
// CPU-NEXT:    %36 = stablehlo.add %cst_11, %cst_11 : tensor<f64>
// CPU-NEXT:    %37 = enzyme.addSampleToTrace(%35 : tensor<f64>) into %33 {symbol = #enzyme.symbol<2>}
// CPU-NEXT:    %38 = enzyme.addWeightToTrace(%36 : tensor<f64>) into %37
// CPU-NEXT:    %39 = enzyme.addRetvalToTrace(%35 : tensor<f64>) into %38
// CPU-NEXT:    %40 = stablehlo.negate %36 : tensor<f64>
// CPU-NEXT:    %41 = stablehlo.reshape %29#2 : (tensor<2xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %42 = "stablehlo.triangular_solve"(%12, %41) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %43 = "stablehlo.triangular_solve"(%12, %42) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf64>, tensor<2x1xf64>) -> tensor<2x1xf64>
// CPU-NEXT:    %44 = stablehlo.reshape %43 : (tensor<2x1xf64>) -> tensor<2xf64>
// CPU-NEXT:    %45 = stablehlo.dot_general %29#2, %44, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2xf64>) -> tensor<f64>
// CPU-NEXT:    %46 = stablehlo.multiply %45, %cst_9 : tensor<f64>
// CPU-NEXT:    %47 = stablehlo.add %40, %46 : tensor<f64>
// CPU-NEXT:    %48 = stablehlo.subtract %22, %47 : tensor<f64>
// CPU-NEXT:    %49 = stablehlo.exponential %48 : tensor<f64>
// CPU-NEXT:    %50 = stablehlo.minimum %49, %cst_10 : tensor<f64>
// CPU-NEXT:    %output_state_15, %output_16 = stablehlo.rng_bit_generator %29#4, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
// CPU-NEXT:    %51 = stablehlo.shift_right_logical %output_16, %c_0 : tensor<ui64>
// CPU-NEXT:    %52 = stablehlo.or %51, %c : tensor<ui64>
// CPU-NEXT:    %53 = stablehlo.bitcast_convert %52 : (tensor<ui64>) -> tensor<f64>
// CPU-NEXT:    %54 = stablehlo.subtract %53, %cst_10 : tensor<f64>
// CPU-NEXT:    %55 = stablehlo.compare  LT, %54, %50,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
// CPU-NEXT:    %56 = enzyme.selectTrace %55, %39, %0 : tensor<i1>
// CPU-NEXT:    return %56, %55, %output_state_15 : !enzyme.Trace, tensor<i1>, tensor<2xui64>
// CPU-NEXT:  }
