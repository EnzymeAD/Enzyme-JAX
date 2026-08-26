// RUN: enzymexlamlir-opt --transform-interpreter %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.add_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.sub_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.and_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.max_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.min_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.or_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.xor_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.mul_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.div_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.rem_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.pow_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.shift_right_logical_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.slice_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.convert_simplify {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.dynamic_slice_to_static {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.dynamic_update_slice_elim {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.if_inline
      transform.apply_patterns.enzyme_hlo.if_to_select
      transform.apply_patterns.enzyme_hlo.divide_sqrt_to_multiply_rsqrt {benefit = 16 : i64}
      transform.apply_patterns.enzyme_hlo.replace_neg_add_with_subtract
      transform.apply_patterns.enzyme_hlo.replace_subtract_neg_with_add
      transform.apply_patterns.enzyme_hlo.binop_const_simplify
      transform.apply_patterns.enzyme_hlo.not_select_simplify
      transform.apply_patterns.enzyme_hlo.common_compare_expression_rewrite
      transform.apply_patterns.enzyme_hlo.compare_select_simplify
      transform.apply_patterns.enzyme_hlo.while_simplify {parameter = true}
      transform.apply_patterns.enzyme_hlo.greedy_while_loop_batch_fission
    } : !transform.any_op
    transform.yield 
  }
  func.func @logpdf(%arg0: tensor<6x6xf64> {enzymexla.memory_effects = []}) -> (tensor<f64>, tensor<6x6xf64>) attributes {enzymexla.memory_effects = []} {
    %cst = stablehlo.constant dense<-10.974465114022347> : tensor<f64>
    %c = stablehlo.constant dense<2> : tensor<i64>
    %cst_0 = stablehlo.constant dense<1.010000e+00> : tensor<f64>
    %c_1 = stablehlo.constant dense<1> : tensor<i32>
    %c_2 = stablehlo.constant dense<4> : tensor<i64>
    %cst_3 = stablehlo.constant dense<0.010000000000000002> : tensor<f64>
    %c_4 = stablehlo.constant dense<6> : tensor<i64>
    %c_5 = stablehlo.constant dense<1> : tensor<i64>
    %c_6 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    %1 = stablehlo.convert %c_6 : (tensor<i64>) -> tensor<f64>
    %2 = stablehlo.convert %c_5 : tensor<i64>
    %3 = stablehlo.convert %c_4 : tensor<i64>
    %4 = stablehlo.convert %c_6 : tensor<i64>
    %5 = stablehlo.convert %cst_3 : tensor<f64>
    %6:7 = stablehlo.while(%iterArg = %4, %iterArg_7 = %2, %iterArg_8 = %3, %iterArg_9 = %2, %iterArg_10 = %0, %iterArg_11 = %1, %iterArg_12 = %5) : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<f64> attributes {enzyme.disable_mincut}
    cond {
      %15 = stablehlo.subtract %iterArg_8, %iterArg_9 : tensor<i64>
      %16 = stablehlo.divide %15, %iterArg_7 : tensor<i64>
      %17 = stablehlo.add %16, %2 : tensor<i64>
      %18 = stablehlo.compare  LT, %iterArg, %17 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %18 : tensor<i1>
    } do {
      %15 = stablehlo.multiply %iterArg, %iterArg_7 : tensor<i64>
      %16 = stablehlo.add %iterArg_9, %15 : tensor<i64>
      %17 = stablehlo.add %iterArg, %2 : tensor<i64>
      %18:8 = stablehlo.while(%iterArg_13 = %4, %iterArg_14 = %2, %iterArg_15 = %2, %iterArg_16 = %3, %iterArg_17 = %iterArg_10, %iterArg_18 = %iterArg_11, %iterArg_19 = %iterArg_12, %iterArg_20 = %16) : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<f64>, tensor<i64> attributes {enzyme.disable_mincut}
      cond {
        %19 = stablehlo.subtract %iterArg_16, %iterArg_15 : tensor<i64>
        %20 = stablehlo.divide %19, %iterArg_14 : tensor<i64>
        %21 = stablehlo.add %20, %2 : tensor<i64>
        %22 = stablehlo.compare  LT, %iterArg_13, %21 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %22 : tensor<i1>
      } do {
        %19 = stablehlo.multiply %iterArg_13, %iterArg_14 : tensor<i64>
        %20 = stablehlo.add %iterArg_15, %19 : tensor<i64>
        %21 = stablehlo.add %iterArg_13, %2 : tensor<i64>
        %22 = stablehlo.convert %c_2 : (tensor<i64>) -> tensor<f64>
        %23 = stablehlo.add %22, %iterArg_19 : tensor<f64>
        %24 = stablehlo.convert %20 : (tensor<i64>) -> tensor<i32>
        %25 = stablehlo.convert %c_1 : tensor<i32>
        %26 = stablehlo.subtract %24, %25 : tensor<i32>
        %27 = stablehlo.convert %iterArg_20 : (tensor<i64>) -> tensor<i32>
        %28 = stablehlo.subtract %27, %25 : tensor<i32>
        %29 = stablehlo.dynamic_slice %iterArg_17, %26, %28, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
        %30 = stablehlo.transpose %29, dims = [1, 0] : (tensor<1x1xf64>) -> tensor<1x1xf64>
        %31 = stablehlo.reshape %30 : (tensor<1x1xf64>) -> tensor<f64>
        %32 = stablehlo.transpose %31, dims = [] : (tensor<f64>) -> tensor<f64>
        %33 = stablehlo.multiply %23, %32 : tensor<f64>
        %34 = stablehlo.compare  LT, %20, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %35:8 = "stablehlo.if"(%34) ({
          %52 = stablehlo.add %20, %2 : tensor<i64>
          %53 = stablehlo.convert %52 : (tensor<i64>) -> tensor<i32>
          %54 = stablehlo.subtract %53, %25 : tensor<i32>
          %55 = stablehlo.dynamic_slice %iterArg_17, %54, %28, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
          %56 = stablehlo.transpose %55, dims = [1, 0] : (tensor<1x1xf64>) -> tensor<1x1xf64>
          %57 = stablehlo.reshape %56 : (tensor<1x1xf64>) -> tensor<f64>
          %58 = stablehlo.transpose %57, dims = [] : (tensor<f64>) -> tensor<f64>
          %59 = stablehlo.subtract %33, %58 : tensor<f64>
          stablehlo.return %iterArg_17, %33, %iterArg_20, %20, %iterArg_17, %59, %iterArg_20, %20 : tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>
        }, {
          stablehlo.return %iterArg_17, %33, %iterArg_20, %20, %iterArg_17, %33, %iterArg_20, %20 : tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>
        }) : (tensor<i1>) -> (tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>)
        %36 = stablehlo.compare  LT, %35#6, %3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %37:8 = "stablehlo.if"(%36) ({
          %52 = stablehlo.add %35#6, %2 : tensor<i64>
          %53 = stablehlo.convert %35#7 : (tensor<i64>) -> tensor<i32>
          %54 = stablehlo.subtract %53, %25 : tensor<i32>
          %55 = stablehlo.convert %52 : (tensor<i64>) -> tensor<i32>
          %56 = stablehlo.subtract %55, %25 : tensor<i32>
          %57 = stablehlo.dynamic_slice %35#4, %54, %56, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
          %58 = stablehlo.transpose %57, dims = [1, 0] : (tensor<1x1xf64>) -> tensor<1x1xf64>
          %59 = stablehlo.reshape %58 : (tensor<1x1xf64>) -> tensor<f64>
          %60 = stablehlo.transpose %59, dims = [] : (tensor<f64>) -> tensor<f64>
          %61 = stablehlo.subtract %35#5, %60 : tensor<f64>
          stablehlo.return %35#4, %35#5, %35#6, %35#7, %35#4, %61, %35#6, %35#7 : tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>
        }, {
          stablehlo.return %35#4, %35#5, %35#6, %35#7, %35#4, %35#5, %35#6, %35#7 : tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>
        }) : (tensor<i1>) -> (tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>)
        %38 = stablehlo.compare  GT, %37#7, %2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %39:8 = "stablehlo.if"(%38) ({
          %52 = stablehlo.subtract %37#7, %2 : tensor<i64>
          %53 = stablehlo.convert %52 : (tensor<i64>) -> tensor<i32>
          %54 = stablehlo.subtract %53, %25 : tensor<i32>
          %55 = stablehlo.convert %37#6 : (tensor<i64>) -> tensor<i32>
          %56 = stablehlo.subtract %55, %25 : tensor<i32>
          %57 = stablehlo.dynamic_slice %37#4, %54, %56, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
          %58 = stablehlo.transpose %57, dims = [1, 0] : (tensor<1x1xf64>) -> tensor<1x1xf64>
          %59 = stablehlo.reshape %58 : (tensor<1x1xf64>) -> tensor<f64>
          %60 = stablehlo.transpose %59, dims = [] : (tensor<f64>) -> tensor<f64>
          %61 = stablehlo.subtract %37#5, %60 : tensor<f64>
          stablehlo.return %37#4, %37#5, %37#6, %37#7, %37#4, %61, %37#6, %37#7 : tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>
        }, {
          stablehlo.return %37#4, %37#5, %37#6, %37#7, %37#4, %37#5, %37#6, %37#7 : tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>
        }) : (tensor<i1>) -> (tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>)
        %40 = stablehlo.compare  GT, %39#6, %2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %41:8 = "stablehlo.if"(%40) ({
          %52 = stablehlo.subtract %39#6, %2 : tensor<i64>
          %53 = stablehlo.convert %39#7 : (tensor<i64>) -> tensor<i32>
          %54 = stablehlo.subtract %53, %25 : tensor<i32>
          %55 = stablehlo.convert %52 : (tensor<i64>) -> tensor<i32>
          %56 = stablehlo.subtract %55, %25 : tensor<i32>
          %57 = stablehlo.dynamic_slice %39#4, %54, %56, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
          %58 = stablehlo.transpose %57, dims = [1, 0] : (tensor<1x1xf64>) -> tensor<1x1xf64>
          %59 = stablehlo.reshape %58 : (tensor<1x1xf64>) -> tensor<f64>
          %60 = stablehlo.transpose %59, dims = [] : (tensor<f64>) -> tensor<f64>
          %61 = stablehlo.subtract %39#5, %60 : tensor<f64>
          stablehlo.return %39#4, %39#5, %39#6, %39#7, %39#4, %61, %39#6, %39#7 : tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>
        }, {
          stablehlo.return %39#4, %39#5, %39#6, %39#7, %39#4, %39#5, %39#6, %39#7 : tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>
        }) : (tensor<i1>) -> (tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<i64>, tensor<i64>)
        %42 = stablehlo.convert %35#3 : (tensor<i64>) -> tensor<i32>
        %43 = stablehlo.subtract %42, %25 : tensor<i32>
        %44 = stablehlo.convert %35#2 : (tensor<i64>) -> tensor<i32>
        %45 = stablehlo.subtract %44, %25 : tensor<i32>
        %46 = stablehlo.dynamic_slice %35#0, %43, %45, sizes = [1, 1] : (tensor<6x6xf64>, tensor<i32>, tensor<i32>) -> tensor<1x1xf64>
        %47 = stablehlo.transpose %46, dims = [1, 0] : (tensor<1x1xf64>) -> tensor<1x1xf64>
        %48 = stablehlo.reshape %47 : (tensor<1x1xf64>) -> tensor<f64>
        %49 = stablehlo.transpose %48, dims = [] : (tensor<f64>) -> tensor<f64>
        %50 = stablehlo.multiply %41#5, %49 : tensor<f64>
        %51 = stablehlo.add %iterArg_18, %50 : tensor<f64>
        stablehlo.return %21, %iterArg_14, %iterArg_15, %iterArg_16, %35#0, %51, %iterArg_19, %35#2 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<f64>, tensor<i64>
      }
      stablehlo.return %17, %iterArg_7, %iterArg_8, %iterArg_9, %18#4, %18#5, %18#6 : tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<6x6xf64>, tensor<f64>, tensor<f64>
    }
    %7 = stablehlo.convert %cst_0 : tensor<f64>
    %8 = stablehlo.divide %6#5, %7 : tensor<f64>
    %9 = stablehlo.negate %8 : tensor<f64>
    %10 = stablehlo.convert %c : (tensor<i64>) -> tensor<f64>
    %11 = stablehlo.divide %9, %10 : tensor<f64>
    %12 = stablehlo.convert %cst : tensor<f64>
    %13 = stablehlo.add %11, %12 : tensor<f64>
    %14 = stablehlo.transpose %6#4, dims = [1, 0] : (tensor<6x6xf64>) -> tensor<6x6xf64>
    return %13, %14 : tensor<f64>, tensor<6x6xf64>
  }
}

// CHECK: func.func @logpdf
