// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu})" | FileCheck %s --check-prefix=SHLO
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(lower-probprog-to-stablehlo{backend=cpu},lower-probprog-trace-ops{backend=cpu})" | FileCheck %s --check-prefix=CPU

module {
  // SHLO:    func.func @test_simple_loop(%arg0: tensor<i64>) -> tensor<f64> {
  // SHLO-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
  // SHLO-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
  // SHLO-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // SHLO-NEXT:    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_1 = %cst) : tensor<i64>, tensor<f64>
  // SHLO-NEXT:    cond {
  // SHLO-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %arg0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // SHLO-NEXT:      stablehlo.return %1 : tensor<i1>
  // SHLO-NEXT:    } do {
  // SHLO-NEXT:      %1 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<f64>
  // SHLO-NEXT:      %2 = stablehlo.add %iterArg_1, %1 : tensor<f64>
  // SHLO-NEXT:      %3 = stablehlo.add %iterArg, %c_0 : tensor<i64>
  // SHLO-NEXT:      stablehlo.return %3, %2 : tensor<i64>, tensor<f64>
  // SHLO-NEXT:    }
  // SHLO-NEXT:    return %0#1 : tensor<f64>
  // SHLO-NEXT:  }
  func.func @test_simple_loop(%n: tensor<i64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %init = stablehlo.constant dense<0.0> : tensor<f64>

    %result = enzyme.for_loop (%c0 : tensor<i64>) to (%n : tensor<i64>) step (%c1 : tensor<i64>)
      iter_args(%init : tensor<f64>)
      -> tensor<f64> {
    ^bb0(%iv: tensor<i64>, %sum_iter: tensor<f64>):
      %iv_f64 = stablehlo.convert %iv : (tensor<i64>) -> tensor<f64>
      %sum_next = stablehlo.add %sum_iter, %iv_f64 : tensor<f64>
      enzyme.yield %sum_next : tensor<f64>
    }

    return %result : tensor<f64>
  }

  // SHLO:  func.func @test_nested_loop(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<f64> {
  // SHLO-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
  // SHLO-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
  // SHLO-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // SHLO-NEXT:    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_1 = %cst) : tensor<i64>, tensor<f64>
  // SHLO-NEXT:    cond {
  // SHLO-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %arg0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // SHLO-NEXT:      stablehlo.return %1 : tensor<i1>
  // SHLO-NEXT:    } do {
  // SHLO-NEXT:      %1:2 = stablehlo.while(%iterArg_2 = %c, %iterArg_3 = %iterArg_1) : tensor<i64>, tensor<f64>
  // SHLO-NEXT:      cond {
  // SHLO-NEXT:        %3 = stablehlo.compare  LT, %iterArg_2, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // SHLO-NEXT:        stablehlo.return %3 : tensor<i1>
  // SHLO-NEXT:      } do {
  // SHLO-NEXT:        %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  // SHLO-NEXT:        %3 = stablehlo.add %iterArg_3, %cst_4 : tensor<f64>
  // SHLO-NEXT:        %4 = stablehlo.add %iterArg_2, %c_0 : tensor<i64>
  // SHLO-NEXT:        stablehlo.return %4, %3 : tensor<i64>, tensor<f64>
  // SHLO-NEXT:      }
  // SHLO-NEXT:      %2 = stablehlo.add %iterArg, %c_0 : tensor<i64>
  // SHLO-NEXT:      stablehlo.return %2, %1#1 : tensor<i64>, tensor<f64>
  // SHLO-NEXT:    }
  // SHLO-NEXT:    return %0#1 : tensor<f64>
  // SHLO-NEXT:  }
  func.func @test_nested_loop(%m: tensor<i64>, %n: tensor<i64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %init = stablehlo.constant dense<0.0> : tensor<f64>

    %result = enzyme.for_loop (%c0 : tensor<i64>) to (%m : tensor<i64>) step (%c1 : tensor<i64>)
      iter_args(%init : tensor<f64>)
      -> tensor<f64> {
    ^bb0(%i: tensor<i64>, %outer_sum_iter: tensor<f64>):
      %inner_result = enzyme.for_loop (%c0 : tensor<i64>) to (%n : tensor<i64>) step (%c1 : tensor<i64>)
        iter_args(%outer_sum_iter : tensor<f64>)
        -> tensor<f64> {
      ^bb1(%j: tensor<i64>, %inner_sum_iter: tensor<f64>):
        %inc = stablehlo.constant dense<1.0> : tensor<f64>
        %inner_sum_next = stablehlo.add %inner_sum_iter, %inc : tensor<f64>
        enzyme.yield %inner_sum_next : tensor<f64>
      }
      enzyme.yield %inner_result : tensor<f64>
    }

    return %result : tensor<f64>
  }

  // SHLO:  func.func @test_loop_with_multiple_iter_args(%arg0: tensor<i64>) -> (tensor<f64>, tensor<f64>) {
  // SHLO-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
  // SHLO-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
  // SHLO-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // SHLO-NEXT:    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  // SHLO-NEXT:    %0:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst, %iterArg_3 = %cst_1) : tensor<i64>, tensor<f64>, tensor<f64>
  // SHLO-NEXT:    cond {
  // SHLO-NEXT:      %1 = stablehlo.compare  LT, %iterArg, %arg0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // SHLO-NEXT:      stablehlo.return %1 : tensor<i1>
  // SHLO-NEXT:    } do {
  // SHLO-NEXT:      %1 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<f64>
  // SHLO-NEXT:      %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
  // SHLO-NEXT:      %2 = stablehlo.add %1, %cst_4 : tensor<f64>
  // SHLO-NEXT:      %3 = stablehlo.add %iterArg_2, %1 : tensor<f64>
  // SHLO-NEXT:      %4 = stablehlo.multiply %iterArg_3, %2 : tensor<f64>
  // SHLO-NEXT:      %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
  // SHLO-NEXT:      stablehlo.return %5, %3, %4 : tensor<i64>, tensor<f64>, tensor<f64>
  // SHLO-NEXT:    }
  // SHLO-NEXT:    return %0#1, %0#2 : tensor<f64>, tensor<f64>
  // SHLO-NEXT:  }
  func.func @test_loop_with_multiple_iter_args(%n: tensor<i64>) -> (tensor<f64>, tensor<f64>) {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %init_sum = stablehlo.constant dense<0.0> : tensor<f64>
    %init_prod = stablehlo.constant dense<1.0> : tensor<f64>

    %sum, %prod = enzyme.for_loop (%c0 : tensor<i64>) to (%n : tensor<i64>) step (%c1 : tensor<i64>)
      iter_args(%init_sum, %init_prod : tensor<f64>, tensor<f64>)
      -> tensor<f64>, tensor<f64> {
    ^bb0(%iv: tensor<i64>, %s_iter: tensor<f64>, %p_iter: tensor<f64>):
      %iv_f64 = stablehlo.convert %iv : (tensor<i64>) -> tensor<f64>
      %one = stablehlo.constant dense<1.0> : tensor<f64>
      %iv_plus_one = stablehlo.add %iv_f64, %one : tensor<f64>

      %s_next = stablehlo.add %s_iter, %iv_f64 : tensor<f64>
      %p_next = stablehlo.multiply %p_iter, %iv_plus_one : tensor<f64>

      enzyme.yield %s_next, %p_next : tensor<f64>, tensor<f64>
    }

    return %sum, %prod : tensor<f64>, tensor<f64>
  }

  // SHLO:  func.func @test_loop_with_trace(%arg0: tensor<i64>) -> (tensor<f64>, !enzyme.Trace) {
  // SHLO-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
  // SHLO-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
  // SHLO-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // SHLO-NEXT:    %0 = enzyme.initTrace : !enzyme.Trace
  // SHLO-NEXT:    %1 = builtin.unrealized_conversion_cast %0 : !enzyme.Trace to tensor<ui64>
  // SHLO-NEXT:    %2:3 = stablehlo.while(%iterArg = %c, %iterArg_1 = %cst, %iterArg_2 = %1) : tensor<i64>, tensor<f64>, tensor<ui64>
  // SHLO-NEXT:    cond {
  // SHLO-NEXT:      %4 = stablehlo.compare  LT, %iterArg, %arg0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // SHLO-NEXT:      stablehlo.return %4 : tensor<i1>
  // SHLO-NEXT:    } do {
  // SHLO-NEXT:      %4 = builtin.unrealized_conversion_cast %iterArg_2 : tensor<ui64> to !enzyme.Trace
  // SHLO-NEXT:      %5 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<f64>
  // SHLO-NEXT:      %6 = stablehlo.add %iterArg_1, %5 : tensor<f64>
  // SHLO-NEXT:      %7 = enzyme.addWeightToTrace(%6 : tensor<f64>) into %4
  // SHLO-NEXT:      %8 = builtin.unrealized_conversion_cast %7 : !enzyme.Trace to tensor<ui64>
  // SHLO-NEXT:      %9 = stablehlo.add %iterArg, %c_0 : tensor<i64>
  // SHLO-NEXT:      stablehlo.return %9, %6, %8 : tensor<i64>, tensor<f64>, tensor<ui64>
  // SHLO-NEXT:    }
  // SHLO-NEXT:    %3 = builtin.unrealized_conversion_cast %2#2 : tensor<ui64> to !enzyme.Trace
  // SHLO-NEXT:    return %2#1, %3 : tensor<f64>, !enzyme.Trace
  // SHLO-NEXT:  }

  // CPU:  func.func @test_loop_with_trace(%arg0: tensor<i64>) -> (tensor<f64>, tensor<ui64>) {
  // CPU-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
  // CPU-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
  // CPU-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // CPU-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<ui64>
  // CPU-NEXT:    %0 = enzymexla.jit_call @enzyme_probprog_init_trace_wrapper_0 (%c_1) {operand_layouts = [dense<> : tensor<0xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<ui64>) -> tensor<ui64>
  // CPU-NEXT:    %1:3 = stablehlo.while(%iterArg = %c, %iterArg_2 = %cst, %iterArg_3 = %0) : tensor<i64>, tensor<f64>, tensor<ui64>
  // CPU-NEXT:    cond {
  // CPU-NEXT:      %2 = stablehlo.compare  LT, %iterArg, %arg0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CPU-NEXT:      stablehlo.return %2 : tensor<i1>
  // CPU-NEXT:    } do {
  // CPU-NEXT:      %2 = stablehlo.convert %iterArg : (tensor<i64>) -> tensor<f64>
  // CPU-NEXT:      %3 = stablehlo.add %iterArg_2, %2 : tensor<f64>
  // CPU-NEXT:      %4 = enzymexla.jit_call @enzyme_probprog_add_weight_to_trace_wrapper_0 (%iterArg_3, %3) {output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]} : (tensor<ui64>, tensor<f64>) -> tensor<ui64>
  // CPU-NEXT:      %5 = stablehlo.add %iterArg, %c_0 : tensor<i64>
  // CPU-NEXT:      stablehlo.return %5, %3, %4 : tensor<i64>, tensor<f64>, tensor<ui64>
  // CPU-NEXT:    }
  // CPU-NEXT:    return %1#1, %1#2 : tensor<f64>, tensor<ui64>
  // CPU-NEXT:  }
  func.func @test_loop_with_trace(%n: tensor<i64>) -> (tensor<f64>, !enzyme.Trace) {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %init_sum = stablehlo.constant dense<0.0> : tensor<f64>
    %init_trace = enzyme.initTrace : !enzyme.Trace

    %sum, %trace = enzyme.for_loop (%c0 : tensor<i64>) to (%n : tensor<i64>) step (%c1 : tensor<i64>)
      iter_args(%init_sum, %init_trace : tensor<f64>, !enzyme.Trace)
      -> tensor<f64>, !enzyme.Trace {
    ^bb0(%iv: tensor<i64>, %s_iter: tensor<f64>, %t_iter: !enzyme.Trace):
      %iv_f64 = stablehlo.convert %iv : (tensor<i64>) -> tensor<f64>
      %s_next = stablehlo.add %s_iter, %iv_f64 : tensor<f64>
      %t_next = enzyme.addWeightToTrace(%s_next : tensor<f64>) into %t_iter
      enzyme.yield %s_next, %t_next : tensor<f64>, !enzyme.Trace
    }

    return %sum, %trace : tensor<f64>, !enzyme.Trace
  }

  // SHLO:  func.func @test_while_loop(%arg0: tensor<i64>) -> tensor<f64> {
  // SHLO-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  // SHLO-NEXT:    %c = stablehlo.constant dense<0> : tensor<i64>
  // SHLO-NEXT:    %c_0 = stablehlo.constant dense<1> : tensor<i64>
  // SHLO-NEXT:    %0:2 = stablehlo.while(%iterArg = %cst, %iterArg_1 = %c) : tensor<f64>, tensor<i64>
  // SHLO-NEXT:    cond {
  // SHLO-NEXT:      %1 = stablehlo.compare  LT, %iterArg_1, %arg0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // SHLO-NEXT:      stablehlo.return %1 : tensor<i1>
  // SHLO-NEXT:    } do {
  // SHLO-NEXT:      %1 = stablehlo.convert %iterArg_1 : (tensor<i64>) -> tensor<f64>
  // SHLO-NEXT:      %2 = stablehlo.add %iterArg, %1 : tensor<f64>
  // SHLO-NEXT:      %3 = stablehlo.add %iterArg_1, %c_0 : tensor<i64>
  // SHLO-NEXT:      stablehlo.return %2, %3 : tensor<f64>, tensor<i64>
  // SHLO-NEXT:    }
  // SHLO-NEXT:    return %0#0 : tensor<f64>
  // SHLO-NEXT:  }
  func.func @test_while_loop(%n: tensor<i64>) -> tensor<f64> {
    %init_sum = stablehlo.constant dense<0.0> : tensor<f64>
    %init_counter = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>

    %sum, %counter = enzyme.while_loop (%init_sum, %init_counter : tensor<f64>, tensor<i64>)
      -> tensor<f64>, tensor<i64>
      condition {
      ^bb0(%s_cond: tensor<f64>, %c_cond: tensor<i64>):
        %cond = stablehlo.compare LT, %c_cond, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
        enzyme.yield %cond : tensor<i1>
      }
      body {
      ^bb0(%s_body: tensor<f64>, %c_body: tensor<i64>):
        %c_f64 = stablehlo.convert %c_body : (tensor<i64>) -> tensor<f64>
        %s_next = stablehlo.add %s_body, %c_f64 : tensor<f64>
        %c_next = stablehlo.add %c_body, %c1 : tensor<i64>
        enzyme.yield %s_next, %c_next : tensor<f64>, tensor<i64>
      }

    return %sum : tensor<f64>
  }
}
