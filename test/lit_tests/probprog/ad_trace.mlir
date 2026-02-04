// RUN: enzymexlamlir-opt %s --outline-enzyme-regions --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --enzyme-simplify-math --cse | FileCheck %s --check-prefix=CPU --dump-input=always

module {
  func.func @test_autodiff_trace(%arg0: tensor<f64>) -> (!enzyme.Trace, tensor<f64>) {
    %c = stablehlo.constant dense<0> : tensor<ui64>
    %trace = builtin.unrealized_conversion_cast %c : tensor<ui64> to !enzyme.Trace
    %seed = stablehlo.constant dense<1.0> : tensor<f64>
    // CPU: %{{.*}}:2 = call @diffetest_autodiff_trace_to_diff0(%arg0, %0, %cst) : (tensor<f64>, !enzyme.Trace, tensor<f64>) -> (!enzyme.Trace, tensor<f64>) 
    %grad:2 = enzyme.autodiff_region(%arg0, %seed) {
    ^bb0(%x: tensor<f64>):
      %trace1 = enzyme.addWeightToTrace %x into %trace : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
      %y = stablehlo.multiply %x, %x : tensor<f64>
      enzyme.yield %y, %trace1 : tensor<f64>, !enzyme.Trace
    } attributes {
      activity = [#enzyme<activity enzyme_active>],
      ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_const>]
    } : (tensor<f64>, tensor<f64>) -> (!enzyme.Trace, tensor<f64>)
    return %grad#0, %grad#1 : !enzyme.Trace, tensor<f64>
  }
}

// CPU:  func.func @test_autodiff_trace_to_diff0(%arg0: tensor<f64>, %arg1: !enzyme.Trace) -> (tensor<f64>, !enzyme.Trace) {
// CPU-NEXT:    %[[V0:.+]] = enzyme.addWeightToTrace %arg0 into %arg1 : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// CPU-NEXT:    %[[V1:.+]] = stablehlo.multiply %arg0, %arg0 : tensor<f64>
// CPU-NEXT:    return %[[V1]], %[[V0]] : tensor<f64>, !enzyme.Trace
// CPU-NEXT:  }

// CPU:  func.func private @diffetest_autodiff_trace_to_diff0(%arg0: tensor<f64>, %arg1: !enzyme.Trace, %arg2: tensor<f64>) -> (!enzyme.Trace, tensor<f64>) {
// CPU-NEXT:    %[[V0:.+]] = enzyme.addWeightToTrace %arg0 into %arg1 : (!enzyme.Trace, tensor<f64>) -> !enzyme.Trace
// CPU-NEXT:    %[[V1:.+]] = stablehlo.multiply %arg2, %arg0 : tensor<f64>
// CPU-NEXT:    %[[V2:.+]] = arith.addf %[[V1]], %[[V1]] : tensor<f64>
// CPU-NEXT:    return %[[V0]], %[[V2]] : !enzyme.Trace, tensor<f64>
// CPU-NEXT:  }