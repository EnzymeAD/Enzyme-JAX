// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=remove_loop_carried_dependencies_from_while_load_operations},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s

// Verify that remove_loop_carried_dependencies_from_while_load_operations does not
// bypass a self-lane carried load when the carried buffer has other reads in the loop
// (such as true loop-carried recurrences). Bypassing in that case would keep the loop init
// live across iterations and force a copy of the buffer on loop entry without removing
// the dependency.

module {
  func.func @durbin_hull(%arg0: tensor<2000x2000xf64>) -> tensor<2000x2000xf64> {
    %c = stablehlo.constant dense<-1> : tensor<i64>
    %c_0 = stablehlo.constant dense<1999> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i64>
    %c_3 = stablehlo.constant dense<2000> : tensor<i64>
    %0:2 = stablehlo.while(%iterArg = %c_2, %iterArg_4 = %arg0) : tensor<i64>, tensor<2000x2000xf64>
    cond {
      %1 = stablehlo.compare LT, %iterArg, %c_3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1:2 = stablehlo.while(%iterArg_5 = %c_1, %iterArg_6 = %iterArg_4) : tensor<i64>, tensor<2000x2000xf64>
      cond {
        %3 = stablehlo.compare LT, %iterArg_5, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %3 : tensor<i1>
      } do {
        %3 = stablehlo.multiply %iterArg_5, %c : tensor<i64>
        %4 = stablehlo.add %iterArg, %3 : tensor<i64>
        %5 = stablehlo.add %4, %c : tensor<i64>
        %6 = stablehlo.compare GE, %5, %c_1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        %7 = stablehlo.dynamic_slice %iterArg_6, %iterArg_5, %iterArg, sizes = [1, 1] : (tensor<2000x2000xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
        %8 = stablehlo.reshape %7 : (tensor<1x1xf64>) -> tensor<f64>
        %9 = stablehlo.add %iterArg, %c : tensor<i64>
        %10 = stablehlo.dynamic_slice %iterArg_6, %iterArg_5, %9, sizes = [1, 1] : (tensor<2000x2000xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
        %11 = stablehlo.reshape %10 : (tensor<1x1xf64>) -> tensor<f64>
        %12 = stablehlo.add %8, %11 : tensor<f64>
        %13 = stablehlo.add %iterArg_5, %c_2 : tensor<i64>
        %14 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f64>) -> tensor<1x1xf64>
        %15 = stablehlo.dynamic_slice %iterArg_6, %13, %iterArg, sizes = [1, 1] : (tensor<2000x2000xf64>, tensor<i64>, tensor<i64>) -> tensor<1x1xf64>
        %16 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
        %17 = stablehlo.select %16, %14, %15 : tensor<1x1xi1>, tensor<1x1xf64>
        %18 = stablehlo.dynamic_update_slice %iterArg_6, %17, %13, %iterArg : (tensor<2000x2000xf64>, tensor<1x1xf64>, tensor<i64>, tensor<i64>) -> tensor<2000x2000xf64>
        stablehlo.return %13, %18 : tensor<i64>, tensor<2000x2000xf64>
      }
      %2 = stablehlo.add %iterArg, %c_2 : tensor<i64>
      stablehlo.return %2, %1#1 : tensor<i64>, tensor<2000x2000xf64>
    }
    return %0#1 : tensor<2000x2000xf64>
  }
}

// CHECK-LABEL: func.func @durbin_hull
// CHECK:         stablehlo.while(%iterArg_5 = %{{.*}}, %iterArg_6 = %iterArg_4)
// CHECK-NOT:     stablehlo.dynamic_slice %iterArg_4
// CHECK:         %[[PLUS1:.*]] = stablehlo.add %iterArg_5, %{{.*}}
// CHECK:         %[[ELSE:.*]] = stablehlo.dynamic_slice %iterArg_6, %[[PLUS1]], %iterArg, sizes = [1, 1]
// CHECK:         stablehlo.select %{{.*}}, %{{.*}}, %[[ELSE]]
