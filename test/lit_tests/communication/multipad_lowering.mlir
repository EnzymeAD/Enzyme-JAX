// RUN: enzymexlamlir-opt --optimize-communication="multipad_custom_call=1" %s | FileCheck %s

sdy.mesh @mesh1 = <["z"=1, "x"=4, "y"=4]>

func.func @main(%arg0: tensor<1519x3056xf64> {sdy.sharding = #sdy.sharding<@mesh1, [{"x"}, {"y"}]>}) -> (tensor<1520x3056xf64>, tensor<1520x3056xf64>) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0:2 = "enzymexla.multi_pad"(%arg0, %cst) <{amount = 1 : i64, dimension = 0 : i32}> {sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"x"}, {"y"}]>, <@mesh1, [{"x"}, {"y"}]>]>} : (tensor<1519x3056xf64>, tensor<f64>) -> (tensor<1520x3056xf64>, tensor<1520x3056xf64>)
    return %0#1, %0#0 : tensor<1520x3056xf64>, tensor<1520x3056xf64>
}

// CHECK-LABEL: func.func @main
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT: %[[CC:.*]]:2 = stablehlo.custom_call @_SPMDInternalOp_MultiPad(%arg0, %cst) {backend_config = "dimension=0,amt=1,bufferize=0"{{.*}}} : (tensor<1519x3056xf64>, tensor<f64>) -> (tensor<1520x3056xf64>, tensor<1520x3056xf64>)
