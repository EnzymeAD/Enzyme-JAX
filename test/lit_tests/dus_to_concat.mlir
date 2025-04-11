// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dus_to_concat" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s

// CHECK-LABEL:           func.func @dus_concat_test2
// CHECK-SAME:                    (%[[VAL_0:.*]]: tensor<144x1024x1024xf64>, %[[VAL_1:.*]]: tensor<128x1008x1008xf64>) -> tensor<144x1024x1024xf64> {
// CHECK:             %[[VAL_2:.*]] = stablehlo.slice %[[VAL_0]] [0:144, 0:1024, 0:8] : (tensor<144x1024x1024xf64>) -> tensor<144x1024x8xf64>
// CHECK:             %[[VAL_3:.*]] = stablehlo.slice %[[VAL_0]] [0:144, 0:1024, 8:1024] : (tensor<144x1024x1024xf64>) -> tensor<144x1024x1016xf64>
// CHECK:             %[[VAL_4:.*]] = stablehlo.slice %[[VAL_3]] [0:144, 0:1024, 0:1008] : (tensor<144x1024x1016xf64>) -> tensor<144x1024x1008xf64>
// CHECK:             %[[VAL_5:.*]] = stablehlo.slice %[[VAL_3]] [0:144, 0:1024, 1008:1016] : (tensor<144x1024x1016xf64>) -> tensor<144x1024x8xf64>
// CHECK:             %[[VAL_6:.*]] = stablehlo.slice %[[VAL_4]] [0:144, 0:8, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<144x8x1008xf64>
// CHECK:             %[[VAL_7:.*]] = stablehlo.slice %[[VAL_4]] [0:144, 8:1024, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<144x1016x1008xf64>
// CHECK:             %[[VAL_8:.*]] = stablehlo.slice %[[VAL_7]] [0:144, 0:1008, 0:1008] : (tensor<144x1016x1008xf64>) -> tensor<144x1008x1008xf64>
// CHECK:             %[[VAL_9:.*]] = stablehlo.slice %[[VAL_7]] [0:144, 1008:1016, 0:1008] : (tensor<144x1016x1008xf64>) -> tensor<144x8x1008xf64>
// CHECK:             %[[VAL_10:.*]] = stablehlo.slice %[[VAL_8]] [0:8, 0:1008, 0:1008] : (tensor<144x1008x1008xf64>) -> tensor<8x1008x1008xf64>
// CHECK:             %[[VAL_11:.*]] = stablehlo.slice %[[VAL_8]] [8:144, 0:1008, 0:1008] : (tensor<144x1008x1008xf64>) -> tensor<136x1008x1008xf64>
// CHECK:             %[[VAL_12:.*]] = stablehlo.slice %[[VAL_11]] [0:128, 0:1008, 0:1008] : (tensor<136x1008x1008xf64>) -> tensor<128x1008x1008xf64>
// CHECK:             %[[VAL_13:.*]] = stablehlo.slice %[[VAL_11]] [128:136, 0:1008, 0:1008] : (tensor<136x1008x1008xf64>) -> tensor<8x1008x1008xf64>
// CHECK:             %[[VAL_14:.*]] = stablehlo.concatenate %[[VAL_12]], %[[VAL_13]], dim = 0 : (tensor<128x1008x1008xf64>, tensor<8x1008x1008xf64>) -> tensor<136x1008x1008xf64>
// CHECK:             %[[VAL_15:.*]] = stablehlo.concatenate %[[VAL_10]], %[[VAL_14]], dim = 0 : (tensor<8x1008x1008xf64>, tensor<136x1008x1008xf64>) -> tensor<144x1008x1008xf64>
// CHECK:             %[[VAL_16:.*]] = stablehlo.concatenate %[[VAL_15]], %[[VAL_9]], dim = 1 : (tensor<144x1008x1008xf64>, tensor<144x8x1008xf64>) -> tensor<144x1016x1008xf64>
// CHECK:             %[[VAL_17:.*]] = stablehlo.concatenate %[[VAL_6]], %[[VAL_16]], dim = 1 : (tensor<144x8x1008xf64>, tensor<144x1016x1008xf64>) -> tensor<144x1024x1008xf64>
// CHECK:             %[[VAL_18:.*]] = stablehlo.concatenate %[[VAL_17]], %[[VAL_5]], dim = 2 : (tensor<144x1024x1008xf64>, tensor<144x1024x8xf64>) -> tensor<144x1024x1016xf64>
// CHECK:             %[[VAL_19:.*]] = stablehlo.concatenate %[[VAL_2]], %[[VAL_18]], dim = 2 : (tensor<144x1024x8xf64>, tensor<144x1024x1016xf64>) -> tensor<144x1024x1024xf64>
// CHECK:             return %[[VAL_19]] : tensor<144x1024x1024xf64>
// CHECK:           }

  func.func @dus_concat_test2(
    %A: tensor<144x1024x1024xf64>,
    %arg3: tensor<128x1008x1008xf64>
  ) -> (tensor<144x1024x1024xf64>)  {
    %c_31 = stablehlo.constant dense<8> : tensor<i64>
    %dus = stablehlo.dynamic_update_slice %A, %arg3, %c_31, %c_31, %c_31 : (tensor<144x1024x1024xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>
    func.return %dus : tensor<144x1024x1024xf64>
  }

// -----

// CHECK-LABEL:  func.func @main
// CHECK-NOT:    stablehlo.dynamic_update_slice

  func.func @main(%arg0: tensor<1024xf64> {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"}, %arg1: tensor<1024xf64> {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"}, %arg2: tensor<1024xf64> {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"}, %arg3: tensor<1024xf64> {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"}, %arg4: tensor<1024xf64> {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"}, %arg5: tensor<1024xf64> {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"}, %arg6: tensor<1024xf64> {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"}, %arg7: tensor<1024xf64> {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"}, %arg8: tensor<f64> {mhlo.sharding = "{replicated}", tf.aliasing_output = 0 : i32}, %arg9: tensor<i64> {mhlo.sharding = "{replicated}", tf.aliasing_output = 1 : i32}, %arg10: tensor<1x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 2 : i32}, %arg11: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 3 : i32}, %arg12: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 4 : i32}, %arg13: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 5 : i32}, %arg14: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 6 : i32}, %arg15: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 7 : i32}, %arg16: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 8 : i32}, %arg17: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 9 : i32}, %arg18: tensor<1x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 10 : i32}, %arg19: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 11 : i32}, %arg20: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 12 : i32}, %arg21: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 13 : i32}, %arg22: tensor<1x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 14 : i32}, %arg23: tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}", tf.aliasing_output = 15 : i32}, %arg24: tensor<i64> {mhlo.sharding = "{replicated}"}) -> (tensor<f64> {mhlo.sharding = "{replicated}"}, tensor<i64> {mhlo.sharding = "{replicated}"}, tensor<1x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<1x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<1x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}, tensor<144x1024x1024xf64> {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"}) {
    %cst = stablehlo.constant dense<3.125000e+01> : tensor<1007xf64>
    %cst_0 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<5.000000e-01> : tensor<128x1007x1008xf64>
    %cst_1 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<5.000000e-01> : tensor<127x1007x1008xf64>
    %cst_2 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<551562.13613372378> : tensor<128x1008x1008xf64>
    %cst_3 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<3.125000e+01> : tensor<127x1007x1008xf64>
    %cst_4 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<3.125000e+01> : tensor<127x1008x1008xf64>
    %cst_5 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<5.000000e-01> : tensor<127x1008x1008xf64>
    %cst_6 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<3.125000e+01> : tensor<128x1008x1008xf64>
    %cst_7 = stablehlo.constant {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} dense<1.000000e+00> : tensor<1008xf64>
    %cst_8 = stablehlo.constant {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} dense<3.125000e+01> : tensor<1008xf64>
    %cst_9 = stablehlo.constant {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} dense<5.5561793028102388E-4> : tensor<1008x1008xf64>
    %cst_10 = stablehlo.constant {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} dense<9.8066499999999994> : tensor<1008x1008xf64>
    %cst_11 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<2.000000e+00> : tensor<128x1008x1008xf64>
    %cst_12 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<5.000000e-01> : tensor<128x1008x1008xf64>
    %cst_13 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<17649.988356279162> : tensor<128x1007x1008xf64>
    %cst_14 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<17649.988356279162> : tensor<128x1008x1008xf64>
    %cst_15 = stablehlo.constant dense<-0.000000e+00> : tensor<1x1010x1010xf64>
    %cst_16 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<551562.13613372378> : tensor<128x1022x1022xf64>
    %cst_17 = stablehlo.constant {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} dense<3.125000e+01> : tensor<1022xf64>
    %cst_18 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<6.000000e+01> : tensor<1x1008x1008xf64>
    %cst_19 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<6.000000e-01> : tensor<1x1008x1008xf64>
    %cst_20 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<1.600000e+00> : tensor<1x1008x1008xf64>
    %cst_21 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<6.000000e+01> : tensor<128x1007x1008xf64>
    %cst_22 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<6.000000e-01> : tensor<128x1007x1008xf64>
    %cst_23 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<1.600000e+00> : tensor<128x1007x1008xf64>
    %cst_24 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<6.000000e+01> : tensor<128x1008x1008xf64>
    %cst_25 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<6.000000e-01> : tensor<128x1008x1008xf64>
    %cst_26 = stablehlo.constant {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} dense<1.600000e+00> : tensor<128x1008x1008xf64>
    %c = stablehlo.constant dense<135> : tensor<i64>
    %c_27 = stablehlo.constant dense<136> : tensor<i64>
    %cst_28 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c_29 = stablehlo.constant dense<7> : tensor<i64>
    %c_30 = stablehlo.constant dense<1016> : tensor<i64>
    %c_31 = stablehlo.constant dense<8> : tensor<i64>
    %cst_32 = stablehlo.constant dense<6.000000e+01> : tensor<f64>
    %c_33 = stablehlo.constant dense<1> : tensor<i64>
    %c_34 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.slice %arg13 [8:137, 1:1023, 1:1023] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<129x1022x1022xf64>
    %1 = stablehlo.slice %arg15 [135:136, 7:1017, 7:1017] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<1x1010x1010xf64>
    %2 = stablehlo.slice %arg16 [8:136, 8:1016, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<128x1008x1008xf64>
    %3 = stablehlo.slice %arg17 [8:136, 8:1016, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<128x1008x1008xf64>
    %4 = stablehlo.slice %arg18 [0:1, 8:1016, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1024x1024xf64>) -> tensor<1x1008x1008xf64>
    %5 = stablehlo.slice %arg19 [8:136, 8:1016, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<128x1008x1008xf64>
    %6 = stablehlo.slice %arg20 [8:136, 8:1016, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<128x1008x1008xf64>
    %7 = stablehlo.slice %arg21 [8:136, 8:1016, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<128x1008x1008xf64>
    %8 = stablehlo.slice %arg22 [0:1, 8:1016, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1024x1024xf64>) -> tensor<1x1008x1008xf64>
    %9 = stablehlo.slice %arg23 [8:136, 8:1016, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<128x1008x1008xf64>
    %10 = stablehlo.slice %arg10 [0:1, 0:1024, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1024x1024xf64>) -> tensor<1x1024x1008xf64>
    %11 = stablehlo.slice %arg11 [0:144, 0:1024, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<144x1024x1008xf64>
    %12 = stablehlo.slice %arg14 [0:144, 0:1024, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<144x1024x1008xf64>
    %13 = stablehlo.slice %arg12 [0:144, 0:1024, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<144x1024x1008xf64>
    %14:17 = stablehlo.while(%iterArg = %c_34, %iterArg_35 = %arg8, %iterArg_36 = %arg9, %iterArg_37 = %10, %iterArg_38 = %11, %iterArg_39 = %13, %iterArg_40 = %0, %iterArg_41 = %12, %iterArg_42 = %1, %iterArg_43 = %2, %iterArg_44 = %3, %iterArg_45 = %4, %iterArg_46 = %5, %iterArg_47 = %6, %iterArg_48 = %7, %iterArg_49 = %8, %iterArg_50 = %9) : tensor<i64>, tensor<f64>, tensor<i64>, tensor<1x1024x1008xf64>, tensor<144x1024x1008xf64>, tensor<144x1024x1008xf64>, tensor<129x1022x1022xf64>, tensor<144x1024x1008xf64>, tensor<1x1010x1010xf64>, tensor<128x1008x1008xf64>, tensor<128x1008x1008xf64>, tensor<1x1008x1008xf64>, tensor<128x1008x1008xf64>, tensor<128x1008x1008xf64>, tensor<128x1008x1008xf64>, tensor<1x1008x1008xf64>, tensor<128x1008x1008xf64> attributes {mhlo.sharding = "{{replicated}, {replicated}, {replicated}, {devices=[1,2,2]<=[2,2]T(1,0)}, {devices=[1,2,2]<=[2,2]T(1,0)}, /*index=5*/{devices=[1,2,2]<=[2,2]T(1,0)}, {devices=[1,2,2]<=[2,2]T(1,0)}, {devices=[1,2,2]<=[2,2]T(1,0)}, {devices=[1,2,2]<=[2,2]T(1,0)}, {devices=[1,2,2]<=[2,2]T(1,0)}, /*index=10*/{devices=[1,2,2]<=[2,2]T(1,0)}, {devices=[1,2,2]<=[2,2]T(1,0)}, {devices=[1,2,2]<=[2,2]T(1,0)}, {devices=[1,2,2]<=[2,2]T(1,0)}, {devices=[1,2,2]<=[2,2]T(1,0)}, /*index=15*/{devices=[1,2,2]<=[2,2]T(1,0)}, {devices=[1,2,2]<=[2,2]T(1,0)}}"}
     cond {
      %37 = stablehlo.compare  LT, %iterArg, %arg24 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %37 : tensor<i1>
    } do {
      %37 = stablehlo.slice %iterArg_40 [128:129, 7:1015, 7:1015] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<129x1022x1022xf64>) -> tensor<1x1008x1008xf64>
      %38 = stablehlo.slice %iterArg_38 [8:136, 8:1016, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1008xf64>) -> tensor<128x1008x1008xf64>
      %39 = stablehlo.multiply %iterArg_43, %cst_26 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %40 = stablehlo.multiply %iterArg_47, %cst_25 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %41 = stablehlo.subtract %39, %40 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %42 = stablehlo.multiply %41, %cst_24 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %43 = stablehlo.add %38, %42 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %44 = stablehlo.slice %iterArg_39 [8:136, 9:1016, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1008xf64>) -> tensor<128x1007x1008xf64>
      %45 = stablehlo.slice %iterArg_44 [0:128, 1:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1007x1008xf64>
      %46 = stablehlo.multiply %45, %cst_23 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %47 = stablehlo.slice %iterArg_48 [0:128, 1:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1007x1008xf64>
      %48 = stablehlo.multiply %47, %cst_22 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %49 = stablehlo.subtract %46, %48 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %50 = stablehlo.multiply %49, %cst_21 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %51 = stablehlo.add %44, %50 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %52 = stablehlo.slice %51 [0:1, 0:1007, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>) -> tensor<1x1007x1008xf64>
      %53 = stablehlo.slice %51 [127:128, 0:1007, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>) -> tensor<1x1007x1008xf64>
      %54 = stablehlo.slice %51 [0:128, 0:1007, 0:1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>) -> tensor<128x1007x1xf64>
      %55 = stablehlo.slice %51 [0:128, 0:1007, 1007:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>) -> tensor<128x1007x1xf64>
      %56 = stablehlo.slice %51 [0:128, 0:1007, 0:1007] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>) -> tensor<128x1007x1007xf64>
      %57 = stablehlo.slice %51 [0:128, 0:1007, 1:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>) -> tensor<128x1007x1007xf64>
      %58 = stablehlo.slice %iterArg_41 [8:136, 8:1016, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1008xf64>) -> tensor<128x1008x1008xf64>
      %59 = stablehlo.multiply %iterArg_46, %cst_26 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %60 = stablehlo.multiply %iterArg_50, %cst_25 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %61 = stablehlo.subtract %59, %60 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %62 = stablehlo.multiply %61, %cst_24 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %63 = stablehlo.add %58, %62 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %64 = stablehlo.slice %iterArg_37 [0:1, 8:1016, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1024x1008xf64>) -> tensor<1x1008x1008xf64>
      %65 = stablehlo.multiply %37, %cst_20 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<1x1008x1008xf64>
      %66 = stablehlo.multiply %iterArg_49, %cst_19 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<1x1008x1008xf64>
      %67 = stablehlo.subtract %65, %66 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<1x1008x1008xf64>
      %68 = stablehlo.multiply %67, %cst_18 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<1x1008x1008xf64>
      %69 = stablehlo.add %64, %68 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<1x1008x1008xf64>
      %70 = stablehlo.add %iterArg_35, %cst_32 : tensor<f64>
      %71 = stablehlo.add %iterArg_36, %c_33 : tensor<i64>
      %72 = stablehlo.slice %69 [0:1, 0:1, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1008x1008xf64>) -> tensor<1x1x1008xf64>
      %73 = stablehlo.slice %69 [0:1, 1007:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1008x1008xf64>) -> tensor<1x1x1008xf64>
      %74 = stablehlo.slice %iterArg_37 [0:1, 0:7, 0:1008] : (tensor<1x1024x1008xf64>) -> tensor<1x7x1008xf64>
      %75 = stablehlo.slice %iterArg_37 [0:1, 1017:1024, 0:1008] : (tensor<1x1024x1008xf64>) -> tensor<1x7x1008xf64>
      %76 = stablehlo.concatenate %74, %72, %69, %73, %75, dim = 1 : (tensor<1x7x1008xf64>, tensor<1x1x1008xf64>, tensor<1x1008x1008xf64>, tensor<1x1x1008xf64>, tensor<1x7x1008xf64>) -> tensor<1x1024x1008xf64>
      %77 = stablehlo.pad %51, %cst_28, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>, tensor<f64>) -> tensor<128x1008x1008xf64>
      %78 = stablehlo.dynamic_update_slice %iterArg_39, %77, %c_31, %c_31, %c_34 : (tensor<144x1024x1008xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
      %79 = stablehlo.pad %51, %cst_28, low = [0, 1, 0], high = [0, 1, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>, tensor<f64>) -> tensor<128x1009x1008xf64>
      %80 = stablehlo.dynamic_update_slice %iterArg_39, %79, %c_31, %c_31, %c_34 : (tensor<144x1024x1008xf64>, tensor<128x1009x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
      %81 = stablehlo.slice %43 [0:1, 0:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<1x1008x1008xf64>
      %82 = stablehlo.pad %52, %cst_28, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1007x1008xf64>, tensor<f64>) -> tensor<1x1008x1008xf64>
      %83 = stablehlo.slice %63 [0:1, 0:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<1x1008x1008xf64>
      %84 = stablehlo.slice %43 [127:128, 0:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<1x1008x1008xf64>
      %85 = stablehlo.pad %53, %cst_28, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1007x1008xf64>, tensor<f64>) -> tensor<1x1008x1008xf64>
      %86 = stablehlo.slice %63 [127:128, 0:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<1x1008x1008xf64>
      %87 = stablehlo.dynamic_update_slice %80, %82, %c_29, %c_31, %c_34 : (tensor<144x1024x1008xf64>, tensor<1x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
      %88 = stablehlo.concatenate %81, %43, %84, dim = 0 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1008x1008xf64>, tensor<128x1008x1008xf64>, tensor<1x1008x1008xf64>) -> tensor<130x1008x1008xf64>
      %89 = stablehlo.dynamic_update_slice %iterArg_38, %88, %c_29, %c_31, %c_34 : (tensor<144x1024x1008xf64>, tensor<130x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
      %90 = stablehlo.dynamic_update_slice %87, %85, %c_27, %c_31, %c_34 : (tensor<144x1024x1008xf64>, tensor<1x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
      %91 = stablehlo.concatenate %83, %63, %86, dim = 0 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1008x1008xf64>, tensor<128x1008x1008xf64>, tensor<1x1008x1008xf64>) -> tensor<130x1008x1008xf64>
      %92 = stablehlo.dynamic_update_slice %iterArg_41, %91, %c_29, %c_31, %c_34 : (tensor<144x1024x1008xf64>, tensor<130x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
      %93 = stablehlo.slice %43 [0:128, 0:1, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1x1008xf64>
      %94 = stablehlo.slice %63 [0:128, 0:1, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1x1008xf64>
      %95 = stablehlo.slice %43 [0:128, 1007:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1x1008xf64>
      %96 = stablehlo.slice %63 [0:128, 1007:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1x1008xf64>
      %97 = stablehlo.dynamic_update_slice %89, %93, %c_31, %c_29, %c_34 : (tensor<144x1024x1008xf64>, tensor<128x1x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
      %98 = stablehlo.dynamic_update_slice %92, %94, %c_31, %c_29, %c_34 : (tensor<144x1024x1008xf64>, tensor<128x1x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
      %99 = stablehlo.dynamic_update_slice %97, %95, %c_31, %c_30, %c_34 : (tensor<144x1024x1008xf64>, tensor<128x1x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
      %100 = stablehlo.dynamic_update_slice %98, %96, %c_31, %c_30, %c_34 : (tensor<144x1024x1008xf64>, tensor<128x1x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1008xf64>
      %101 = stablehlo.slice %arg2 [2:1024] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1022xf64>
      %102 = stablehlo.multiply %101, %cst_17 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : tensor<1022xf64>
      %103 = stablehlo.slice %arg2 [1:1023] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1022xf64>
      %104 = stablehlo.multiply %103, %cst_17 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : tensor<1022xf64>
      %105 = stablehlo.slice %arg4 [1:1023] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1022xf64>
      %106 = stablehlo.slice %99 [8:136, 1:1023, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x1008xf64>
      %107 = stablehlo.slice %99 [8:136, 1:1023, 1002:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x6xf64>
      %108 = stablehlo.slice %99 [8:136, 1:1023, 0:8] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x8xf64>
      %109 = stablehlo.concatenate %107, %106, %108, dim = 2 : (tensor<128x1022x6xf64>, tensor<128x1022x1008xf64>, tensor<128x1022x8xf64>) -> tensor<128x1022x1022xf64>
      %110 = stablehlo.multiply %109, %cst_16 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1022x1022xf64>
      %111 = stablehlo.slice %99 [8:136, 1:1023, 1001:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x7xf64>
      %112 = stablehlo.slice %99 [8:136, 1:1023, 0:7] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x7xf64>
      %113 = stablehlo.concatenate %111, %106, %112, dim = 2 : (tensor<128x1022x7xf64>, tensor<128x1022x1008xf64>, tensor<128x1022x7xf64>) -> tensor<128x1022x1022xf64>
      %114 = stablehlo.multiply %113, %cst_16 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1022x1022xf64>
      %115 = stablehlo.subtract %110, %114 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1022x1022xf64>
      %116 = stablehlo.broadcast_in_dim %102, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1022xf64>) -> tensor<128x1022x1022xf64>
      %117 = stablehlo.slice %80 [8:136, 2:1024, 1001:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x7xf64>
      %118 = stablehlo.slice %80 [8:136, 2:1024, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x1008xf64>
      %119 = stablehlo.slice %80 [8:136, 2:1024, 0:7] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x7xf64>
      %120 = stablehlo.concatenate %117, %118, %119, dim = 2 : (tensor<128x1022x7xf64>, tensor<128x1022x1008xf64>, tensor<128x1022x7xf64>) -> tensor<128x1022x1022xf64>
      %121 = stablehlo.multiply %116, %120 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1022x1022xf64>
      %122 = stablehlo.broadcast_in_dim %104, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1022xf64>) -> tensor<128x1022x1022xf64>
      %123 = stablehlo.slice %80 [8:136, 1:1023, 1001:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x7xf64>
      %124 = stablehlo.slice %80 [8:136, 1:1023, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x1008xf64>
      %125 = stablehlo.slice %80 [8:136, 1:1023, 0:7] : (tensor<144x1024x1008xf64>) -> tensor<128x1022x7xf64>
      %126 = stablehlo.concatenate %123, %124, %125, dim = 2 : (tensor<128x1022x7xf64>, tensor<128x1022x1008xf64>, tensor<128x1022x7xf64>) -> tensor<128x1022x1022xf64>
      %127 = stablehlo.multiply %122, %126 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1022x1022xf64>
      %128 = stablehlo.subtract %121, %127 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1022x1022xf64>
      %129 = stablehlo.add %115, %128 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1022x1022xf64>
      %130 = stablehlo.broadcast_in_dim %105, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1022xf64>) -> tensor<128x1022x1022xf64>
      %131 = stablehlo.divide %129, %130 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1022x1022xf64>
      %132 = "stablehlo.reduce_window"(%131, %cst_28) <{base_dilations = array<i64: 1, 1, 1>, padding = dense<[[127, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>, window_dilations = array<i64: 1, 1, 1>, window_dimensions = array<i64: 128, 1, 1>, window_strides = array<i64: 1, 1, 1>}> ({
      ^bb0(%arg25: tensor<f64>, %arg26: tensor<f64>):
        %413 = stablehlo.add %arg25, %arg26 : tensor<f64>
        stablehlo.return %413 : tensor<f64>
      }) {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1022x1022xf64>, tensor<f64>) -> tensor<128x1022x1022xf64>
      %133 = stablehlo.negate %132 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1022x1022xf64>
      %134 = stablehlo.pad %133, %cst_28, low = [1, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1022x1022xf64>, tensor<f64>) -> tensor<129x1022x1022xf64>
      %135 = stablehlo.dynamic_update_slice %arg15, %cst_15, %c, %c_29, %c_29 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<1x1010x1010xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>
      %136 = stablehlo.slice %arg1 [7:1015] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %137 = stablehlo.slice %arg1 [8:1016] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %138 = stablehlo.slice %arg7 [8:1016] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %139 = stablehlo.slice %arg5 [8:1016] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %140 = stablehlo.slice %arg2 [8:1016] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %141 = stablehlo.slice %arg4 [8:1016] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %142 = stablehlo.slice %arg1 [9:1017] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %143 = stablehlo.slice %arg7 [9:1017] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %144 = stablehlo.slice %arg2 [9:1017] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %145 = stablehlo.reshape %76 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : (tensor<1x1024x1008xf64>) -> tensor<1024x1008xf64>
      %146 = stablehlo.slice %69 [0:1, 0:1008, 1007:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1008x1008xf64>) -> tensor<1x1008x1xf64>
      %147 = stablehlo.reshape %146 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : (tensor<1x1008x1xf64>) -> tensor<1008x1xf64>
      %148 = stablehlo.slice %145 [8:1016, 0:1007] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : (tensor<1024x1008xf64>) -> tensor<1008x1007xf64>
      %149 = stablehlo.concatenate %147, %148, dim = 1 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : (tensor<1008x1xf64>, tensor<1008x1007xf64>) -> tensor<1008x1008xf64>
      %150 = stablehlo.slice %145 [8:1016, 0:1008] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : (tensor<1024x1008xf64>) -> tensor<1008x1008xf64>
      %151 = stablehlo.slice %97 [8:136, 7:1015, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1008x1008xf64>
      %152 = stablehlo.slice %135 [8:136, 8:1016, 7:1015] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<128x1008x1008xf64>
      %153 = stablehlo.slice %43 [0:128, 0:1008, 1007:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1008x1xf64>
      %154 = stablehlo.slice %43 [0:128, 0:1008, 0:1007] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1008x1007xf64>
      %155 = stablehlo.concatenate %153, %154, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1xf64>, tensor<128x1008x1007xf64>) -> tensor<128x1008x1008xf64>
      %156 = stablehlo.pad %55, %cst_28, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1xf64>, tensor<f64>) -> tensor<128x1008x1xf64>
      %157 = stablehlo.pad %56, %cst_28, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1007xf64>, tensor<f64>) -> tensor<128x1008x1007xf64>
      %158 = stablehlo.concatenate %156, %157, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1xf64>, tensor<128x1008x1007xf64>) -> tensor<128x1008x1008xf64>
      %159 = stablehlo.slice %135 [8:136, 8:1016, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<128x1008x1008xf64>
      %160 = stablehlo.slice %43 [0:128, 0:1008, 1:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1008x1007xf64>
      %161 = stablehlo.slice %43 [0:128, 0:1008, 0:1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1008x1xf64>
      %162 = stablehlo.concatenate %160, %161, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1007xf64>, tensor<128x1008x1xf64>) -> tensor<128x1008x1008xf64>
      %163 = stablehlo.pad %55, %cst_28, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1xf64>, tensor<f64>) -> tensor<128x1008x1xf64>
      %164 = stablehlo.pad %56, %cst_28, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1007xf64>, tensor<f64>) -> tensor<128x1008x1007xf64>
      %165 = stablehlo.concatenate %163, %164, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1xf64>, tensor<128x1008x1007xf64>) -> tensor<128x1008x1008xf64>
      %166 = stablehlo.multiply %51, %cst_13 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %167 = stablehlo.pad %166, %cst_28, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>, tensor<f64>) -> tensor<128x1008x1008xf64>
      %168 = stablehlo.multiply %158, %cst_14 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %169 = stablehlo.subtract %167, %168 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %170 = stablehlo.broadcast_in_dim %137, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %171 = stablehlo.multiply %170, %43 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %172 = stablehlo.broadcast_in_dim %136, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %173 = stablehlo.multiply %172, %151 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %174 = stablehlo.subtract %171, %173 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %175 = stablehlo.subtract %169, %174 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %176 = stablehlo.broadcast_in_dim %138, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %177 = stablehlo.divide %175, %176 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %178 = stablehlo.pad %166, %cst_28, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>, tensor<f64>) -> tensor<128x1008x1008xf64>
      %179 = stablehlo.multiply %165, %cst_14 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %180 = stablehlo.subtract %178, %179 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %181 = stablehlo.broadcast_in_dim %142, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %182 = stablehlo.slice %99 [8:136, 9:1017, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1008x1008xf64>
      %183 = stablehlo.multiply %181, %182 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %184 = stablehlo.subtract %183, %171 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %185 = stablehlo.subtract %180, %184 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %186 = stablehlo.broadcast_in_dim %143, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %187 = stablehlo.divide %185, %186 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %188 = stablehlo.add %177, %187 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %189 = stablehlo.multiply %188, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %190 = stablehlo.negate %189 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %191 = stablehlo.broadcast_in_dim %140, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %192 = stablehlo.multiply %191, %158 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %193 = stablehlo.broadcast_in_dim %144, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %194 = stablehlo.multiply %193, %165 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %195 = stablehlo.add %192, %194 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %196 = stablehlo.multiply %195, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %197 = stablehlo.slice %191 [0:128, 1:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1007x1008xf64>
      %198 = stablehlo.multiply %197, %51 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %199 = stablehlo.slice %193 [0:128, 0:1007, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1007x1008xf64>
      %200 = stablehlo.multiply %199, %51 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %201 = stablehlo.slice %200 [0:128, 0:1, 0:1008] : (tensor<128x1007x1008xf64>) -> tensor<128x1x1008xf64>
      %202 = stablehlo.slice %200 [0:128, 1:1007, 0:1008] : (tensor<128x1007x1008xf64>) -> tensor<128x1006x1008xf64>
      %203 = stablehlo.slice %198 [0:128, 0:1006, 0:1008] : (tensor<128x1007x1008xf64>) -> tensor<128x1006x1008xf64>
      %204 = stablehlo.add %203, %202 : tensor<128x1006x1008xf64>
      %205 = stablehlo.slice %198 [0:128, 1006:1007, 0:1008] : (tensor<128x1007x1008xf64>) -> tensor<128x1x1008xf64>
      %206 = stablehlo.concatenate %201, %204, %205, dim = 1 : (tensor<128x1x1008xf64>, tensor<128x1006x1008xf64>, tensor<128x1x1008xf64>) -> tensor<128x1008x1008xf64>
      %207 = stablehlo.multiply %206, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %208 = stablehlo.add %196, %207 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %209 = stablehlo.multiply %208, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %210 = stablehlo.multiply %190, %209 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %211 = stablehlo.divide %210, %170 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %212 = stablehlo.multiply %43, %43 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %213 = stablehlo.multiply %162, %162 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %214 = stablehlo.add %212, %213 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %215 = stablehlo.multiply %214, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %216 = stablehlo.multiply %51, %51 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %217 = stablehlo.slice %216 [0:128, 0:1, 0:1008] : (tensor<128x1007x1008xf64>) -> tensor<128x1x1008xf64>
      %218 = stablehlo.slice %216 [0:128, 1:1007, 0:1008] : (tensor<128x1007x1008xf64>) -> tensor<128x1006x1008xf64>
      %219 = stablehlo.slice %216 [0:128, 0:1006, 0:1008] : (tensor<128x1007x1008xf64>) -> tensor<128x1006x1008xf64>
      %220 = stablehlo.add %219, %218 : tensor<128x1006x1008xf64>
      %221 = stablehlo.slice %216 [0:128, 1006:1007, 0:1008] : (tensor<128x1007x1008xf64>) -> tensor<128x1x1008xf64>
      %222 = stablehlo.concatenate %217, %220, %221, dim = 1 : (tensor<128x1x1008xf64>, tensor<128x1006x1008xf64>, tensor<128x1x1008xf64>) -> tensor<128x1008x1008xf64>
      %223 = stablehlo.multiply %222, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %224 = stablehlo.add %215, %223 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %225 = stablehlo.divide %224, %cst_11 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %226 = stablehlo.multiply %155, %155 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %227 = stablehlo.add %226, %212 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %228 = stablehlo.multiply %227, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %229 = stablehlo.multiply %158, %158 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %230 = stablehlo.multiply %165, %165 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %231 = stablehlo.add %229, %230 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %232 = stablehlo.multiply %231, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %233 = stablehlo.add %228, %232 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %234 = stablehlo.divide %233, %cst_11 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %235 = stablehlo.subtract %225, %234 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %236 = stablehlo.divide %235, %170 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %237 = stablehlo.broadcast_in_dim %141, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %238 = stablehlo.slice %133 [0:127, 7:1015, 6:1014] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1022x1022xf64>) -> tensor<127x1008x1008xf64>
      %239 = stablehlo.slice %237 [1:128, 0:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<127x1008x1008xf64>
      %240 = stablehlo.multiply %239, %238 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1008x1008xf64>
      %241 = stablehlo.slice %133 [0:127, 7:1015, 7:1015] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1022x1022xf64>) -> tensor<127x1008x1008xf64>
      %242 = stablehlo.multiply %239, %241 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1008x1008xf64>
      %243 = stablehlo.add %242, %240 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1008x1008xf64>
      %244 = stablehlo.multiply %243, %cst_5 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1008x1008xf64>
      %245 = stablehlo.slice %43 [0:127, 0:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<127x1008x1008xf64>
      %246 = stablehlo.slice %43 [1:128, 0:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<127x1008x1008xf64>
      %247 = stablehlo.subtract %246, %245 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1008x1008xf64>
      %248 = stablehlo.divide %247, %cst_4 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1008x1008xf64>
      %249 = stablehlo.multiply %244, %248 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1008x1008xf64>
      %250 = stablehlo.slice %133 [0:128, 7:1015, 6:1014] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1022x1022xf64>) -> tensor<128x1008x1008xf64>
      %251 = stablehlo.multiply %237, %250 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %252 = stablehlo.slice %133 [0:128, 7:1015, 7:1015] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1022x1022xf64>) -> tensor<128x1008x1008xf64>
      %253 = stablehlo.multiply %237, %252 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %254 = stablehlo.add %251, %253 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %255 = stablehlo.multiply %254, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %256 = stablehlo.concatenate %246, %84, dim = 0 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<127x1008x1008xf64>, tensor<1x1008x1008xf64>) -> tensor<128x1008x1008xf64>
      %257 = stablehlo.subtract %256, %43 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %258 = stablehlo.divide %257, %cst_6 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %259 = stablehlo.multiply %255, %258 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %260 = stablehlo.slice %259 [0:1, 0:1008, 0:1008] : (tensor<128x1008x1008xf64>) -> tensor<1x1008x1008xf64>
      %261 = stablehlo.slice %259 [1:128, 0:1008, 0:1008] : (tensor<128x1008x1008xf64>) -> tensor<127x1008x1008xf64>
      %262 = stablehlo.add %261, %249 : tensor<127x1008x1008xf64>
      %263 = stablehlo.concatenate %260, %262, dim = 0 : (tensor<1x1008x1008xf64>, tensor<127x1008x1008xf64>) -> tensor<128x1008x1008xf64>
      %264 = stablehlo.multiply %263, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %265 = stablehlo.broadcast_in_dim %139, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %266 = stablehlo.divide %264, %265 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %267 = stablehlo.add %211, %266 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %268 = stablehlo.add %267, %236 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %269 = stablehlo.negate %268 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %270 = stablehlo.subtract %150, %149 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : tensor<1008x1008xf64>
      %271 = stablehlo.broadcast_in_dim %137, dims = [0] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<1008x1008xf64>
      %272 = stablehlo.divide %270, %271 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : tensor<1008x1008xf64>
      %273 = stablehlo.multiply %272, %cst_10 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : tensor<1008x1008xf64>
      %274 = stablehlo.broadcast_in_dim %273, dims = [1, 2] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008x1008xf64>) -> tensor<128x1008x1008xf64>
      %275 = stablehlo.subtract %269, %274 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %276 = stablehlo.subtract %159, %152 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %277 = stablehlo.divide %276, %170 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %278 = stablehlo.subtract %275, %277 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %279 = stablehlo.slice %arg4 [7:1015] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %280 = stablehlo.slice %arg6 [8:1016] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf64>) -> tensor<1008xf64>
      %281 = stablehlo.slice %145 [7:1015, 0:1008] {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : (tensor<1024x1008xf64>) -> tensor<1008x1008xf64>
      %282 = stablehlo.slice %135 [8:136, 7:1015, 8:1016] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>) -> tensor<128x1008x1008xf64>
      %283 = stablehlo.slice %78 [8:136, 7:1015, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1008x1008xf64>
      %284 = stablehlo.slice %97 [8:136, 7:1015, 1:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1008x1007xf64>
      %285 = stablehlo.slice %97 [8:136, 7:1015, 0:1] : (tensor<144x1024x1008xf64>) -> tensor<128x1008x1xf64>
      %286 = stablehlo.concatenate %284, %285, dim = 2 : (tensor<128x1008x1007xf64>, tensor<128x1008x1xf64>) -> tensor<128x1008x1008xf64>
      %287 = stablehlo.pad %57, %cst_28, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1007xf64>, tensor<f64>) -> tensor<128x1008x1007xf64>
      %288 = stablehlo.pad %54, %cst_28, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1xf64>, tensor<f64>) -> tensor<128x1008x1xf64>
      %289 = stablehlo.concatenate %287, %288, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1007xf64>, tensor<128x1008x1xf64>) -> tensor<128x1008x1008xf64>
      %290 = stablehlo.multiply %289, %cst_14 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %291 = stablehlo.subtract %290, %167 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %292 = stablehlo.multiply %170, %162 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %293 = stablehlo.multiply %172, %286 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %294 = stablehlo.subtract %292, %293 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %295 = stablehlo.subtract %291, %294 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %296 = stablehlo.divide %295, %176 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %297 = stablehlo.add %177, %296 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %298 = stablehlo.multiply %297, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %299 = stablehlo.multiply %151, %cst_14 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %300 = stablehlo.multiply %286, %cst_14 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %301 = stablehlo.add %299, %300 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %302 = stablehlo.multiply %301, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %303 = stablehlo.multiply %43, %cst_14 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %304 = stablehlo.multiply %162, %cst_14 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %305 = stablehlo.add %303, %304 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %306 = stablehlo.multiply %305, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %307 = stablehlo.add %302, %306 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %308 = stablehlo.multiply %307, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %309 = stablehlo.multiply %298, %308 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %310 = stablehlo.divide %309, %cst_14 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %311 = stablehlo.multiply %151, %151 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %312 = stablehlo.multiply %286, %286 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %313 = stablehlo.add %311, %312 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %314 = stablehlo.multiply %313, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %315 = stablehlo.multiply %283, %283 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %316 = stablehlo.slice %315 [0:128, 0:1, 0:1008] : (tensor<128x1008x1008xf64>) -> tensor<128x1x1008xf64>
      %317 = stablehlo.slice %315 [0:128, 1:1008, 0:1008] : (tensor<128x1008x1008xf64>) -> tensor<128x1007x1008xf64>
      %318 = stablehlo.add %317, %216 : tensor<128x1007x1008xf64>
      %319 = stablehlo.concatenate %316, %318, dim = 1 : (tensor<128x1x1008xf64>, tensor<128x1007x1008xf64>) -> tensor<128x1008x1008xf64>
      %320 = stablehlo.multiply %319, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %321 = stablehlo.add %314, %320 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %322 = stablehlo.divide %321, %cst_11 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %323 = stablehlo.subtract %225, %322 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %324 = stablehlo.divide %323, %cst_14 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %325 = stablehlo.broadcast_in_dim %279, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %326 = stablehlo.slice %51 [1:128, 0:1007, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>) -> tensor<127x1007x1008xf64>
      %327 = stablehlo.slice %51 [0:127, 0:1007, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>) -> tensor<127x1007x1008xf64>
      %328 = stablehlo.subtract %326, %327 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1007x1008xf64>
      %329 = stablehlo.divide %328, %cst_3 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1007x1008xf64>
      %330 = stablehlo.slice %242 [0:127, 1:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<127x1008x1008xf64>) -> tensor<127x1007x1008xf64>
      %331 = stablehlo.slice %325 [1:128, 1:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<127x1007x1008xf64>
      %332 = stablehlo.slice %133 [0:127, 7:1014, 7:1015] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1022x1022xf64>) -> tensor<127x1007x1008xf64>
      %333 = stablehlo.multiply %331, %332 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1007x1008xf64>
      %334 = stablehlo.add %330, %333 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1007x1008xf64>
      %335 = stablehlo.multiply %334, %cst_1 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1007x1008xf64>
      %336 = stablehlo.multiply %335, %329 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1007x1008xf64>
      %337 = stablehlo.pad %336, %cst_28, low = [1, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<127x1007x1008xf64>, tensor<f64>) -> tensor<128x1008x1008xf64>
      %338 = stablehlo.slice %133 [0:128, 6:1014, 7:1015] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1022x1022xf64>) -> tensor<128x1008x1008xf64>
      %339 = stablehlo.multiply %325, %338 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %340 = stablehlo.add %339, %253 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %341 = stablehlo.multiply %340, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %342 = stablehlo.slice %90 [9:137, 8:1016, 0:1008] : (tensor<144x1024x1008xf64>) -> tensor<128x1008x1008xf64>
      %343 = stablehlo.subtract %342, %77 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %344 = stablehlo.divide %343, %cst_6 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %345 = stablehlo.multiply %341, %344 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %346 = stablehlo.add %337, %345 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %347 = stablehlo.multiply %346, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %348 = stablehlo.broadcast_in_dim %280, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %349 = stablehlo.divide %347, %348 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %350 = stablehlo.add %310, %349 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %351 = stablehlo.add %350, %324 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %352 = stablehlo.negate %351 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %353 = stablehlo.subtract %150, %281 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : tensor<1008x1008xf64>
      %354 = stablehlo.multiply %353, %cst_9 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0)}"} : tensor<1008x1008xf64>
      %355 = stablehlo.broadcast_in_dim %354, dims = [1, 2] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008x1008xf64>) -> tensor<128x1008x1008xf64>
      %356 = stablehlo.subtract %352, %355 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %357 = stablehlo.subtract %159, %282 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %358 = stablehlo.divide %357, %cst_14 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %359 = stablehlo.subtract %356, %358 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %360 = stablehlo.multiply %141, %cst_8 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : tensor<1008xf64>
      %361 = stablehlo.divide %cst_7, %360 {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : tensor<1008xf64>
      %362 = stablehlo.multiply %63, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %363 = stablehlo.broadcast_in_dim %361, dims = [1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1008xf64>) -> tensor<128x1008x1008xf64>
      %364 = stablehlo.multiply %162, %cst_2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %365 = stablehlo.slice %63 [0:128, 0:1008, 1:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1008x1007xf64>
      %366 = stablehlo.slice %63 [0:128, 0:1008, 0:1] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1008x1xf64>
      %367 = stablehlo.concatenate %365, %366, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1007xf64>, tensor<128x1008x1xf64>) -> tensor<128x1008x1008xf64>
      %368 = stablehlo.multiply %367, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %369 = stablehlo.add %362, %368 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %370 = stablehlo.multiply %364, %369 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %371 = stablehlo.multiply %43, %cst_2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %372 = stablehlo.slice %63 [0:128, 0:1008, 1007:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1008x1xf64>
      %373 = stablehlo.slice %63 [0:128, 0:1008, 0:1007] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1008x1007xf64>
      %374 = stablehlo.concatenate %372, %373, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1xf64>, tensor<128x1008x1007xf64>) -> tensor<128x1008x1008xf64>
      %375 = stablehlo.multiply %374, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %376 = stablehlo.add %375, %362 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %377 = stablehlo.multiply %371, %376 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %378 = stablehlo.subtract %370, %377 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %379 = stablehlo.slice %arg2 [9:1016] : (tensor<1024xf64>) -> tensor<1007xf64>
      %380 = stablehlo.multiply %379, %cst {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"} : tensor<1007xf64>
      %381 = stablehlo.broadcast_in_dim %380, dims = [1] : (tensor<1007xf64>) -> tensor<128x1007x1008xf64>
      %382 = stablehlo.multiply %381, %51 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %383 = stablehlo.slice %362 [0:128, 0:1007, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1007x1008xf64>
      %384 = stablehlo.slice %63 [0:128, 1:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1007x1008xf64>
      %385 = stablehlo.multiply %384, %cst_0 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %386 = stablehlo.add %383, %385 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %387 = stablehlo.multiply %382, %386 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %388 = stablehlo.pad %387, %cst_28, low = [0, 0, 0], high = [0, 1, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>, tensor<f64>) -> tensor<128x1008x1008xf64>
      %389 = stablehlo.slice %63 [0:128, 0:1007, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1007x1008xf64>
      %390 = stablehlo.multiply %389, %cst_0 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %391 = stablehlo.slice %362 [0:128, 1:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<128x1007x1008xf64>
      %392 = stablehlo.add %390, %391 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %393 = stablehlo.multiply %382, %392 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1007x1008xf64>
      %394 = stablehlo.pad %393, %cst_28, low = [0, 1, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1007x1008xf64>, tensor<f64>) -> tensor<128x1008x1008xf64>
      %395 = stablehlo.subtract %388, %394 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %396 = stablehlo.add %378, %395 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %397 = stablehlo.slice %63 [1:128, 0:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<127x1008x1008xf64>
      %398 = stablehlo.concatenate %397, %86, dim = 0 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<127x1008x1008xf64>, tensor<1x1008x1008xf64>) -> tensor<128x1008x1008xf64>
      %399 = stablehlo.multiply %398, %cst_12 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %400 = stablehlo.add %362, %399 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %401 = stablehlo.multiply %253, %400 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %402 = stablehlo.slice %63 [0:127, 0:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<127x1008x1008xf64>
      %403 = stablehlo.multiply %402, %cst_5 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1008x1008xf64>
      %404 = stablehlo.slice %362 [1:128, 0:1008, 0:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<128x1008x1008xf64>) -> tensor<127x1008x1008xf64>
      %405 = stablehlo.add %403, %404 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1008x1008xf64>
      %406 = stablehlo.multiply %242, %405 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<127x1008x1008xf64>
      %407 = stablehlo.pad %406, %cst_28, low = [1, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<127x1008x1008xf64>, tensor<f64>) -> tensor<128x1008x1008xf64>
      %408 = stablehlo.subtract %401, %407 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %409 = stablehlo.add %396, %408 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %410 = stablehlo.multiply %363, %409 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %411 = stablehlo.negate %410 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : tensor<128x1008x1008xf64>
      %412 = stablehlo.add %iterArg, %c_33 : tensor<i64>
      stablehlo.return %412, %70, %71, %76, %99, %90, %134, %100, %cst_15, %278, %359, %37, %411, %iterArg_43, %iterArg_44, %37, %iterArg_46 : tensor<i64>, tensor<f64>, tensor<i64>, tensor<1x1024x1008xf64>, tensor<144x1024x1008xf64>, tensor<144x1024x1008xf64>, tensor<129x1022x1022xf64>, tensor<144x1024x1008xf64>, tensor<1x1010x1010xf64>, tensor<128x1008x1008xf64>, tensor<128x1008x1008xf64>, tensor<1x1008x1008xf64>, tensor<128x1008x1008xf64>, tensor<128x1008x1008xf64>, tensor<128x1008x1008xf64>, tensor<1x1008x1008xf64>, tensor<128x1008x1008xf64>
    }
    %15 = stablehlo.slice %14#5 [0:144, 0:1024, 0:8] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1008xf64>) -> tensor<144x1024x8xf64>
    %16 = stablehlo.slice %14#5 [0:144, 0:1024, 1000:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1008xf64>) -> tensor<144x1024x8xf64>
    %17 = stablehlo.concatenate %15, %14#5, %16, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x8xf64>, tensor<144x1024x1008xf64>, tensor<144x1024x8xf64>) -> tensor<144x1024x1024xf64>
    %18 = stablehlo.slice %14#3 [0:1, 0:1024, 0:8] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1024x1008xf64>) -> tensor<1x1024x8xf64>
    %19 = stablehlo.slice %14#3 [0:1, 0:1024, 1000:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1024x1008xf64>) -> tensor<1x1024x8xf64>
    %20 = stablehlo.concatenate %18, %14#3, %19, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1024x8xf64>, tensor<1x1024x1008xf64>, tensor<1x1024x8xf64>) -> tensor<1x1024x1024xf64>
    %21 = stablehlo.slice %14#4 [0:144, 0:1024, 0:8] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1008xf64>) -> tensor<144x1024x8xf64>
    %22 = stablehlo.slice %14#4 [0:144, 0:1024, 1000:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1008xf64>) -> tensor<144x1024x8xf64>
    %23 = stablehlo.concatenate %21, %14#4, %22, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x8xf64>, tensor<144x1024x1008xf64>, tensor<144x1024x8xf64>) -> tensor<144x1024x1024xf64>
    %24 = stablehlo.slice %14#7 [0:144, 0:1024, 0:8] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1008xf64>) -> tensor<144x1024x8xf64>
    %25 = stablehlo.slice %14#7 [0:144, 0:1024, 1000:1008] {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1008xf64>) -> tensor<144x1024x8xf64>
    %26 = stablehlo.concatenate %24, %14#7, %25, dim = 2 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x8xf64>, tensor<144x1024x1008xf64>, tensor<144x1024x8xf64>) -> tensor<144x1024x1024xf64>
    %27 = stablehlo.dynamic_update_slice %arg13, %14#6, %c_31, %c_33, %c_33 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<129x1022x1022xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>
    %28 = stablehlo.dynamic_update_slice %arg15, %14#8, %c, %c_29, %c_29 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<1x1010x1010xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>
    %29 = stablehlo.dynamic_update_slice %arg16, %14#9, %c_31, %c_31, %c_31 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>
    %30 = stablehlo.dynamic_update_slice %arg17, %14#10, %c_31, %c_31, %c_31 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>
    %31 = stablehlo.dynamic_update_slice %arg18, %14#11, %c_34, %c_31, %c_31 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1024x1024xf64>, tensor<1x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1024x1024xf64>
    %32 = stablehlo.dynamic_update_slice %arg19, %14#12, %c_31, %c_31, %c_31 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>
    %33 = stablehlo.dynamic_update_slice %arg20, %14#13, %c_31, %c_31, %c_31 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>
    %34 = stablehlo.dynamic_update_slice %arg21, %14#14, %c_31, %c_31, %c_31 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>
    %35 = stablehlo.dynamic_update_slice %arg22, %14#15, %c_34, %c_31, %c_31 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<1x1024x1024xf64>, tensor<1x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1024x1024xf64>
    %36 = stablehlo.dynamic_update_slice %arg23, %14#16, %c_31, %c_31, %c_31 {mhlo.sharding = "{devices=[1,2,2]<=[2,2]T(1,0)}"} : (tensor<144x1024x1024xf64>, tensor<128x1008x1008xf64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<144x1024x1024xf64>
    return %14#1, %14#2, %20, %23, %17, %27, %26, %28, %29, %30, %31, %32, %33, %34, %35, %36 : tensor<f64>, tensor<i64>, tensor<1x1024x1024xf64>, tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>, tensor<1x1024x1024xf64>, tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>, tensor<144x1024x1024xf64>, tensor<1x1024x1024xf64>, tensor<144x1024x1024xf64>
  }
