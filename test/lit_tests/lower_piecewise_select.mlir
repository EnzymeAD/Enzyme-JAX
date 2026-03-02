// RUN: enzymexlamlir-opt %s --pass-pipeline="any(enzyme-hlo-generate-td{patterns=lower_piecewise_select},transform-interpreter,enzyme-hlo-remove-transform)"
module {
// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[ARG0:.*]]: tensor<20x6144x12272xf64>) -> tensor<20x6144x12272xf64> {
// CHECK:           %[[VAL_0:.*]] = stablehlo.constant dense<9> : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_1:.*]] = stablehlo.constant dense<13> : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_2:.*]] = stablehlo.constant dense<7> : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_3:.*]] = stablehlo.constant dense<12272> : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_4:.*]] = stablehlo.constant dense<0> : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_5:.*]] = stablehlo.constant dense<6137> : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_6:.*]] = stablehlo.constant dense<6136> : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_7:.*]] = stablehlo.constant dense<12> : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_8:.*]] = stablehlo.constant dense<8> : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_9:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK:           %[[VAL_10:.*]] = stablehlo.broadcast %[[VAL_9]], sizes = [20, 6144, 12272] : (tensor<f64>) -> tensor<20x6144x12272xf64>
// CHECK:           %[[VAL_11:.*]] = stablehlo.iota dim = 0 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_12:.*]] = stablehlo.compare  GE, %[[VAL_11]], %[[VAL_8]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_13:.*]] = stablehlo.compare  LT, %[[VAL_11]], %[[VAL_7]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_14:.*]] = stablehlo.and %[[VAL_12]], %[[VAL_13]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_15:.*]] = stablehlo.iota dim = 1 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_16:.*]] = stablehlo.compare  GE, %[[VAL_15]], %[[VAL_6]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_17:.*]] = stablehlo.compare  LT, %[[VAL_15]], %[[VAL_5]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_18:.*]] = stablehlo.and %[[VAL_16]], %[[VAL_17]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_19:.*]] = stablehlo.and %[[VAL_14]], %[[VAL_18]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_20:.*]] = stablehlo.iota dim = 2 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_21:.*]] = stablehlo.compare  GE, %[[VAL_20]], %[[VAL_4]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_22:.*]] = stablehlo.compare  LT, %[[VAL_20]], %[[VAL_3]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_23:.*]] = stablehlo.and %[[VAL_21]], %[[VAL_22]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_24:.*]] = stablehlo.and %[[VAL_19]], %[[VAL_23]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_25:.*]] = stablehlo.iota dim = 0 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_26:.*]] = stablehlo.compare  GE, %[[VAL_25]], %[[VAL_2]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_27:.*]] = stablehlo.compare  LT, %[[VAL_25]], %[[VAL_1]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_28:.*]] = stablehlo.and %[[VAL_26]], %[[VAL_27]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_29:.*]] = stablehlo.iota dim = 1 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_30:.*]] = stablehlo.compare  GE, %[[VAL_29]], %[[VAL_8]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_31:.*]] = stablehlo.compare  LT, %[[VAL_29]], %[[VAL_0]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_32:.*]] = stablehlo.and %[[VAL_30]], %[[VAL_31]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_33:.*]] = stablehlo.and %[[VAL_28]], %[[VAL_32]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_34:.*]] = stablehlo.iota dim = 2 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_35:.*]] = stablehlo.compare  GE, %[[VAL_34]], %[[VAL_4]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_36:.*]] = stablehlo.compare  LT, %[[VAL_34]], %[[VAL_3]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_37:.*]] = stablehlo.and %[[VAL_35]], %[[VAL_36]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_38:.*]] = stablehlo.and %[[VAL_33]], %[[VAL_37]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_39:.*]] = stablehlo.or %[[VAL_24]], %[[VAL_38]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_40:.*]] = stablehlo.iota dim = 0 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_41:.*]] = stablehlo.compare  GE, %[[VAL_40]], %[[VAL_7]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_42:.*]] = stablehlo.compare  LT, %[[VAL_40]], %[[VAL_1]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_43:.*]] = stablehlo.and %[[VAL_41]], %[[VAL_42]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_44:.*]] = stablehlo.iota dim = 1 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_45:.*]] = stablehlo.compare  GE, %[[VAL_44]], %[[VAL_0]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_46:.*]] = stablehlo.compare  LT, %[[VAL_44]], %[[VAL_6]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_47:.*]] = stablehlo.and %[[VAL_45]], %[[VAL_46]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_48:.*]] = stablehlo.and %[[VAL_43]], %[[VAL_47]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_49:.*]] = stablehlo.iota dim = 2 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_50:.*]] = stablehlo.compare  GE, %[[VAL_49]], %[[VAL_4]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_51:.*]] = stablehlo.compare  LT, %[[VAL_49]], %[[VAL_3]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_52:.*]] = stablehlo.and %[[VAL_50]], %[[VAL_51]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_53:.*]] = stablehlo.and %[[VAL_48]], %[[VAL_52]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_54:.*]] = stablehlo.or %[[VAL_39]], %[[VAL_53]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_55:.*]] = stablehlo.iota dim = 0 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_56:.*]] = stablehlo.compare  GE, %[[VAL_55]], %[[VAL_2]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_57:.*]] = stablehlo.compare  LT, %[[VAL_55]], %[[VAL_8]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_58:.*]] = stablehlo.and %[[VAL_56]], %[[VAL_57]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_59:.*]] = stablehlo.iota dim = 1 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_60:.*]] = stablehlo.compare  GE, %[[VAL_59]], %[[VAL_0]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_61:.*]] = stablehlo.compare  LT, %[[VAL_59]], %[[VAL_6]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_62:.*]] = stablehlo.and %[[VAL_60]], %[[VAL_61]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_63:.*]] = stablehlo.and %[[VAL_58]], %[[VAL_62]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_64:.*]] = stablehlo.iota dim = 2 : tensor<20x6144x12272xi32>
// CHECK:           %[[VAL_65:.*]] = stablehlo.compare  GE, %[[VAL_64]], %[[VAL_4]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_66:.*]] = stablehlo.compare  LT, %[[VAL_64]], %[[VAL_3]] : (tensor<20x6144x12272xi32>, tensor<20x6144x12272xi32>) -> tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_67:.*]] = stablehlo.and %[[VAL_65]], %[[VAL_66]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_68:.*]] = stablehlo.and %[[VAL_63]], %[[VAL_67]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_69:.*]] = stablehlo.or %[[VAL_54]], %[[VAL_68]] : tensor<20x6144x12272xi1>
// CHECK:           %[[VAL_70:.*]] = stablehlo.select %[[VAL_69]], %[[VAL_10]], %[[ARG0]] : tensor<20x6144x12272xi1>, tensor<20x6144x12272xf64>
// CHECK:           return %[[VAL_70]] : tensor<20x6144x12272xf64>
// CHECK:         }
  func.func @foo(%arg0: tensor<20x6144x12272xf64>) -> tensor<20x6144x12272xf64> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %0 = stablehlo.broadcast %cst, sizes = [20, 6144, 12272] : (tensor<f64>) -> tensor<20x6144x12272xf64>
    %1 = "enzymexla.piecewise_select"(%0, %arg0) <{boxes = [[8, 12, 6136, 6137, 0, 12272], [7, 13, 8, 9, 0, 12272], [12, 13, 9, 6136, 0, 12272], [7, 8, 9, 6136, 0, 12272]]}> : (tensor<20x6144x12272xf64>, tensor<20x6144x12272xf64>) -> tensor<20x6144x12272xf64>
    return %1 : tensor<20x6144x12272xf64>
  }
}
