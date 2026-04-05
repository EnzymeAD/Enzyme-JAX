// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first expansion-size=2" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last expansion-size=2" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple expansion-size=2" %s | FileCheck --check-prefix=TUPLE %s

func.func @compare_f64_eq(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>) -> tensor<4x4xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xi1>
  return %0 : tensor<4x4xi1>
}

// FIRST-LABEL: @compare_f64_eq
// FIRST: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// FIRST: %[[SUB1:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<4x4xf64>
// FIRST: %[[C2:.*]] = stablehlo.convert %[[SUB1]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[CAT1:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// FIRST: %[[C3:.*]] = stablehlo.convert %arg1 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[C4:.*]] = stablehlo.convert %[[C3]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// FIRST: %[[SUB2:.*]] = stablehlo.subtract %arg1, %[[C4]] : tensor<4x4xf64>
// FIRST: %[[C5:.*]] = stablehlo.convert %[[SUB2]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[R3:.*]] = stablehlo.reshape %[[C3]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[R4:.*]] = stablehlo.reshape %[[C5]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[CAT2:.*]] = stablehlo.concatenate %[[R3]], %[[R4]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// FIRST: %[[CMP:.*]] = stablehlo.compare EQ, %[[CAT1]], %[[CAT2]] : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xi1>
// FIRST: %[[INIT:.*]] = stablehlo.constant dense<true> : tensor<i1>
// FIRST: %[[OUT:.*]] = stablehlo.reduce(%[[CMP]] init: %[[INIT]]) applies stablehlo.and across dimensions = [0] : (tensor<2x4x4xi1>, tensor<i1>) -> tensor<4x4xi1>
// FIRST: return %[[OUT]] : tensor<4x4xi1>

// LAST-LABEL: @compare_f64_eq
// LAST: %[[CMP:.*]] = stablehlo.compare EQ, %{{.*}}, %{{.*}} : (tensor<4x4x2xf32>, tensor<4x4x2xf32>) -> tensor<4x4x2xi1>
// LAST: %[[INIT:.*]] = stablehlo.constant dense<true> : tensor<i1>
// LAST: %[[OUT:.*]] = stablehlo.reduce(%[[CMP]] init: %[[INIT]]) applies stablehlo.and across dimensions = [2] : (tensor<4x4x2xi1>, tensor<i1>) -> tensor<4x4xi1>
// LAST: return %[[OUT]] : tensor<4x4xi1>

// TUPLE-LABEL: @compare_f64_eq
// TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[HIGH_EQ:.*]] = stablehlo.compare EQ, %[[HIGH]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[LOW_EQ:.*]] = stablehlo.compare EQ, %[[LOW]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[HIGH_RESHAPED:.*]] = stablehlo.reshape %[[HIGH_EQ]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
// TUPLE: %[[LOW_RESHAPED:.*]] = stablehlo.reshape %[[LOW_EQ]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
// TUPLE: %[[OUT:.*]] = stablehlo.and %[[HIGH_RESHAPED]], %[[LOW_RESHAPED]] : tensor<4x4xi1>
// TUPLE: return %[[OUT]] : tensor<4x4xi1>


func.func @compare_f64_ne(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>) -> tensor<4x4xi1> {
  %0 = stablehlo.compare NE, %arg0, %arg1 : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xi1>
  return %0 : tensor<4x4xi1>
}

// FIRST-LABEL: @compare_f64_ne
// FIRST: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// FIRST: %[[SUB1:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<4x4xf64>
// FIRST: %[[C2:.*]] = stablehlo.convert %[[SUB1]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[CAT1:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// FIRST: %[[C3:.*]] = stablehlo.convert %arg1 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[C4:.*]] = stablehlo.convert %[[C3]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// FIRST: %[[SUB2:.*]] = stablehlo.subtract %arg1, %[[C4]] : tensor<4x4xf64>
// FIRST: %[[C5:.*]] = stablehlo.convert %[[SUB2]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[R3:.*]] = stablehlo.reshape %[[C3]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[R4:.*]] = stablehlo.reshape %[[C5]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[CAT2:.*]] = stablehlo.concatenate %[[R3]], %[[R4]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// FIRST: %[[CMP:.*]] = stablehlo.compare NE, %[[CAT1]], %[[CAT2]] : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xi1>
// FIRST: %[[INIT:.*]] = stablehlo.constant dense<false> : tensor<i1>
// FIRST: %[[OUT:.*]] = stablehlo.reduce(%[[CMP]] init: %[[INIT]]) applies stablehlo.or across dimensions = [0] : (tensor<2x4x4xi1>, tensor<i1>) -> tensor<4x4xi1>
// FIRST: return %[[OUT]] : tensor<4x4xi1>

// LAST-LABEL: @compare_f64_ne
// LAST: %[[CMP:.*]] = stablehlo.compare NE, %{{.*}}, %{{.*}} : (tensor<4x4x2xf32>, tensor<4x4x2xf32>) -> tensor<4x4x2xi1>
// LAST: %[[INIT:.*]] = stablehlo.constant dense<false> : tensor<i1>
// LAST: %[[OUT:.*]] = stablehlo.reduce(%[[CMP]] init: %[[INIT]]) applies stablehlo.or across dimensions = [2] : (tensor<4x4x2xi1>, tensor<i1>) -> tensor<4x4xi1>
// LAST: return %[[OUT]] : tensor<4x4xi1>

// TUPLE-LABEL: @compare_f64_ne
// TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[HIGH_NE:.*]] = stablehlo.compare NE, %[[HIGH]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[LOW_NE:.*]] = stablehlo.compare NE, %[[LOW]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[HIGH_RESHAPED:.*]] = stablehlo.reshape %[[HIGH_NE]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
// TUPLE: %[[LOW_RESHAPED:.*]] = stablehlo.reshape %[[LOW_NE]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
// TUPLE: %[[OUT:.*]] = stablehlo.or %[[HIGH_RESHAPED]], %[[LOW_RESHAPED]] : tensor<4x4xi1>
// TUPLE: return %[[OUT]] : tensor<4x4xi1>


func.func @compare_f64_ge(%arg0: tensor<4x4xf64>, %arg1: tensor<4x4xf64>) -> tensor<4x4xi1> {
  %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xi1>
  return %0 : tensor<4x4xi1>
}

// FIRST-LABEL: @compare_f64_ge
// FIRST: %[[C0:.*]] = stablehlo.convert %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[C1:.*]] = stablehlo.convert %[[C0]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// FIRST: %[[SUB1:.*]] = stablehlo.subtract %arg0, %[[C1]] : tensor<4x4xf64>
// FIRST: %[[C2:.*]] = stablehlo.convert %[[SUB1]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[R1:.*]] = stablehlo.reshape %[[C0]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[R2:.*]] = stablehlo.reshape %[[C2]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[CAT1:.*]] = stablehlo.concatenate %[[R1]], %[[R2]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// FIRST: %[[C3:.*]] = stablehlo.convert %arg1 : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[C4:.*]] = stablehlo.convert %[[C3]] : (tensor<4x4xf32>) -> tensor<4x4xf64>
// FIRST: %[[SUB2:.*]] = stablehlo.subtract %arg1, %[[C4]] : tensor<4x4xf64>
// FIRST: %[[C5:.*]] = stablehlo.convert %[[SUB2]] : (tensor<4x4xf64>) -> tensor<4x4xf32>
// FIRST: %[[R3:.*]] = stablehlo.reshape %[[C3]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[R4:.*]] = stablehlo.reshape %[[C5]] : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[CAT2:.*]] = stablehlo.concatenate %[[R3]], %[[R4]], dim = 0 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<2x4x4xf32>
// FIRST: %[[X_HI:.*]] = stablehlo.slice %[[CAT1]] [0:1, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[X_LO:.*]] = stablehlo.slice %[[CAT1]] [1:2, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[Y_HI:.*]] = stablehlo.slice %[[CAT2]] [0:1, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[Y_LO:.*]] = stablehlo.slice %[[CAT2]] [1:2, 0:4, 0:4] : (tensor<2x4x4xf32>) -> tensor<1x4x4xf32>
// FIRST: %[[GT:.*]] = stablehlo.compare GT, %[[X_HI]], %[[Y_HI]] : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xi1>
// FIRST: %[[EQ:.*]] = stablehlo.compare EQ, %[[X_HI]], %[[Y_HI]] : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xi1>
// FIRST: %[[GE:.*]] = stablehlo.compare GE, %[[X_LO]], %[[Y_LO]] : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xi1>
// FIRST: %[[SEL:.*]] = stablehlo.select %[[EQ]], %[[GE]], %[[GT]] : tensor<1x4x4xi1>, tensor<1x4x4xi1>
// FIRST: %[[OUT:.*]] = stablehlo.reshape %[[SEL]] : (tensor<1x4x4xi1>) -> tensor<4x4xi1>
// FIRST: return %[[OUT]] : tensor<4x4xi1>

// LAST-LABEL: @compare_f64_ge
// LAST: %[[HIGH:.*]] = stablehlo.slice %{{.*}} [0:4, 0:4, 0:1] : (tensor<4x4x2xf32>) -> tensor<4x4x1xf32>
// LAST: %[[LOW:.*]] = stablehlo.slice %{{.*}} [0:4, 0:4, 1:2] : (tensor<4x4x2xf32>) -> tensor<4x4x1xf32>
// LAST: %[[HIGH_GT:.*]] = stablehlo.compare GT, %[[HIGH]], %{{.*}} : (tensor<4x4x1xf32>, tensor<4x4x1xf32>) -> tensor<4x4x1xi1>
// LAST: %[[HIGH_EQ:.*]] = stablehlo.compare EQ, %[[HIGH]], %{{.*}} : (tensor<4x4x1xf32>, tensor<4x4x1xf32>) -> tensor<4x4x1xi1>
// LAST: %[[LOW_GE:.*]] = stablehlo.compare GE, %[[LOW]], %{{.*}} : (tensor<4x4x1xf32>, tensor<4x4x1xf32>) -> tensor<4x4x1xi1>
// LAST: %[[OUT:.*]] = stablehlo.select %[[HIGH_EQ]], %[[LOW_GE]], %[[HIGH_GT]] : tensor<4x4x1xi1>
// LAST: return %{{.*}} : tensor<4x4xi1>

// TUPLE-LABEL: @compare_f64_ge
// TUPLE: %[[HIGH:.*]] = stablehlo.get_tuple_element %{{.*}}[0] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[LOW:.*]] = stablehlo.get_tuple_element %{{.*}}[1] : (tuple<tensor<4x4xf32>, tensor<4x4xf32>>) -> tensor<4x4xf32>
// TUPLE: %[[HIGH_GT:.*]] = stablehlo.compare GT, %[[HIGH]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[HIGH_EQ:.*]] = stablehlo.compare EQ, %[[HIGH]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[LOW_GE:.*]] = stablehlo.compare GE, %[[LOW]], %{{.*}} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// TUPLE: %[[OUT:.*]] = stablehlo.select %[[HIGH_EQ]], %[[LOW_GE]], %[[HIGH_GT]] : tensor<4x4xi1>
// TUPLE: return %[[OUT]] : tensor<4x4xi1>
