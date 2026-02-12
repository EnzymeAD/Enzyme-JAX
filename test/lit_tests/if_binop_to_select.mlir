// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

module {
  // Case 1: if(pred) { add(base, other) } else { base }
  func.func @hoist_add_true(%pred: tensor<i1>, %base: tensor<f32>, %other: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.add %base, %other : tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%base) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK-LABEL: func.func @hoist_add_true
  // CHECK-SAME: (%[[PRED:.+]]: tensor<i1>, %[[BASE:.+]]: tensor<f32>, %[[OTHER:.+]]: tensor<f32>)
  // CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[SEL:.+]] = stablehlo.select %[[PRED]], %[[OTHER]], %[[ZERO]] : tensor<i1>, tensor<f32>
  // CHECK: %[[ADD:.+]] = stablehlo.add %[[BASE]], %[[SEL]] : tensor<f32>
  // CHECK: return %[[ADD]]

  // Case 2: if(pred) { base } else { add(base, other) }
  func.func @hoist_add_false(%pred: tensor<i1>, %base: tensor<f32>, %other: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%base) : (tensor<f32>) -> ()
    }, {
      %1 = stablehlo.add %base, %other : tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK-LABEL: func.func @hoist_add_false
  // CHECK-SAME: (%[[PRED:.+]]: tensor<i1>, %[[BASE:.+]]: tensor<f32>, %[[OTHER:.+]]: tensor<f32>)
  // CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[SEL:.+]] = stablehlo.select %[[PRED]], %[[ZERO]], %[[OTHER]] : tensor<i1>, tensor<f32>
  // CHECK: %[[ADD:.+]] = stablehlo.add %[[BASE]], %[[SEL]] : tensor<f32>
  // CHECK: return %[[ADD]]

  // Case with Mul and identity 1.0
  func.func @hoist_mul(%pred: tensor<i1>, %base: tensor<f32>, %other: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.mul %other, %base : tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%base) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK-LABEL: func.func @hoist_mul
  // CHECK: %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[SEL:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %[[ONE]]
  // CHECK: %[[MUL:.+]] = stablehlo.mul %{{.+}}, %[[SEL]]

  // Case with Max and identity -Inf
  func.func @hoist_max(%pred: tensor<i1>, %base: tensor<f32>, %other: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.maximum %base, %other : tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%base) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK-LABEL: func.func @hoist_max
  // CHECK: %[[NEG_INF:.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK: %[[SEL:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %[[NEG_INF]]
  // CHECK: %[[MAX:.+]] = stablehlo.maximum %{{.+}}, %[[SEL]]

  // Case with Min and identity +Inf
  func.func @hoist_min(%pred: tensor<i1>, %base: tensor<f32>, %other: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.minimum %base, %other : tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%base) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK-LABEL: func.func @hoist_min
  // CHECK: %[[POS_INF:.+]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
  // CHECK: %[[SEL:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %[[POS_INF]]
  // CHECK: %[[MIN:.+]] = stablehlo.minimum %{{.+}}, %[[SEL]]

  // Case with multiple results
  func.func @hoist_multiple(%pred: tensor<i1>, %base1: tensor<f32>, %other1: tensor<f32>, %val2: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0:2 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.add %base1, %other1 : tensor<f32>
      "stablehlo.return"(%1, %val2) : (tensor<f32>, tensor<f32>) -> ()
    }, {
       %c2 = stablehlo.constant dense<2.0> : tensor<f32>
      "stablehlo.return"(%base1, %c2) : (tensor<f32>, tensor<f32>) -> ()
    }) : (tensor<i1>) -> (tensor<f32>, tensor<f32>)
    return %0#0, %0#1 : tensor<f32>, tensor<f32>
  }
  // CHECK-LABEL: func.func @hoist_multiple
  // CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[SEL1:.+]] = stablehlo.select %[[PRED]], %[[OTHER1]], %[[ZERO]]
  // CHECK: %[[HOISTED1:.+]] = stablehlo.add %[[BASE1]], %[[SEL1]]
  // CHECK: %[[IF_RES:.+]] = "stablehlo.if"(%[[PRED]])
  // CHECK:   %[[C2:.+]] = stablehlo.constant dense<2.000000e+00>
  // CHECK:   stablehlo.return %[[C2]]
  // CHECK: return %[[HOISTED1]], %[[IF_RES]]

  // Case with broadcasting
  func.func @hoist_add_broadcast(%pred: tensor<i1>, %base: tensor<10xf32>, %other: tensor<10xf32>) -> tensor<10xf32> {
    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.add %base, %other : tensor<10xf32>
      "stablehlo.return"(%1) : (tensor<10xf32>) -> ()
    }, {
      "stablehlo.return"(%base) : (tensor<10xf32>) -> ()
    }) : (tensor<i1>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
  // CHECK-LABEL: func.func @hoist_add_broadcast
  // CHECK: %[[ZERO_SCALAR:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[ZERO_VEC:.+]] = stablehlo.broadcast_in_dim %[[ZERO_SCALAR]], dims = [] : (tensor<f32>) -> tensor<10xf32>
  // CHECK: %[[SEL:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %[[ZERO_VEC]] : tensor<i1>, tensor<10xf32>
  // CHECK: %[[ADD:.+]] = stablehlo.add %{{.+}}, %[[SEL]]

  // TEST: Commutativity - add(other, base)
  func.func @hoist_add_swapped(%pred: tensor<i1>, %base: tensor<f32>, %other: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.add %other, %base : tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%base) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK-LABEL: func.func @hoist_add_swapped
  // CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[SEL:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %[[ZERO]]
  // CHECK: %[[ADD:.+]] = stablehlo.add %[[SEL]], %{{.+}}
  // CHECK: return %[[ADD]]

  // TEST: Non-commutative - subtraction with base as LHS (HOISTABLE)
  func.func @hoist_sub_lhs(%pred: tensor<i1>, %base: tensor<f32>, %other: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.subtract %base, %other : tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%base) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK-LABEL: func.func @hoist_sub_lhs
  // CHECK: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[SEL:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %[[ZERO]]
  // CHECK: %[[SUB:.+]] = stablehlo.subtract %{{.+}}, %[[SEL]]
  // CHECK: return %[[SUB]]

  // TEST: Non-commutative - subtraction with base as RHS (NOT HOISTABLE)
  func.func @no_hoist_sub_rhs(%pred: tensor<i1>, %base: tensor<f32>, %other: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.subtract %other, %base : tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%base) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK-LABEL: func.func @no_hoist_sub_rhs
  // CHECK: "stablehlo.if"
  // CHECK:   stablehlo.subtract
  // CHECK:   stablehlo.return

  // TEST: Non-commutative - division with base as LHS (HOISTABLE)
  func.func @hoist_div_lhs(%pred: tensor<i1>, %base: tensor<f32>, %other: tensor<f32>) -> tensor<f32> {
    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.divide %base, %other : tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }, {
      "stablehlo.return"(%base) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
    return %0 : tensor<f32>
  }
  // CHECK-LABEL: func.func @hoist_div_lhs
  // CHECK: %[[ONE:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[SEL:.+]] = stablehlo.select %{{.+}}, %{{.+}}, %[[ONE]]
  // CHECK: %[[DIV:.+]] = stablehlo.divide %{{.+}}, %[[SEL]]
  // CHECK: return %[[DIV]]
}
