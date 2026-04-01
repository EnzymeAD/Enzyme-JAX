// RUN: enzymexlamlir-opt %s -parse-optimization-rules | FileCheck %s

module {
  tessera.optimizations {
    tessera.optimization "eigen.inv(eigen.inv(x)) -> x"
    tessera.optimization "eigen.mag(arith.negf(x),y,z) -> eigen.mag(x,y,z)"
  }
}

// CHECK: module @patterns

// CHECK: pdl.pattern : benefit(1) {
// CHECK-NEXT:   %[[X0:.*]] = operand
// CHECK-NEXT:   %[[NEGF:.*]] = attribute = @arith.negf
// CHECK-NEXT:   %[[T0:.*]] = type
// CHECK-NEXT:   %[[NEGF_CALL:.*]] = operation "tessera.call"(%[[X0]] : !pdl.value) {"callee" = %[[NEGF]]} -> (%[[T0]] : !pdl.type)
// CHECK-NEXT:   %[[NEGF_RES:.*]] = result 0 of %[[NEGF_CALL]]
// CHECK-NEXT:   %[[Y:.*]] = operand
// CHECK-NEXT:   %[[Z:.*]] = operand
// CHECK-NEXT:   %[[MAG:.*]] = attribute = @eigen.mag
// CHECK-NEXT:   %[[T1:.*]] = type
// CHECK-NEXT:   %[[MAG_CALL:.*]] = operation "tessera.call"(%[[NEGF_RES]], %[[Y]], %[[Z]] : !pdl.value, !pdl.value, !pdl.value) {"callee" = %[[MAG]]} -> (%[[T1]] : !pdl.type)
// CHECK-NEXT:   %{{.*}} = result 0 of %[[MAG_CALL]]
// CHECK-NEXT:   rewrite %[[MAG_CALL]] {
// CHECK-NEXT:     %[[MAG2:.*]] = attribute = @eigen.mag
// CHECK-NEXT:     %[[T2:.*]] = type
// CHECK-NEXT:     %[[NEW_CALL:.*]] = operation "tessera.call"(%[[X0]], %[[Y]], %[[Z]] : !pdl.value, !pdl.value, !pdl.value) {"callee" = %[[MAG2]]} -> (%[[T2]] : !pdl.type)
// CHECK-NEXT:     %{{.*}} = result 0 of %[[NEW_CALL]]
// CHECK-NEXT:     replace %[[MAG_CALL]] with %[[NEW_CALL]]
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: pdl.pattern : benefit(1) {
// CHECK-NEXT:   %[[X0:.*]] = operand
// CHECK-NEXT:   %[[INV1:.*]] = attribute = @eigen.inv
// CHECK-NEXT:   %[[T0:.*]] = type
// CHECK-NEXT:   %[[INV1_CALL:.*]] = operation "tessera.call"(%[[X0]] : !pdl.value) {"callee" = %[[INV1]]} -> (%[[T0]] : !pdl.type)
// CHECK-NEXT:   %[[INV1_RES:.*]] = result 0 of %[[INV1_CALL]]
// CHECK-NEXT:   %[[INV2:.*]] = attribute = @eigen.inv
// CHECK-NEXT:   %[[T1:.*]] = type
// CHECK-NEXT:   %[[INV2_CALL:.*]] = operation "tessera.call"(%[[INV1_RES]] : !pdl.value) {"callee" = %[[INV2]]} -> (%[[T1]] : !pdl.type)
// CHECK-NEXT:   %{{.*}} = result 0 of %[[INV2_CALL]]
// CHECK-NEXT:   rewrite %[[INV2_CALL]] {
// CHECK-NEXT:     replace %[[INV2_CALL]] with(%[[X0]] : !pdl.value)
// CHECK-NEXT:   }
// CHECK-NEXT: }
