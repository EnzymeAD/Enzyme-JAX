// RUN: enzymexlamlir-opt %s -parse-optimization-rules | FileCheck %s

module {
  tessera.optimizations {
    tessera.optimization "tessera.shorthand(x, .5) -> tessera.mulf(x, 2.)"
    tessera.optimization "arith.mulf(x, -1.0) -> arith.negf(x)"
    tessera.optimization "tessera.sqrt(x) -> tessera.pow(x, 0.5)"
    tessera.optimization "tessera.circle(r) -> tessera.mulf(r, 3.141592653589793)"
    tessera.optimization "tessera.scale(x) -> tessera.mul(x, 3000000000)"
    tessera.optimization "tessera.wide(x, 3000000000) -> tessera.mul(x, x)"
    tessera.optimization "eigen.inv(eigen.inv(x)) -> x"
    tessera.optimization "eigen.mag(arith.negf(x),y,z) -> eigen.mag(x,y,z)"
    tessera.optimization "tessera.pow(x, 2) -> tessera.mul(x, x)"
  }
}

// CHECK: module @patterns

// CHECK: pdl.pattern : benefit(1) {
// CHECK-NEXT:   %[[X0:.*]] = operand
// CHECK-NEXT:   %[[T0:.*]] = type
// CHECK-NEXT:   %[[C0:.*]] = operation  -> (%[[T0]] : !pdl.type)
// CHECK-NEXT:   %[[TWO:.*]] = attribute = 2 : i32
// CHECK-NEXT:   apply_native_constraint "isConstantEqualTo"(%[[C0]], %[[TWO]] : !pdl.operation, !pdl.attribute)
// CHECK-NEXT:   %[[RES0:.*]] = result 0 of %[[C0]]
// CHECK-NEXT:   %[[POW:.*]] = attribute = @tessera.pow
// CHECK-NEXT:   %[[T1:.*]] = type
// CHECK-NEXT:   %[[POW_CALL:.*]] = operation "tessera.call"(%[[X0]], %[[RES0]] : !pdl.value, !pdl.value)  {"callee" = %[[POW]]} -> (%[[T1]] : !pdl.type)
// CHECK-NEXT:   %[[POW_RES:.*]] = result 0 of %[[POW_CALL]]
// CHECK-NEXT:   rewrite %[[POW_CALL]] {
// CHECK-NEXT:     %[[MUL:.*]] = attribute = @tessera.mul
// CHECK-NEXT:     %[[T2:.*]] = type
// CHECK-NEXT:     %[[MUL_CALL:.*]] = operation "tessera.call"(%[[X0]], %[[X0]] : !pdl.value, !pdl.value)  {"callee" = %[[MUL]]} -> (%[[T2]] : !pdl.type)
// CHECK-NEXT:     %[[MUL_RES:.*]] = result 0 of %[[MUL_CALL]]
// CHECK-NEXT:     replace %[[POW_CALL]] with %[[MUL_CALL]]
// CHECK-NEXT:   }
// CHECK-NEXT: }

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

// CHECK: pdl.pattern : benefit(1) {
// CHECK-NEXT:   %[[X0:.*]] = operand
// CHECK-NEXT:   %[[T0:.*]] = type
// CHECK-NEXT:   %[[C0:.*]] = operation  -> (%[[T0]] : !pdl.type)
// CHECK-NEXT:   %[[BIG:.*]] = attribute = 3000000000 : i64
// CHECK-NEXT:   apply_native_constraint "isConstantEqualTo"(%[[C0]], %[[BIG]] : !pdl.operation, !pdl.attribute)
// CHECK-NEXT:   %[[RES0:.*]] = result 0 of %[[C0]]
// CHECK-NEXT:   %[[WIDE:.*]] = attribute = @tessera.wide
// CHECK-NEXT:   %[[T1:.*]] = type
// CHECK-NEXT:   %[[WIDE_CALL:.*]] = operation "tessera.call"(%[[X0]], %[[RES0]] : !pdl.value, !pdl.value)  {"callee" = %[[WIDE]]} -> (%[[T1]] : !pdl.type)
// CHECK-NEXT:   %{{.*}} = result 0 of %[[WIDE_CALL]]
// CHECK-NEXT:   rewrite %[[WIDE_CALL]] {
// CHECK-NEXT:     %[[MUL:.*]] = attribute = @tessera.mul
// CHECK-NEXT:     %[[T2:.*]] = type
// CHECK-NEXT:     %[[MUL_CALL:.*]] = operation "tessera.call"(%[[X0]], %[[X0]] : !pdl.value, !pdl.value)  {"callee" = %[[MUL]]} -> (%[[T2]] : !pdl.type)
// CHECK-NEXT:     %{{.*}} = result 0 of %[[MUL_CALL]]
// CHECK-NEXT:     replace %[[WIDE_CALL]] with %[[MUL_CALL]]
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: pdl.pattern : benefit(1) {
// CHECK-NEXT:   %[[X0:.*]] = operand
// CHECK-NEXT:   %[[SCALE:.*]] = attribute = @tessera.scale
// CHECK-NEXT:   %[[T0:.*]] = type
// CHECK-NEXT:   %[[SCALE_CALL:.*]] = operation "tessera.call"(%[[X0]] : !pdl.value) {"callee" = %[[SCALE]]} -> (%[[T0]] : !pdl.type)
// CHECK-NEXT:   %{{.*}} = result 0 of %[[SCALE_CALL]]
// CHECK-NEXT:   rewrite %[[SCALE_CALL]] {
// CHECK-NEXT:     %[[BIG:.*]] = attribute = 3000000000 : i64
// CHECK-NEXT:     %[[CST:.*]] = operation "llvm.mlir.constant" {{.*}}"value" = %[[BIG]]{{.*}}
// CHECK-NEXT:     %[[CST_RES:.*]] = result 0 of %[[CST]]
// CHECK-NEXT:     %[[MUL:.*]] = attribute = @tessera.mul
// CHECK-NEXT:     %[[T1:.*]] = type
// CHECK-NEXT:     %[[MUL_CALL:.*]] = operation "tessera.call"(%[[X0]], %[[CST_RES]] : !pdl.value, !pdl.value)  {"callee" = %[[MUL]]} -> (%[[T1]] : !pdl.type)
// CHECK-NEXT:     %{{.*}} = result 0 of %[[MUL_CALL]]
// CHECK-NEXT:     replace %[[SCALE_CALL]] with %[[MUL_CALL]]
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: pdl.pattern : benefit(1) {
// CHECK-NEXT:   %[[R:.*]] = operand
// CHECK-NEXT:   %[[CIRCLE:.*]] = attribute = @tessera.circle
// CHECK-NEXT:   %[[T0:.*]] = type
// CHECK-NEXT:   %[[CIRCLE_CALL:.*]] = operation "tessera.call"(%[[R]] : !pdl.value) {"callee" = %[[CIRCLE]]} -> (%[[T0]] : !pdl.type)
// CHECK-NEXT:   %{{.*}} = result 0 of %[[CIRCLE_CALL]]
// CHECK-NEXT:   rewrite %[[CIRCLE_CALL]] {
// CHECK-NEXT:     %[[PI:.*]] = attribute = {{.*}} : f64
// CHECK-NEXT:     %[[CST:.*]] = operation "llvm.mlir.constant" {{.*}}"value" = %[[PI]]{{.*}}
// CHECK-NEXT:     %[[CST_RES:.*]] = result 0 of %[[CST]]
// CHECK-NEXT:     %[[MULF:.*]] = attribute = @tessera.mulf
// CHECK-NEXT:     %[[T1:.*]] = type
// CHECK-NEXT:     %[[MULF_CALL:.*]] = operation "tessera.call"(%[[R]], %[[CST_RES]] : !pdl.value, !pdl.value)  {"callee" = %[[MULF]]} -> (%[[T1]] : !pdl.type)
// CHECK-NEXT:     %{{.*}} = result 0 of %[[MULF_CALL]]
// CHECK-NEXT:     replace %[[CIRCLE_CALL]] with %[[MULF_CALL]]
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: pdl.pattern : benefit(1) {
// CHECK-NEXT:   %[[X0:.*]] = operand
// CHECK-NEXT:   %[[SQRT:.*]] = attribute = @tessera.sqrt
// CHECK-NEXT:   %[[T0:.*]] = type
// CHECK-NEXT:   %[[SQRT_CALL:.*]] = operation "tessera.call"(%[[X0]] : !pdl.value) {"callee" = %[[SQRT]]} -> (%[[T0]] : !pdl.type)
// CHECK-NEXT:   %{{.*}} = result 0 of %[[SQRT_CALL]]
// CHECK-NEXT:   rewrite %[[SQRT_CALL]] {
// CHECK-NEXT:     %[[POINT5:.*]] = attribute = 5.000000e-01 : f32
// CHECK-NEXT:     %[[CST:.*]] = operation "llvm.mlir.constant" {{.*}}"value" = %[[POINT5]]{{.*}}
// CHECK-NEXT:     %[[CST_RES:.*]] = result 0 of %[[CST]]
// CHECK-NEXT:     %[[POW:.*]] = attribute = @tessera.pow
// CHECK-NEXT:     %[[T1:.*]] = type
// CHECK-NEXT:     %[[POW_CALL:.*]] = operation "tessera.call"(%[[X0]], %[[CST_RES]] : !pdl.value, !pdl.value)  {"callee" = %[[POW]]} -> (%[[T1]] : !pdl.type)
// CHECK-NEXT:     %{{.*}} = result 0 of %[[POW_CALL]]
// CHECK-NEXT:     replace %[[SQRT_CALL]] with %[[POW_CALL]]
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: pdl.pattern : benefit(1) {
// CHECK-NEXT:   %[[X0:.*]] = operand
// CHECK-NEXT:   %[[T0:.*]] = type
// CHECK-NEXT:   %[[C0:.*]] = operation  -> (%[[T0]] : !pdl.type)
// CHECK-NEXT:   %[[NEG1:.*]] = attribute = -1.000000e+00 : f64
// CHECK-NEXT:   apply_native_constraint "isFloatConstantEqualTo"(%[[C0]], %[[NEG1]] : !pdl.operation, !pdl.attribute)
// CHECK-NEXT:   %[[RES0:.*]] = result 0 of %[[C0]]
// CHECK-NEXT:   %[[MULF:.*]] = attribute = @arith.mulf
// CHECK-NEXT:   %[[T1:.*]] = type
// CHECK-NEXT:   %[[MULF_CALL:.*]] = operation "tessera.call"(%[[X0]], %[[RES0]] : !pdl.value, !pdl.value)  {"callee" = %[[MULF]]} -> (%[[T1]] : !pdl.type)
// CHECK-NEXT:   %{{.*}} = result 0 of %[[MULF_CALL]]
// CHECK-NEXT:   rewrite %[[MULF_CALL]] {
// CHECK-NEXT:     %[[NEGF:.*]] = attribute = @arith.negf
// CHECK-NEXT:     %[[T2:.*]] = type
// CHECK-NEXT:     %[[NEGF_CALL:.*]] = operation "tessera.call"(%[[X0]] : !pdl.value) {"callee" = %[[NEGF]]} -> (%[[T2]] : !pdl.type)
// CHECK-NEXT:     %{{.*}} = result 0 of %[[NEGF_CALL]]
// CHECK-NEXT:     replace %[[MULF_CALL]] with %[[NEGF_CALL]]
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK: pdl.pattern : benefit(1) {
// CHECK-NEXT:   %[[X0:.*]] = operand
// CHECK-NEXT:   %[[T0:.*]] = type
// CHECK-NEXT:   %[[C0:.*]] = operation  -> (%[[T0]] : !pdl.type)
// CHECK-NEXT:   %[[POINT5:.*]] = attribute = 5.000000e-01 : f64
// CHECK-NEXT:   apply_native_constraint "isFloatConstantEqualTo"(%[[C0]], %[[POINT5]] : !pdl.operation, !pdl.attribute)
// CHECK-NEXT:   %[[RES0:.*]] = result 0 of %[[C0]]
// CHECK-NEXT:   %[[SH:.*]] = attribute = @tessera.shorthand
// CHECK-NEXT:   %[[T1:.*]] = type
// CHECK-NEXT:   %[[SH_CALL:.*]] = operation "tessera.call"(%[[X0]], %[[RES0]] : !pdl.value, !pdl.value)  {"callee" = %[[SH]]} -> (%[[T1]] : !pdl.type)
// CHECK-NEXT:   %{{.*}} = result 0 of %[[SH_CALL]]
// CHECK-NEXT:   rewrite %[[SH_CALL]] {
// CHECK-NEXT:     %[[TWO:.*]] = attribute = 2.000000e+00 : f32
// CHECK-NEXT:     %[[CST:.*]] = operation "llvm.mlir.constant" {{.*}}"value" = %[[TWO]]{{.*}}
// CHECK-NEXT:     %[[CST_RES:.*]] = result 0 of %[[CST]]
// CHECK-NEXT:     %[[MULF:.*]] = attribute = @tessera.mulf
// CHECK-NEXT:     %[[T2:.*]] = type
// CHECK-NEXT:     %[[MULF_CALL:.*]] = operation "tessera.call"(%[[X0]], %[[CST_RES]] : !pdl.value, !pdl.value)  {"callee" = %[[MULF]]} -> (%[[T2]] : !pdl.type)
// CHECK-NEXT:     %{{.*}} = result 0 of %[[MULF_CALL]]
// CHECK-NEXT:     replace %[[SH_CALL]] with %[[MULF_CALL]]
// CHECK-NEXT:   }
// CHECK-NEXT: }
