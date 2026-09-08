// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reshape_while;reshape_bitcast_convert" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reshape_while;reshape_bitcast_convert" --transform-interpreter --enzyme-hlo-remove-transform --inline --canonicalize --symbol-dce --drop-unsupported-attributes | stablehlo-translate --interpret
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt --inline --canonicalize --symbol-dce --drop-unsupported-attributes | stablehlo-translate --interpret

// Both buffers enter and leave the function flat, but the body uses 2x3
// tensors. Carry those shapes through the loop. The byte view of x must also
// use the new shape while keeping the four bytes of each i32 together.
// Numerically, each iteration computes x' = 2*x + 1 and y' = y + x.
// CHECK-LABEL: func.func @buffers(
// CHECK: stablehlo.reshape {{.*}} : (tensor<6xi32>) -> tensor<2x3xi32>
// CHECK: stablehlo.reshape {{.*}} : (tensor<6xi32>) -> tensor<2x3xi32>
// CHECK: stablehlo.while({{.*}}) : tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>
// CHECK: } do {
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.bitcast_convert {{.*}} : (tensor<2x3xi32>) -> tensor<2x3x4xi8>
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.return {{.*}} : tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>
// CHECK: }
// CHECK: stablehlo.reshape {{.*}} : (tensor<2x3xi32>) -> tensor<6xi32>
// CHECK: stablehlo.reshape {{.*}} : (tensor<2x3xi32>) -> tensor<6xi32>
func.func @buffers(%n: tensor<i32>, %x: tensor<6xi32>, %y: tensor<6xi32>) -> (tensor<6xi32>, tensor<6xi32>) {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %ones = stablehlo.constant dense<1> : tensor<2x3xi32>
  %r:3 = stablehlo.while(%i = %zero, %a = %x, %b = %y) : tensor<i32>, tensor<6xi32>, tensor<6xi32>
  cond {
    %c = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %av = stablehlo.reshape %a : (tensor<6xi32>) -> tensor<2x3xi32>
    %bv = stablehlo.reshape %b : (tensor<6xi32>) -> tensor<2x3xi32>
    // Full optimization may introduce non-view users and leave this loop flat.
    // Keep its byte view static for interpretation; only_bitcast tests a
    // dynamic intermediate when the argument has exclusively bitcast users.
    %bytes = stablehlo.bitcast_convert %a : (tensor<6xi32>) -> tensor<6x4xi8>
    %byte_view = stablehlo.reshape %bytes : (tensor<6x4xi8>) -> tensor<2x3x4xi8>
    %roundtrip = stablehlo.bitcast_convert %byte_view : (tensor<2x3x4xi8>) -> tensor<2x3xi32>
    %twice = stablehlo.add %av, %roundtrip : tensor<2x3xi32>
    %next_a = stablehlo.add %twice, %ones : tensor<2x3xi32>
    %next_b = stablehlo.add %bv, %av : tensor<2x3xi32>
    %af = stablehlo.reshape %next_a : (tensor<2x3xi32>) -> tensor<6xi32>
    %bf = stablehlo.reshape %next_b : (tensor<2x3xi32>) -> tensor<6xi32>
    %next_i = stablehlo.add %i, %one : tensor<i32>
    stablehlo.return %next_i, %af, %bf : tensor<i32>, tensor<6xi32>, tensor<6xi32>
  }
  return %r#1, %r#2 : tensor<6xi32>, tensor<6xi32>
}

// A matching body reshape is insufficient: the condition also consumes the
// original argument directly with a slice. Keep this carried type unchanged.
// CHECK-LABEL: func.func @condition_use(
// CHECK: stablehlo.while({{.*}}) : tensor<i32>, tensor<6xi32>
// CHECK: cond {
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.slice {{.*}} : (tensor<6xi32>) -> tensor<1xi32>
// CHECK: } do {
// CHECK: stablehlo.reshape {{.*}} : (tensor<6xi32>) -> tensor<2x3xi32>
// CHECK: stablehlo.return {{.*}} : tensor<i32>, tensor<6xi32>
func.func @condition_use(%n: tensor<i32>, %x: tensor<6xi32>) -> tensor<6xi32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %limit = stablehlo.constant dense<100> : tensor<i32>
  %ones = stablehlo.constant dense<1> : tensor<2x3xi32>
  %r:2 = stablehlo.while(%i = %zero, %a = %x) : tensor<i32>, tensor<6xi32>
  cond {
    %first = stablehlo.slice %a [0:1] : (tensor<6xi32>) -> tensor<1xi32>
    %value = stablehlo.reshape %first : (tensor<1xi32>) -> tensor<i32>
    %c0 = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c1 = stablehlo.compare LT, %value, %limit, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %c = stablehlo.and %c0, %c1 : tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %av = stablehlo.reshape %a : (tensor<6xi32>) -> tensor<2x3xi32>
    %next = stablehlo.add %av, %ones : tensor<2x3xi32>
    %flat = stablehlo.reshape %next : (tensor<2x3xi32>) -> tensor<6xi32>
    %next_i = stablehlo.add %i, %one : tensor<i32>
    stablehlo.return %next_i, %flat : tensor<i32>, tensor<6xi32>
  }
  return %r#1 : tensor<6xi32>
}

// A matching reshape in the body must not hide an additional arithmetic user
// of the original argument. This computes x' = 3*x and keeps the flat type.
// CHECK-LABEL: func.func @mixed_body_use(
// CHECK: stablehlo.while({{.*}}) : tensor<i32>, tensor<6xi32>
// CHECK: } do {
// CHECK: stablehlo.add {{.*}} : tensor<6xi32>
// CHECK: stablehlo.return {{.*}} : tensor<i32>, tensor<6xi32>
func.func @mixed_body_use(%n: tensor<i32>, %x: tensor<6xi32>) -> tensor<6xi32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %r:2 = stablehlo.while(%i = %zero, %a = %x) : tensor<i32>, tensor<6xi32>
  cond {
    %c = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %av = stablehlo.reshape %a : (tensor<6xi32>) -> tensor<2x3xi32>
    %twice = stablehlo.add %a, %a : tensor<6xi32>
    %twice_view = stablehlo.reshape %twice : (tensor<6xi32>) -> tensor<2x3xi32>
    %next = stablehlo.add %av, %twice_view : tensor<2x3xi32>
    %flat = stablehlo.reshape %next : (tensor<2x3xi32>) -> tensor<6xi32>
    %next_i = stablehlo.add %i, %one : tensor<i32>
    stablehlo.return %next_i, %flat : tensor<i32>, tensor<6xi32>
  }
  return %r#1 : tensor<6xi32>
}

// All users are bitcast converts, with no direct reshape of the argument.
// This must still hoist: the byte view can use the new carried shape directly.
// CHECK-LABEL: func.func @only_bitcast(
// CHECK: stablehlo.while({{.*}}) : tensor<i32>, tensor<2x3xi32>
// CHECK: } do {
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.bitcast_convert {{.*}} : (tensor<2x3xi32>) -> tensor<2x3x4xi8>
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.return {{.*}} : tensor<i32>, tensor<2x3xi32>
func.func @only_bitcast(%n: tensor<i32>, %x: tensor<6xi32>) -> tensor<6xi32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %ones = stablehlo.constant dense<1> : tensor<2x3xi32>
  %r:2 = stablehlo.while(%i = %zero, %a = %x) : tensor<i32>, tensor<6xi32>
  cond {
    %c = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %bytes = stablehlo.bitcast_convert %a : (tensor<6xi32>) -> tensor<?x4xi8>
    %byte_view = stablehlo.reshape %bytes : (tensor<?x4xi8>) -> tensor<2x3x4xi8>
    %value = stablehlo.bitcast_convert %byte_view : (tensor<2x3x4xi8>) -> tensor<2x3xi32>
    %next = stablehlo.add %value, %ones : tensor<2x3xi32>
    %flat = stablehlo.reshape %next : (tensor<2x3xi32>) -> tensor<6xi32>
    %next_i = stablehlo.add %i, %one : tensor<i32>
    stablehlo.return %next_i, %flat : tensor<i32>, tensor<6xi32>
  }
  return %r#1 : tensor<6xi32>
}

// Each yielded value is an input view of the other argument. Updating both
// arguments together must preserve this exchange and must terminate rewriting.
// CHECK-LABEL: func.func @swap(
// CHECK: stablehlo.while({{.*}}) : tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>
// CHECK: } do {
// CHECK-NOT: stablehlo.reshape
// CHECK: stablehlo.return {{.*}} : tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>
func.func @swap(%n: tensor<i32>, %x: tensor<6xi32>, %y: tensor<6xi32>) -> (tensor<6xi32>, tensor<6xi32>) {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %r:3 = stablehlo.while(%i = %zero, %a = %x, %b = %y) : tensor<i32>, tensor<6xi32>, tensor<6xi32>
  cond {
    %c = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %av = stablehlo.reshape %a : (tensor<6xi32>) -> tensor<2x3xi32>
    %bv = stablehlo.reshape %b : (tensor<6xi32>) -> tensor<2x3xi32>
    %af = stablehlo.reshape %bv : (tensor<2x3xi32>) -> tensor<6xi32>
    %bf = stablehlo.reshape %av : (tensor<2x3xi32>) -> tensor<6xi32>
    %next_i = stablehlo.add %i, %one : tensor<i32>
    stablehlo.return %next_i, %af, %bf : tensor<i32>, tensor<6xi32>, tensor<6xi32>
  }
  return %r#1, %r#2 : tensor<6xi32>, tensor<6xi32>
}

// An unused input has no incompatible consumers. Its yielded reshape can
// still move out of the loop, preserving the input when there are zero trips.
// CHECK-LABEL: func.func @unused_input(
// CHECK: stablehlo.while({{.*}}) : tensor<i32>, tensor<2x3xi32>
func.func @unused_input(%n: tensor<i32>, %x: tensor<6xi32>) -> tensor<6xi32> {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %r:2 = stablehlo.while(%i = %zero, %a = %x) : tensor<i32>, tensor<6xi32>
  cond {
    %c = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %data = stablehlo.broadcast_in_dim %i, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %flat = stablehlo.reshape %data : (tensor<2x3xi32>) -> tensor<6xi32>
    %next_i = stablehlo.add %i, %one : tensor<i32>
    stablehlo.return %next_i, %flat : tensor<i32>, tensor<6xi32>
  }
  return %r#1 : tensor<6xi32>
}

func.func @main() {
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %three = stablehlo.constant dense<3> : tensor<i32>
  %x = stablehlo.constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
  %y = stablehlo.constant dense<0> : tensor<6xi32>
  %x1 = stablehlo.constant dense<[3, 5, 7, 9, 11, 13]> : tensor<6xi32>
  %x3 = stablehlo.constant dense<[15, 23, 31, 39, 47, 55]> : tensor<6xi32>
  %y3 = stablehlo.constant dense<[11, 18, 25, 32, 39, 46]> : tensor<6xi32>
  %c3 = stablehlo.constant dense<[4, 5, 6, 7, 8, 9]> : tensor<6xi32>
  %m3 = stablehlo.constant dense<[27, 54, 81, 108, 135, 162]> : tensor<6xi32>
  %u3 = stablehlo.constant dense<2> : tensor<6xi32>
  %r0:2 = func.call @buffers(%zero, %x, %y) : (tensor<i32>, tensor<6xi32>, tensor<6xi32>) -> (tensor<6xi32>, tensor<6xi32>)
  %r1:2 = func.call @buffers(%one, %x, %y) : (tensor<i32>, tensor<6xi32>, tensor<6xi32>) -> (tensor<6xi32>, tensor<6xi32>)
  %r3:2 = func.call @buffers(%three, %x, %y) : (tensor<i32>, tensor<6xi32>, tensor<6xi32>) -> (tensor<6xi32>, tensor<6xi32>)
  check.expect_eq %r0#0, %x : tensor<6xi32>
  check.expect_eq %r0#1, %y : tensor<6xi32>
  check.expect_eq %r1#0, %x1 : tensor<6xi32>
  check.expect_eq %r1#1, %x : tensor<6xi32>
  check.expect_eq %r3#0, %x3 : tensor<6xi32>
  check.expect_eq %r3#1, %y3 : tensor<6xi32>
  %c0 = func.call @condition_use(%zero, %x) : (tensor<i32>, tensor<6xi32>) -> tensor<6xi32>
  %c = func.call @condition_use(%three, %x) : (tensor<i32>, tensor<6xi32>) -> tensor<6xi32>
  check.expect_eq %c0, %x : tensor<6xi32>
  check.expect_eq %c, %c3 : tensor<6xi32>
  %m = func.call @mixed_body_use(%three, %x) : (tensor<i32>, tensor<6xi32>) -> tensor<6xi32>
  check.expect_eq %m, %m3 : tensor<6xi32>
  %b0 = func.call @only_bitcast(%zero, %x) : (tensor<i32>, tensor<6xi32>) -> tensor<6xi32>
  %b = func.call @only_bitcast(%three, %x) : (tensor<i32>, tensor<6xi32>) -> tensor<6xi32>
  check.expect_eq %b0, %x : tensor<6xi32>
  check.expect_eq %b, %c3 : tensor<6xi32>
  %u0 = func.call @unused_input(%zero, %x) : (tensor<i32>, tensor<6xi32>) -> tensor<6xi32>
  %u = func.call @unused_input(%three, %x) : (tensor<i32>, tensor<6xi32>) -> tensor<6xi32>
  check.expect_eq %u0, %x : tensor<6xi32>
  check.expect_eq %u, %u3 : tensor<6xi32>
  %s:2 = func.call @swap(%three, %x, %y) : (tensor<i32>, tensor<6xi32>, tensor<6xi32>) -> (tensor<6xi32>, tensor<6xi32>)
  check.expect_eq %s#0, %y : tensor<6xi32>
  check.expect_eq %s#1, %x : tensor<6xi32>
  return
}
