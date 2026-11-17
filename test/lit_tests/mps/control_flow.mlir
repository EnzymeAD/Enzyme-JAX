// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @call(
func.func @call(%x: tensor<4xf32>, %y: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.call @callee(%{{.+}}, %{{.+}}) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  %0:2 = mps.call @callee(%x, %y) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @if(
func.func @if(%p: tensor<i1>, %x: tensor<4xf32>, %y: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.if %{{.+}} : tensor<i1> -> tensor<4xf32> {
  %0 = mps.if %p : tensor<i1> -> tensor<4xf32> {
    // CHECK: mps.yield %{{.+}} : tensor<4xf32>
    mps.yield %x : tensor<4xf32>
  // CHECK: } else {
  } else {
    // CHECK: mps.yield %{{.+}} : tensor<4xf32>
    mps.yield %y : tensor<4xf32>
  }
  // CHECK: mps.if %{{.+}} : tensor<i1> {
  mps.if %p : tensor<i1> {
    // CHECK: mps.yield
    mps.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @for(
func.func @for(%lb: tensor<i32>, %ub: tensor<i32>, %step: tensor<i32>, %x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.for(%{{.+}}, %{{.+}}, %{{.+}}) iter_args(%{{.+}}) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>) -> tensor<4xf32> {
  %0 = mps.for(%lb, %ub, %step) iter_args(%x) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: ^bb0(%{{.+}}: tensor<i32>, %{{.+}}: tensor<4xf32>):
  ^bb0(%i: tensor<i32>, %acc: tensor<4xf32>):
    // CHECK: mps.yield %{{.+}} : tensor<4xf32>
    mps.yield %acc : tensor<4xf32>
  }
  // CHECK: mps.for(%{{.+}}, %{{.+}}, %{{.+}}) : (tensor<i32>, tensor<i32>, tensor<i32>) -> () {
  mps.for(%lb, %ub, %step) : (tensor<i32>, tensor<i32>, tensor<i32>) -> () {
  // CHECK: ^bb0(%{{.+}}: tensor<i32>):
  ^bb0(%i: tensor<i32>):
    // CHECK: mps.yield
    mps.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @while(
func.func @while(%p: tensor<i1>, %x: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.while(%{{.+}}) : (tensor<4xf32>) -> tensor<4xf32> {
  %0 = mps.while(%x) : (tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: ^bb0(%{{.+}}: tensor<4xf32>):
  ^bb0(%a: tensor<4xf32>):
    // CHECK: mps.condition(%{{.+}} : tensor<i1>) %{{.+}} : tensor<4xf32>
    mps.condition(%p : tensor<i1>) %a : tensor<4xf32>
  // CHECK: } do {
  } do {
  // CHECK: ^bb0(%{{.+}}: tensor<4xf32>):
  ^bb0(%a: tensor<4xf32>):
    // CHECK: mps.yield %{{.+}} : tensor<4xf32>
    mps.yield %a : tensor<4xf32>
  }
  return
}
