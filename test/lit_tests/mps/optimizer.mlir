// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @stochastic_gradient_descent(
func.func @stochastic_gradient_descent(%lr: tensor<f32>, %v: tensor<4xf32>, %g: tensor<4xf32>) {
  // CHECK: %{{.+}} = mps.stochastic_gradient_descent %{{.+}}, %{{.+}}, %{{.+}} : (tensor<f32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %0 = mps.stochastic_gradient_descent %lr, %v, %g : (tensor<f32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @apply_stochastic_gradient_descent(
func.func @apply_stochastic_gradient_descent(%lr: tensor<f32>, %var: tensor<4xf32>, %g: tensor<4xf32>) {
  // CHECK: mps.apply_stochastic_gradient_descent %{{.+}}, %{{.+}}, %{{.+}} : (tensor<f32>, tensor<4xf32>, tensor<4xf32>) -> ()
  mps.apply_stochastic_gradient_descent %lr, %var, %g : (tensor<f32>, tensor<4xf32>, tensor<4xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @adam(
func.func @adam(%lr: tensor<f32>, %b1: tensor<f32>, %b2: tensor<f32>, %eps: tensor<f32>, %b1p: tensor<f32>, %b2p: tensor<f32>, %v: tensor<4xf32>, %m: tensor<4xf32>, %vel: tensor<4xf32>, %g: tensor<4xf32>, %maxvel: tensor<4xf32>) {
  // CHECK: %{{.+}}, %{{.+}}, %{{.+}} = mps.adam %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
  %0:3 = mps.adam %lr, %b1, %b2, %eps, %b1p, %b2p, %v, %m, %vel, %g : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
  // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} = mps.adam %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
  %1:4 = mps.adam %lr, %b1, %b2, %eps, %b1p, %b2p, %v, %m, %vel, %maxvel, %g : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @adam_with_current_learning_rate(
func.func @adam_with_current_learning_rate(%lr: tensor<f32>, %b1: tensor<f32>, %b2: tensor<f32>, %eps: tensor<f32>, %v: tensor<4xf32>, %m: tensor<4xf32>, %vel: tensor<4xf32>, %g: tensor<4xf32>, %maxvel: tensor<4xf32>) {
  // CHECK: %{{.+}}, %{{.+}}, %{{.+}} = mps.adam_with_current_learning_rate %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
  %0:3 = mps.adam_with_current_learning_rate %lr, %b1, %b2, %eps, %v, %m, %vel, %g : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
  // CHECK: %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} = mps.adam_with_current_learning_rate %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
  %1:4 = mps.adam_with_current_learning_rate %lr, %b1, %b2, %eps, %v, %m, %vel, %maxvel, %g : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
  return
}
