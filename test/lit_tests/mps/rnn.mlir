// RUN: enzymexlamlir-opt --split-input-file %s | enzymexlamlir-opt --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @single_gate_rnn(
func.func @single_gate_rnn(%x: tensor<5x2x4xf32>, %r: tensor<3x3xf32>, %w: tensor<3x4xf32>, %b: tensor<3xf32>, %h0: tensor<2x3xf32>, %mask: tensor<2x3xf32>, %xh: tensor<5x2x3xf32>) {
  // CHECK: %{{.+}} = "mps.single_gate_rnn"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{activation = #mps.rnn_activation<tanh>, bidirectional = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, reverse = false, training = false}> : (tensor<5x2x4xf32>, tensor<3x3xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<5x2x3xf32>
  %0 = "mps.single_gate_rnn"(%x, %r, %w, %b, %h0, %mask) <{activation = #mps.rnn_activation<tanh>, bidirectional = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>, reverse = false, training = false}> : (tensor<5x2x4xf32>, tensor<3x3xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<5x2x3xf32>
  // CHECK: %{{.+}} = "mps.single_gate_rnn"(%{{.+}}, %{{.+}}) <{activation = #mps.rnn_activation<relu>, bidirectional = false, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>, reverse = true, training = true}> : (tensor<5x2x3xf32>, tensor<3x3xf32>) -> (tensor<5x2x3xf32>, tensor<5x2x3xf32>)
  %1:2 = "mps.single_gate_rnn"(%xh, %r) <{activation = #mps.rnn_activation<relu>, bidirectional = false, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>, reverse = true, training = true}> : (tensor<5x2x3xf32>, tensor<3x3xf32>) -> (tensor<5x2x3xf32>, tensor<5x2x3xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @single_gate_rnn_gradients(
func.func @single_gate_rnn_gradients(%x: tensor<5x2x4xf32>, %r: tensor<3x3xf32>, %xh: tensor<5x2x3xf32>, %w: tensor<3x4xf32>, %b: tensor<3xf32>, %h0: tensor<2x3xf32>) {
  // CHECK: %{{.+}} = "mps.single_gate_rnn_gradients"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{activation = #mps.rnn_activation<tanh>, bidirectional = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 1, 1, 0>, reverse = false, training = true}> : (tensor<5x2x4xf32>, tensor<3x3xf32>, tensor<5x2x3xf32>, tensor<5x2x3xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<2x3xf32>) -> (tensor<5x2x4xf32>, tensor<3x3xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<2x3xf32>)
  %0:5 = "mps.single_gate_rnn_gradients"(%x, %r, %xh, %xh, %w, %b, %h0) <{activation = #mps.rnn_activation<tanh>, bidirectional = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 1, 1, 0>, reverse = false, training = true}> : (tensor<5x2x4xf32>, tensor<3x3xf32>, tensor<5x2x3xf32>, tensor<5x2x3xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<2x3xf32>) -> (tensor<5x2x4xf32>, tensor<3x3xf32>, tensor<3x4xf32>, tensor<3xf32>, tensor<2x3xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @lstm(
func.func @lstm(%x: tensor<5x2x4xf32>, %r: tensor<12x3xf32>, %w: tensor<12x4xf32>, %b: tensor<12xf32>, %h0: tensor<2x3xf32>, %c0: tensor<2x3xf32>, %p: tensor<12xf32>) {
  // CHECK: %{{.+}} = "mps.lstm"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{activation = #mps.rnn_activation<tanh>, bidirectional = false, cell_gate_activation = #mps.rnn_activation<tanh>, forget_gate_activation = #mps.rnn_activation<sigmoid>, forget_gate_last = false, input_gate_activation = #mps.rnn_activation<sigmoid>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 1>, output_gate_activation = #mps.rnn_activation<hardSigmoid>, produce_cell = true, reverse = false, training = true}> : (tensor<5x2x4xf32>, tensor<12x3xf32>, tensor<12x4xf32>, tensor<12xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<12xf32>) -> (tensor<5x2x3xf32>, tensor<5x2x3xf32>, tensor<5x2x12xf32>)
  %0:3 = "mps.lstm"(%x, %r, %w, %b, %h0, %c0, %p) <{activation = #mps.rnn_activation<tanh>, bidirectional = false, cell_gate_activation = #mps.rnn_activation<tanh>, forget_gate_activation = #mps.rnn_activation<sigmoid>, forget_gate_last = false, input_gate_activation = #mps.rnn_activation<sigmoid>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 1>, output_gate_activation = #mps.rnn_activation<hardSigmoid>, produce_cell = true, reverse = false, training = true}> : (tensor<5x2x4xf32>, tensor<12x3xf32>, tensor<12x4xf32>, tensor<12xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<12xf32>) -> (tensor<5x2x3xf32>, tensor<5x2x3xf32>, tensor<5x2x12xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @lstm_gradients(
func.func @lstm_gradients(%x: tensor<5x2x4xf32>, %r: tensor<12x3xf32>, %h: tensor<5x2x3xf32>, %z: tensor<5x2x12xf32>, %w: tensor<12x4xf32>, %b: tensor<12xf32>) {
  // CHECK: %{{.+}} = "mps.lstm_gradients"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{activation = #mps.rnn_activation<tanh>, bidirectional = false, cell_gate_activation = #mps.rnn_activation<tanh>, forget_gate_activation = #mps.rnn_activation<sigmoid>, forget_gate_last = false, input_gate_activation = #mps.rnn_activation<sigmoid>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0>, output_gate_activation = #mps.rnn_activation<sigmoid>, produce_cell = true, reverse = false, training = true}> : (tensor<5x2x4xf32>, tensor<12x3xf32>, tensor<5x2x3xf32>, tensor<5x2x12xf32>, tensor<5x2x3xf32>, tensor<12x4xf32>, tensor<12xf32>) -> (tensor<5x2x4xf32>, tensor<12x3xf32>, tensor<12x4xf32>, tensor<12xf32>)
  %0:4 = "mps.lstm_gradients"(%x, %r, %h, %z, %h, %w, %b) <{activation = #mps.rnn_activation<tanh>, bidirectional = false, cell_gate_activation = #mps.rnn_activation<tanh>, forget_gate_activation = #mps.rnn_activation<sigmoid>, forget_gate_last = false, input_gate_activation = #mps.rnn_activation<sigmoid>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0>, output_gate_activation = #mps.rnn_activation<sigmoid>, produce_cell = true, reverse = false, training = true}> : (tensor<5x2x4xf32>, tensor<12x3xf32>, tensor<5x2x3xf32>, tensor<5x2x12xf32>, tensor<5x2x3xf32>, tensor<12x4xf32>, tensor<12xf32>) -> (tensor<5x2x4xf32>, tensor<12x3xf32>, tensor<12x4xf32>, tensor<12xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @gru(
func.func @gru(%x: tensor<5x2x4xf32>, %r: tensor<9x3xf32>, %w: tensor<9x4xf32>, %b: tensor<9xf32>, %h0: tensor<2x3xf32>, %b2: tensor<3xf32>) {
  // CHECK: %{{.+}} = "mps.gru"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{bidirectional = false, flip_z = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, output_gate_activation = #mps.rnn_activation<tanh>, reset_after = true, reset_gate_activation = #mps.rnn_activation<sigmoid>, reset_gate_first = false, reverse = false, training = true, update_gate_activation = #mps.rnn_activation<sigmoid>}> : (tensor<5x2x4xf32>, tensor<9x3xf32>, tensor<9x4xf32>, tensor<9xf32>, tensor<2x3xf32>, tensor<3xf32>) -> (tensor<5x2x3xf32>, tensor<5x2x9xf32>)
  %0:2 = "mps.gru"(%x, %r, %w, %b, %h0, %b2) <{bidirectional = false, flip_z = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1>, output_gate_activation = #mps.rnn_activation<tanh>, reset_after = true, reset_gate_activation = #mps.rnn_activation<sigmoid>, reset_gate_first = false, reverse = false, training = true, update_gate_activation = #mps.rnn_activation<sigmoid>}> : (tensor<5x2x4xf32>, tensor<9x3xf32>, tensor<9x4xf32>, tensor<9xf32>, tensor<2x3xf32>, tensor<3xf32>) -> (tensor<5x2x3xf32>, tensor<5x2x9xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @gru_gradients(
func.func @gru_gradients(%x: tensor<5x2x4xf32>, %r: tensor<9x3xf32>, %h: tensor<5x2x3xf32>, %z: tensor<5x2x9xf32>, %w: tensor<9x4xf32>, %b: tensor<9xf32>) {
  // CHECK: %{{.+}} = "mps.gru_gradients"(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) <{bidirectional = false, flip_z = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0>, output_gate_activation = #mps.rnn_activation<tanh>, reset_after = false, reset_gate_activation = #mps.rnn_activation<sigmoid>, reset_gate_first = true, reverse = false, training = true, update_gate_activation = #mps.rnn_activation<sigmoid>}> : (tensor<5x2x4xf32>, tensor<9x3xf32>, tensor<5x2x3xf32>, tensor<5x2x9xf32>, tensor<5x2x3xf32>, tensor<9x4xf32>, tensor<9xf32>) -> (tensor<5x2x4xf32>, tensor<9x3xf32>, tensor<9x4xf32>, tensor<9xf32>)
  %0:4 = "mps.gru_gradients"(%x, %r, %h, %z, %h, %w, %b) <{bidirectional = false, flip_z = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0>, output_gate_activation = #mps.rnn_activation<tanh>, reset_after = false, reset_gate_activation = #mps.rnn_activation<sigmoid>, reset_gate_first = true, reverse = false, training = true, update_gate_activation = #mps.rnn_activation<sigmoid>}> : (tensor<5x2x4xf32>, tensor<9x3xf32>, tensor<5x2x3xf32>, tensor<5x2x9xf32>, tensor<5x2x3xf32>, tensor<9x4xf32>, tensor<9xf32>) -> (tensor<5x2x4xf32>, tensor<9x3xf32>, tensor<9x4xf32>, tensor<9xf32>)
  return
}
