// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=while_split_concat_carry" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Two columns kept in one buffer, both rewritten every iteration and each
// read as its own column: the loop carries the two columns.
// CHECK-LABEL: func.func @columns
// CHECK-SAME:    (%[[X:.+]]: tensor<1000x1xf64>)
// CHECK-DAG:     %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<1000x2xf64>
// CHECK-DAG:     %[[HALF:.+]] = stablehlo.constant dense<5.000000e-01> : tensor<1000x1xf64>
// CHECK-DAG:     %[[I0:.+]] = stablehlo.slice %[[ZERO]] [0:1000, 0:1]
// CHECK-DAG:     %[[I1:.+]] = stablehlo.slice %[[ZERO]] [0:1000, 1:2]
// CHECK:         %[[W:.+]]:3 = stablehlo.while(%[[I:.+]] = %{{.+}}, %[[A:.+]] = %[[I0]], %[[B:.+]] = %[[I1]]) : tensor<i64>, tensor<1000x1xf64>, tensor<1000x1xf64>
// CHECK:         } do {
// CHECK-NOT:       stablehlo.slice
// CHECK:           %[[SUM:.+]] = stablehlo.add %[[X]], %[[A]] : tensor<1000x1xf64>
// CHECK-NEXT:      %[[PROD:.+]] = stablehlo.multiply %[[HALF]], %[[SUM]] : tensor<1000x1xf64>
// CHECK-NEXT:      stablehlo.return %{{.+}}, %[[PROD]], %[[SUM]] : tensor<i64>, tensor<1000x1xf64>, tensor<1000x1xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[W]]#1 : tensor<1000x1xf64>
func.func @columns(%x: tensor<1000x1xf64>) -> tensor<1000x1xf64> {
  %zero = stablehlo.constant dense<0.0> : tensor<1000x2xf64>
  %half = stablehlo.constant dense<0.5> : tensor<1000x1xf64>
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c100 = stablehlo.constant dense<100> : tensor<i64>
  %w:2 = stablehlo.while(%i = %c0, %S = %zero) : tensor<i64>, tensor<1000x2xf64>
  cond {
    %lt = stablehlo.compare LT, %i, %c100 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %n = stablehlo.add %i, %c1 : tensor<i64>
    %col0 = stablehlo.slice %S [0:1000, 0:1] : (tensor<1000x2xf64>) -> tensor<1000x1xf64>
    %sum = stablehlo.add %x, %col0 : tensor<1000x1xf64>
    %prod = stablehlo.multiply %half, %sum : tensor<1000x1xf64>
    %next = stablehlo.concatenate %prod, %sum, dim = 1 : (tensor<1000x1xf64>, tensor<1000x1xf64>) -> tensor<1000x2xf64>
    stablehlo.return %n, %next : tensor<i64>, tensor<1000x2xf64>
  }
  %r = stablehlo.slice %w#1 [0:1000, 0:1] : (tensor<1000x2xf64>) -> tensor<1000x1xf64>
  return %r : tensor<1000x1xf64>
}

// Pieces of unequal size, read in part and with an offset, by the condition
// too, with another carried value on either side.
// CHECK-LABEL: func.func @offsets
// CHECK-SAME:    (%[[X:.+]]: tensor<8x4xf32>, %[[INIT:.+]]: tensor<8x7xf32>, %[[LIM:.+]]: tensor<2x3xf32>)
// CHECK-DAG:     %[[I0:.+]] = stablehlo.slice %[[INIT]] [0:8, 0:4] : (tensor<8x7xf32>) -> tensor<8x4xf32>
// CHECK-DAG:     %[[I1:.+]] = stablehlo.slice %[[INIT]] [0:8, 4:7] : (tensor<8x7xf32>) -> tensor<8x3xf32>
// CHECK:         %[[W:.+]]:4 = stablehlo.while(%[[I:.+]] = %{{.+}}, %[[A:.+]] = %[[I0]], %[[B:.+]] = %[[I1]], %[[J:.+]] = %{{.+}}) : tensor<i64>, tensor<8x4xf32>, tensor<8x3xf32>, tensor<i64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %[[PROBE:.+]] = stablehlo.slice %[[B]] [0:2, 0:3] : (tensor<8x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:      stablehlo.compare LT, %[[PROBE]], %[[LIM]]
// CHECK:         } do {
// CHECK-NEXT:      %[[N:.+]] = stablehlo.add %[[I]], %{{.+}} : tensor<i64>
// CHECK-NEXT:      %[[M:.+]] = stablehlo.add %[[J]], %[[N]] : tensor<i64>
// CHECK-NEXT:      %[[BB:.+]] = stablehlo.slice %[[B]] [0:8, 1:3] : (tensor<8x3xf32>) -> tensor<8x2xf32>
// CHECK-NEXT:      %[[A2:.+]] = stablehlo.add %[[A]], %[[X]] : tensor<8x4xf32>
// CHECK-NEXT:      %[[B2:.+]] = stablehlo.multiply %[[BB]], %[[BB]] : tensor<8x2xf32>
// CHECK-NEXT:      %[[PAD:.+]] = stablehlo.pad %[[B2]]
// CHECK-NEXT:      stablehlo.return %[[N]], %[[A2]], %[[PAD]], %[[M]] : tensor<i64>, tensor<8x4xf32>, tensor<8x3xf32>, tensor<i64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[R:.+]] = stablehlo.slice %[[W]]#2 [3:5, 0:3] : (tensor<8x3xf32>) -> tensor<2x3xf32>
// CHECK-NEXT:    return %[[W]]#0, %[[R]], %[[W]]#3
func.func @offsets(%x: tensor<8x4xf32>, %init: tensor<8x7xf32>, %lim: tensor<2x3xf32>) -> (tensor<i64>, tensor<2x3xf32>, tensor<i64>) {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c100 = stablehlo.constant dense<100> : tensor<i64>
  %c0f = stablehlo.constant dense<0.0> : tensor<f32>
  %w:3 = stablehlo.while(%i = %c0, %S = %init, %j = %c0) : tensor<i64>, tensor<8x7xf32>, tensor<i64>
  cond {
    %probe = stablehlo.slice %S [0:2, 4:7] : (tensor<8x7xf32>) -> tensor<2x3xf32>
    %cmp = stablehlo.compare LT, %probe, %lim : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
    %true = stablehlo.constant dense<true> : tensor<i1>
    %all = stablehlo.reduce(%cmp init: %true) applies stablehlo.and across dimensions = [0, 1] : (tensor<2x3xi1>, tensor<i1>) -> tensor<i1>
    stablehlo.return %all : tensor<i1>
  } do {
    %n = stablehlo.add %i, %c1 : tensor<i64>
    %m = stablehlo.add %j, %n : tensor<i64>
    %a = stablehlo.slice %S [0:8, 0:4] : (tensor<8x7xf32>) -> tensor<8x4xf32>
    %b = stablehlo.slice %S [0:8, 5:7] : (tensor<8x7xf32>) -> tensor<8x2xf32>
    %a2 = stablehlo.add %a, %x : tensor<8x4xf32>
    %b2 = stablehlo.multiply %b, %b : tensor<8x2xf32>
    %pad = stablehlo.pad %b2, %c0f, low = [0, 1], high = [0, 0], interior = [0, 0] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8x3xf32>
    %next = stablehlo.concatenate %a2, %pad, dim = 1 : (tensor<8x4xf32>, tensor<8x3xf32>) -> tensor<8x7xf32>
    stablehlo.return %n, %next, %m : tensor<i64>, tensor<8x7xf32>, tensor<i64>
  }
  %r = stablehlo.slice %w#1 [3:5, 4:7] : (tensor<8x7xf32>) -> tensor<2x3xf32>
  return %w#0, %r, %w#2 : tensor<i64>, tensor<2x3xf32>, tensor<i64>
}

// A slice straddles two pieces: unchanged.
// CHECK-LABEL: func.func @straddle
// CHECK:         stablehlo.while(%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) : tensor<i64>, tensor<1000x2xf64>
// CHECK:         stablehlo.concatenate
func.func @straddle(%x: tensor<1000x1xf64>) -> tensor<1000x2xf64> {
  %zero = stablehlo.constant dense<0.0> : tensor<1000x2xf64>
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c100 = stablehlo.constant dense<100> : tensor<i64>
  %w:2 = stablehlo.while(%i = %c0, %S = %zero) : tensor<i64>, tensor<1000x2xf64>
  cond {
    %lt = stablehlo.compare LT, %i, %c100 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %n = stablehlo.add %i, %c1 : tensor<i64>
    %col0 = stablehlo.slice %S [0:1000, 0:1] : (tensor<1000x2xf64>) -> tensor<1000x1xf64>
    %sum = stablehlo.add %x, %col0 : tensor<1000x1xf64>
    %next = stablehlo.concatenate %sum, %col0, dim = 1 : (tensor<1000x1xf64>, tensor<1000x1xf64>) -> tensor<1000x2xf64>
    stablehlo.return %n, %next : tensor<i64>, tensor<1000x2xf64>
  }
  %r = stablehlo.slice %w#1 [0:1000, 0:2] : (tensor<1000x2xf64>) -> tensor<1000x2xf64>
  return %r : tensor<1000x2xf64>
}

// The body reads the carried value whole: unchanged.
// CHECK-LABEL: func.func @whole
// CHECK:         stablehlo.while(%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) : tensor<i64>, tensor<1000x2xf64>
// CHECK:         stablehlo.concatenate
func.func @whole(%x: tensor<1000x2xf64>) -> tensor<1000x1xf64> {
  %zero = stablehlo.constant dense<0.0> : tensor<1000x2xf64>
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c100 = stablehlo.constant dense<100> : tensor<i64>
  %w:2 = stablehlo.while(%i = %c0, %S = %zero) : tensor<i64>, tensor<1000x2xf64>
  cond {
    %lt = stablehlo.compare LT, %i, %c100 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %lt : tensor<i1>
  } do {
    %n = stablehlo.add %i, %c1 : tensor<i64>
    %sum = stablehlo.add %x, %S : tensor<1000x2xf64>
    %col0 = stablehlo.slice %sum [0:1000, 0:1] : (tensor<1000x2xf64>) -> tensor<1000x1xf64>
    %col1 = stablehlo.slice %sum [0:1000, 1:2] : (tensor<1000x2xf64>) -> tensor<1000x1xf64>
    %next = stablehlo.concatenate %col1, %col0, dim = 1 : (tensor<1000x1xf64>, tensor<1000x1xf64>) -> tensor<1000x2xf64>
    stablehlo.return %n, %next : tensor<i64>, tensor<1000x2xf64>
  }
  %r = stablehlo.slice %w#1 [0:1000, 0:1] : (tensor<1000x2xf64>) -> tensor<1000x1xf64>
  return %r : tensor<1000x1xf64>
}
