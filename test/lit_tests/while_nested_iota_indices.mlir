// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=greedy_while_loop_batch_fission" --transform-interpreter --enzyme-hlo-remove-transform --enzyme-batch --enzyme-batch-to-stablehlo --canonicalize --verify-each | FileCheck %s

// Each output is x[p] * sum(i * j * weights[i + 2*(j-1)]), with i in [1,2]
// and j in [1,2,3] using Julia's one-based column-major indexing. The row
// and column lookups are inside loops nested beneath the p loop. They are
// not affine functions of p and must not be assigned zero-scale mappings.
// A default APInt offset is one bit wide: an invented iota start of 1 then
// becomes -1, collapsing the row index and producing gather[-2,0,2].
// CHECK-LABEL: func.func @main
// CHECK-NOT: stablehlo.constant dense<-1> : tensor<5xi64>
// CHECK: {{^    return }}
module {
  func.func @main(%weights: tensor<6xf64>, %xs: tensor<5xf64>) -> (tensor<5xf64>, tensor<6xf64>, tensor<5xf64>) {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %two = stablehlo.constant dense<2> : tensor<i64>
    %three = stablehlo.constant dense<3> : tensor<i64>
    %five = stablehlo.constant dense<5> : tensor<i64>
    %rows = stablehlo.constant dense<[1, 2]> : tensor<2xi64>
    %columns = stablehlo.constant dense<[1, 2, 3]> : tensor<3xi64>
    %fzero = stablehlo.constant dense<0.0> : tensor<f64>
    %zeros = stablehlo.constant dense<0.0> : tensor<5xf64>
    %outer:2 = stablehlo.while(%p = %zero, %out = %zeros) : tensor<i64>, tensor<5xf64>
    cond {
      %continue = stablehlo.compare LT, %p, %five : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %continue : tensor<i1>
    } do {
      %nextp = stablehlo.add %p, %one : tensor<i64>
      %xv = stablehlo.dynamic_slice %xs, %p, sizes = [1] : (tensor<5xf64>, tensor<i64>) -> tensor<1xf64>
      %x = stablehlo.reshape %xv : (tensor<1xf64>) -> tensor<f64>
      %middle:2 = stablehlo.while(%i = %zero, %total = %fzero) : tensor<i64>, tensor<f64>
      cond {
        %continue = stablehlo.compare LT, %i, %two : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %continue : tensor<i1>
      } do {
        %nexti = stablehlo.add %i, %one : tensor<i64>
        %rowv = stablehlo.dynamic_slice %rows, %i, sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
        %row = stablehlo.reshape %rowv : (tensor<1xi64>) -> tensor<i64>
        %inner:2 = stablehlo.while(%j = %zero, %partial = %fzero) : tensor<i64>, tensor<f64>
        cond {
          %continue = stablehlo.compare LT, %j, %three : (tensor<i64>, tensor<i64>) -> tensor<i1>
          stablehlo.return %continue : tensor<i1>
        } do {
          %nextj = stablehlo.add %j, %one : tensor<i64>
          %columnv = stablehlo.dynamic_slice %columns, %j, sizes = [1] : (tensor<3xi64>, tensor<i64>) -> tensor<1xi64>
          %column = stablehlo.reshape %columnv : (tensor<1xi64>) -> tensor<i64>
          %col0 = stablehlo.subtract %column, %one : tensor<i64>
          %offset = stablehlo.multiply %col0, %two : tensor<i64>
          %index1 = stablehlo.add %row, %offset : tensor<i64>
          %index = stablehlo.subtract %index1, %one : tensor<i64>
          %wv = stablehlo.dynamic_slice %weights, %index, sizes = [1] : (tensor<6xf64>, tensor<i64>) -> tensor<1xf64>
          %w = stablehlo.reshape %wv : (tensor<1xf64>) -> tensor<f64>
          %colf = stablehlo.convert %column : (tensor<i64>) -> tensor<f64>
          %term = stablehlo.multiply %colf, %w : tensor<f64>
          %sum = stablehlo.add %partial, %term : tensor<f64>
          stablehlo.return %nextj, %sum : tensor<i64>, tensor<f64>
        }
        %rowf = stablehlo.convert %row : (tensor<i64>) -> tensor<f64>
        %term = stablehlo.multiply %rowf, %inner#1 : tensor<f64>
        %sum = stablehlo.add %total, %term : tensor<f64>
        stablehlo.return %nexti, %sum : tensor<i64>, tensor<f64>
      }
      %value = stablehlo.multiply %x, %middle#1 : tensor<f64>
      %v = stablehlo.broadcast_in_dim %value, dims = [] : (tensor<f64>) -> tensor<1xf64>
      %updated = stablehlo.dynamic_update_slice %out, %v, %p : (tensor<5xf64>, tensor<1xf64>, tensor<i64>) -> tensor<5xf64>
      stablehlo.return %nextp, %updated : tensor<i64>, tensor<5xf64>
    }
    return %outer#1, %weights, %xs : tensor<5xf64>, tensor<6xf64>, tensor<5xf64>
  }


// A slice with a known affine start still participates in batching.
// CHECK-LABEL: func.func @known_affine
// CHECK: stablehlo.iota dim = 0 : tensor<5xf64>
// CHECK: {{^    return }}
  func.func @known_affine(%weights: tensor<5xf64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %five = stablehlo.constant dense<5> : tensor<i64>
    %indices = stablehlo.constant dense<[1, 2, 3, 4, 5]> : tensor<5xi64>
    %fzero = stablehlo.constant dense<0.0> : tensor<f64>
    %loop:2 = stablehlo.while(%i = %zero, %total = %fzero) : tensor<i64>, tensor<f64>
    cond {
      %continue = stablehlo.compare LT, %i, %five : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %continue : tensor<i1>
    } do {
      %next = stablehlo.add %i, %one : tensor<i64>
      %iv = stablehlo.dynamic_slice %indices, %i, sizes = [1] : (tensor<5xi64>, tensor<i64>) -> tensor<1xi64>
      %index = stablehlo.reshape %iv : (tensor<1xi64>) -> tensor<i64>
      %findex = stablehlo.convert %index : (tensor<i64>) -> tensor<f64>
      %wv = stablehlo.dynamic_slice %weights, %i, sizes = [1] : (tensor<5xf64>, tensor<i64>) -> tensor<1xf64>
      %w = stablehlo.reshape %wv : (tensor<1xf64>) -> tensor<f64>
      %value = stablehlo.multiply %findex, %w : tensor<f64>
      %sum = stablehlo.add %total, %value : tensor<f64>
      stablehlo.return %next, %sum : tensor<i64>, tensor<f64>
    }
    return %loop#1 : tensor<f64>
  }
}
