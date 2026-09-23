// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=greedy_while_loop_batch_fission;merge_consecutive_reshapes;reshape_op_canon;while_elementwise_reduction_to_reduce;while_deadresult;while_is_copy_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Each load has two consumers that never re-join; the loop must fully batch.

func.func @gain_pairs(%x: tensor<4428xf64>) -> (tensor<f64>, tensor<101xf64>) {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
  %zeros = stablehlo.constant dense<0.000000e+00> : tensor<101xf64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c101 = stablehlo.constant dense<101> : tensor<i64>
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c4227 = stablehlo.constant dense<4227> : tensor<i64>
  %c2 = stablehlo.constant dense<2> : tensor<i64>
  %cst8 = stablehlo.constant dense<8.000000e+00> : tensor<f64>
  %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
  %mhalf = stablehlo.constant dense<-5.000000e-01> : tensor<f64>
  %r:3 = stablehlo.while(%i = %c0, %acc = %zero, %ph = %zeros) : tensor<i64>, tensor<f64>, tensor<101xf64>
  cond {
    %p = stablehlo.compare LT, %i, %c101 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %p : tensor<i1>
  } do {
    %0 = stablehlo.multiply %i, %c2 : tensor<i64>
    %1 = stablehlo.add %c4227, %0 : tensor<i64>
    %2 = stablehlo.add %c1, %i : tensor<i64>
    %3 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.subtract %3, %c : tensor<i32>
    %5 = stablehlo.dynamic_slice %x, %4, sizes = [1] : (tensor<4428xf64>, tensor<i32>) -> tensor<1xf64>
    %6 = stablehlo.add %1, %c1 : tensor<i64>
    %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
    %8 = stablehlo.subtract %7, %c : tensor<i32>
    %9 = stablehlo.dynamic_slice %x, %8, sizes = [1] : (tensor<4428xf64>, tensor<i32>) -> tensor<1xf64>
    %10 = stablehlo.multiply %5, %5 : tensor<1xf64>
    %11 = stablehlo.multiply %9, %9 : tensor<1xf64>
    %12 = stablehlo.add %10, %11 : tensor<1xf64>
    %13 = stablehlo.log %12 : tensor<1xf64>
    %14 = stablehlo.reshape %13 : (tensor<1xf64>) -> tensor<f64>
    %15 = stablehlo.multiply %14, %half : tensor<f64>
    %16 = stablehlo.multiply %15, %15 : tensor<f64>
    %17 = stablehlo.negate %16 : tensor<f64>
    %18 = stablehlo.multiply %17, %cst8 : tensor<f64>
    %19 = stablehlo.multiply %14, %mhalf : tensor<f64>
    %20 = stablehlo.add %18, %19 : tensor<f64>
    %21 = stablehlo.atan2 %5, %9 : tensor<1xf64>
    %22 = stablehlo.add %acc, %20 : tensor<f64>
    %23 = stablehlo.convert %2 : (tensor<i64>) -> tensor<i32>
    %24 = stablehlo.subtract %23, %c : tensor<i32>
    %25 = stablehlo.dynamic_update_slice %ph, %21, %24 : (tensor<101xf64>, tensor<1xf64>, tensor<i32>) -> tensor<101xf64>
    stablehlo.return %2, %22, %25 : tensor<i64>, tensor<f64>, tensor<101xf64>
  }
  return %r#1, %r#2 : tensor<f64>, tensor<101xf64>
}

// CHECK-LABEL: func.func @gain_pairs
// CHECK-NOT:     stablehlo.while
// CHECK:         %[[RE0:.+]] = stablehlo.slice %arg0 [4226:4427] : (tensor<4428xf64>) -> tensor<201xf64>
// CHECK-NEXT:    %[[RE1:.+]] = stablehlo.slice %[[RE0]] [0:201:2] : (tensor<201xf64>) -> tensor<101xf64>
// CHECK-NEXT:    %[[RE:.+]] = stablehlo.reshape %[[RE1]] : (tensor<101xf64>) -> tensor<101x1xf64>
// CHECK-NEXT:    %[[IM0:.+]] = stablehlo.slice %arg0 [4227:4428] : (tensor<4428xf64>) -> tensor<201xf64>
// CHECK-NEXT:    %[[IM1:.+]] = stablehlo.slice %[[IM0]] [0:201:2] : (tensor<201xf64>) -> tensor<101xf64>
// CHECK-NEXT:    %[[IM:.+]] = stablehlo.reshape %[[IM1]] : (tensor<101xf64>) -> tensor<101x1xf64>
// CHECK-NEXT:    %[[PHASE:.+]] = stablehlo.atan2 %[[RE]], %[[IM]] : tensor<101x1xf64>
// CHECK:         %[[AMP2:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<101x1xf64>
// CHECK-NEXT:    %[[LOGAMP:.+]] = stablehlo.log %[[AMP2]] : tensor<101x1xf64>
// CHECK:         %[[TERMS:.+]] = stablehlo.add %{{.+}}, %{{.+}} : tensor<101xf64>
// CHECK-NEXT:    %[[SUM:.+]] = stablehlo.reduce(%[[TERMS]] init: %{{.+}}) applies stablehlo.add across dimensions = [0] : (tensor<101xf64>, tensor<f64>) -> tensor<f64>
// CHECK:         %[[PHT:.+]] = stablehlo.transpose %[[PHASE]], dims = [0, 1] : (tensor<101x1xf64>) -> tensor<101x1xf64>
// CHECK-NEXT:    %[[PHFLAT:.+]] = stablehlo.reshape %[[PHT]] : (tensor<101x1xf64>) -> tensor<101xf64>
// CHECK:         %[[OUT:.+]] = stablehlo.dynamic_update_slice %{{.+}}, %{{.+}}, %{{.+}} : (tensor<101xf64>, tensor<101xf64>, tensor<i32>) -> tensor<101xf64>
// CHECK-NOT:     stablehlo.while
