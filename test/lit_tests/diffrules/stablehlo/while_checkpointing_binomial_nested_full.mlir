// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<i64>) -> tensor<f32> {
    %c0_i64 = stablehlo.constant dense<0> : tensor<i64>
    %c1_i64 = stablehlo.constant dense<1> : tensor<i64>

    %outer:2 = stablehlo.while(%iv0 = %c0_i64, %carried = %arg0) : tensor<i64>, tensor<f32>
     attributes {
      enzyme.enable_checkpointing = true,
      enzyme.checkpoint_period = 4,
      enzyme.binomial_checkpointing
    } cond {
      %cond = stablehlo.compare LT, %iv0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %inner:2 = stablehlo.while(%iv1 = %c0_i64, %carried2 = %carried) : tensor<i64>, tensor<f32>
       cond {
        %cond2 = stablehlo.compare LT, %iv1, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %cond2 : tensor<i1>
      } do {
        %0 = stablehlo.multiply %arg0, %carried2 : tensor<f32>
        %iv1next = stablehlo.add %iv1, %c1_i64 : tensor<i64>
        stablehlo.return %iv1next, %0 : tensor<i64>, tensor<f32>
      }
      %iv0next = stablehlo.add %iv0, %c1_i64 : tensor<i64>
      stablehlo.return %iv0next, %inner#1 : tensor<i64>, tensor<f32>
    } 
    return %outer#1 : tensor<f32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<f32>, %arg1: tensor<i64>, %arg2: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<0xf32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.minimum %c_3, %arg1 : tensor<i64>
// CHECK-NEXT:    %1:5 = stablehlo.while(%iterArg = %c_5, %iterArg_6 = %c_5, %iterArg_7 = %arg0, %iterArg_8 = %cst_2, %iterArg_9 = %c_1) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %4 = stablehlo.compare LT, %iterArg, %0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4 = stablehlo.reshape %iterArg_7 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:      %5 = stablehlo.dynamic_update_slice %iterArg_8, %4, %iterArg : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
// CHECK-NEXT:      %6 = stablehlo.reshape %iterArg_6 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %7 = stablehlo.dynamic_update_slice %iterArg_9, %6, %iterArg : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:      %8 = stablehlo.subtract %arg1, %iterArg_6 : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.subtract %0, %iterArg : tensor<i64>
// CHECK-NEXT:      %10 = stablehlo.minimum %9, %8 : tensor<i64>
// CHECK-NEXT:      %11 = enzyme.binomial_progress %8, %10 : tensor<i64>
// CHECK-NEXT:      %12:2 = stablehlo.while(%iterArg_10 = %c_5, %iterArg_11 = %iterArg_7) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %15 = stablehlo.compare LT, %iterArg_10, %11 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %15 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %15:2 = stablehlo.while(%iterArg_12 = %c_5, %iterArg_13 = %iterArg_11) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %17 = stablehlo.compare LT, %iterArg_12, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %17 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %17 = stablehlo.multiply %arg0, %iterArg_13 : tensor<f32>
// CHECK-NEXT:          %18 = stablehlo.add %iterArg_12, %c_4 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %18, %17 : tensor<i64>, tensor<f32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %16 = stablehlo.add %iterArg_10, %c_4 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %16, %15#1 : tensor<i64>, tensor<f32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %13 = stablehlo.add %iterArg_6, %11 : tensor<i64>
// CHECK-NEXT:      %14 = stablehlo.add %iterArg, %c_4 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %14, %13, %12#1, %5, %7 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2:6 = stablehlo.while(%iterArg = %c_5, %iterArg_6 = %0, %iterArg_7 = %arg2, %iterArg_8 = %1#3, %iterArg_9 = %1#4, %iterArg_10 = %cst) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>, tensor<f32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %4 = stablehlo.compare LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4 = stablehlo.subtract %iterArg_6, %c_4 : tensor<i64>
// CHECK-NEXT:      %5 = stablehlo.subtract %arg1, %iterArg : tensor<i64>
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %iterArg_8, %4, sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:      %8 = stablehlo.dynamic_slice %iterArg_9, %4, sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %10:5 = stablehlo.while(%iterArg_11 = %9, %iterArg_12 = %4, %iterArg_13 = %7, %iterArg_14 = %iterArg_8, %iterArg_15 = %iterArg_9) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %17 = stablehlo.add %iterArg_11, %c_4 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.compare LT, %17, %5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %18 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %17 = stablehlo.subtract %5, %iterArg_11 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.subtract %0, %iterArg_12 : tensor<i64>
// CHECK-NEXT:        %19 = stablehlo.minimum %18, %17 : tensor<i64>
// CHECK-NEXT:        %20 = enzyme.binomial_progress %17, %19 : tensor<i64>
// CHECK-NEXT:        %21 = stablehlo.reshape %iterArg_13 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:        %22 = stablehlo.dynamic_update_slice %iterArg_14, %21, %iterArg_12 : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
// CHECK-NEXT:        %23 = stablehlo.reshape %iterArg_11 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %24 = stablehlo.dynamic_update_slice %iterArg_15, %23, %iterArg_12 : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:        %25 = stablehlo.add %iterArg_11, %20 : tensor<i64>
// CHECK-NEXT:        %26 = stablehlo.compare EQ, %25, %5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %27 = stablehlo.subtract %25, %c_4 : tensor<i64>
// CHECK-NEXT:        %28 = stablehlo.select %26, %27, %25 : tensor<i1>, tensor<i64>
// CHECK-NEXT:        %29:2 = stablehlo.while(%iterArg_16 = %iterArg_11, %iterArg_17 = %iterArg_13) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %31 = stablehlo.compare LT, %iterArg_16, %28 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %31 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %31:2 = stablehlo.while(%iterArg_18 = %c_5, %iterArg_19 = %iterArg_17) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:          cond {
// CHECK-NEXT:            %33 = stablehlo.compare LT, %iterArg_18, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:            stablehlo.return %33 : tensor<i1>
// CHECK-NEXT:          } do {
// CHECK-NEXT:            %33 = stablehlo.multiply %arg0, %iterArg_19 : tensor<f32>
// CHECK-NEXT:            %34 = stablehlo.add %iterArg_18, %c_4 : tensor<i64>
// CHECK-NEXT:            stablehlo.return %34, %33 : tensor<i64>, tensor<f32>
// CHECK-NEXT:          }
// CHECK-NEXT:          %32 = stablehlo.add %iterArg_16, %c_4 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %32, %31#1 : tensor<i64>, tensor<f32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %30 = stablehlo.add %iterArg_12, %c_4 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %25, %30, %29#1, %22, %24 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %11 = stablehlo.reshape %arg1 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %12 = stablehlo.dynamic_pad %cst_0, %cst, %c, %11, %c : (tensor<0xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
// CHECK-NEXT:      %13:3 = stablehlo.while(%iterArg_11 = %c_5, %iterArg_12 = %10#2, %iterArg_13 = %12) : tensor<i64>, tensor<f32>, tensor<?xf32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %17 = stablehlo.compare LT, %iterArg_11, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %17 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %17 = stablehlo.reshape %iterArg_12 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:        %18 = stablehlo.dynamic_update_slice %iterArg_13, %17, %iterArg_11 : (tensor<?xf32>, tensor<1xf32>, tensor<i64>) -> tensor<?xf32>
// CHECK-NEXT:        %19 = stablehlo.multiply %arg0, %iterArg_12 : tensor<f32>
// CHECK-NEXT:        %20 = stablehlo.add %iterArg_11, %c_4 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %20, %19, %18 : tensor<i64>, tensor<f32>, tensor<?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %14 = stablehlo.subtract %arg1, %c_4 : tensor<i64>
// CHECK-NEXT:      %15:3 = stablehlo.while(%iterArg_11 = %c_5, %iterArg_12 = %iterArg_7, %iterArg_13 = %iterArg_10) : tensor<i64>, tensor<f32>, tensor<f32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %17 = stablehlo.compare LT, %iterArg_11, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %17 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %17 = stablehlo.subtract %14, %iterArg_11 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.dynamic_slice %13#2, %17, sizes = [1] : (tensor<?xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:        %19 = stablehlo.reshape %18 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:        %20 = stablehlo.add %iterArg_11, %c_4 : tensor<i64>
// CHECK-NEXT:        %21 = stablehlo.multiply %iterArg_12, %19 : tensor<f32>
// CHECK-NEXT:        %22 = stablehlo.add %iterArg_13, %21 : tensor<f32>
// CHECK-NEXT:        %23 = stablehlo.multiply %iterArg_12, %arg0 : tensor<f32>
// CHECK-NEXT:        stablehlo.return %20, %23, %22 : tensor<i64>, tensor<f32>, tensor<f32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %16 = stablehlo.add %iterArg, %c_4 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %16, %10#1, %15#1, %10#3, %10#4, %15#2 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>, tensor<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %3 = stablehlo.add %2#5, %2#2 : tensor<f32>
// CHECK-NEXT:    return %3 : tensor<f32>
// CHECK-NEXT:  }
