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
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.minimum %c_2, %arg1 : tensor<i64>
// CHECK-NEXT:    %1:5 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_4, %iterArg_6 = %arg0, %iterArg_7 = %cst_1, %iterArg_8 = %c_0) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %4 = stablehlo.compare LT, %iterArg, %0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4 = stablehlo.reshape %iterArg_6 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:      %5 = stablehlo.dynamic_update_slice %iterArg_7, %4, %iterArg : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
// CHECK-NEXT:      %6 = stablehlo.reshape %iterArg_5 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %7 = stablehlo.dynamic_update_slice %iterArg_8, %6, %iterArg : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:      %8 = stablehlo.subtract %arg1, %iterArg_5 : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.subtract %0, %iterArg : tensor<i64>
// CHECK-NEXT:      %10 = stablehlo.minimum %9, %8 : tensor<i64>
// CHECK-NEXT:      %11 = enzyme.binomial_progress %8, %10 : tensor<i64>
// CHECK-NEXT:      %12:2 = stablehlo.while(%iterArg_9 = %c_4, %iterArg_10 = %iterArg_6) : tensor<i64>, tensor<f32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %15 = stablehlo.compare LT, %iterArg_9, %11 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %15 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %15:2 = stablehlo.while(%iterArg_11 = %c_4, %iterArg_12 = %iterArg_10) : tensor<i64>, tensor<f32>
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %17 = stablehlo.compare LT, %iterArg_11, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %17 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %17 = stablehlo.multiply %arg0, %iterArg_12 : tensor<f32>
// CHECK-NEXT:          %18 = stablehlo.add %iterArg_11, %c_3 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %18, %17 : tensor<i64>, tensor<f32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %16 = stablehlo.add %iterArg_9, %c_3 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %16, %15#1 : tensor<i64>, tensor<f32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %13 = stablehlo.add %iterArg_5, %11 : tensor<i64>
// CHECK-NEXT:      %14 = stablehlo.add %iterArg, %c_3 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %14, %13, %12#1, %5, %7 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2:6 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %0, %iterArg_6 = %arg2, %iterArg_7 = %1#3, %iterArg_8 = %1#4, %iterArg_9 = %cst) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>, tensor<f32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %4 = stablehlo.compare LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %4 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %4 = stablehlo.subtract %iterArg_5, %c_3 : tensor<i64>
// CHECK-NEXT:      %5 = stablehlo.subtract %arg1, %iterArg : tensor<i64>
// CHECK-NEXT:      %6 = stablehlo.dynamic_slice %iterArg_7, %4, sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:      %7 = stablehlo.reshape %6 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:      %8 = stablehlo.dynamic_slice %iterArg_8, %4, sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %9 = stablehlo.reshape %8 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %10:5 = stablehlo.while(%iterArg_10 = %9, %iterArg_11 = %4, %iterArg_12 = %7, %iterArg_13 = %iterArg_7, %iterArg_14 = %iterArg_8) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %18 = stablehlo.add %iterArg_10, %c_3 : tensor<i64>
// CHECK-NEXT:        %19 = stablehlo.compare LT, %18, %5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %19 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %18 = stablehlo.subtract %5, %iterArg_10 : tensor<i64>
// CHECK-NEXT:        %19 = stablehlo.subtract %0, %iterArg_11 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.minimum %19, %18 : tensor<i64>
// CHECK-NEXT:        %21 = enzyme.binomial_progress %18, %20 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.reshape %iterArg_12 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:        %23 = stablehlo.dynamic_update_slice %iterArg_13, %22, %iterArg_11 : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
// CHECK-NEXT:        %24 = stablehlo.reshape %iterArg_10 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %25 = stablehlo.dynamic_update_slice %iterArg_14, %24, %iterArg_11 : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:        %26 = stablehlo.add %iterArg_10, %21 : tensor<i64>
// CHECK-NEXT:        %27 = stablehlo.compare EQ, %26, %5 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %28 = stablehlo.subtract %26, %c_3 : tensor<i64>
// CHECK-NEXT:        %29 = stablehlo.select %27, %28, %26 : tensor<i1>, tensor<i64>
// CHECK-NEXT:        %30:2 = stablehlo.while(%iterArg_15 = %iterArg_10, %iterArg_16 = %iterArg_12) : tensor<i64>, tensor<f32>
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %32 = stablehlo.compare LT, %iterArg_15, %29 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %32 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %32:2 = stablehlo.while(%iterArg_17 = %c_4, %iterArg_18 = %iterArg_16) : tensor<i64>, tensor<f32>
// CHECK-NEXT:          cond {
// CHECK-NEXT:            %34 = stablehlo.compare LT, %iterArg_17, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:            stablehlo.return %34 : tensor<i1>
// CHECK-NEXT:          } do {
// CHECK-NEXT:            %34 = stablehlo.multiply %arg0, %iterArg_18 : tensor<f32>
// CHECK-NEXT:            %35 = stablehlo.add %iterArg_17, %c_3 : tensor<i64>
// CHECK-NEXT:            stablehlo.return %35, %34 : tensor<i64>, tensor<f32>
// CHECK-NEXT:          }
// CHECK-NEXT:          %33 = stablehlo.add %iterArg_15, %c_3 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %33, %32#1 : tensor<i64>, tensor<f32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %31 = stablehlo.add %iterArg_11, %c_3 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %26, %31, %30#1, %23, %25 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %11 = stablehlo.reshape %arg1 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %12 = tensor.empty() : tensor<0xf32>
// CHECK-NEXT:      %13 = stablehlo.dynamic_pad %12, %cst, %c, %11, %c : (tensor<0xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
// CHECK-NEXT:      %14:3 = stablehlo.while(%iterArg_10 = %c_4, %iterArg_11 = %10#2, %iterArg_12 = %13) : tensor<i64>, tensor<f32>, tensor<?xf32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %18 = stablehlo.compare LT, %iterArg_10, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %18 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %18 = stablehlo.reshape %iterArg_11 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:        %19 = stablehlo.dynamic_update_slice %iterArg_12, %18, %iterArg_10 : (tensor<?xf32>, tensor<1xf32>, tensor<i64>) -> tensor<?xf32>
// CHECK-NEXT:        %20 = stablehlo.multiply %arg0, %iterArg_11 : tensor<f32>
// CHECK-NEXT:        %21 = stablehlo.add %iterArg_10, %c_3 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %21, %20, %19 : tensor<i64>, tensor<f32>, tensor<?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %15 = stablehlo.subtract %arg1, %c_3 : tensor<i64>
// CHECK-NEXT:      %16:3 = stablehlo.while(%iterArg_10 = %c_4, %iterArg_11 = %iterArg_6, %iterArg_12 = %iterArg_9) : tensor<i64>, tensor<f32>, tensor<f32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %18 = stablehlo.compare LT, %iterArg_10, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %18 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %18 = stablehlo.subtract %15, %iterArg_10 : tensor<i64>
// CHECK-NEXT:        %19 = stablehlo.dynamic_slice %14#2, %18, sizes = [1] : (tensor<?xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:        %20 = stablehlo.reshape %19 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:        %21 = stablehlo.add %iterArg_10, %c_3 : tensor<i64>
// CHECK-NEXT:        %22 = stablehlo.multiply %iterArg_11, %20 : tensor<f32>
// CHECK-NEXT:        %23 = stablehlo.add %iterArg_12, %22 : tensor<f32>
// CHECK-NEXT:        %24 = stablehlo.multiply %iterArg_11, %arg0 : tensor<f32>
// CHECK-NEXT:        stablehlo.return %21, %24, %23 : tensor<i64>, tensor<f32>, tensor<f32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %17 = stablehlo.add %iterArg, %c_3 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %17, %10#1, %16#1, %10#3, %10#4, %16#2 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>, tensor<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %3 = stablehlo.add %2#5, %2#2 : tensor<f32>
// CHECK-NEXT:    return %3 : tensor<f32>
// CHECK-NEXT:  }

