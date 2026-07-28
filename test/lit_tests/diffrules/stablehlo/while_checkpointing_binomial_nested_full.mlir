// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<i64>) -> tensor<f32> {
    %c0_i64 = stablehlo.constant dense<0> : tensor<i64>
    %c1_i64 = stablehlo.constant dense<1> : tensor<i64>

    %outer:2 = stablehlo.while(%iv0 = %c0_i64, %carried = %arg0) : tensor<i64>, tensor<f32>
     attributes {
      enzymexla.enable_checkpointing = true,
      enzymexla.checkpoint_period = 4,
      enzymexla.binomial_checkpointing
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
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0:5 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %arg0, %iterArg_6 = %cst_2, %iterArg_7 = %c_4, %iterArg_8 = %c_1) : tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<i64>, tensor<4xi64>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %3 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %3 = stablehlo.reshape %iterArg_5 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:      %4 = stablehlo.dynamic_update_slice %iterArg_6, %3, %iterArg : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
// CHECK-NEXT:      %5 = stablehlo.subtract %arg1, %iterArg_7 : tensor<i64>
// CHECK-NEXT:      %6 = stablehlo.subtract %c_0, %iterArg {enzymexla.bounds = {{.+}}} : tensor<i64>
// CHECK-NEXT:      %7 = enzymexla.math.binomial_progress(%5, %6) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:      %8 = stablehlo.add %iterArg_7, %7 : tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.reshape %iterArg_7 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %10 = stablehlo.dynamic_update_slice %iterArg_8, %9, %iterArg : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:      %11:2 = stablehlo.while(%iterArg_9 = %c_4, %iterArg_10 = %iterArg_5) : tensor<i64>, tensor<f32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %13 = stablehlo.compare LT, %iterArg_9, %7 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %13 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %13:2 = stablehlo.while(%iterArg_11 = %c_4, %iterArg_12 = %iterArg_10) : tensor<i64>, tensor<f32>
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %15 = stablehlo.compare LT, %iterArg_11, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %15 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %15 = stablehlo.multiply %arg0, %iterArg_12 : tensor<f32>
// CHECK-NEXT:          %16 = stablehlo.add %iterArg_11, %c_3 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %16, %15 : tensor<i64>, tensor<f32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %14 = stablehlo.add %iterArg_9, %c_3 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %14, %13#1 : tensor<i64>, tensor<f32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %12 = stablehlo.add %iterArg, %c_3 {enzymexla.bounds = {{.+}}} : tensor<i64>
// CHECK-NEXT:      stablehlo.return %12, %11#1, %4, %8, %10 : tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<i64>, tensor<4xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %1:6 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_0, %iterArg_6 = %0#4, %iterArg_7 = %0#2, %iterArg_8 = %arg2, %iterArg_9 = %cst) : tensor<i64>, tensor<i64>, tensor<4xi64>, tensor<4xf32>, tensor<f32>, tensor<f32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %3 = stablehlo.compare LT, %iterArg, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %3 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %3 = stablehlo.add %iterArg, %c_3 : tensor<i64>
// CHECK-NEXT:      %4 = stablehlo.subtract %iterArg_5, %c_3 : tensor<i64>
// CHECK-NEXT:      %5 = stablehlo.dynamic_slice %iterArg_7, %4, sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:      %6 = stablehlo.reshape %5 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:      %7 = stablehlo.dynamic_slice %iterArg_6, %4, sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %8 = stablehlo.reshape %7 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %9 = stablehlo.subtract %arg1, %iterArg : tensor<i64>
// CHECK-NEXT:      %10:5 = stablehlo.while(%iterArg_10 = %8, %iterArg_11 = %4, %iterArg_12 = %iterArg_6, %iterArg_13 = %6, %iterArg_14 = %iterArg_7) : tensor<i64>, tensor<i64>, tensor<4xi64>, tensor<f32>, tensor<4xf32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %17 = stablehlo.add %iterArg_10, %c_3 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.compare LT, %17, %9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %18 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %17 = stablehlo.subtract %9, %iterArg_10 : tensor<i64>
// CHECK-NEXT:        %18 = stablehlo.subtract %c_0, %iterArg_11 : tensor<i64>
// CHECK-NEXT:        %19 = enzymexla.math.binomial_progress(%17, %18) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.reshape %iterArg_13 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:        %21 = stablehlo.dynamic_update_slice %iterArg_14, %20, %iterArg_11 : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
// CHECK-NEXT:        %22 = stablehlo.reshape %iterArg_10 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %23 = stablehlo.dynamic_update_slice %iterArg_12, %22, %iterArg_11 : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:        %24 = stablehlo.add %iterArg_10, %19 : tensor<i64>
// CHECK-NEXT:        %25 = stablehlo.compare EQ, %24, %9 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %26 = stablehlo.convert %25 : (tensor<i1>) -> tensor<i64>
// CHECK-NEXT:        %27 = stablehlo.subtract %24, %26 : tensor<i64>
// CHECK-NEXT:        %28:2 = stablehlo.while(%iterArg_15 = %iterArg_10, %iterArg_16 = %iterArg_13) : tensor<i64>, tensor<f32>
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %30 = stablehlo.compare LT, %iterArg_15, %27 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %30 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %30:2 = stablehlo.while(%iterArg_17 = %c_4, %iterArg_18 = %iterArg_16) : tensor<i64>, tensor<f32>
// CHECK-NEXT:          cond {
// CHECK-NEXT:            %32 = stablehlo.compare LT, %iterArg_17, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:            stablehlo.return %32 : tensor<i1>
// CHECK-NEXT:          } do {
// CHECK-NEXT:            %32 = stablehlo.multiply %arg0, %iterArg_18 : tensor<f32>
// CHECK-NEXT:            %33 = stablehlo.add %iterArg_17, %c_3 : tensor<i64>
// CHECK-NEXT:            stablehlo.return %33, %32 : tensor<i64>, tensor<f32>
// CHECK-NEXT:          }
// CHECK-NEXT:          %31 = stablehlo.add %iterArg_15, %c_3 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %31, %30#1 : tensor<i64>, tensor<f32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %29 = stablehlo.add %iterArg_11, %c_3 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %24, %29, %23, %28#1, %21 : tensor<i64>, tensor<i64>, tensor<4xi64>, tensor<f32>, tensor<4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %11 = stablehlo.reshape %arg1 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %12 = tensor.empty() : tensor<0xf32>
// CHECK-NEXT:      %13 = stablehlo.dynamic_pad %12, %cst, %c, %11, %c : (tensor<0xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
// CHECK-NEXT:      %14:3 = stablehlo.while(%iterArg_10 = %c_4, %iterArg_11 = %10#3, %iterArg_12 = %13) : tensor<i64>, tensor<f32>, tensor<?xf32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %17 = stablehlo.compare LT, %iterArg_10, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %17 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %17 = stablehlo.reshape %iterArg_11 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:        %18 = stablehlo.dynamic_update_slice %iterArg_12, %17, %iterArg_10 : (tensor<?xf32>, tensor<1xf32>, tensor<i64>) -> tensor<?xf32>
// CHECK-NEXT:        %19 = stablehlo.multiply %arg0, %iterArg_11 : tensor<f32>
// CHECK-NEXT:        %20 = stablehlo.add %iterArg_10, %c_3 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %20, %19, %18 : tensor<i64>, tensor<f32>, tensor<?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %15 = stablehlo.subtract %arg1, %c_3 : tensor<i64>
// CHECK-NEXT:      %16:4 = stablehlo.while(%iterArg_10 = %c_4, %iterArg_11 = %iterArg_8, %iterArg_12 = %iterArg_9, %iterArg_13 = %15) : tensor<i64>, tensor<f32>, tensor<f32>, tensor<i64>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %17 = stablehlo.compare LT, %iterArg_10, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %17 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %17 = stablehlo.dynamic_slice %14#2, %iterArg_13, sizes = [1] : (tensor<?xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:        %18 = stablehlo.reshape %17 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:        %19 = stablehlo.add %iterArg_10, %c_3 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.multiply %iterArg_11, %18 : tensor<f32>
// CHECK-NEXT:        %21 = stablehlo.add %iterArg_12, %20 : tensor<f32>
// CHECK-NEXT:        %22 = stablehlo.multiply %iterArg_11, %arg0 : tensor<f32>
// CHECK-NEXT:        %23 = stablehlo.subtract %iterArg_13, %c_3 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %19, %22, %21, %23 : tensor<i64>, tensor<f32>, tensor<f32>, tensor<i64>
// CHECK-NEXT:      }
// CHECK-NEXT:      stablehlo.return %3, %10#1, %10#2, %10#4, %16#1, %16#2 : tensor<i64>, tensor<i64>, tensor<4xi64>, tensor<4xf32>, tensor<f32>, tensor<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = stablehlo.add %1#5, %1#4 : tensor<f32>
// CHECK-NEXT:    return %2 : tensor<f32>
// CHECK-NEXT:  }
