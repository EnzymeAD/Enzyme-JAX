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

// Checkpointed forward/recomputation loops retain their induction comparison.
// Unannotated dynamic reverse loops can carry the condition after differentiation;
// check the initial predicate, the returned next predicate, and all data results.
// CHECK-LABEL: func.func @main(%arg0: tensor<f32>, %arg1: tensor<i64>, %arg2: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:     %c_2 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:     %c_3 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:     %c_4 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0 = stablehlo.minimum %c_2, %arg1 : tensor<i64>
// CHECK-NEXT:     %1:5 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %c_4, %iterArg_6 = %arg0, %iterArg_7 = %cst_1, %iterArg_8 = %c_0) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:     cond {
// CHECK-NEXT:       %5 = stablehlo.compare LT, %iterArg, %0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %5 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %5 = stablehlo.reshape %iterArg_6 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:       %6 = stablehlo.dynamic_update_slice %iterArg_7, %5, %iterArg : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
// CHECK-NEXT:       %7 = stablehlo.reshape %iterArg_5 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:       %8 = stablehlo.dynamic_update_slice %iterArg_8, %7, %iterArg : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:       %9 = stablehlo.subtract %arg1, %iterArg_5 : tensor<i64>
// CHECK-NEXT:       %10 = stablehlo.subtract %0, %iterArg : tensor<i64>
// CHECK-NEXT:       %11 = stablehlo.minimum %10, %9 : tensor<i64>
// CHECK-NEXT:       %12 = enzyme.binomial_progress %9, %11 : tensor<i64>
// CHECK-NEXT:       %13:2 = stablehlo.while(%iterArg_9 = %c_4, %iterArg_10 = %iterArg_6) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %16 = stablehlo.compare LT, %iterArg_9, %12 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %16 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %16:2 = stablehlo.while(%iterArg_11 = %c_4, %iterArg_12 = %iterArg_10) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:         cond {
// CHECK-NEXT:           %18 = stablehlo.compare LT, %iterArg_11, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:           stablehlo.return %18 : tensor<i1>
// CHECK-NEXT:         } do {
// CHECK-NEXT:           %18 = stablehlo.multiply %arg0, %iterArg_12 : tensor<f32>
// CHECK-NEXT:           %19 = stablehlo.add %iterArg_11, %c_3 : tensor<i64>
// CHECK-NEXT:           stablehlo.return %19, %18 : tensor<i64>, tensor<f32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %17 = stablehlo.add %iterArg_9, %c_3 : tensor<i64>
// CHECK-NEXT:         stablehlo.return %17, %16#1 : tensor<i64>, tensor<f32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %14 = stablehlo.add %iterArg_5, %12 : tensor<i64>
// CHECK-NEXT:       %15 = stablehlo.add %iterArg, %c_3 : tensor<i64>
// CHECK-NEXT:       stablehlo.return %15, %14, %13#1, %6, %8 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = stablehlo.compare LT, %c_4, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:     %3:7 = stablehlo.while(%iterArg = %c_4, %iterArg_5 = %0, %iterArg_6 = %arg2, %iterArg_7 = %1#3, %iterArg_8 = %1#4, %iterArg_9 = %cst, %iterArg_10 = %2) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>, tensor<f32>, tensor<i1>
// CHECK-NEXT:     cond {
// CHECK-NEXT:       stablehlo.return %iterArg_10 : tensor<i1>
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %5 = stablehlo.subtract %iterArg_5, %c_3 : tensor<i64>
// CHECK-NEXT:       %6 = stablehlo.subtract %arg1, %iterArg : tensor<i64>
// CHECK-NEXT:       %7 = stablehlo.dynamic_slice %iterArg_7, %5, sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:       %8 = stablehlo.reshape %7 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:       %9 = stablehlo.dynamic_slice %iterArg_8, %5, sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:       %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:       %11:5 = stablehlo.while(%iterArg_11 = %10, %iterArg_12 = %5, %iterArg_13 = %8, %iterArg_14 = %iterArg_7, %iterArg_15 = %iterArg_8) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %21 = stablehlo.add %iterArg_11, %c_3 : tensor<i64>
// CHECK-NEXT:         %22 = stablehlo.compare LT, %21, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %22 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %21 = stablehlo.subtract %6, %iterArg_11 : tensor<i64>
// CHECK-NEXT:         %22 = stablehlo.subtract %0, %iterArg_12 : tensor<i64>
// CHECK-NEXT:         %23 = stablehlo.minimum %22, %21 : tensor<i64>
// CHECK-NEXT:         %24 = enzyme.binomial_progress %21, %23 : tensor<i64>
// CHECK-NEXT:         %25 = stablehlo.reshape %iterArg_13 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:         %26 = stablehlo.dynamic_update_slice %iterArg_14, %25, %iterArg_12 : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
// CHECK-NEXT:         %27 = stablehlo.reshape %iterArg_11 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:         %28 = stablehlo.dynamic_update_slice %iterArg_15, %27, %iterArg_12 : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:         %29 = stablehlo.add %iterArg_11, %24 : tensor<i64>
// CHECK-NEXT:         %30 = stablehlo.compare EQ, %29, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         %31 = stablehlo.subtract %29, %c_3 : tensor<i64>
// CHECK-NEXT:         %32 = stablehlo.select %30, %31, %29 : tensor<i1>, tensor<i64>
// CHECK-NEXT:         %33:2 = stablehlo.while(%iterArg_16 = %iterArg_11, %iterArg_17 = %iterArg_13) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:         cond {
// CHECK-NEXT:           %35 = stablehlo.compare LT, %iterArg_16, %32 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:           stablehlo.return %35 : tensor<i1>
// CHECK-NEXT:         } do {
// CHECK-NEXT:           %35:2 = stablehlo.while(%iterArg_18 = %c_4, %iterArg_19 = %iterArg_17) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:           cond {
// CHECK-NEXT:             %37 = stablehlo.compare LT, %iterArg_18, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:             stablehlo.return %37 : tensor<i1>
// CHECK-NEXT:           } do {
// CHECK-NEXT:             %37 = stablehlo.multiply %arg0, %iterArg_19 : tensor<f32>
// CHECK-NEXT:             %38 = stablehlo.add %iterArg_18, %c_3 : tensor<i64>
// CHECK-NEXT:             stablehlo.return %38, %37 : tensor<i64>, tensor<f32>
// CHECK-NEXT:           }
// CHECK-NEXT:           %36 = stablehlo.add %iterArg_16, %c_3 : tensor<i64>
// CHECK-NEXT:           stablehlo.return %36, %35#1 : tensor<i64>, tensor<f32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %34 = stablehlo.add %iterArg_12, %c_3 : tensor<i64>
// CHECK-NEXT:         stablehlo.return %29, %34, %33#1, %26, %28 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>
// CHECK-NEXT:       }
// CHECK-NEXT:       %12 = stablehlo.reshape %arg1 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:       %13 = tensor.empty() : tensor<0xf32>
// CHECK-NEXT:       %14 = stablehlo.dynamic_pad %13, %cst, %c, %12, %c : (tensor<0xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
// CHECK-NEXT:       %15:3 = stablehlo.while(%iterArg_11 = %c_4, %iterArg_12 = %11#2, %iterArg_13 = %14) : tensor<i64>, tensor<f32>, tensor<?xf32>
// CHECK-NEXT:       cond {
// CHECK-NEXT:         %21 = stablehlo.compare LT, %iterArg_11, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %21 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %21 = stablehlo.reshape %iterArg_12 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:         %22 = stablehlo.dynamic_update_slice %iterArg_13, %21, %iterArg_11 : (tensor<?xf32>, tensor<1xf32>, tensor<i64>) -> tensor<?xf32>
// CHECK-NEXT:         %23 = stablehlo.multiply %arg0, %iterArg_12 : tensor<f32>
// CHECK-NEXT:         %24 = stablehlo.add %iterArg_11, %c_3 : tensor<i64>
// CHECK-NEXT:         stablehlo.return %24, %23, %22 : tensor<i64>, tensor<f32>, tensor<?xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       %16 = stablehlo.subtract %arg1, %c_3 : tensor<i64>
// CHECK-NEXT:       %17 = stablehlo.compare LT, %c_4, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       %18:4 = stablehlo.while(%iterArg_11 = %c_4, %iterArg_12 = %iterArg_6, %iterArg_13 = %iterArg_9, %iterArg_14 = %17) : tensor<i64>, tensor<f32>, tensor<f32>, tensor<i1>
// CHECK-NEXT:       cond {
// CHECK-NEXT:         stablehlo.return %iterArg_14 : tensor<i1>
// CHECK-NEXT:       } do {
// CHECK-NEXT:         %21 = stablehlo.subtract %16, %iterArg_11 : tensor<i64>
// CHECK-NEXT:         %22 = stablehlo.dynamic_slice %15#2, %21, sizes = [1] : (tensor<?xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:         %23 = stablehlo.reshape %22 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:         %24 = stablehlo.add %iterArg_11, %c_3 : tensor<i64>
// CHECK-NEXT:         %25 = stablehlo.multiply %iterArg_12, %23 : tensor<f32>
// CHECK-NEXT:         %26 = stablehlo.add %iterArg_13, %25 : tensor<f32>
// CHECK-NEXT:         %27 = stablehlo.multiply %iterArg_12, %arg0 : tensor<f32>
// CHECK-NEXT:         %28 = stablehlo.compare LT, %24, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:         stablehlo.return %24, %27, %26, %28 : tensor<i64>, tensor<f32>, tensor<f32>, tensor<i1>
// CHECK-NEXT:       }
// CHECK-NEXT:       %19 = stablehlo.add %iterArg, %c_3 : tensor<i64>
// CHECK-NEXT:       %20 = stablehlo.compare LT, %19, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:       stablehlo.return %19, %11#1, %18#1, %11#3, %11#4, %18#2, %20 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>, tensor<f32>, tensor<i1>
// CHECK-NEXT:     }
// CHECK-NEXT:     %4 = stablehlo.add %3#5, %3#2 : tensor<f32>
// CHECK-NEXT:     return %4 : tensor<f32>
// CHECK-NEXT:   }
