// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @opt_seq_short(%2850: tensor<4x1518x3056xf64>, %2852: tensor<4x1518x3056xf64>, %2858: tensor<4x1518x3056xf64>) -> tensor<4x1518x3056xf64> {
    %cst_151 = stablehlo.constant dense<0.58333333333333326> : tensor<4x1518x3056xf64>
    %2851 = stablehlo.multiply %2850, %cst_151 : tensor<4x1518x3056xf64>
    %2853 = stablehlo.multiply %2852, %cst_151 : tensor<4x1518x3056xf64>
    %2859 = stablehlo.add %2853, %2858 : tensor<4x1518x3056xf64>
    %2860 = stablehlo.add %2851, %2859 : tensor<4x1518x3056xf64>
    return %2860 : tensor<4x1518x3056xf64>
}

// CHECK: func.func @opt_seq_short(%arg0: tensor<4x1518x3056xf64>, %arg1: tensor<4x1518x3056xf64>, %arg2: tensor<4x1518x3056xf64>) -> tensor<4x1518x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.58333333333333326> : tensor<4x1518x3056xf64>
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg1 : tensor<4x1518x3056xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %cst, %0 : tensor<4x1518x3056xf64>
// CHECK-NEXT:     %2 = stablehlo.add %arg2, %1 : tensor<4x1518x3056xf64>
// CHECK-NEXT:     return %2 : tensor<4x1518x3056xf64>
// CHECK-NEXT: }

func.func @opt_seq_long(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
    %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
    %2 = stablehlo.add %arg3, %1 : tensor<4xf64>
    %3 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
    %4 = stablehlo.add %3, %2 : tensor<4xf64>
    %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
    return %5 : tensor<4xf64>
}

// CHECK: func.func @opt_seq_long(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.add %arg1, %arg0 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %arg5, %0 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.add %arg2, %1 : tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.add %arg3, %2 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %arg4 : tensor<4xf64>
// CHECK-NEXT:     return %4 : tensor<4xf64>
// CHECK-NEXT: }

func.func @opt_tree(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
    %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
    %2 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
    %3 = stablehlo.add %2, %arg3 : tensor<4xf64>
    %4 = stablehlo.add %3, %1 : tensor<4xf64>
    %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
    return %5 : tensor<4xf64>
}

// CHECK: func.func @opt_tree(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.add %arg1, %arg0 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %arg5, %0 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.add %arg2, %1 : tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.add %arg3, %2 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %arg4 : tensor<4xf64>
// CHECK-NEXT:     return %4 : tensor<4xf64>
// CHECK-NEXT: }

func.func @opt_multi(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
    %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
    %2 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
    %3 = stablehlo.add %2, %arg3 : tensor<4xf64>
    %4 = stablehlo.add %3, %1 : tensor<4xf64>
    %5 = stablehlo.multiply %arg4, %arg5 : tensor<4xf64>
    %6 = stablehlo.add %4, %5 : tensor<4xf64>
    return %6: tensor<4xf64>
}

// CHECK: func.func @opt_multi(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.add %arg1, %arg0 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %arg4 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.multiply %arg5, %1 : tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.add %arg3, %arg2 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %2 : tensor<4xf64>
// CHECK-NEXT:     return %4 : tensor<4xf64>
// CHECK-NEXT: }

func.func @no_opt_benefit(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg2 : tensor<4xf64>
    %1 = stablehlo.multiply %arg2, %arg1 : tensor<4xf64>
    %2 = stablehlo.add %0, %1 : tensor<4xf64>
    return %2 : tensor<4xf64>
}

// CHECK: func.func @no_opt_benefit(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg2 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %arg2, %arg1 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.add %0, %1 : tensor<4xf64>
// CHECK-NEXT:     return %2 : tensor<4xf64>
// CHECK-NEXT: }

func.func @no_opt_multiuse_add_inbetween(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
    %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
    %2 = stablehlo.add %arg3, %1 : tensor<4xf64>
    %3 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
    %4 = stablehlo.add %3, %2 : tensor<4xf64>
    %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
    %6 = stablehlo.divide %5, %2 : tensor<4xf64>
    return %6 : tensor<4xf64>
}

// CHECK: func.func @no_opt_multiuse_add_inbetween(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.add %arg3, %1 : tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %2 : tensor<4xf64>
// CHECK-NEXT:     %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
// CHECK-NEXT:     %6 = stablehlo.divide %5, %2 : tensor<4xf64>
// CHECK-NEXT:     return %6 : tensor<4xf64>
// CHECK-NEXT: }

func.func @no_opt_multiuse_mul(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
    %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
    %2 = stablehlo.add %arg3, %1 : tensor<4xf64>
    %3 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
    %4 = stablehlo.add %3, %2 : tensor<4xf64>
    %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
    %6 = stablehlo.divide %5, %0 : tensor<4xf64>
    return %6 : tensor<4xf64>
}

// CHECK: func.func @no_opt_multiuse_mul(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.add %arg3, %1 : tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %2 : tensor<4xf64>
// CHECK-NEXT:     %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
// CHECK-NEXT:     %6 = stablehlo.divide %5, %0 : tensor<4xf64>
// CHECK-NEXT:     return %6 : tensor<4xf64>
// CHECK-NEXT: }

func.func @no_opt_single_mul(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf64>
    %1 = stablehlo.add %0, %0 : tensor<4xf64>
    return %1 : tensor<4xf64>
}

// CHECK: func.func @no_opt_single_mul(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %0 : tensor<4xf64>
// CHECK-NEXT:     return %1 : tensor<4xf64>
// CHECK-NEXT: }

// same number of ops but number of muls goes from 7 to 3
func.func @case_gb25_loop(%9: tensor<f64>, %12: tensor<f64>, %22: tensor<f64>, %25: tensor<f64>, %37: tensor<f64>, %34: tensor<f64>, %46: tensor<f64>, %63: tensor<f64>, %69: tensor<f64>, %73: tensor<f64>, %86: tensor<f64>, %93: tensor<f64>, %184: tensor<f64>) -> tensor<f64> {
    %cst = stablehlo.constant dense<8.3174536423027376> : tensor<f64>
    %cst_3 = stablehlo.constant dense<-175.65119408165049> : tensor<f64>
    %cst_4 = stablehlo.constant dense<0.10132118364233778> : tensor<f64>
    %cst_13 = stablehlo.constant dense<-3776.4484124697938> : tensor<f64>
    %cst_15 = stablehlo.constant dense<4.1587268211513688> : tensor<f64>
    %cst_21 = stablehlo.constant dense<48.376411884489357> : tensor<f64>
    %cst_26 = stablehlo.constant dense<2392.6120982565681> : tensor<f64>
    %cst_28 = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
    %cst_34 = stablehlo.constant dense<5.000000e-01> : tensor<f64>

    %13 = stablehlo.multiply %cst_28, %12 : tensor<f64>
    %14 = stablehlo.add %9, %13 : tensor<f64>
    %17 = stablehlo.add %14, %cst_15 : tensor<f64>

    %26 = stablehlo.multiply %cst_28, %25 : tensor<f64>
    %27 = stablehlo.add %22, %26 : tensor<f64>

    %38 = stablehlo.multiply %cst_28, %37 : tensor<f64>
    %39 = stablehlo.add %34, %38 : tensor<f64>
    %42 = stablehlo.add %39, %27 : tensor<f64>
    %43 = stablehlo.add %cst, %42 : tensor<f64>
    %44 = stablehlo.add %17, %43 : tensor<f64>
    %48 = stablehlo.add %44, %46 : tensor<f64>
    %67 = stablehlo.add %48, %63 : tensor<f64>

    %70 = stablehlo.multiply %69, %cst_34 : tensor<f64>
    %74 = stablehlo.add %70, %cst_13 : tensor<f64>
    %75 = stablehlo.add %74, %73 : tensor<f64>

    %87 = stablehlo.multiply %86, %cst_34 : tensor<f64>
    %88 = stablehlo.add %87, %cst_21 : tensor<f64>

    %94 = stablehlo.multiply %93, %cst_4 : tensor<f64>
    %95 = stablehlo.add %94, %cst_3 : tensor<f64>
    %96 = stablehlo.add %88, %95 : tensor<f64>
    %97 = stablehlo.add %75, %96 : tensor<f64>

    %185 = stablehlo.multiply %184, %cst_34 : tensor<f64>
    %186 = stablehlo.add %185, %cst_26 : tensor<f64>
    %187 = stablehlo.add %186, %97 : tensor<f64>
    %188 = stablehlo.add %187, %67 : tensor<f64>

    return %188 : tensor<f64>
}

// CHECK: func.func @case_gb25_loop(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<f64>, %arg5: tensor<f64>, %arg6: tensor<f64>, %arg7: tensor<f64>, %arg8: tensor<f64>, %arg9: tensor<f64>, %arg10: tensor<f64>, %arg11: tensor<f64>, %arg12: tensor<f64>) -> tensor<f64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<8.3174536423027376> : tensor<f64>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<-175.65119408165049> : tensor<f64>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.10132118364233778> : tensor<f64>
// CHECK-NEXT:     %cst_2 = stablehlo.constant dense<-3776.4484124697938> : tensor<f64>
// CHECK-NEXT:     %cst_3 = stablehlo.constant dense<4.1587268211513688> : tensor<f64>
// CHECK-NEXT:     %cst_4 = stablehlo.constant dense<48.376411884489357> : tensor<f64>
// CHECK-NEXT:     %cst_5 = stablehlo.constant dense<2392.6120982565681> : tensor<f64>
// CHECK-NEXT:     %cst_6 = stablehlo.constant dense<-2.000000e+00> : tensor<f64>
// CHECK-NEXT:     %cst_7 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
// CHECK-NEXT:     %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f64>
// CHECK-NEXT:     %0 = stablehlo.add %arg1, %arg4 : tensor<f64>
// CHECK-NEXT:     %1 = stablehlo.add %arg0, %cst_8 : tensor<f64>
// CHECK-NEXT:     %2 = stablehlo.add %1, %cst_3 : tensor<f64>
// CHECK-NEXT:     %3 = stablehlo.add %0, %arg3 : tensor<f64>
// CHECK-NEXT:     %4 = stablehlo.multiply %cst_6, %3 : tensor<f64>
// CHECK-NEXT:     %5 = stablehlo.add %arg2, %4 : tensor<f64>
// CHECK-NEXT:     %6 = stablehlo.add %arg5, %cst_8 : tensor<f64>
// CHECK-NEXT:     %7 = stablehlo.add %6, %5 : tensor<f64>
// CHECK-NEXT:     %8 = stablehlo.add %cst, %7 : tensor<f64>
// CHECK-NEXT:     %9 = stablehlo.add %2, %8 : tensor<f64>
// CHECK-NEXT:     %10 = stablehlo.add %9, %arg6 : tensor<f64>
// CHECK-NEXT:     %11 = stablehlo.add %10, %arg7 : tensor<f64>
// CHECK-NEXT:     %12 = stablehlo.add %arg12, %arg8 : tensor<f64>
// CHECK-NEXT:     %13 = stablehlo.add %12, %arg10 : tensor<f64>
// CHECK-NEXT:     %14 = stablehlo.multiply %cst_7, %13 : tensor<f64>
// CHECK-NEXT:     %15 = stablehlo.add %cst_2, %cst_8 : tensor<f64>
// CHECK-NEXT:     %16 = stablehlo.add %15, %arg9 : tensor<f64>
// CHECK-NEXT:     %17 = stablehlo.add %cst_4, %14 : tensor<f64>
// CHECK-NEXT:     %18 = stablehlo.multiply %arg11, %cst_1 : tensor<f64>
// CHECK-NEXT:     %19 = stablehlo.add %18, %cst_0 : tensor<f64>
// CHECK-NEXT:     %20 = stablehlo.add %17, %19 : tensor<f64>
// CHECK-NEXT:     %21 = stablehlo.add %16, %20 : tensor<f64>
// CHECK-NEXT:     %22 = stablehlo.add %cst_5, %cst_8 : tensor<f64>
// CHECK-NEXT:     %23 = stablehlo.add %22, %21 : tensor<f64>
// CHECK-NEXT:     %24 = stablehlo.add %23, %11 : tensor<f64>
// CHECK-NEXT:     return %24 : tensor<f64>
// CHECK-NEXT: }
