// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Test 1: Basic chain - syrk with output_uplo=F feeds into another syrk with uplo=U
// Expected: output_uplo should be set to U, child syrk's uplo should also be U
func.func @test_basic_chain_upper(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>

    // First SYRK with output_uplo=F
    %0 = blas.syrk %cst, %arg0, %cst_0, %cst_1 {
        output_uplo = #blas.uplo<any>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<any>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Second SYRK with uplo=U requires the input C to have U layout
    %1 = blas.syrk %cst, %arg0, %cst_0, %0 {
        output_uplo = #blas.uplo<any>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<upper>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %1 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @test_basic_chain_upper
// CHECK: blas.syrk
// CHECK-SAME: output_uplo = #blas.uplo<upper>
// CHECK: blas.syrk
// CHECK-SAME: uplo = #blas.uplo<upper>


// Test 2: Chain with lower triangular - syrk feeds into syrk with uplo=L
// Expected: output_uplo should be set to L
func.func @test_basic_chain_lower(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>

    %0 = blas.syrk %cst, %arg0, %cst_0, %cst_1 {
        output_uplo = #blas.uplo<any>,
        transpose = #blas.transpose<none>,
        uplo = #blas.uplo<any>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %1 = blas.syrk %cst, %arg0, %cst_0, %0 {
        output_uplo = #blas.uplo<lower>,
        transpose = #blas.transpose<none>,
        uplo = #blas.uplo<lower>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %1 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @test_basic_chain_lower
// CHECK: blas.syrk
// CHECK-SAME: output_uplo = #blas.uplo<lower>
// CHECK: blas.syrk
// CHECK-SAME: uplo = #blas.uplo<lower>


// Test 3: Chain with elementwise ops in between
// Expected: Should still propagate through elementwise ops
func.func @test_elementwise_chain(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = stablehlo.constant dense<3.444500e+00> : tensor<32x32xf32>
    %cst_0 = stablehlo.constant dense<2.031500e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>
    %cst_4 = stablehlo.constant dense<-4.775000e+00> : tensor<32x32xf32>

    // First SYRK with output_uplo=F
    %0 = blas.syrk %cst_2, %arg0, %cst_1, %cst_3 {
        output_uplo = #blas.uplo<any>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<any>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Elementwise operations
    %1 = stablehlo.add %0, %arg1 : tensor<32x32xf32>
    %2 = stablehlo.multiply %cst_4, %1 : tensor<32x32xf32>

    // Second SYRK uses %2 as C (derived from %0)
    %3 = blas.syrk %cst_0, %arg0, %cst_2, %2 {
        output_uplo = #blas.uplo<any>,
        uplo = #blas.uplo<upper>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %3 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @test_elementwise_chain
// CHECK: blas.syrk
// CHECK-SAME: output_uplo = #blas.uplo<upper>
// CHECK: stablehlo.add
// CHECK: stablehlo.multiply
// CHECK: blas.syrk
// CHECK-SAME: uplo = #blas.uplo<upper>


// Test 4: Multiple child syrk ops with same uplo
// Expected: All children get the same uplo propagated
func.func @test_multiple_children_same_uplo(%arg0: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>

    // Parent SYRK
    %0 = blas.syrk %cst, %arg0, %cst_0, %cst_1 {
        output_uplo = #blas.uplo<any>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<any>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Two child SYRKs, both with uplo=L
    %1 = blas.syrk %cst, %arg0, %cst_0, %0 {
        output_uplo = #blas.uplo<lower>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<lower>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %2 = blas.syrk %cst, %arg0, %cst_0, %0 {
        output_uplo = #blas.uplo<lower>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<lower>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %1, %2 : tensor<32x32xf32>, tensor<32x32xf32>
}

// CHECK-LABEL: func.func @test_multiple_children_same_uplo
// CHECK: blas.syrk
// CHECK-SAME: output_uplo = #blas.uplo<lower>
// CHECK: blas.syrk
// CHECK-SAME: uplo = #blas.uplo<lower>
// CHECK: blas.syrk
// CHECK-SAME: uplo = #blas.uplo<lower>


// Test 5: Conflicting uplos - should NOT transform
// Expected: No transformation when children have conflicting uplos (U and L)
func.func @test_conflicting_uplos(%arg0: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>

    // Parent SYRK
    %0 = blas.syrk %cst, %arg0, %cst_0, %cst_1 {
        output_uplo = #blas.uplo<any>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<any>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Two child SYRKs with CONFLICTING uplos
    %1 = blas.syrk %cst, %arg0, %cst_0, %0 {
        output_uplo = #blas.uplo<upper>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<upper>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %2 = blas.syrk %cst, %arg0, %cst_0, %0 {
        output_uplo = #blas.uplo<lower>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<lower>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %1, %2 : tensor<32x32xf32>, tensor<32x32xf32>
}

// CHECK-LABEL: func.func @test_conflicting_uplos
// Parent SYRK should still have output_uplo = F due to conflict
// CHECK: blas.syrk
// CHECK-SAME: output_uplo = #blas.uplo<any>


// Test 6: Non-elementwise user blocks optimization
// Expected: No transformation when result is used by non-elementwise op
func.func @test_non_elementwise_user(%arg0: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>

    // Parent SYRK
    %0 = blas.syrk %cst, %arg0, %cst_0, %cst_1 {
        output_uplo = #blas.uplo<any>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<any>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // One user is a syrk
    %1 = blas.syrk %cst, %arg0, %cst_0, %0 {
        output_uplo = #blas.uplo<upper>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<upper>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Return the original result - this creates an escape that prevents optimization
    return %1, %0 : tensor<32x32xf32>, tensor<32x32xf32>
}

// CHECK-LABEL: func.func @test_non_elementwise_user
// Parent SYRK should still have output_uplo = F due to non-syrk user
// CHECK: blas.syrk
// CHECK-SAME: output_uplo = #blas.uplo<any>


// Test 7: All children have uplo=F, choose based on output_uplo majority (upper wins)
func.func @test_all_f_children_upper_majority(%arg0: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>

    // Parent SYRK
    %0 = blas.syrk %cst, %arg0, %cst_0, %cst_1 {
        output_uplo = #blas.uplo<any>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<any>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Children with uplo=F but different output_uplo preferences
    %1 = blas.syrk %cst, %arg0, %cst_0, %0 {
        output_uplo = #blas.uplo<upper>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<any>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %2 = blas.syrk %cst, %arg0, %cst_0, %0 {
        output_uplo = #blas.uplo<upper>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<any>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %1, %2 : tensor<32x32xf32>, tensor<32x32xf32>
}

// CHECK-LABEL: func.func @test_all_f_children_upper_majority
// CHECK: blas.syrk
// CHECK-SAME: output_uplo = #blas.uplo<upper>
// CHECK: blas.syrk
// CHECK-SAME: uplo = #blas.uplo<upper>
// CHECK: blas.syrk
// CHECK-SAME: uplo = #blas.uplo<upper>


// Test 8: Syrk used as A operand should not block the optimization
// The optimization only looks at uses as C operand
func.func @test_syrk_as_a_operand(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<32x32xf32>

    // Parent SYRK
    %0 = blas.syrk %cst, %arg0, %cst_0, %cst_1 {
        output_uplo = #blas.uplo<any>,
        transpose = #blas.transpose<transpose>,
        uplo = #blas.uplo<any>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Child SYRK uses result as A, not C - should fail the pattern
    %1 = blas.syrk %cst, %0, %cst_0, %cst_1 {
        output_uplo = #blas.uplo<upper>,
        transpose = #blas.transpose<none>,
        uplo = #blas.uplo<upper>
    } : (tensor<f32>, tensor<32x32xf32>, tensor<f32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %1 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @test_syrk_as_a_operand
// Should NOT transform since child syrk uses result as A operand, not C
// CHECK: blas.syrk
// CHECK-SAME: output_uplo = #blas.uplo<any>
