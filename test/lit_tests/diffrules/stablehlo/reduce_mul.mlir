// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_mul outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=reduce_mul_complex outfn= argTys=enzyme_active,enzyme_active retTys=enzyme_active mode=ReverseModeCombined" --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --verify-each=0 | FileCheck %s --check-prefix=COMPLEX
// RUN: enzymexlamlir-opt %s --enzyme --arith-raise --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --chlo-legalize-to-stablehlo --enzyme-hlo-opt --verify-each=0 | stablehlo-translate - --interpret --allow-unregistered-dialect

module {
  func.func @reduce_mul(%a: tensor<9xf32>, %init: tensor<f32>) -> tensor<f32> {
    %6 = "stablehlo.reduce"(%a, %init) <{dimensions = array<i64: 0>}> ({
        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
        %8 = "stablehlo.multiply"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%8) : (tensor<f32>) -> ()
        }) : (tensor<9xf32>, tensor<f32>) -> tensor<f32>
    return %6 : tensor<f32>
  }

  func.func @reduce_mul_complex(%a: tensor<2xcomplex<f32>>, %init: tensor<complex<f32>>) -> tensor<complex<f32>> {
    %0 = stablehlo.reduce(%a init: %init) applies stablehlo.multiply across dimensions = [0] : (tensor<2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
    return %0 : tensor<complex<f32>>
  }

  func.func @main() {
    %ca = stablehlo.constant dense<[(1.0, 2.0), (3.0, -1.0)]> : tensor<2xcomplex<f32>>
    %cinit = stablehlo.constant dense<(1.0, 0.0)> : tensor<complex<f32>>
    %cdres = stablehlo.constant dense<(1.0, 0.0)> : tensor<complex<f32>>

    %crev:3 = enzyme.autodiff @reduce_mul_complex(%ca, %cinit, %cdres) <{
      activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
      ret_activity=[#enzyme.activity<enzyme_active>]
    }> : (tensor<2xcomplex<f32>>, tensor<complex<f32>>, tensor<complex<f32>>) -> (tensor<complex<f32>>, tensor<2xcomplex<f32>>, tensor<complex<f32>>)

    check.expect_almost_eq_const %crev#0, dense<(5.0, 5.0)> : tensor<complex<f32>>
    check.expect_almost_eq_const %crev#1, dense<[(3.0, 1.0), (1.0, -2.0)]> : tensor<2xcomplex<f32>>
    check.expect_almost_eq_const %crev#2, dense<(5.0, -5.0)> : tensor<complex<f32>>
    return
  }
}

// CHECK:  func.func @reduce_mul(%arg0: tensor<9xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<9xf32>, tensor<f32>) {
// CHECK-NEXT:    %c = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:    %c_0 = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<1> : tensor<9xi32>
// CHECK-NEXT:    %c_2 = stablehlo.constant dense<0> : tensor<9xi32>
// CHECK-NEXT:    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<9xf32>
// CHECK-NEXT:    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<9xf32>
// CHECK-NEXT:    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %0 = stablehlo.compare EQ, %arg0, %cst_4 : (tensor<9xf32>, tensor<9xf32>) -> tensor<9xi1>
// CHECK-NEXT:    %1 = stablehlo.compare EQ, %arg1, %cst_5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:    %2 = stablehlo.select %0, %cst_3, %arg0 : tensor<9xi1>, tensor<9xf32>
// CHECK-NEXT:    %3 = stablehlo.select %1, %cst, %arg1 : tensor<i1>, tensor<f32>
// CHECK-NEXT:    %4 = stablehlo.select %0, %c_1, %c_2 : tensor<9xi1>, tensor<9xi32>
// CHECK-NEXT:    %5 = stablehlo.select %1, %c, %c_0 : tensor<i1>, tensor<i32>
// CHECK-NEXT:    %6:2 = stablehlo.reduce(%2 init: %3), (%4 init: %5) across dimensions = [0] : (tensor<9xf32>, tensor<9xi32>, tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>)
// CHECK-NEXT:     reducer(%arg3: tensor<f32>, %arg5: tensor<f32>) (%arg4: tensor<i32>, %arg6: tensor<i32>)  {
// CHECK-NEXT:      %18 = stablehlo.multiply %arg3, %arg5 : tensor<f32>
// CHECK-NEXT:      %19 = stablehlo.add %arg4, %arg6 : tensor<i32>
// CHECK-NEXT:      stablehlo.return %18, %19 : tensor<f32>, tensor<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %7 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<9xf32>
// CHECK-NEXT:    %8 = stablehlo.broadcast_in_dim %6#0, dims = [] : (tensor<f32>) -> tensor<9xf32>
// CHECK-NEXT:    %9 = stablehlo.broadcast_in_dim %6#1, dims = [] : (tensor<i32>) -> tensor<9xi32>
// CHECK-NEXT:    %10 = stablehlo.divide %8, %2 : tensor<9xf32>
// CHECK-NEXT:    %11 = stablehlo.compare EQ, %9, %4 : (tensor<9xi32>, tensor<9xi32>) -> tensor<9xi1>
// CHECK-NEXT:    %12 = stablehlo.select %11, %10, %cst_4 : tensor<9xi1>, tensor<9xf32>
// CHECK-NEXT:    %13 = stablehlo.multiply %7, %12 : tensor<9xf32>
// CHECK-NEXT:    %14 = stablehlo.divide %6#0, %3 : tensor<f32>
// CHECK-NEXT:    %15 = stablehlo.compare EQ, %6#1, %5 : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:    %16 = stablehlo.select %15, %14, %cst_5 : tensor<i1>, tensor<f32>
// CHECK-NEXT:    %17 = stablehlo.multiply %16, %arg2 : tensor<f32>
// CHECK-NEXT:    return %13, %17 : tensor<9xf32>, tensor<f32>
// CHECK-NEXT:  }

// COMPLEX:  func.func @reduce_mul_complex(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<complex<f32>>, %arg2: tensor<complex<f32>>) -> (tensor<2xcomplex<f32>>, tensor<complex<f32>>) {
// COMPLEX:    %6:2 = stablehlo.reduce(%2 init: %3), (%4 init: %5) across dimensions = [0]
// COMPLEX:    %10 = stablehlo.divide %8, %2
// COMPLEX:    %12 = stablehlo.select %11, %10, %cst_4
// COMPLEX:    %13 = chlo.conj %12 : tensor<2xcomplex<f32>> -> tensor<2xcomplex<f32>>
// COMPLEX:    %14 = stablehlo.multiply %7, %13 : tensor<2xcomplex<f32>>
// COMPLEX:    %15 = stablehlo.divide %6#0, %3
// COMPLEX:    %17 = stablehlo.select %16, %15, %cst_5
// COMPLEX:    %18 = chlo.conj %17
// COMPLEX:    %19 = stablehlo.multiply %18, %arg2
// COMPLEX:    return %14, %19 : tensor<2xcomplex<f32>>, tensor<complex<f32>>
// COMPLEX-NEXT:  }
