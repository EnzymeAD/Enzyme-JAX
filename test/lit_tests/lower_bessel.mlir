
// RUN: enzymexlamlir-opt --lower-enzymexla-bessel %s | FileCheck %s
// RUN: enzymexlamlir-opt --lower-enzymexla-bessel --enzyme-hlo-opt %s | FileCheck %s --check-prefix=CONSTPROP

// this doesn't actually work, the function is not inlined and const-propped.
// CHECK-CONSTPROP: stablehlo.constant dense<6.671318618245173> : tensor<10x2xf64>

// CHECK:      func.func private @special_besseli(%arg0: tensor<10x2xf64>, %arg1: tensor<10x2xf64>) -> tensor<10x2xf64> {
// CHECK-NEXT:   %cst = stablehlo.constant dense<-1.000000e+00> : tensor<10x2xf64>
// CHECK-NEXT:   %cst_0 = stablehlo.constant dense<1.000000e-15> : tensor<10x2xf64>
// CHECK-NEXT:   %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:   %c_1 = stablehlo.constant dense<true> : tensor<10x2xi1>
// CHECK-NEXT:   %c_2 = stablehlo.constant dense<200> : tensor<10x2xi32>
// CHECK-NEXT:   %c_3 = stablehlo.constant dense<1> : tensor<10x2xi32>
// CHECK-NEXT:   %c_4 = stablehlo.constant dense<0> : tensor<10x2xi32>
// CHECK-NEXT:   %cst_5 = stablehlo.constant dense<0.63661977236758138> : tensor<10x2xf64>
// CHECK-NEXT:   %cst_6 = stablehlo.constant dense<3.1415926535897931> : tensor<10x2xf64>
// CHECK-NEXT:   %cst_7 = stablehlo.constant dense<2.000000e+00> : tensor<10x2xf64>
// CHECK-NEXT:   %cst_8 = stablehlo.constant dense<2.500000e-01> : tensor<10x2xf64>
// CHECK-NEXT:   %cst_9 = stablehlo.constant dense<5.000000e-01> : tensor<10x2xf64>
// CHECK-NEXT:   %cst_10 = stablehlo.constant dense<1.000000e+00> : tensor<10x2xf64>
// CHECK-NEXT:   %cst_11 = stablehlo.constant dense<0.000000e+00> : tensor<10x2xf64>
// CHECK-NEXT:   %0 = stablehlo.abs %arg0 : tensor<10x2xf64>
// CHECK-NEXT:   %1 = stablehlo.abs %arg1 : tensor<10x2xf64>
// CHECK-NEXT:   %2 = stablehlo.compare  GE, %arg0, %cst_11 : (tensor<10x2xf64>, tensor<10x2xf64>) -> tensor<10x2xi1>
// CHECK-NEXT:   %3 = stablehlo.compare  GE, %arg1, %cst_11 : (tensor<10x2xf64>, tensor<10x2xf64>) -> tensor<10x2xi1>
// CHECK-NEXT:   %4 = stablehlo.multiply %1, %1 : tensor<10x2xf64>
// CHECK-NEXT:   %5 = stablehlo.multiply %4, %cst_8 : tensor<10x2xf64>
// CHECK-NEXT:   %6:4 = stablehlo.while(%iterArg = %c_4, %iterArg_12 = %cst_11, %iterArg_13 = %cst_10, %iterArg_14 = %c_1) : tensor<10x2xi32>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xi1>
// CHECK-NEXT:   cond {
// CHECK-NEXT:     %30 = stablehlo.compare  LT, %iterArg, %c_2 : (tensor<10x2xi32>, tensor<10x2xi32>) -> tensor<10x2xi1>
// CHECK-NEXT:     %31 = stablehlo.reduce(%iterArg_14 init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<10x2xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:     %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<i1>) -> tensor<10x2xi1>
// CHECK-NEXT:     %33 = stablehlo.and %30, %32 : tensor<10x2xi1>
// CHECK-NEXT:     %34 = stablehlo.reduce(%33 init: %c) applies stablehlo.or across dimensions = [0, 1] : (tensor<10x2xi1>, tensor<i1>) -> tensor<i1>
// CHECK-NEXT:     stablehlo.return %34 : tensor<i1>
// CHECK-NEXT:   } do {
// CHECK-NEXT:     %30 = stablehlo.add %iterArg_12, %iterArg_13 : tensor<10x2xf64>
// CHECK-NEXT:     %31 = stablehlo.convert %iterArg : (tensor<10x2xi32>) -> tensor<10x2xf64>
// CHECK-NEXT:     %32 = stablehlo.add %31, %cst_10 : tensor<10x2xf64>
// CHECK-NEXT:     %33 = stablehlo.add %0, %32 : tensor<10x2xf64>
// CHECK-NEXT:     %34 = stablehlo.multiply %33, %32 : tensor<10x2xf64>
// CHECK-NEXT:     %35 = stablehlo.divide %5, %34 : tensor<10x2xf64>
// CHECK-NEXT:     %36 = stablehlo.multiply %iterArg_13, %35 : tensor<10x2xf64>
// CHECK-NEXT:     %37 = stablehlo.abs %36 : tensor<10x2xf64>
// CHECK-NEXT:     %38 = stablehlo.abs %30 : tensor<10x2xf64>
// CHECK-NEXT:     %39 = stablehlo.divide %37, %38 : tensor<10x2xf64>
// CHECK-NEXT:     %40 = stablehlo.compare  GT, %39, %cst_0 : (tensor<10x2xf64>, tensor<10x2xf64>) -> tensor<10x2xi1>
// CHECK-NEXT:     %41 = stablehlo.compare  LT, %38, %cst_0 : (tensor<10x2xf64>, tensor<10x2xf64>) -> tensor<10x2xi1>
// CHECK-NEXT:     %42 = stablehlo.or %40, %41 : tensor<10x2xi1>
// CHECK-NEXT:     %43 = stablehlo.and %iterArg_14, %42 : tensor<10x2xi1>
// CHECK-NEXT:     %44 = stablehlo.add %iterArg, %c_3 : tensor<10x2xi32>
// CHECK-NEXT:     stablehlo.return %44, %30, %36, %43 : tensor<10x2xi32>, tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xi1>
// CHECK-NEXT:   }
// CHECK-NEXT:   %7 = stablehlo.multiply %1, %cst_9 : tensor<10x2xf64>
// CHECK-NEXT:   %8 = stablehlo.power %7, %0 : tensor<10x2xf64>
// CHECK-NEXT:   %9 = stablehlo.add %0, %cst_10 : tensor<10x2xf64>
// CHECK-NEXT:   %10 = chlo.lgamma %9 : tensor<10x2xf64> -> tensor<10x2xf64>
// CHECK-NEXT:   %11 = stablehlo.exponential %10 : tensor<10x2xf64>
// CHECK-NEXT:   %12 = stablehlo.divide %8, %11 : tensor<10x2xf64>
// CHECK-NEXT:   %13 = stablehlo.multiply %6#1, %12 : tensor<10x2xf64>
// CHECK-NEXT:   %14 = stablehlo.floor %0 : tensor<10x2xf64>
// CHECK-NEXT:   %15 = stablehlo.compare  EQ, %0, %14 : (tensor<10x2xf64>, tensor<10x2xf64>) -> tensor<10x2xi1>
// CHECK-NEXT:   %16 = stablehlo.multiply %0, %cst_9 : tensor<10x2xf64>
// CHECK-NEXT:   %17 = stablehlo.floor %16 : tensor<10x2xf64>
// CHECK-NEXT:   %18 = stablehlo.multiply %17, %cst_7 : tensor<10x2xf64>
// CHECK-NEXT:   %19 = stablehlo.compare  EQ, %0, %18 : (tensor<10x2xf64>, tensor<10x2xf64>) -> tensor<10x2xi1>
// CHECK-NEXT:   %20 = stablehlo.select %19, %cst_10, %cst : tensor<10x2xi1>, tensor<10x2xf64>
// CHECK-NEXT:   %21 = stablehlo.multiply %20, %13 : tensor<10x2xf64>
// CHECK-NEXT:   %22 = stablehlo.select %15, %21, %cst_11 : tensor<10x2xi1>, tensor<10x2xf64>
// CHECK-NEXT:   %23 = stablehlo.select %3, %13, %22 : tensor<10x2xi1>, tensor<10x2xf64>
// CHECK-NEXT:   %24 = stablehlo.multiply %cst_6, %0 : tensor<10x2xf64>
// CHECK-NEXT:   %25 = stablehlo.sine %24 : tensor<10x2xf64>
// CHECK-NEXT:   %26 = stablehlo.multiply %cst_5, %25 : tensor<10x2xf64>
// CHECK-NEXT:   %27 = stablehlo.multiply %26, %13 : tensor<10x2xf64>
// CHECK-NEXT:   %28 = stablehlo.add %13, %27 : tensor<10x2xf64>
// CHECK-NEXT:   %29 = stablehlo.select %2, %23, %28 : tensor<10x2xi1>, tensor<10x2xf64>
// CHECK-NEXT:   return %29 : tensor<10x2xf64>
// CHECK-NEXT: }

// CHECK: %0 = call @special_besseli(%{{.*}}, %{{.*}}) : (tensor<10x2xf64>, tensor<10x2xf64>) -> tensor<10x2xf64>


func.func @main(%x: tensor<10x2xf64>, %y: tensor<10x2xf64>) -> (tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xf64>) {
  %a = stablehlo.constant dense<4.2> : tensor<10x2xf64>
  %order = stablehlo.constant dense<2.3> : tensor<10x2xf64>
  
  %b = "enzymexla.special.besseli"(%order, %a) : (tensor<10x2xf64>, tensor<10x2xf64>) -> (tensor<10x2xf64>)
  %c = "enzymexla.special.besseli"(%order, %b) : (tensor<10x2xf64>, tensor<10x2xf64>) -> (tensor<10x2xf64>)
  %d = "enzymexla.special.besselj"(%order, %b) : (tensor<10x2xf64>, tensor<10x2xf64>) -> (tensor<10x2xf64>)
  return %b, %c, %d : tensor<10x2xf64>, tensor<10x2xf64>, tensor<10x2xf64>
}
