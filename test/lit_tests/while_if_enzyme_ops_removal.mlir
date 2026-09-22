// RUN: enzymexlamlir-opt %s --remove-unnecessary-enzyme-ops | FileCheck %s

// remove-unnecessary-enzyme-ops on the IR the enzyme pass leaves for a counted
// stablehlo.while whose body updates gradients under a stablehlo.if (Reactant's
// output for such a traced loop, captured right before the removal pass; `%9`
// is the augmented forward loop, `%13` the reverse loop).
//
// The reverse `stablehlo.if` is processed before the reverse loop. Its
// IfOpEnzymeOpsRemover reroutes the branch-local enzyme.get / enzyme.set through
// gets placed before the if, new if results and sets placed after it -- and
// must erase the original nested ops while doing so, otherwise
// WhileOpEnzymeOpsRemover finds an enzyme.set that is not a direct child of the
// loop body and the pass fails ("had set op which was not a direct
// descendant"). The forward if hands its push (`%1`, the slice index) out
// through a new result, with a zero dummy on the other branch, and the loop
// then batches both caches over the trip count.

module {
  func.func private @rev_cond_sum(%arg0: tensor<6xf64>, %arg1: tensor<f64>) -> (tensor<6xf64>, tensor<6xf64>) {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_0 = stablehlo.constant dense<6> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %c_3 = stablehlo.constant dense<[false, true, true, true, true, true]> : tensor<6xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<1xf64>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<6xf64>
    %0 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<6xf64>>
    "enzyme.set"(%0, %cst_5) : (!enzyme.Gradient<tensor<6xf64>>, tensor<6xf64>) -> ()
    %1 = "enzyme.init"() : () -> !enzyme.Cache<tensor<i32>>
    %2 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<1xf64>>
    "enzyme.set"(%2, %cst_4) : (!enzyme.Gradient<tensor<1xf64>>, tensor<1xf64>) -> ()
    %3 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<f64>>
    "enzyme.set"(%3, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
    %4 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<f64>>
    "enzyme.set"(%4, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
    %5 = "enzyme.init"() : () -> !enzyme.Cache<tensor<i1>>
    %6 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<f64>>
    "enzyme.set"(%6, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
    %7 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<f64>>
    "enzyme.set"(%7, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
    %8 = "enzyme.init"() : () -> !enzyme.Gradient<tensor<f64>>
    "enzyme.set"(%8, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
    %9:2 = stablehlo.while(%iterArg = %c_1, %iterArg_6 = %cst) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut}
    cond {
      %15 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %15 : tensor<i1>
    } do {
      %15 = stablehlo.add %c, %iterArg : tensor<i64>
      %16 = stablehlo.dynamic_slice %c_3, %iterArg, sizes = [1] : (tensor<6xi1>, tensor<i64>) -> tensor<1xi1>
      %17 = stablehlo.reshape %16 : (tensor<1xi1>) -> tensor<i1>
      "enzyme.push"(%5, %17) : (!enzyme.Cache<tensor<i1>>, tensor<i1>) -> ()
      %18 = "stablehlo.if"(%17) ({
        %19 = stablehlo.convert %15 : (tensor<i64>) -> tensor<i32>
        %20 = stablehlo.subtract %19, %c_2 : tensor<i32>
        "enzyme.push"(%1, %20) : (!enzyme.Cache<tensor<i32>>, tensor<i32>) -> ()
        %21 = stablehlo.dynamic_slice %arg0, %20, sizes = [1] : (tensor<6xf64>, tensor<i32>) -> tensor<1xf64>
        %22 = stablehlo.reshape %21 : (tensor<1xf64>) -> tensor<f64>
        %23 = stablehlo.add %iterArg_6, %22 : tensor<f64>
        stablehlo.return %23 : tensor<f64>
      }, {
        stablehlo.return %iterArg_6 : tensor<f64>
      }) : (tensor<i1>) -> tensor<f64>
      stablehlo.return %15, %18 : tensor<i64>, tensor<f64>
    }
    %10 = "enzyme.get"(%8) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
    %11 = stablehlo.add %10, %arg1 : tensor<f64>
    "enzyme.set"(%8, %11) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
    %12 = "enzyme.get"(%8) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
    "enzyme.set"(%8, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
    %13:2 = stablehlo.while(%iterArg = %c_1, %iterArg_6 = %12) : tensor<i64>, tensor<f64>
    cond {
      %15 = stablehlo.compare LT, %iterArg, %c_0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %15 : tensor<i1>
    } do {
      %15 = stablehlo.add %iterArg, %c : tensor<i64>
      "enzyme.set"(%7, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
      "enzyme.set"(%6, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
      %16 = "enzyme.get"(%6) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
      %17 = stablehlo.add %16, %iterArg_6 : tensor<f64>
      "enzyme.set"(%6, %17) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
      %18 = "enzyme.pop"(%5) : (!enzyme.Cache<tensor<i1>>) -> tensor<i1>
      "stablehlo.if"(%18) ({
        %20 = "enzyme.get"(%6) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
        %21 = "enzyme.get"(%4) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
        %22 = stablehlo.add %21, %20 : tensor<f64>
        "enzyme.set"(%4, %22) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
        %23 = "enzyme.get"(%4) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
        "enzyme.set"(%4, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
        %24 = "enzyme.get"(%7) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
        %25 = stablehlo.add %24, %23 : tensor<f64>
        "enzyme.set"(%7, %25) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
        %26 = "enzyme.get"(%3) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
        %27 = stablehlo.add %26, %23 : tensor<f64>
        "enzyme.set"(%3, %27) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
        %28 = "enzyme.get"(%3) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
        "enzyme.set"(%3, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
        %29 = stablehlo.reshape %28 : (tensor<f64>) -> tensor<1xf64>
        %30 = "enzyme.get"(%2) : (!enzyme.Gradient<tensor<1xf64>>) -> tensor<1xf64>
        %31 = stablehlo.add %30, %29 : tensor<1xf64>
        "enzyme.set"(%2, %31) : (!enzyme.Gradient<tensor<1xf64>>, tensor<1xf64>) -> ()
        %32 = "enzyme.get"(%2) : (!enzyme.Gradient<tensor<1xf64>>) -> tensor<1xf64>
        "enzyme.set"(%2, %cst_4) : (!enzyme.Gradient<tensor<1xf64>>, tensor<1xf64>) -> ()
        %33 = "enzyme.pop"(%1) : (!enzyme.Cache<tensor<i32>>) -> tensor<i32>
        %34 = stablehlo.dynamic_update_slice %cst_5, %32, %33 : (tensor<6xf64>, tensor<1xf64>, tensor<i32>) -> tensor<6xf64>
        %35 = "enzyme.get"(%0) : (!enzyme.Gradient<tensor<6xf64>>) -> tensor<6xf64>
        %36 = stablehlo.add %35, %34 : tensor<6xf64>
        "enzyme.set"(%0, %36) : (!enzyme.Gradient<tensor<6xf64>>, tensor<6xf64>) -> ()
        stablehlo.return 
      }, {
        %20 = "enzyme.get"(%6) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
        %21 = "enzyme.get"(%7) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
        %22 = stablehlo.add %21, %20 : tensor<f64>
        "enzyme.set"(%7, %22) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
        stablehlo.return 
      }) : (tensor<i1>) -> ()
      %19 = "enzyme.get"(%7) : (!enzyme.Gradient<tensor<f64>>) -> tensor<f64>
      "enzyme.set"(%7, %cst) : (!enzyme.Gradient<tensor<f64>>, tensor<f64>) -> ()
      stablehlo.return %15, %19 : tensor<i64>, tensor<f64>
    }
    %14 = "enzyme.get"(%0) : (!enzyme.Gradient<tensor<6xf64>>) -> tensor<6xf64>
    return %arg0, %14 : tensor<6xf64>, tensor<6xf64>
  }
}

// CHECK-LABEL:   func.func private @rev_cond_sum(
// CHECK-NOT:       enzyme.
//
// Forward loop: counter, primal accumulator, the predicate cache and the slice
// index cache, both batched over the six iterations; the branch stays lazy and
// yields the index it pushed.
// CHECK:           %[[FWD:.+]]:4 = stablehlo.while(%iterArg = %{{.*}}, %iterArg_{{[0-9]+}} = %{{.*}}, %iterArg_{{[0-9]+}} = %{{.*}}, %iterArg_{{[0-9]+}} = %{{.*}}) : tensor<i64>, tensor<f64>, tensor<6xi1>, tensor<6xi32>
// CHECK:             stablehlo.dynamic_update_slice %{{.*}} : (tensor<6xi1>, tensor<1xi1>, tensor<i64>) -> tensor<6xi1>
// CHECK:             "stablehlo.if"
// CHECK:               stablehlo.dynamic_slice %arg0
// CHECK:               stablehlo.return %{{.*}}, %{{.*}} : tensor<f64>, tensor<i32>
// CHECK:             }, {
// CHECK:               stablehlo.return %{{.*}}, %{{.*}} : tensor<f64>, tensor<i32>
// CHECK:             })
// CHECK:             stablehlo.dynamic_update_slice %{{.*}} : (tensor<6xi32>, tensor<1xi32>, tensor<i64>) -> tensor<6xi32>
// CHECK-NOT:       enzyme.
//
// Reverse loop: the gradients set under the if are iteration arguments now, the
// caches are read back with the reverse counter, and the reverse if returns the
// updated gradients (the input adjoint is the tensor<6xf64> one).
// CHECK:           stablehlo.while(%iterArg = %{{.*}}) : tensor<i64>, {{.*}}tensor<6xf64>, tensor<i64>
// CHECK:             stablehlo.dynamic_slice %[[FWD]]#2, %{{.*}} : (tensor<6xi1>, tensor<i64>) -> tensor<1xi1>
// CHECK:             stablehlo.dynamic_slice %[[FWD]]#3, %{{.*}} : (tensor<6xi32>, tensor<i64>) -> tensor<1xi32>
// CHECK:             "stablehlo.if"
// CHECK:               stablehlo.dynamic_update_slice %{{.*}} : (tensor<6xf64>, tensor<1xf64>, tensor<i32>) -> tensor<6xf64>
// CHECK:               stablehlo.return %{{.*}} : tensor<f64>, tensor<f64>, tensor<f64>, tensor<1xf64>, tensor<6xf64>
// CHECK:             }, {
// CHECK:               stablehlo.return %{{.*}} : tensor<f64>, tensor<f64>, tensor<f64>, tensor<1xf64>, tensor<6xf64>
// CHECK:             })
// CHECK-NOT:       enzyme.
// CHECK:           return %arg0, %{{.*}} : tensor<6xf64>, tensor<6xf64>
