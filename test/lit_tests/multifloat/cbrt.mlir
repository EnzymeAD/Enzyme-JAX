// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" %s | FileCheck --check-prefix=FIRST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" %s | FileCheck --check-prefix=LAST %s
// RUN: enzymexlamlir-opt --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" %s | FileCheck --check-prefix=TUPLE %s
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=first" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=last" | stablehlo-translate - --interpret --allow-unregistered-dialect
// RUN: enzymexlamlir-opt %s --multi-float-conversion="source-type=f64 target-type=f32 concat-dimension=tuple" | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @cbrt(%arg0: tensor<3xf64>) -> tensor<3xf64> {
  // The f64 cbrt is lowered to the double-double Newton kernel, whose f32 seed
  // is a single stablehlo.cbrt on the high limb. So a cbrt op remains, but not
  // on the source f64 type.
  // FIRST-LABEL: @cbrt
  // FIRST: stablehlo.cbrt
  // FIRST-NOT: stablehlo.cbrt {{.*}} : tensor<3xf64>

  // LAST-LABEL: @cbrt
  // LAST: stablehlo.cbrt
  // LAST-NOT: stablehlo.cbrt {{.*}} : tensor<3xf64>

  // TUPLE-LABEL: @cbrt
  // TUPLE: stablehlo.cbrt
  // TUPLE-NOT: stablehlo.cbrt {{.*}} : tensor<3xf64>

  %0 = stablehlo.cbrt %arg0 : tensor<3xf64>
  return %0 : tensor<3xf64>
}

func.func @main() attributes {enzyme.no_multifloat} {
  // Perfect cubes converge exactly (seed is exact, Newton residual is 0):
  // cbrt(8)=2, cbrt(-27)=-3 (odd function → negatives are fine), cbrt(64)=4.
  %cst = stablehlo.constant dense<[8.0, -27.0, 64.0]> : tensor<3xf64>
  %expected = stablehlo.constant dense<[2.0, -3.0, 4.0]> : tensor<3xf64>
  %res = func.call @cbrt(%cst) : (tensor<3xf64>) -> tensor<3xf64>
  "check.expect_close"(%res, %expected) {max_ulp_difference = 3 : ui64} : (tensor<3xf64>, tensor<3xf64>) -> ()

  // cbrt(0) = 0 exactly (special-cased to avoid 0/0 in the Newton update).
  %z = stablehlo.constant dense<0.0> : tensor<3xf64>
  %resz = func.call @cbrt(%z) : (tensor<3xf64>) -> tensor<3xf64>
  "check.expect_close"(%resz, %z) {max_ulp_difference = 0 : ui64} : (tensor<3xf64>, tensor<3xf64>) -> ()
  return
}
