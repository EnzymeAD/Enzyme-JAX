// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=true})" %s | FileCheck %s --check-prefix=NONAN
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=no_nan_add_sub_simplify(1)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=NONAN
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{no_nan=false})" %s | FileCheck %s --check-prefix=NAN
// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=no_nan_add_sub_simplify(0)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s --check-prefix=NAN


func.func @t1(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<3xf64>
  %1 = stablehlo.subtract %arg1, %0 : tensor<3xf64>
  return %1 : tensor<3xf64>
}


// NONAN-LABEL:   @t1(
// NONAN-SAME:    %[[ARG0:.+]]: tensor<3xf64>, %[[ARG1:.+]]: tensor<3xf64>) -> tensor<3xf64> {
// NONAN-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[ARG0]] : tensor<3xf64>
// NONAN-NEXT:    return %[[NEG]] : tensor<3xf64>

// NAN-LABEL:  @t1(
// NAN-SAME:  %[[ARG0:.+]]: tensor<3xf64>, %[[ARG1:.+]]: tensor<3xf64>) -> tensor<3xf64> {
// NAN-NEXT:    %[[ADD:.+]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<3xf64>
// NAN-NEXT:    %[[SUB:.+]] = stablehlo.subtract %[[ARG1]], %[[ADD]] : tensor<3xf64>
// NAN-NEXT:    return %[[SUB]] : tensor<3xf64>


func.func @t2(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<3xf64>
  %1 = stablehlo.subtract %0, %arg1 : tensor<3xf64>
  return %1 : tensor<3xf64>
}

// NONAN-LABEL:  @t2(
// NONAN-SAME:  %[[ARG0:.+]]: tensor<3xf64>, %[[ARG1:.+]]: tensor<3xf64>) -> tensor<3xf64> {
// NONAN-NEXT:    return %[[ARG0]] : tensor<3xf64>

// NAN-LABEL:  @t2(
// NAN-SAME:  %[[ARG0:.+]]: tensor<3xf64>, %[[ARG1:.+]]: tensor<3xf64>) -> tensor<3xf64> {
// NAN-NEXT:    %[[ADD:.+]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<3xf64>
// NAN-NEXT:    %[[SUB:.+]] = stablehlo.subtract %[[ADD]], %[[ARG1]] : tensor<3xf64>
// NAN-NEXT:    return %[[SUB]] : tensor<3xf64>


func.func @t3(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
  %0 = stablehlo.add %arg1, %arg0 : tensor<3xf64>
  %1 = stablehlo.subtract %arg1, %0 : tensor<3xf64>
  return %1 : tensor<3xf64>
}

// NONAN-LABEL:  @t3(
// NONAN-SAME:  %[[ARG0:.+]]: tensor<3xf64>, %[[ARG1:.+]]: tensor<3xf64>) -> tensor<3xf64> {
// NONAN-NEXT:    %[[NEG:.+]] = stablehlo.negate %[[ARG0]] : tensor<3xf64>
// NONAN-NEXT:    return %[[NEG]] : tensor<3xf64>

// NAN-LABEL:  @t3(
// NAN-SAME:  %[[ARG0:.+]]: tensor<3xf64>, %[[ARG1:.+]]: tensor<3xf64>) -> tensor<3xf64> {
// NAN-NEXT:    %[[ADD:.+]] = stablehlo.add %[[ARG1]], %[[ARG0]] : tensor<3xf64>
// NAN-NEXT:    %[[SUB:.+]] = stablehlo.subtract %[[ARG1]], %[[ADD]] : tensor<3xf64>
// NAN-NEXT:    return %[[SUB]] : tensor<3xf64>


func.func @t4(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) -> tensor<3xf64> {
  %0 = stablehlo.add %arg1, %arg0 : tensor<3xf64>
  %1 = stablehlo.subtract %0, %arg1 : tensor<3xf64>
  return %1 : tensor<3xf64>
}

// NONAN-LABEL:  @t4(
// NONAN-SAME:  %[[ARG0:.+]]: tensor<3xf64>, %[[ARG1:.+]]: tensor<3xf64>) -> tensor<3xf64> {
// NONAN-NEXT:    return %[[ARG0]] : tensor<3xf64>

// NAN-LABEL:  @t4(
// NAN-SAME:  %[[ARG0:.+]]: tensor<3xf64>, %[[ARG1:.+]]: tensor<3xf64>) -> tensor<3xf64> {
// NAN-NEXT:    %[[ADD:.+]] = stablehlo.add %[[ARG1]], %[[ARG0]] : tensor<3xf64>
// NAN-NEXT:    %[[SUB:.+]] = stablehlo.subtract %[[ADD]], %[[ARG1]] : tensor<3xf64>
// NAN-NEXT:    return %[[SUB]] : tensor<3xf64>

func.func @t5(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi32> {
  %0 = stablehlo.add %arg1, %arg0 : tensor<3xi32>
  %1 = stablehlo.subtract %0, %arg1 : tensor<3xi32>
  return %1 : tensor<3xi32>
}

// NONAN-LABEL:  @t5(
// NONAN-SAME:  %[[ARG0:.+]]: tensor<3xi32>, %[[ARG1:.+]]: tensor<3xi32>) -> tensor<3xi32> {
// NONAN-NEXT:    return %[[ARG0]] : tensor<3xi32>

// NAN-LABEL:  @t5(
// NAN-SAME:  %[[ARG0:.+]]: tensor<3xi32>, %[[ARG1:.+]]: tensor<3xi32>) -> tensor<3xi32> {
// NAN-NEXT:    return %[[ARG0]] : tensor<3xi32>

func.func @t6(%arg0: tensor<3xi2>, %arg1: tensor<3xi2>) -> tensor<3xi2> {
  %0 = stablehlo.add %arg1, %arg0 : tensor<3xi2>
  %1 = stablehlo.subtract %0, %arg1 : tensor<3xi2>
  return %1 : tensor<3xi2>
}

// NONAN-LABEL:  @t6(
// NONAN-SAME:  %[[ARG0:.+]]: tensor<3xi2>, %[[ARG1:.+]]: tensor<3xi2>) -> tensor<3xi2> {
// NONAN-NEXT:    return %[[ARG0]] : tensor<3xi2>

// NAN-LABEL:  @t6(
// NAN-SAME:  %[[ARG0:.+]]: tensor<3xi2>, %[[ARG1:.+]]: tensor<3xi2>) -> tensor<3xi2> {
// NAN-NEXT:    return %[[ARG0]] : tensor<3xi2>

func.func @t7(%arg0: tensor<3xui32>, %arg1: tensor<3xui32>) -> tensor<3xui32> {
  %0 = stablehlo.add %arg1, %arg0 : tensor<3xui32>
  %1 = stablehlo.subtract %0, %arg1 : tensor<3xui32>
  return %1 : tensor<3xui32>
}

// NONAN-LABEL:  @t7(
// NONAN-SAME:  %[[ARG0:.+]]: tensor<3xui32>, %[[ARG1:.+]]: tensor<3xui32>) -> tensor<3xui32> {
// NONAN-NEXT:    return %[[ARG0]] : tensor<3xui32>

// NAN-LABEL:  @t7(
// NAN-SAME:  %[[ARG0:.+]]: tensor<3xui32>, %[[ARG1:.+]]: tensor<3xui32>) -> tensor<3xui32> {
// NAN-NEXT:    return %[[ARG0]] : tensor<3xui32>
