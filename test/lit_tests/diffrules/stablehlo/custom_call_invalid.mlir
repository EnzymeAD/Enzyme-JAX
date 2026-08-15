// RUN: not enzymexlamlir-opt %s --split-input-file --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const mode=ReverseModeCombined" 2>&1 | FileCheck %s

// Every way of getting the attributes or the reverse signature wrong, each of
// which would otherwise be a silently wrong gradient.

// enzyme.reverse naming a symbol that is not in the module.
func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = @nowhere,
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'enzyme.reverse' refers to 'nowhere', which is not a func.func in this module

// -----

// An operand index past the end of the operand list.
func.func @scale_rev(%x: tensor<4xf32>, %m: tensor<4xi1>, %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> tensor<4xf32> {
  func.return %dy : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 7>
  } : (tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'enzyme.active_operands' index 7 is out of range for a custom call with 2 operands

// -----

// The same operand listed twice.
func.func @scale_rev(%x: tensor<4xf32>, %m: tensor<4xi1>, %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  func.return %dy, %dy : tensor<4xf32>, tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0, 0>
  } : (tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'enzyme.active_operands' lists operand 0 more than once

// -----

// A boolean operand explicitly marked active. Integers accumulate through
// arith.addi rather than failing, so this has to be refused, not tolerated.
func.func @scale_rev(%x: tensor<4xf32>, %m: tensor<4xi1>, %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> tensor<4xi1> {
  %c = stablehlo.constant dense<false> : tensor<4xi1>
  func.return %c : tensor<4xi1>
}

func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 1>
  } : (tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'enzyme.active_operands' lists operand 1 of type tensor<4xi1>, which cannot carry a cotangent

// -----

// enzyme.active_operands given as something other than a dense i64 array.
func.func @scale_rev(%x: tensor<4xf32>, %m: tensor<4xi1>, %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> tensor<4xf32> {
  func.return %dy : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = 0 : i64
  } : (tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'enzyme.active_operands' must be a dense i64 array attribute

// -----

// enzyme.reverse given as a string rather than a symbol reference.
func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = "scale_rev",
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'enzyme.reverse' must be a symbol reference to a func.func

// -----

// Argument 0 does not have the primal operand's type.
func.func @scale_rev(%x: tensor<8xf32>, %m: tensor<4xi1>, %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> tensor<4xf32> {
  func.return %dy : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'scale_rev' argument 0 has type tensor<8xf32>, expected the primal operand type tensor<4xf32>

// -----

// The primal result slot has the wrong type.
func.func @scale_rev(%x: tensor<4xf32>, %m: tensor<4xi1>, %y: tensor<8xf32>,
                     %dy: tensor<4xf32>) -> tensor<4xf32> {
  func.return %dy : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'scale_rev' argument 2 has type tensor<8xf32>, expected the primal result type tensor<4xf32>

// -----

// The cotangent slot has the wrong type.
func.func @scale_rev(%x: tensor<4xf32>, %m: tensor<4xi1>, %y: tensor<4xf32>,
                     %dy: tensor<8xf32>) -> tensor<4xf32> {
  %c = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
  func.return %c : tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'scale_rev' argument 3 has type tensor<8xf32>, expected the cotangent type tensor<4xf32>

// -----

// One active operand, but two cotangents returned.
func.func @scale_rev(%x: tensor<4xf32>, %m: tensor<4xi1>, %y: tensor<4xf32>,
                     %dy: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  func.return %dy, %dy : tensor<4xf32>, tensor<4xf32>
}

func.func @main(%x : tensor<4xf32>, %mask : tensor<4xi1>) -> tensor<4xf32> {
  %y = stablehlo.custom_call @scale(%x, %mask) {
    enzyme.reverse = @scale_rev,
    enzyme.active_operands = array<i64: 0>
  } : (tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
  func.return %y : tensor<4xf32>
}

// CHECK: 'scale_rev' returns 2 value(s), expected one cotangent per active operand, i.e. 1
