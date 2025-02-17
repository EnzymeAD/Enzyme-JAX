// RUN: enzymexlamlir-opt --enzyme='postpasses=enzyme-hlo-generate-td{create-module=true patterns=canonicalization<1>}' %s | FileCheck %s --check-prefixes=CHECK,GENERATE
// RUN: enzymexlamlir-opt --enzyme='postpasses=enzyme-hlo-generate-td{create-module=true patterns=canonicalization<1>},enzyme-consuming-transform-interpreter' %s | FileCheck %s --check-prefixes=CHECK,INTERP

// CHECK-LABEL: @square
func.func @square(%x: complex<f64>) -> complex<f64> {
  %next = complex.mul %x, %x : complex<f64>
  return %next : complex<f64>
}

// CHECK-LABEL: @dsquare
func.func @dsquare(%x: complex<f64>, %dx: complex<f64>) -> complex<f64> {
  // CHECK: call @fwddiffesquare
  %r = enzyme.fwddiff @square(%x, %dx) { activity=[#enzyme<activity enzyme_dup>], ret_activity=[#enzyme<activity enzyme_dupnoneed>] } : (complex<f64>, complex<f64>) -> complex<f64>
  return %r : complex<f64>
}

// CHECK:    func private @fwddiffesquare
// GENERATE:   module attributes {transform.with_named_sequence}
// GENERATE:     transform.named_sequence @__transform_main
// GENERATE:       transform.apply_patterns
// GENERATE:         transform.apply_patterns.canonicalization
// CHECK:      complex.mul
// CHECK:      complex.mul
// CHECK:      complex.add
// GENERATE:   complex.mul
// INTERP-NOT: complex.mul
