// RUN: enzymexlamlir-opt %s -llvm-to-tessera -split-input-file -verify-diagnostics | FileCheck %s

// A tessera op marking that cannot become a valid tessera.define is a mistake
// in the annotation, not in the program. The function is left as an ordinary
// one, with a warning saying why, and everything else is lifted as usual: the
// mistake costs only the rewrites the marking would have allowed.

// The argument list must name every argument of the function.
module {
  // CHECK-LABEL: llvm.func @count_mismatch(
  // CHECK-NOT: tessera_op
  // expected-warning @+1 {{ignoring tessera op marking 'lib.f(x)': its argument list names 1 argument(s), but the function takes 2}}
  llvm.func @count_mismatch(%a: f64, %b: f64) -> f64 attributes {pure_tessera_op = "lib.f(x)"} {
    llvm.return %a : f64
  }

  // CHECK: tessera.define @lib.ok(
  llvm.func @ok(%a: f64) -> f64 attributes {pure_tessera_op = "lib.ok(x)"} {
    llvm.return %a : f64
  }

  // CHECK-LABEL: llvm.func @main
  // CHECK: llvm.call @count_mismatch
  // CHECK: tessera.call @lib.ok
  llvm.func @main(%x: f64) -> f64 {
    %0 = llvm.call @count_mismatch(%x, %x) : (f64, f64) -> f64
    %1 = llvm.call @ok(%0) : (f64) -> f64
    llvm.return %1 : f64
  }
}

// -----

// A marker must be one of val=in, val=out and val=inout.
module {
  // CHECK-LABEL: llvm.func @bad_marker(
  // CHECK-NOT: tessera_op
  // expected-warning @+1 {{argument 'x:val=sideways' has invalid marker 'val=sideways', expected :val=in, :val=out, or :val=inout}}
  llvm.func @bad_marker(%a: !llvm.ptr) -> f64 attributes {tessera_op = "lib.f(x:val=sideways):globals=0"} {
    %0 = llvm.load %a : !llvm.ptr -> f64
    llvm.return %0 : f64
  }
}

// -----

// Only an argument passed through a pointer has anything to lift.
module {
  // CHECK-LABEL: llvm.func @marked_value(
  // CHECK-NOT: tessera_op
  // expected-warning @+1 {{argument 'x:val=in' is marked, but is not passed through a pointer}}
  llvm.func @marked_value(%a: f64) -> f64 attributes {tessera_op = "lib.f(x:val=in):globals=0"} {
    llvm.return %a : f64
  }
}

// -----

// A marked argument needs the global the plugin emits for its type.
module {
  // CHECK-LABEL: llvm.func @missing_type(
  // CHECK-NOT: tessera_op
  // expected-warning @+1 {{no lifting entry found for arg at index 7}}
  llvm.func @missing_type(%a: !llvm.ptr) -> f64 attributes {tessera_op = "lib.f(x:val=in):globals=7"} {
    %0 = llvm.load %a : !llvm.ptr -> f64
    llvm.return %0 : f64
  }
}

// -----

// Two functions cannot both be the same tessera op; which one a rule meant
// cannot be known, so neither is lifted.
module {
  // CHECK-LABEL: llvm.func @first(
  // CHECK-NOT: tessera_op
  // expected-warning @+1 {{ignoring tessera op marking 'lib.f(x)': another symbol already has the name 'lib.f'}}
  llvm.func @first(%a: f64) -> f64 attributes {pure_tessera_op = "lib.f(x)"} {
    llvm.return %a : f64
  }

  // CHECK-LABEL: llvm.func @second(
  // CHECK-NOT: tessera_op
  // CHECK-NOT: tessera.define
  // expected-warning @+1 {{ignoring tessera op marking 'lib.f(x)': another symbol already has the name 'lib.f'}}
  llvm.func @second(%a: f32) -> f32 attributes {pure_tessera_op = "lib.f(x)"} {
    llvm.return %a : f32
  }
}
