// Regression test for #3962: K identical constants are the iter_args of an
// scf.for with an N-op body. Deduplicating them replaces K - 1 operands of the
// loop, and the greedy driver used to walk the whole body on each replacement,
// taking K x N time (well over the 60 s test timeout instead of a few seconds).

// RUN: awk -v K=120000 -v N=120000 'BEGIN { \
// RUN:   print "func.func @main(%%x: f32) -> f32 {"; \
// RUN:   print "  %%lb = arith.constant 0 : index"; \
// RUN:   print "  %%ub = arith.constant 10 : index"; \
// RUN:   print "  %%one = arith.constant 1 : index"; \
// RUN:   for (k = 0; k < K; k++) print "  %%c" k " = arith.constant 1.0 : f32"; \
// RUN:   ORS = ""; \
// RUN:   print "  %%r:" K " = scf.for %%i = %%lb to %%ub step %%one iter_args(%%a0 = %%c0"; \
// RUN:   for (k = 1; k < K; k++) print ", %%a" k " = %%c" k; \
// RUN:   print ") -> (f32"; \
// RUN:   for (k = 1; k < K; k++) print ", f32"; \
// RUN:   ORS = "\n"; \
// RUN:   print ") {"; \
// RUN:   print "    %%v0 = arith.addf %%a0, %%x : f32"; \
// RUN:   for (n = 1; n < N; n++) print "    %%v" n " = arith." (n %% 2 ? "mulf" : "addf") " %%v" (n - 1) ", %%x : f32"; \
// RUN:   ORS = ""; \
// RUN:   print "    scf.yield %%v" (N - 1); \
// RUN:   for (k = 1; k < K; k++) print ", %%a" k; \
// RUN:   print " : f32"; \
// RUN:   for (k = 1; k < K; k++) print ", f32"; \
// RUN:   ORS = "\n"; \
// RUN:   print ""; \
// RUN:   print "  }"; \
// RUN:   print "  return %%r#0 : f32"; \
// RUN:   print "}" }' | enzymexlamlir-opt --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK: scf.for
// CHECK: scf.yield
