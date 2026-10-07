// RUN: enzymexlamlir-opt %s --enzyme-checkpoint-scopes | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-checkpoint-scopes 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN

// enzyme_ad.jax.checkpoint(schedule, budget) is the JAX name scope
// enzyme_checkpoint[schedule,budget]: the stablehlo.while JAX lowers directly
// in it, named .../enzyme_checkpoint[schedule,budget]/while, is checkpointed
// with that schedule of enzyme/checkpoint_schedule.h. JAX transformations wrap
// the scope's name in theirs.

module {
  func.func @binomial(%x: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %v = %x) : tensor<i64>, tensor<f32>
    cond {
      %c10 = stablehlo.constant dense<10> : tensor<i64>
      %p = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %j = stablehlo.add %i, %c1 : tensor<i64>
      %w = stablehlo.sine %v : tensor<f32>
      stablehlo.return %j, %w : tensor<i64>, tensor<f32>
    } loc("jit(f)/enzyme_checkpoint[binomial,4]/while")
    return %r#1 : tensor<f32>
  }

  // CHECK-LABEL: func.func @binomial(
  // CHECK:         stablehlo.while{{.*}} attributes {enzyme.binomial_checkpointing, enzyme.checkpoint_period = 4 : i64, enzyme.enable_checkpointing = true}

  func.func @revolve_default(%x: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %v = %x) : tensor<i64>, tensor<f32>
    cond {
      %c10 = stablehlo.constant dense<10> : tensor<i64>
      %p = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %j = stablehlo.add %i, %c1 : tensor<i64>
      %w = stablehlo.sine %v : tensor<f32>
      stablehlo.return %j, %w : tensor<i64>, tensor<f32>
    } loc("jit(f)/enzyme_checkpoint[revolve,0]/while")
    return %r#1 : tensor<f32>
  }

  // CHECK-LABEL: func.func @revolve_default(
  // CHECK:         stablehlo.while{{.*}} attributes {enzyme.binomial_checkpointing, enzyme.enable_checkpointing = true}

  func.func @periodic_vmap(%x: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %v = %x) : tensor<i64>, tensor<f32>
    cond {
      %c10 = stablehlo.constant dense<10> : tensor<i64>
      %p = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %j = stablehlo.add %i, %c1 : tensor<i64>
      %w = stablehlo.sine %v : tensor<f32>
      stablehlo.return %j, %w : tensor<i64>, tensor<f32>
    } loc("jit(f)/vmap(enzyme_checkpoint[periodic,3])/while")
    return %r#1 : tensor<f32>
  }

  // CHECK-LABEL: func.func @periodic_vmap(
  // CHECK:         stablehlo.while{{.*}} attributes {enzyme.checkpoint_period = 3 : i64, enzyme.enable_checkpointing = true}

  func.func @transpose_jvp(%x: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %v = %x) : tensor<i64>, tensor<f32>
    cond {
      %c10 = stablehlo.constant dense<10> : tensor<i64>
      %p = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %j = stablehlo.add %i, %c1 : tensor<i64>
      %w = stablehlo.sine %v : tensor<f32>
      stablehlo.return %j, %w : tensor<i64>, tensor<f32>
    } loc("jit(f)/transpose(jvp(enzyme_checkpoint[binomial,2]))/while")
    return %r#1 : tensor<f32>
  }

  // CHECK-LABEL: func.func @transpose_jvp(
  // CHECK:         stablehlo.while{{.*}} attributes {enzyme.binomial_checkpointing, enzyme.checkpoint_period = 2 : i64, enzyme.enable_checkpointing = true}

  func.func @no_budget(%x: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %v = %x) : tensor<i64>, tensor<f32>
    cond {
      %c10 = stablehlo.constant dense<10> : tensor<i64>
      %p = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %j = stablehlo.add %i, %c1 : tensor<i64>
      %w = stablehlo.sine %v : tensor<f32>
      stablehlo.return %j, %w : tensor<i64>, tensor<f32>
    } loc("enzyme_checkpoint[binomial]/while")
    return %r#1 : tensor<f32>
  }

  // CHECK-LABEL: func.func @no_budget(
  // CHECK:         stablehlo.while{{.*}} attributes {enzyme.binomial_checkpointing, enzyme.enable_checkpointing = true}

  func.func @store_all(%x: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %v = %x) : tensor<i64>, tensor<f32>
    cond {
      %c10 = stablehlo.constant dense<10> : tensor<i64>
      %p = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %j = stablehlo.add %i, %c1 : tensor<i64>
      %w = stablehlo.sine %v : tensor<f32>
      stablehlo.return %j, %w : tensor<i64>, tensor<f32>
    } loc("jit(f)/enzyme_checkpoint[store_all,0]/while")
    return %r#1 : tensor<f32>
  }

  // CHECK-LABEL: func.func @store_all(
  // CHECK:         stablehlo.while{{.*}} attributes {enzyme.enable_checkpointing = false}

  func.func @nested(%x: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %v = %x) : tensor<i64>, tensor<f32>
    cond {
      %c10 = stablehlo.constant dense<10> : tensor<i64>
      %p = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %j = stablehlo.add %i, %c1 : tensor<i64>
      %w = stablehlo.sine %v : tensor<f32>
      %s:2 = stablehlo.while(%k = %c0, %u = %w) : tensor<i64>, tensor<f32>
      cond {
        %c2 = stablehlo.constant dense<2> : tensor<i64>
        %q = stablehlo.compare LT, %k, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %q : tensor<i1>
      } do {
        %c1b = stablehlo.constant dense<1> : tensor<i64>
        %k1 = stablehlo.add %k, %c1b : tensor<i64>
        stablehlo.return %k1, %u : tensor<i64>, tensor<f32>
      } loc("jit(f)/enzyme_checkpoint[binomial,4]/while/body/while")
      stablehlo.return %j, %w : tensor<i64>, tensor<f32>
    } loc("jit(f)/enzyme_checkpoint[binomial,4]/while")
    return %r#1 : tensor<f32>
  }

  // The loop in the marked one's body is not marked.
  // CHECK-LABEL: func.func @nested(
  // CHECK:         stablehlo.while{{.*}} attributes {enzyme.binomial_checkpointing, enzyme.checkpoint_period = 4 : i64, enzyme.enable_checkpointing = true}
  // CHECK:           stablehlo.while
  // CHECK-NOT:       attributes
  // CHECK:           cond

  func.func @explicit(%x: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %v = %x) : tensor<i64>, tensor<f32> attributes {enzyme.enable_checkpointing = true, enzyme.checkpoint_period = 7 : i64}
    cond {
      %c10 = stablehlo.constant dense<10> : tensor<i64>
      %p = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %j = stablehlo.add %i, %c1 : tensor<i64>
      %w = stablehlo.sine %v : tensor<f32>
      stablehlo.return %j, %w : tensor<i64>, tensor<f32>
    } loc("jit(f)/enzyme_checkpoint[binomial,4]/while")
    return %r#1 : tensor<f32>
  }

  // Attributes put there already (Reactant's @trace) stay as they are.
  // CHECK-LABEL: func.func @explicit(
  // CHECK:         stablehlo.while{{.*}} attributes {enzyme.checkpoint_period = 7 : i64, enzyme.enable_checkpointing = true}

  func.func @unmarked(%x: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %v = %x) : tensor<i64>, tensor<f32>
    cond {
      %c10 = stablehlo.constant dense<10> : tensor<i64>
      %p = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %j = stablehlo.add %i, %c1 : tensor<i64>
      %w = stablehlo.sine %v : tensor<f32>
      stablehlo.return %j, %w : tensor<i64>, tensor<f32>
    } loc("jit(f)/checkpoint[binomial,4]/while")
    return %r#1 : tensor<f32>
  }

  // CHECK-LABEL: func.func @unmarked(
  // CHECK:         stablehlo.while
  // CHECK-NOT:     enzyme.enable_checkpointing
  // CHECK:         return

  func.func @unknown(%x: tensor<f32>) -> tensor<f32> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %v = %x) : tensor<i64>, tensor<f32>
    cond {
      %c10 = stablehlo.constant dense<10> : tensor<i64>
      %p = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %p : tensor<i1>
    } do {
      %c1 = stablehlo.constant dense<1> : tensor<i64>
      %j = stablehlo.add %i, %c1 : tensor<i64>
      %w = stablehlo.sine %v : tensor<f32>
      stablehlo.return %j, %w : tensor<i64>, tensor<f32>
    } loc("jit(f)/enzyme_checkpoint[fastest,4]/while")
    return %r#1 : tensor<f32>
  }

  // The JAX name is all the location there is.
  // WARN: warning: loc("jit(f)/enzyme_checkpoint[fastest,4]/while"): unknown checkpointing request 'enzyme_checkpoint[fastest,4]'
  // WARN-NOT: warning
  // CHECK-LABEL: func.func @unknown(
  // CHECK-NOT:     enzyme.enable_checkpointing
  // CHECK:         return
}
