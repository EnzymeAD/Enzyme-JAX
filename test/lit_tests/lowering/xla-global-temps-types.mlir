// RUN: enzymexlamlir-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(convert-polygeist-to-llvm{backend=xla-gpu})"

module {
  // expected-error @+1 {{unsupported XLA element type 'index'}}
  enzymexla.temp_alloc "private" @bound : memref<index, 1>
}

// -----

module {
  // expected-error @+1 {{unsupported XLA element type 'i128'}}
  enzymexla.temp_alloc "private" @bound : memref<i128, 1>
}

// -----

module {
  // expected-error @+1 {{unsupported XLA element type 'complex<f16>'}}
  enzymexla.temp_alloc "private" @bound : memref<complex<f16>, 1>
}
