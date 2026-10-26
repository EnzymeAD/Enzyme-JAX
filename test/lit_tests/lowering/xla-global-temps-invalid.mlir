// RUN: enzymexlamlir-opt %s --split-input-file --verify-diagnostics

module {
  // expected-error @+1 {{requires private visibility}}
  enzymexla.temp_alloc @bound : memref<i32, 1>
}

// -----

module {
  // expected-error @+1 {{requires a static shape}}
  enzymexla.temp_alloc "private" @bound : memref<?xi32, 1>
}

// -----

module {
  // expected-error @+1 {{requires an identity layout}}
  enzymexla.temp_alloc "private" @bound : memref<4xi32, strided<[2]>, 1>
}

// -----

module {
  // expected-error @+1 {{requires device memory space 1}}
  enzymexla.temp_alloc "private" @bound : memref<i32>
}

// -----

module {
  // expected-error @+1 {{requires device memory space 1}}
  enzymexla.temp_alloc "private" @bound : memref<i32, "device">
}

// -----

module {
  func.func @missing() -> memref<i32, 1> {
    // expected-error @+1 {{'bound' does not reference a temp_alloc declaration}}
    %device = enzymexla.get_global_temp @bound : memref<i32, 1>
    return %device : memref<i32, 1>
  }
}

// -----

module {
  enzymexla.temp_alloc "private" @bound : memref<i32, 1>
  func.func @mismatch() -> memref<i64, 1> {
    // expected-error @+1 {{result type does not match the temp_alloc type}}
    %device = enzymexla.get_global_temp @bound : memref<i64, 1>
    return %device : memref<i64, 1>
  }
}
