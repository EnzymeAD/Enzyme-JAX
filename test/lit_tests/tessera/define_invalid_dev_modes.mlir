// RUN: enzymexlamlir-opt %s -split-input-file -verify-diagnostics

// What tessera.define requires of devModes. llvm-to-tessera drops a marking
// that would break these, so only hand-written IR or a pass bug reaches them.

// expected-error @+1 {{devModes size (1) must match argModes size (2)}}
tessera.define private @size(!llvm.ptr, !llvm.ptr) attributes {argModes = [unit, unit], devModes = [#tessera.dir<in>], pure = false}

// -----

// expected-error @+1 {{devModes entry 0 must be an ArgDirectionAttr or UnitAttr}}
tessera.define private @entry(!llvm.ptr) attributes {argModes = [unit], devModes = [1 : i64], pure = false}

// -----

// expected-error @+1 {{argument 0 cannot be both lifted and a device argument}}
tessera.define private @lifted(!llvm.ptr) attributes {argModes = [{dir = #tessera.dir<in>, type = f64}], devModes = [#tessera.dir<in>], pure = false}

// -----

// expected-error @+1 {{devModes entry 0 must be for a pointer argument}}
tessera.define private @scalar(i32) attributes {argModes = [unit], devModes = [#tessera.dir<in>], pure = false}