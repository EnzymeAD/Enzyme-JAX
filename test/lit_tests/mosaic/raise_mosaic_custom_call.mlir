// RUN: enzymexlamlir-opt %s --split-input-file --raise-mosaic-custom-call | FileCheck %s --check-prefix=RAISE
// RUN: enzymexlamlir-opt %s --split-input-file --raise-mosaic-custom-call -lower-mosaic -canonicalize | FileCheck %s --check-prefix=ROUNDTRIP

module {
  // Real JAX Pallas `mosaic_boolean_constant` `@tpu_custom_call` payload with
  // serialized `stable_mosaic.*` bytecode, unaliased output, and frontend attrs.
  // RAISE-LABEL: func.func @boolean_const_kernel
  // ROUNDTRIP-LABEL: func.func @boolean_const_kernel
  func.func @boolean_const_kernel(%arg0: tensor<8x128xi32>, %arg1: tensor<8x128xi32>) -> tensor<8x128xi32> {
    // RAISE: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<8x128xi32>
    // RAISE: %[[CALL:.*]] = enzymexla_mosaic_ext.call @mosaic_module::@mosaic_module_inner::@kernel clusters in(%{{.*}}, %{{.*}}, %{{.*}}) blocks in(%{{.*}}, %{{.*}}, %{{.*}}) (%arg0, %arg1, %[[ZERO]])
    // RAISE-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]
    // RAISE-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 2, operand_tuple_indices = []>]
    // RAISE-SAME: xla_side_effect_free
    // RAISE-SAME: mhlo.frontend_attributes = {kernel_metadata = "{}"}
    // RAISE: return %[[CALL]] : tensor<8x128xi32>
    // ROUNDTRIP-NOT: stablehlo.constant
    // ROUNDTRIP: %[[CC:.*]] = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1)
    // ROUNDTRIP-SAME: backend_config = "{\22custom_call_config\22:{\22body\22:\22{{[A-Za-z0-9+/=]+}}\22,\22needs_layout_passes\22:true,\22serialization_format\22:1}}"
    // ROUNDTRIP-SAME: kernel_name = "kernel"
    // ROUNDTRIP-SAME: mhlo.frontend_attributes = {kernel_metadata = "{}"}
    // ROUNDTRIP-SAME: operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>]
    // ROUNDTRIP-NOT: output_operand_aliases
    // ROUNDTRIP-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]
    // ROUNDTRIP-NEXT: return %[[CC]] : tensor<8x128xi32>
    %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1) {
      backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSZ29vZ2xlMy10cnVuawABJQcBAwUBAwcDEwkLDQ8RExUXGQORcREBbw8HEw8PCxcLExMLDwsLDxsLDwszCwsLC4ULCwsPDx8LDw8XExMLDwsPCw8LCw8LFyMLDwtTCxMFA2EBERsHGyMPDwsbAloEHVlbHwMDCzkdGTsdGUMFGxcPegIjBR0DAws9AwM/QQUfEQsBBSEFIx1LTQMFISMVJQUlEQshBScDCykrLS8zFzUXFTcFKQEBBSsND2FmZmluZV9tYXA8KGQwLCBkMSkgLT4gKGQwLCBkMSk+AAUtBS8FMREDAR0bDSUBCQAAAAAFMxELBR0bRRcPegI3AwMLSSUFA/8FNR1PDQU3HVNVBTkdVw0FOwU9HV1fBT8XD3oCEQMHY2VnaWttBUERDQAFQyMJCSEBAAAAAQAAAAIAAAAAAAAABUUjCQEBI3RwdS5tZW1vcnlfc3BhY2U8dm1lbT4AJwUhAgQJAycFIQIEDRdvBSECBAkxAQICAQIEAQkFBwcHBwEEtgIFAREDHwcDAQUJEQMnBwMtVwcHAwcDBwMDAwcFAwMDAwcFAwMFBgcDAQcBBwkDAwcRAwEHBwcTAwUFCw0DAwkFAwMDAwkFAwMFBgkDAQcDERMDAwkRAwEHBwkTAwUFFRcDAx1HAwULBh0DBQUZGw0GUQMFBQ8dAwMBBQMDAwMBBQMDBQYBAwEHBSEjDwYBAwEDHwMDAREDAQcHARMDBQUlKREFAWEJJwUhIxMAAwYDAQUBAN4GRxEpCQsNBwkJCxULIyEdKQ8tCQsThQ0ZIxkVFxUXGR8PCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLmxvYWQAYXJpdGguY21waQBmdW5jLmZ1bmMAYXJpdGgueG9yaQBhcml0aC5vcmkAYXJpdGguZXh0dWkAdHB1LnZlY3Rvcl9zdG9yZQBmdW5jLnJldHVybgB2YWx1ZQB0aGlyZF9wYXJ0eS9weS9qYXgvdGVzdHMvcGFsbGFzL2V4cG9ydF9iYWNrX2NvbXBhdF9wYWxsYXNfdGVzdC5weQBzeW1fbmFtZQBnZXQ6AGdldABzdGFibGVfbW9zYWljLnZlcnNpb24Aa2VybmVsAGRpbWVuc2lvbl9zZW1hbnRpY3MAZnVuY3Rpb25fdHlwZQBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHByZWRpY2F0ZQBub3Q6AG5vdABvcjoAb3IAc3dhcDoAc3dhcABhZGQAb3BlcmFuZFNlZ21lbnRTaXplcwBzdHJpZGVzAA==\22, \22serialization_format\22: 1, \22needs_layout_passes\22: true}}",
      kernel_name = "kernel",
      mhlo.frontend_attributes = {kernel_metadata = "{}"},
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>]
    } : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi32>
    return %0 : tensor<8x128xi32>
  }
  // RAISE-LABEL: enzymexla_mosaic_ext.module @mosaic_module
  // RAISE: builtin.module @mosaic_module_inner
  // RAISE-NOT: stable_mosaic
  // RAISE: func.func @kernel(%arg0: memref<8x128xi32, #tpu.memory_space<vmem>>, %arg1: memref<8x128xi32, #tpu.memory_space<vmem>>, %arg2: memref<8x128xi32, #tpu.memory_space<vmem>>)
  // RAISE: vector.load
  // RAISE: tpu.vector_store
}

// -----

module {
  // Real JAX Pallas `mosaic_matmul` `@tpu_custom_call` payload with 3D
  // `iteration_bounds`, `window_params` (`@transform_0..2`), `scratch_operands`,
  // `tpu.matmul` upgrade from older serde version, and extra `implicit_sharding`
  // JSON metadata in `backend_config`.
  // RAISE-LABEL: func.func @matmul
  // ROUNDTRIP-LABEL: func.func @matmul
  func.func @matmul(%arg0: tensor<1024x512xf32>, %arg1: tensor<512x256xf32>) -> tensor<1024x256xf32> {
    // RAISE: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1024x256xf32>
    // RAISE: %[[CALL:.*]] = enzymexla_mosaic_ext.call @mosaic_module::@mosaic_module_inner::@matmul_kernel
    // RAISE-SAME: (%arg0, %arg1, %[[ZERO]])
    // RAISE-SAME: backend_config = "{\22implicit_sharding\22:{\22type\22:\22MANUAL\22}}"
    // RAISE: return %[[CALL]] : tensor<1024x256xf32>
    // ROUNDTRIP-NOT: stablehlo.constant
    // ROUNDTRIP: %[[CC:.*]] = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1)
    // ROUNDTRIP-SAME: implicit_sharding
    // ROUNDTRIP-SAME: MANUAL
    // ROUNDTRIP-SAME: kernel_name = "matmul_kernel"
    // ROUNDTRIP-NOT: output_operand_aliases
    // ROUNDTRIP-SAME: result_layouts = [dense<[1, 0]> : tensor<2xindex>]
    // ROUNDTRIP-NEXT: return %[[CC]] : tensor<1024x256xf32>
    %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1) {
      backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSZ29vZ2xlMy10cnVuawABLQkBAwUHAQMJAxkLDQ8RExUXGRsdHyED58ETAbkHEwsTCwsLDwsPDw8LC1MLDw8PDwsPDw8LCwsLExMPDxMPGwsPC0MLFwuFC3MLCwsLFxsLGwsbCxsbGw8LExMPEw8LCxMPExMTHwsTGwsLEwsPCxMTEwsTDwsTEwUHjZFhBwNZARMPBx8nDwcLKyMCZggfAwMLiwUjAwMLdwUlBScFKR15ewUrHSmnHSmrHSm3BS0FLyMJBSEAAQAAAAAAAAABAAAAAAAADREdhzkdETsdEY0dEY8FMR0RqREJAREJBQUzBTUFNwU5FwU7BxcFQyMdlZcRDQAXrRcLHbO1AwVHSQlLBTsRCQ0FPQMPT1ENU1dZWy1dLwlfYWMFPwEHubm7DQ9hZmZpbmVfbWFwPChkMCwgZDEpIC0+IChkMCwgZDEpPgAFQSMJBzEEAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAAFQwVFBUcFSQEHZWltAwUZZxsdCTEDBRlrGx0JMwMFGW8bHQk1AwUNHwkxAwUNHwkzAwUNHwk1EQEBBUsXBTsXAwMLfxEBBQMDNy0dhTkFTQVPAwM3LxEDARcFRQ0XBUcNAwMLkyUFCQAAAAAFURcFQ0EDBZs/nT8FUwVVAwOhvwVXHaU7BVkXBUMFFwVRKRcFUQUFWwMDC7ETCwEFXRcFPycXBT8JI3RwdS5kaW1lbnNpb25fc2VtYW50aWNzPHBhcmFsbGVsPgAjdHB1LmRpbWVuc2lvbl9zZW1hbnRpY3M8YXJiaXRyYXJ5PgAjdHB1Lm1lbW9yeV9zcGFjZTx2bWVtPgAjYXJpdGguZmFzdG1hdGg8bm9uZT4AAQICAycFAggCCAsXvQUCCAIIC1UBAgQLAQkFDwEBAQcHBwcBBQcBAQEFAQEEIgYFAREBRQcDAREHEQFNBwNHgw8BAQEBAQEHAQcBBwEHAQMDDwcDAQMDD30DAQMDDwcDAQ0HD4EDDQUFDxEGgwMBAxUDAyEHAwENByGJAw0FFxkTFCEDGwkDCx0DA0OvAwsZBkMDBQNHAwMXAwMDAwMXAwMDBQYXAwUHDUtNCwQXCUkNS00PAEEDAQUPAEEDAyMDAwMDAyMDAwMFBiMDBQcNHR8DAyUDAwMDAyUDAwMFBiUDBQcHIyUDAycDAwMDAycDAwMFBicDBQcJKSsDAz2RAwUVBz2ZAwUHJy0vFwejnwMFBSExAwMTAwMDAwMTAwMDBQYTAwUHDTU3CwQTCTMNNTcDAysDAwMDAysDAwMFBisDBQcNOz0DAxUDAwMDAxUDAwMFBhUDBQcLQUMLBBUJPwtBQwkAAQcRAXEHAwkLBwEBAQEBAQMDAQcDAQkEAQUBBQcRAXMHAwkLBwEBAQEBAQMDAQcDAQkEAQUFAwcRAXUHAwkLBwEBAQEBAQMDAQcDAQkEAQUBAwYDAQUBAO4JXyUFCxMdHRsNLQkdCyMhIykdLRUZGRkNHSULHQ0TcyMXFw8ZFRcbGRUZHw8NCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBhcml0aABtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLmxvYWQAZnVuYy5mdW5jAGZ1bmMucmV0dXJuAHZlY3Rvci5zdG9yZQBhcml0aC5jbXBpAHNjZi55aWVsZABhcml0aC5leHR1aQBzY2YuaWYAdHB1Lm1hdG11bABhcml0aC5hZGRmAHZlY3Rvci5icm9hZGNhc3QAdGhpcmRfcGFydHkvcHkvamF4L2V4cGVyaW1lbnRhbC9wYWxsYXMvb3BzL3RwdS9tYXRtdWwucHkAc3ltX25hbWUAdmFsdWUAZnVuY3Rpb25fdHlwZQAvZ2V0AHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL3N3YXAAdHJhbnNmb3JtXzAAdHJhbnNmb3JtXzEAdHJhbnNmb3JtXzIAcHJlZGljYXRlAHN0YWJsZV9tb3NhaWMudmVyc2lvbgBtYXRtdWxfa2VybmVsAGRpbWVuc2lvbl9zZW1hbnRpY3MAaXRlcmF0aW9uX2JvdW5kcwBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHdpbmRvd19wYXJhbXMAL2VxAC9jb252ZXJ0X2VsZW1lbnRfdHlwZQAvY29uZAAvZG90X2dlbmVyYWwAdHJhbnNwb3NlX2xocwB0cmFuc3Bvc2VfcmhzAGZhc3RtYXRoAC9hZGQALQAvYnJvYWRjYXN0X2luX2RpbQA=\22, \22serialization_format\22: 1, \22needs_layout_passes\22: true}, \22implicit_sharding\22: {\22type\22: \22MANUAL\22}}",
      kernel_name = "matmul_kernel",
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>]
    } : (tensor<1024x512xf32>, tensor<512x256xf32>) -> tensor<1024x256xf32>
    return %0 : tensor<1024x256xf32>
  }
  // RAISE-LABEL: enzymexla_mosaic_ext.module @mosaic_module
  // RAISE: func.func @matmul_kernel
  // RAISE-SAME: iteration_bounds =
  // RAISE-SAME: window_params =
  // RAISE: tpu.matmul
  // RAISE: func.func @transform_0
  // RAISE: func.func @transform_1
  // RAISE: func.func @transform_2
}

// -----

module {
  // Aliased `tpu_custom_call` (`output_operand_aliases` present): raising must
  // not synthesize a zero operand, and lowering preserves the exact alias.
  // RAISE-LABEL: func.func @aliased_kernel
  // ROUNDTRIP-LABEL: func.func @aliased_kernel
  func.func @aliased_kernel(%arg0: tensor<8x128xi32>, %arg1: tensor<8x128xi32>) -> tensor<8x128xi32> {
    // RAISE-NOT: stablehlo.constant dense<0> : tensor<8x128xi32>
    // RAISE: %[[CALL:.*]] = enzymexla_mosaic_ext.call @mosaic_module::@mosaic_module_inner::@inplace_kernel clusters in(%{{.*}}, %{{.*}}, %{{.*}}) blocks in(%{{.*}}, %{{.*}}, %{{.*}}) (%arg0, %arg1)
    // RAISE-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]
    // RAISE-NOT: xla_side_effect_free
    // ROUNDTRIP: %[[CC:.*]] = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1)
    // ROUNDTRIP-SAME: has_side_effect = true
    // ROUNDTRIP-SAME: kernel_name = "inplace_kernel"
    // ROUNDTRIP-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]
    // ROUNDTRIP-NEXT: return %[[CC]] : tensor<8x128xi32>
    %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1) {
      backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSZ29vZ2xlMy10cnVuawABJQcBAwUBAwcDEwkLDQ8RExUXGQORcREBbw8HEw8PCxcLExMLDwsLDxsLDwszCwsLC4ULCwsPDx8LDw8XExMLDwsPCw8LCw8LFyMLDwtTCxMFA2EBERsHGyMPDwsbAloEHVlbHwMDCzkdGTsdGUMFGxcPegIjBR0DAws9AwM/QQUfEQsBBSEFIx1LTQMFISMVJQUlEQshBScDCykrLS8zFzUXFTcFKQEBBSsND2FmZmluZV9tYXA8KGQwLCBkMSkgLT4gKGQwLCBkMSk+AAUtBS8FMREDAR0bDSUBCQAAAAAFMxELBR0bRRcPegI3AwMLSSUFA/8FNR1PDQU3HVNVBTkdVw0FOwU9HV1fBT8XD3oCEQMHY2VnaWttBUERDQAFQyMJCSEBAAAAAQAAAAIAAAAAAAAABUUjCQEBI3RwdS5tZW1vcnlfc3BhY2U8dm1lbT4AJwUhAgQJAycFIQIEDRdvBSECBAkxAQICAQIEAQkFBwcHBwEEtgIFAREDHwcDAQUJEQMnBwMtVwcHAwcDBwMDAwcFAwMDAwcFAwMFBgcDAQcBBwkDAwcRAwEHBwcTAwUFCw0DAwkFAwMDAwkFAwMFBgkDAQcDERMDAwkRAwEHBwkTAwUFFRcDAx1HAwULBh0DBQUZGw0GUQMFBQ8dAwMBBQMDAwMBBQMDBQYBAwEHBSEjDwYBAwEDHwMDAREDAQcHARMDBQUlKREFAWEJJwUhIxMAAwYDAQUBAN4GRxEpCQsNBwkJCxULIyEdKQ8tCQsThQ0ZIxkVFxUXGR8PCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLmxvYWQAYXJpdGguY21waQBmdW5jLmZ1bmMAYXJpdGgueG9yaQBhcml0aC5vcmkAYXJpdGguZXh0dWkAdHB1LnZlY3Rvcl9zdG9yZQBmdW5jLnJldHVybgB2YWx1ZQB0aGlyZF9wYXJ0eS9weS9qYXgvdGVzdHMvcGFsbGFzL2V4cG9ydF9iYWNrX2NvbXBhdF9wYWxsYXNfdGVzdC5weQBzeW1fbmFtZQBnZXQ6AGdldABzdGFibGVfbW9zYWljLnZlcnNpb24Aa2VybmVsAGRpbWVuc2lvbl9zZW1hbnRpY3MAZnVuY3Rpb25fdHlwZQBzY2FsYXJfcHJlZmV0Y2gAc2NyYXRjaF9vcGVyYW5kcwBtYWluAHByZWRpY2F0ZQBub3Q6AG5vdABvcjoAb3IAc3dhcDoAc3dhcABhZGQAb3BlcmFuZFNlZ21lbnRTaXplcwBzdHJpZGVzAA==\22, \22serialization_format\22: 1, \22needs_layout_passes\22: true}}",
      has_side_effect = true,
      kernel_name = "inplace_kernel",
      operand_layouts = [dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>],
      output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>],
      result_layouts = [dense<[1, 0]> : tensor<2xindex>]
    } : (tensor<8x128xi32>, tensor<8x128xi32>) -> tensor<8x128xi32>
    return %0 : tensor<8x128xi32>
  }
}
