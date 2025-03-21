module @jit_predict_sequence attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1968x1024xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<79x1024xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg4: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg5: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg6: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg7: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg8: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg9: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg10: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg11: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg12: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg13: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg14: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg15: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg16: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg17: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg18: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg19: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg20: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg21: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg22: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg23: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg24: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg25: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg26: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg27: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg28: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg29: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg30: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg31: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg32: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg33: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg34: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg35: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg36: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg37: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg38: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg39: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg40: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg41: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg42: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg43: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg44: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg45: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg46: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg47: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg48: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg49: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg50: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg51: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg52: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg53: tensor<128xf32> {mhlo.sharding = "{replicated}"}, %arg54: tensor<1024x128xf32> {mhlo.sharding = "{replicated}"}, %arg55: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg56: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg57: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg58: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg59: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg60: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg61: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg62: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg63: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg64: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg65: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg66: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg67: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg68: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg69: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg70: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg71: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg72: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg73: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg74: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg75: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg76: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg77: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg78: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg79: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg80: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg81: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg82: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg83: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg84: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg85: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg86: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg87: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg88: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg89: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg90: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg91: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg92: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg93: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg94: tensor<33x79xi32>) -> (tensor<33x79x128xf32> {jax.result_info = ""}) {
    %0 = call @apply_fn(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94) : (tensor<1968x1024xf32>, tensor<79x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<128xf32>, tensor<1024x128xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<33x79xi32>) -> tensor<33x79x128xf32>
    return %0 : tensor<33x79x128xf32>
  }
  func.func private @apply_fn(%arg0: tensor<1968x1024xf32>, %arg1: tensor<79x1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<1024xf32>, %arg9: tensor<1024xf32>, %arg10: tensor<1024xf32>, %arg11: tensor<1024xf32>, %arg12: tensor<1024xf32>, %arg13: tensor<1024xf32>, %arg14: tensor<1024xf32>, %arg15: tensor<1024xf32>, %arg16: tensor<1024xf32>, %arg17: tensor<1024xf32>, %arg18: tensor<1024xf32>, %arg19: tensor<1024xf32>, %arg20: tensor<1024xf32>, %arg21: tensor<1024xf32>, %arg22: tensor<1024xf32>, %arg23: tensor<1024xf32>, %arg24: tensor<1024xf32>, %arg25: tensor<1024xf32>, %arg26: tensor<1024xf32>, %arg27: tensor<1024xf32>, %arg28: tensor<1024xf32>, %arg29: tensor<1024xf32>, %arg30: tensor<1024xf32>, %arg31: tensor<1024xf32>, %arg32: tensor<1024xf32>, %arg33: tensor<1024xf32>, %arg34: tensor<1024xf32>, %arg35: tensor<1024xf32>, %arg36: tensor<1024x4096xf32>, %arg37: tensor<1024x4096xf32>, %arg38: tensor<1024x4096xf32>, %arg39: tensor<4096x1024xf32>, %arg40: tensor<1024x4096xf32>, %arg41: tensor<1024x4096xf32>, %arg42: tensor<4096x1024xf32>, %arg43: tensor<1024x4096xf32>, %arg44: tensor<1024x4096xf32>, %arg45: tensor<4096x1024xf32>, %arg46: tensor<1024x4096xf32>, %arg47: tensor<1024x4096xf32>, %arg48: tensor<4096x1024xf32>, %arg49: tensor<4096x1024xf32>, %arg50: tensor<1024x4096xf32>, %arg51: tensor<1024x4096xf32>, %arg52: tensor<4096x1024xf32>, %arg53: tensor<128xf32>, %arg54: tensor<1024x128xf32>, %arg55: tensor<1024x4096xf32>, %arg56: tensor<1024x4096xf32>, %arg57: tensor<4096x1024xf32>, %arg58: tensor<1024x4096xf32>, %arg59: tensor<1024x4096xf32>, %arg60: tensor<4096x1024xf32>, %arg61: tensor<1024x4096xf32>, %arg62: tensor<1024x1024xf32>, %arg63: tensor<1024x1024xf32>, %arg64: tensor<1024x1024xf32>, %arg65: tensor<1024x1024xf32>, %arg66: tensor<1024x1024xf32>, %arg67: tensor<1024x1024xf32>, %arg68: tensor<1024x1024xf32>, %arg69: tensor<1024x1024xf32>, %arg70: tensor<1024x1024xf32>, %arg71: tensor<1024x1024xf32>, %arg72: tensor<1024x1024xf32>, %arg73: tensor<1024x1024xf32>, %arg74: tensor<1024x1024xf32>, %arg75: tensor<1024x1024xf32>, %arg76: tensor<1024x1024xf32>, %arg77: tensor<1024x1024xf32>, %arg78: tensor<1024x1024xf32>, %arg79: tensor<1024x1024xf32>, %arg80: tensor<1024x1024xf32>, %arg81: tensor<1024x1024xf32>, %arg82: tensor<1024x1024xf32>, %arg83: tensor<1024x1024xf32>, %arg84: tensor<1024x1024xf32>, %arg85: tensor<1024x1024xf32>, %arg86: tensor<1024x1024xf32>, %arg87: tensor<1024x1024xf32>, %arg88: tensor<1024x1024xf32>, %arg89: tensor<1024x1024xf32>, %arg90: tensor<1024x1024xf32>, %arg91: tensor<1024x1024xf32>, %arg92: tensor<1024x1024xf32>, %arg93: tensor<1024x1024xf32>, %arg94: tensor<33x79xi32>) -> tensor<33x79x128xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %cst_2 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %c = stablehlo.constant dense<79> : tensor<i32>
    %cst_4 = stablehlo.constant dense<1.024000e+03> : tensor<f32>
    %c_5 = stablehlo.constant dense<1968> : tensor<i32>
    %c_6 = stablehlo.constant dense<0> : tensor<i32>
    %c_7 = stablehlo.constant dense<0> : tensor<ui8>
    %0 = stablehlo.broadcast_in_dim %c_7, dims = [] : (tensor<ui8>) -> tensor<33x1xui8>
    %1 = stablehlo.convert %0 : (tensor<33x1xui8>) -> tensor<33x1xi32>
    %2 = stablehlo.concatenate %1, %arg94, dim = 1 : (tensor<33x1xi32>, tensor<33x79xi32>) -> tensor<33x80xi32>
    %3 = stablehlo.slice %2 [0:33, 0:79] : (tensor<33x80xi32>) -> tensor<33x79xi32>
    %4 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<33x79xi32>
    %5 = stablehlo.compare  LT, %3, %4,  SIGNED : (tensor<33x79xi32>, tensor<33x79xi32>) -> tensor<33x79xi1>
    %6 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<33x79xi32>
    %7 = stablehlo.add %3, %6 : tensor<33x79xi32>
    %8 = stablehlo.select %5, %7, %3 : tensor<33x79xi1>, tensor<33x79xi32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<33x79xi32>) -> tensor<33x79x1xi32>
    %10 = "stablehlo.gather"(%arg0, %9) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1024>}> : (tensor<1968x1024xf32>, tensor<33x79x1xi32>) -> tensor<33x79x1024xf32>
    %11 = stablehlo.sqrt %cst_4 : tensor<f32>
    %12 = stablehlo.convert %11 : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<33x79x1024xf32>
    %14 = stablehlo.multiply %10, %13 : tensor<33x79x1024xf32>
    %15 = stablehlo.iota dim = 0 : tensor<79xi32>
    %16 = stablehlo.broadcast_in_dim %c_6, dims = [] : (tensor<i32>) -> tensor<79xi32>
    %17 = stablehlo.compare  LT, %15, %16,  SIGNED : (tensor<79xi32>, tensor<79xi32>) -> tensor<79xi1>
    %18 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<79xi32>
    %19 = stablehlo.add %15, %18 : tensor<79xi32>
    %20 = stablehlo.select %17, %19, %15 : tensor<79xi1>, tensor<79xi32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0] : (tensor<79xi32>) -> tensor<79x1xi32>
    %22 = "stablehlo.gather"(%arg1, %21) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1024>}> : (tensor<79x1024xf32>, tensor<79x1xi32>) -> tensor<79x1024xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [1, 2] : (tensor<79x1024xf32>) -> tensor<1x79x1024xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0, 1, 2] : (tensor<1x79x1024xf32>) -> tensor<33x79x1024xf32>
    %25 = stablehlo.add %14, %24 : tensor<33x79x1024xf32>
    %26 = stablehlo.reduce(%25 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %28 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %29 = stablehlo.divide %27, %28 : tensor<33x79x1xf32>
    %30 = call @_var(%25, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %31 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %32 = stablehlo.add %30, %31 : tensor<33x79x1xf32>
    %33 = stablehlo.rsqrt %32 : tensor<33x79x1xf32>
    %34 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %36 = stablehlo.broadcast_in_dim %33, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %37 = stablehlo.multiply %35, %36 : tensor<33x79x1024xf32>
    %38 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %39 = stablehlo.subtract %25, %38 : tensor<33x79x1024xf32>
    %40 = stablehlo.multiply %37, %39 : tensor<33x79x1024xf32>
    %41 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %43 = stablehlo.add %40, %42 : tensor<33x79x1024xf32>
    %44 = stablehlo.dot_general %43, %arg62, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %45 = stablehlo.dot_general %43, %arg63, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %46 = stablehlo.dot_general %43, %arg64, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %47 = stablehlo.reshape %44 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %48 = stablehlo.reshape %45 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %49 = stablehlo.reshape %46 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %50 = stablehlo.dot_general %47, %48, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %51 = stablehlo.sqrt %cst_1 : tensor<f32>
    %52 = stablehlo.divide %cst_0, %51 : tensor<f32>
    %53 = stablehlo.convert %52 : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %55 = stablehlo.multiply %50, %54 : tensor<33x8x79x79xf32>
    %56 = stablehlo.reduce(%55 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %57 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %58 = stablehlo.maximum %57, %56 : tensor<33x8x79xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %61 = stablehlo.subtract %55, %60 : tensor<33x8x79x79xf32>
    %62 = stablehlo.exponential %61 : tensor<33x8x79x79xf32>
    %63 = stablehlo.reduce(%62 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %64 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %65 = stablehlo.broadcast_in_dim %64, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %66 = stablehlo.divide %62, %65 : tensor<33x8x79x79xf32>
    %67 = stablehlo.dot_general %49, %66, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %68 = stablehlo.transpose %67, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %69 = stablehlo.reshape %68 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %70 = stablehlo.dot_general %69, %arg65, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %71 = stablehlo.add %25, %70 : tensor<33x79x1024xf32>
    %72 = stablehlo.reduce(%71 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %74 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %75 = stablehlo.divide %73, %74 : tensor<33x79x1xf32>
    %76 = call @_var(%71, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %77 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %78 = stablehlo.add %76, %77 : tensor<33x79x1xf32>
    %79 = stablehlo.rsqrt %78 : tensor<33x79x1xf32>
    %80 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %82 = stablehlo.broadcast_in_dim %79, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %83 = stablehlo.multiply %81, %82 : tensor<33x79x1024xf32>
    %84 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %85 = stablehlo.subtract %71, %84 : tensor<33x79x1024xf32>
    %86 = stablehlo.multiply %83, %85 : tensor<33x79x1024xf32>
    %87 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %89 = stablehlo.add %86, %88 : tensor<33x79x1024xf32>
    %90 = stablehlo.dot_general %89, %arg36, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %91 = stablehlo.dot_general %89, %arg37, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %92 = call @silu(%90) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %93 = stablehlo.multiply %92, %91 : tensor<33x79x4096xf32>
    %94 = stablehlo.dot_general %93, %arg48, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %95 = stablehlo.add %71, %94 : tensor<33x79x1024xf32>
    %96 = stablehlo.reduce(%95 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %98 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %99 = stablehlo.divide %97, %98 : tensor<33x79x1xf32>
    %100 = call @_var(%95, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %101 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %102 = stablehlo.add %100, %101 : tensor<33x79x1xf32>
    %103 = stablehlo.rsqrt %102 : tensor<33x79x1xf32>
    %104 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %106 = stablehlo.broadcast_in_dim %103, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %107 = stablehlo.multiply %105, %106 : tensor<33x79x1024xf32>
    %108 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %109 = stablehlo.subtract %95, %108 : tensor<33x79x1024xf32>
    %110 = stablehlo.multiply %107, %109 : tensor<33x79x1024xf32>
    %111 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %113 = stablehlo.add %110, %112 : tensor<33x79x1024xf32>
    %114 = stablehlo.dot_general %113, %arg66, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %115 = stablehlo.dot_general %113, %arg67, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %116 = stablehlo.dot_general %113, %arg68, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %117 = stablehlo.reshape %114 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %118 = stablehlo.reshape %115 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %119 = stablehlo.reshape %116 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %120 = stablehlo.dot_general %117, %118, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %121 = stablehlo.sqrt %cst_1 : tensor<f32>
    %122 = stablehlo.divide %cst_0, %121 : tensor<f32>
    %123 = stablehlo.convert %122 : tensor<f32>
    %124 = stablehlo.broadcast_in_dim %123, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %125 = stablehlo.multiply %120, %124 : tensor<33x8x79x79xf32>
    %126 = stablehlo.reduce(%125 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %127 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %128 = stablehlo.maximum %127, %126 : tensor<33x8x79xf32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %131 = stablehlo.subtract %125, %130 : tensor<33x8x79x79xf32>
    %132 = stablehlo.exponential %131 : tensor<33x8x79x79xf32>
    %133 = stablehlo.reduce(%132 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %134 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %135 = stablehlo.broadcast_in_dim %134, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %136 = stablehlo.divide %132, %135 : tensor<33x8x79x79xf32>
    %137 = stablehlo.dot_general %119, %136, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %138 = stablehlo.transpose %137, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %139 = stablehlo.reshape %138 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %140 = stablehlo.dot_general %139, %arg69, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %141 = stablehlo.add %95, %140 : tensor<33x79x1024xf32>
    %142 = stablehlo.reduce(%141 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %143 = stablehlo.broadcast_in_dim %142, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %144 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %145 = stablehlo.divide %143, %144 : tensor<33x79x1xf32>
    %146 = call @_var(%141, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %147 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %148 = stablehlo.add %146, %147 : tensor<33x79x1xf32>
    %149 = stablehlo.rsqrt %148 : tensor<33x79x1xf32>
    %150 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %151 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %152 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %153 = stablehlo.multiply %151, %152 : tensor<33x79x1024xf32>
    %154 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %155 = stablehlo.subtract %141, %154 : tensor<33x79x1024xf32>
    %156 = stablehlo.multiply %153, %155 : tensor<33x79x1024xf32>
    %157 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %159 = stablehlo.add %156, %158 : tensor<33x79x1024xf32>
    %160 = stablehlo.dot_general %159, %arg55, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %161 = stablehlo.dot_general %159, %arg56, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %162 = call @silu(%160) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %163 = stablehlo.multiply %162, %161 : tensor<33x79x4096xf32>
    %164 = stablehlo.dot_general %163, %arg57, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %165 = stablehlo.add %141, %164 : tensor<33x79x1024xf32>
    %166 = stablehlo.reduce(%165 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %168 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %169 = stablehlo.divide %167, %168 : tensor<33x79x1xf32>
    %170 = call @_var(%165, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %171 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %172 = stablehlo.add %170, %171 : tensor<33x79x1xf32>
    %173 = stablehlo.rsqrt %172 : tensor<33x79x1xf32>
    %174 = stablehlo.broadcast_in_dim %arg25, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %175 = stablehlo.broadcast_in_dim %174, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %176 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %177 = stablehlo.multiply %175, %176 : tensor<33x79x1024xf32>
    %178 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %179 = stablehlo.subtract %165, %178 : tensor<33x79x1024xf32>
    %180 = stablehlo.multiply %177, %179 : tensor<33x79x1024xf32>
    %181 = stablehlo.broadcast_in_dim %arg24, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %183 = stablehlo.add %180, %182 : tensor<33x79x1024xf32>
    %184 = stablehlo.dot_general %183, %arg70, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %185 = stablehlo.dot_general %183, %arg71, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %186 = stablehlo.dot_general %183, %arg72, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %187 = stablehlo.reshape %184 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %188 = stablehlo.reshape %185 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %189 = stablehlo.reshape %186 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %190 = stablehlo.dot_general %187, %188, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %191 = stablehlo.sqrt %cst_1 : tensor<f32>
    %192 = stablehlo.divide %cst_0, %191 : tensor<f32>
    %193 = stablehlo.convert %192 : tensor<f32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %195 = stablehlo.multiply %190, %194 : tensor<33x8x79x79xf32>
    %196 = stablehlo.reduce(%195 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %197 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %198 = stablehlo.maximum %197, %196 : tensor<33x8x79xf32>
    %199 = stablehlo.broadcast_in_dim %198, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %200 = stablehlo.broadcast_in_dim %199, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %201 = stablehlo.subtract %195, %200 : tensor<33x8x79x79xf32>
    %202 = stablehlo.exponential %201 : tensor<33x8x79x79xf32>
    %203 = stablehlo.reduce(%202 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %204 = stablehlo.broadcast_in_dim %203, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %205 = stablehlo.broadcast_in_dim %204, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %206 = stablehlo.divide %202, %205 : tensor<33x8x79x79xf32>
    %207 = stablehlo.dot_general %189, %206, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %208 = stablehlo.transpose %207, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %209 = stablehlo.reshape %208 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %210 = stablehlo.dot_general %209, %arg73, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %211 = stablehlo.add %165, %210 : tensor<33x79x1024xf32>
    %212 = stablehlo.reduce(%211 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %214 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %215 = stablehlo.divide %213, %214 : tensor<33x79x1xf32>
    %216 = call @_var(%211, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %217 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %218 = stablehlo.add %216, %217 : tensor<33x79x1xf32>
    %219 = stablehlo.rsqrt %218 : tensor<33x79x1xf32>
    %220 = stablehlo.broadcast_in_dim %arg27, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %222 = stablehlo.broadcast_in_dim %219, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %223 = stablehlo.multiply %221, %222 : tensor<33x79x1024xf32>
    %224 = stablehlo.broadcast_in_dim %215, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %225 = stablehlo.subtract %211, %224 : tensor<33x79x1024xf32>
    %226 = stablehlo.multiply %223, %225 : tensor<33x79x1024xf32>
    %227 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %229 = stablehlo.add %226, %228 : tensor<33x79x1024xf32>
    %230 = stablehlo.dot_general %229, %arg58, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %231 = stablehlo.dot_general %229, %arg59, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %232 = call @silu(%230) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %233 = stablehlo.multiply %232, %231 : tensor<33x79x4096xf32>
    %234 = stablehlo.dot_general %233, %arg60, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %235 = stablehlo.add %211, %234 : tensor<33x79x1024xf32>
    %236 = stablehlo.reduce(%235 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %237 = stablehlo.broadcast_in_dim %236, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %238 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %239 = stablehlo.divide %237, %238 : tensor<33x79x1xf32>
    %240 = call @_var(%235, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %241 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %242 = stablehlo.add %240, %241 : tensor<33x79x1xf32>
    %243 = stablehlo.rsqrt %242 : tensor<33x79x1xf32>
    %244 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %245 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %246 = stablehlo.broadcast_in_dim %243, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %247 = stablehlo.multiply %245, %246 : tensor<33x79x1024xf32>
    %248 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %249 = stablehlo.subtract %235, %248 : tensor<33x79x1024xf32>
    %250 = stablehlo.multiply %247, %249 : tensor<33x79x1024xf32>
    %251 = stablehlo.broadcast_in_dim %arg28, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %252 = stablehlo.broadcast_in_dim %251, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %253 = stablehlo.add %250, %252 : tensor<33x79x1024xf32>
    %254 = stablehlo.dot_general %253, %arg74, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %255 = stablehlo.dot_general %253, %arg75, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %256 = stablehlo.dot_general %253, %arg76, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %257 = stablehlo.reshape %254 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %258 = stablehlo.reshape %255 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %259 = stablehlo.reshape %256 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %260 = stablehlo.dot_general %257, %258, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %261 = stablehlo.sqrt %cst_1 : tensor<f32>
    %262 = stablehlo.divide %cst_0, %261 : tensor<f32>
    %263 = stablehlo.convert %262 : tensor<f32>
    %264 = stablehlo.broadcast_in_dim %263, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %265 = stablehlo.multiply %260, %264 : tensor<33x8x79x79xf32>
    %266 = stablehlo.reduce(%265 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %267 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %268 = stablehlo.maximum %267, %266 : tensor<33x8x79xf32>
    %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %270 = stablehlo.broadcast_in_dim %269, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %271 = stablehlo.subtract %265, %270 : tensor<33x8x79x79xf32>
    %272 = stablehlo.exponential %271 : tensor<33x8x79x79xf32>
    %273 = stablehlo.reduce(%272 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %274 = stablehlo.broadcast_in_dim %273, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %276 = stablehlo.divide %272, %275 : tensor<33x8x79x79xf32>
    %277 = stablehlo.dot_general %259, %276, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %278 = stablehlo.transpose %277, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %279 = stablehlo.reshape %278 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %280 = stablehlo.dot_general %279, %arg77, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %281 = stablehlo.add %235, %280 : tensor<33x79x1024xf32>
    %282 = stablehlo.reduce(%281 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %283 = stablehlo.broadcast_in_dim %282, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %284 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %285 = stablehlo.divide %283, %284 : tensor<33x79x1xf32>
    %286 = call @_var(%281, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %287 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %288 = stablehlo.add %286, %287 : tensor<33x79x1xf32>
    %289 = stablehlo.rsqrt %288 : tensor<33x79x1xf32>
    %290 = stablehlo.broadcast_in_dim %arg31, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %291 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %292 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %293 = stablehlo.multiply %291, %292 : tensor<33x79x1024xf32>
    %294 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %295 = stablehlo.subtract %281, %294 : tensor<33x79x1024xf32>
    %296 = stablehlo.multiply %293, %295 : tensor<33x79x1024xf32>
    %297 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %299 = stablehlo.add %296, %298 : tensor<33x79x1024xf32>
    %300 = stablehlo.dot_general %299, %arg61, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %301 = stablehlo.dot_general %299, %arg38, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %302 = call @silu(%300) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %303 = stablehlo.multiply %302, %301 : tensor<33x79x4096xf32>
    %304 = stablehlo.dot_general %303, %arg39, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %305 = stablehlo.add %281, %304 : tensor<33x79x1024xf32>
    %306 = stablehlo.reduce(%305 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %308 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %309 = stablehlo.divide %307, %308 : tensor<33x79x1xf32>
    %310 = call @_var(%305, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %311 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %312 = stablehlo.add %310, %311 : tensor<33x79x1xf32>
    %313 = stablehlo.rsqrt %312 : tensor<33x79x1xf32>
    %314 = stablehlo.broadcast_in_dim %arg33, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %316 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %317 = stablehlo.multiply %315, %316 : tensor<33x79x1024xf32>
    %318 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %319 = stablehlo.subtract %305, %318 : tensor<33x79x1024xf32>
    %320 = stablehlo.multiply %317, %319 : tensor<33x79x1024xf32>
    %321 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %323 = stablehlo.add %320, %322 : tensor<33x79x1024xf32>
    %324 = stablehlo.dot_general %323, %arg78, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %325 = stablehlo.dot_general %323, %arg79, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %326 = stablehlo.dot_general %323, %arg80, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %327 = stablehlo.reshape %324 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %328 = stablehlo.reshape %325 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %329 = stablehlo.reshape %326 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %330 = stablehlo.dot_general %327, %328, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %331 = stablehlo.sqrt %cst_1 : tensor<f32>
    %332 = stablehlo.divide %cst_0, %331 : tensor<f32>
    %333 = stablehlo.convert %332 : tensor<f32>
    %334 = stablehlo.broadcast_in_dim %333, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %335 = stablehlo.multiply %330, %334 : tensor<33x8x79x79xf32>
    %336 = stablehlo.reduce(%335 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %337 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %338 = stablehlo.maximum %337, %336 : tensor<33x8x79xf32>
    %339 = stablehlo.broadcast_in_dim %338, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %340 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %341 = stablehlo.subtract %335, %340 : tensor<33x8x79x79xf32>
    %342 = stablehlo.exponential %341 : tensor<33x8x79x79xf32>
    %343 = stablehlo.reduce(%342 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %345 = stablehlo.broadcast_in_dim %344, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %346 = stablehlo.divide %342, %345 : tensor<33x8x79x79xf32>
    %347 = stablehlo.dot_general %329, %346, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %348 = stablehlo.transpose %347, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %349 = stablehlo.reshape %348 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %350 = stablehlo.dot_general %349, %arg81, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %351 = stablehlo.add %305, %350 : tensor<33x79x1024xf32>
    %352 = stablehlo.reduce(%351 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %353 = stablehlo.broadcast_in_dim %352, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %354 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %355 = stablehlo.divide %353, %354 : tensor<33x79x1xf32>
    %356 = call @_var(%351, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %357 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %358 = stablehlo.add %356, %357 : tensor<33x79x1xf32>
    %359 = stablehlo.rsqrt %358 : tensor<33x79x1xf32>
    %360 = stablehlo.broadcast_in_dim %arg35, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %361 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %362 = stablehlo.broadcast_in_dim %359, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %363 = stablehlo.multiply %361, %362 : tensor<33x79x1024xf32>
    %364 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %365 = stablehlo.subtract %351, %364 : tensor<33x79x1024xf32>
    %366 = stablehlo.multiply %363, %365 : tensor<33x79x1024xf32>
    %367 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %369 = stablehlo.add %366, %368 : tensor<33x79x1024xf32>
    %370 = stablehlo.dot_general %369, %arg40, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %371 = stablehlo.dot_general %369, %arg41, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %372 = call @silu(%370) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %373 = stablehlo.multiply %372, %371 : tensor<33x79x4096xf32>
    %374 = stablehlo.dot_general %373, %arg42, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %375 = stablehlo.add %351, %374 : tensor<33x79x1024xf32>
    %376 = stablehlo.reduce(%375 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %377 = stablehlo.broadcast_in_dim %376, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %378 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %379 = stablehlo.divide %377, %378 : tensor<33x79x1xf32>
    %380 = call @_var(%375, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %381 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %382 = stablehlo.add %380, %381 : tensor<33x79x1xf32>
    %383 = stablehlo.rsqrt %382 : tensor<33x79x1xf32>
    %384 = stablehlo.broadcast_in_dim %arg7, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %385 = stablehlo.broadcast_in_dim %384, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %386 = stablehlo.broadcast_in_dim %383, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %387 = stablehlo.multiply %385, %386 : tensor<33x79x1024xf32>
    %388 = stablehlo.broadcast_in_dim %379, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %389 = stablehlo.subtract %375, %388 : tensor<33x79x1024xf32>
    %390 = stablehlo.multiply %387, %389 : tensor<33x79x1024xf32>
    %391 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %392 = stablehlo.broadcast_in_dim %391, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %393 = stablehlo.add %390, %392 : tensor<33x79x1024xf32>
    %394 = stablehlo.dot_general %393, %arg82, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %395 = stablehlo.dot_general %393, %arg83, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %396 = stablehlo.dot_general %393, %arg84, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %397 = stablehlo.reshape %394 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %398 = stablehlo.reshape %395 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %399 = stablehlo.reshape %396 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %400 = stablehlo.dot_general %397, %398, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %401 = stablehlo.sqrt %cst_1 : tensor<f32>
    %402 = stablehlo.divide %cst_0, %401 : tensor<f32>
    %403 = stablehlo.convert %402 : tensor<f32>
    %404 = stablehlo.broadcast_in_dim %403, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %405 = stablehlo.multiply %400, %404 : tensor<33x8x79x79xf32>
    %406 = stablehlo.reduce(%405 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %407 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %408 = stablehlo.maximum %407, %406 : tensor<33x8x79xf32>
    %409 = stablehlo.broadcast_in_dim %408, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %410 = stablehlo.broadcast_in_dim %409, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %411 = stablehlo.subtract %405, %410 : tensor<33x8x79x79xf32>
    %412 = stablehlo.exponential %411 : tensor<33x8x79x79xf32>
    %413 = stablehlo.reduce(%412 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %414 = stablehlo.broadcast_in_dim %413, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %415 = stablehlo.broadcast_in_dim %414, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %416 = stablehlo.divide %412, %415 : tensor<33x8x79x79xf32>
    %417 = stablehlo.dot_general %399, %416, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %418 = stablehlo.transpose %417, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %419 = stablehlo.reshape %418 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %420 = stablehlo.dot_general %419, %arg85, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %421 = stablehlo.add %375, %420 : tensor<33x79x1024xf32>
    %422 = stablehlo.reduce(%421 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %423 = stablehlo.broadcast_in_dim %422, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %424 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %425 = stablehlo.divide %423, %424 : tensor<33x79x1xf32>
    %426 = call @_var(%421, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %427 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %428 = stablehlo.add %426, %427 : tensor<33x79x1xf32>
    %429 = stablehlo.rsqrt %428 : tensor<33x79x1xf32>
    %430 = stablehlo.broadcast_in_dim %arg9, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %431 = stablehlo.broadcast_in_dim %430, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %432 = stablehlo.broadcast_in_dim %429, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %433 = stablehlo.multiply %431, %432 : tensor<33x79x1024xf32>
    %434 = stablehlo.broadcast_in_dim %425, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %435 = stablehlo.subtract %421, %434 : tensor<33x79x1024xf32>
    %436 = stablehlo.multiply %433, %435 : tensor<33x79x1024xf32>
    %437 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %438 = stablehlo.broadcast_in_dim %437, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %439 = stablehlo.add %436, %438 : tensor<33x79x1024xf32>
    %440 = stablehlo.dot_general %439, %arg43, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %441 = stablehlo.dot_general %439, %arg44, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %442 = call @silu(%440) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %443 = stablehlo.multiply %442, %441 : tensor<33x79x4096xf32>
    %444 = stablehlo.dot_general %443, %arg45, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %445 = stablehlo.add %421, %444 : tensor<33x79x1024xf32>
    %446 = stablehlo.reduce(%445 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %447 = stablehlo.broadcast_in_dim %446, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %448 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %449 = stablehlo.divide %447, %448 : tensor<33x79x1xf32>
    %450 = call @_var(%445, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %451 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %452 = stablehlo.add %450, %451 : tensor<33x79x1xf32>
    %453 = stablehlo.rsqrt %452 : tensor<33x79x1xf32>
    %454 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %455 = stablehlo.broadcast_in_dim %454, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %456 = stablehlo.broadcast_in_dim %453, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %457 = stablehlo.multiply %455, %456 : tensor<33x79x1024xf32>
    %458 = stablehlo.broadcast_in_dim %449, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %459 = stablehlo.subtract %445, %458 : tensor<33x79x1024xf32>
    %460 = stablehlo.multiply %457, %459 : tensor<33x79x1024xf32>
    %461 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %463 = stablehlo.add %460, %462 : tensor<33x79x1024xf32>
    %464 = stablehlo.dot_general %463, %arg86, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %465 = stablehlo.dot_general %463, %arg87, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %466 = stablehlo.dot_general %463, %arg88, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %467 = stablehlo.reshape %464 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %468 = stablehlo.reshape %465 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %469 = stablehlo.reshape %466 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %470 = stablehlo.dot_general %467, %468, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %471 = stablehlo.sqrt %cst_1 : tensor<f32>
    %472 = stablehlo.divide %cst_0, %471 : tensor<f32>
    %473 = stablehlo.convert %472 : tensor<f32>
    %474 = stablehlo.broadcast_in_dim %473, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %475 = stablehlo.multiply %470, %474 : tensor<33x8x79x79xf32>
    %476 = stablehlo.reduce(%475 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %477 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %478 = stablehlo.maximum %477, %476 : tensor<33x8x79xf32>
    %479 = stablehlo.broadcast_in_dim %478, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %480 = stablehlo.broadcast_in_dim %479, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %481 = stablehlo.subtract %475, %480 : tensor<33x8x79x79xf32>
    %482 = stablehlo.exponential %481 : tensor<33x8x79x79xf32>
    %483 = stablehlo.reduce(%482 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %484 = stablehlo.broadcast_in_dim %483, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %485 = stablehlo.broadcast_in_dim %484, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %486 = stablehlo.divide %482, %485 : tensor<33x8x79x79xf32>
    %487 = stablehlo.dot_general %469, %486, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %488 = stablehlo.transpose %487, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %489 = stablehlo.reshape %488 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %490 = stablehlo.dot_general %489, %arg89, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %491 = stablehlo.add %445, %490 : tensor<33x79x1024xf32>
    %492 = stablehlo.reduce(%491 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %493 = stablehlo.broadcast_in_dim %492, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %494 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %495 = stablehlo.divide %493, %494 : tensor<33x79x1xf32>
    %496 = call @_var(%491, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %497 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %498 = stablehlo.add %496, %497 : tensor<33x79x1xf32>
    %499 = stablehlo.rsqrt %498 : tensor<33x79x1xf32>
    %500 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %501 = stablehlo.broadcast_in_dim %500, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %502 = stablehlo.broadcast_in_dim %499, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %503 = stablehlo.multiply %501, %502 : tensor<33x79x1024xf32>
    %504 = stablehlo.broadcast_in_dim %495, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %505 = stablehlo.subtract %491, %504 : tensor<33x79x1024xf32>
    %506 = stablehlo.multiply %503, %505 : tensor<33x79x1024xf32>
    %507 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %508 = stablehlo.broadcast_in_dim %507, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %509 = stablehlo.add %506, %508 : tensor<33x79x1024xf32>
    %510 = stablehlo.dot_general %509, %arg46, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %511 = stablehlo.dot_general %509, %arg47, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %512 = call @silu(%510) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %513 = stablehlo.multiply %512, %511 : tensor<33x79x4096xf32>
    %514 = stablehlo.dot_general %513, %arg49, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %515 = stablehlo.add %491, %514 : tensor<33x79x1024xf32>
    %516 = stablehlo.reduce(%515 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %517 = stablehlo.broadcast_in_dim %516, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %518 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %519 = stablehlo.divide %517, %518 : tensor<33x79x1xf32>
    %520 = call @_var(%515, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %521 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %522 = stablehlo.add %520, %521 : tensor<33x79x1xf32>
    %523 = stablehlo.rsqrt %522 : tensor<33x79x1xf32>
    %524 = stablehlo.broadcast_in_dim %arg15, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %525 = stablehlo.broadcast_in_dim %524, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %526 = stablehlo.broadcast_in_dim %523, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %527 = stablehlo.multiply %525, %526 : tensor<33x79x1024xf32>
    %528 = stablehlo.broadcast_in_dim %519, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %529 = stablehlo.subtract %515, %528 : tensor<33x79x1024xf32>
    %530 = stablehlo.multiply %527, %529 : tensor<33x79x1024xf32>
    %531 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %532 = stablehlo.broadcast_in_dim %531, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %533 = stablehlo.add %530, %532 : tensor<33x79x1024xf32>
    %534 = stablehlo.dot_general %533, %arg90, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %535 = stablehlo.dot_general %533, %arg91, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %536 = stablehlo.dot_general %533, %arg92, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %537 = stablehlo.reshape %534 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %538 = stablehlo.reshape %535 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %539 = stablehlo.reshape %536 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %540 = stablehlo.dot_general %537, %538, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %541 = stablehlo.sqrt %cst_1 : tensor<f32>
    %542 = stablehlo.divide %cst_0, %541 : tensor<f32>
    %543 = stablehlo.convert %542 : tensor<f32>
    %544 = stablehlo.broadcast_in_dim %543, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %545 = stablehlo.multiply %540, %544 : tensor<33x8x79x79xf32>
    %546 = stablehlo.reduce(%545 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %547 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %548 = stablehlo.maximum %547, %546 : tensor<33x8x79xf32>
    %549 = stablehlo.broadcast_in_dim %548, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %550 = stablehlo.broadcast_in_dim %549, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %551 = stablehlo.subtract %545, %550 : tensor<33x8x79x79xf32>
    %552 = stablehlo.exponential %551 : tensor<33x8x79x79xf32>
    %553 = stablehlo.reduce(%552 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %554 = stablehlo.broadcast_in_dim %553, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %555 = stablehlo.broadcast_in_dim %554, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %556 = stablehlo.divide %552, %555 : tensor<33x8x79x79xf32>
    %557 = stablehlo.dot_general %539, %556, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %558 = stablehlo.transpose %557, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %559 = stablehlo.reshape %558 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %560 = stablehlo.dot_general %559, %arg93, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %561 = stablehlo.add %515, %560 : tensor<33x79x1024xf32>
    %562 = stablehlo.reduce(%561 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %563 = stablehlo.broadcast_in_dim %562, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %564 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %565 = stablehlo.divide %563, %564 : tensor<33x79x1xf32>
    %566 = call @_var(%561, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %567 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %568 = stablehlo.add %566, %567 : tensor<33x79x1xf32>
    %569 = stablehlo.rsqrt %568 : tensor<33x79x1xf32>
    %570 = stablehlo.broadcast_in_dim %arg17, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %571 = stablehlo.broadcast_in_dim %570, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %572 = stablehlo.broadcast_in_dim %569, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %573 = stablehlo.multiply %571, %572 : tensor<33x79x1024xf32>
    %574 = stablehlo.broadcast_in_dim %565, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %575 = stablehlo.subtract %561, %574 : tensor<33x79x1024xf32>
    %576 = stablehlo.multiply %573, %575 : tensor<33x79x1024xf32>
    %577 = stablehlo.broadcast_in_dim %arg16, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %578 = stablehlo.broadcast_in_dim %577, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %579 = stablehlo.add %576, %578 : tensor<33x79x1024xf32>
    %580 = stablehlo.dot_general %579, %arg50, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %581 = stablehlo.dot_general %579, %arg51, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %582 = call @silu(%580) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %583 = stablehlo.multiply %582, %581 : tensor<33x79x4096xf32>
    %584 = stablehlo.dot_general %583, %arg52, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %585 = stablehlo.add %561, %584 : tensor<33x79x1024xf32>
    %586 = stablehlo.reduce(%585 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %587 = stablehlo.broadcast_in_dim %586, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %588 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %589 = stablehlo.divide %587, %588 : tensor<33x79x1xf32>
    %590 = call @_var(%585, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %591 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %592 = stablehlo.add %590, %591 : tensor<33x79x1xf32>
    %593 = stablehlo.rsqrt %592 : tensor<33x79x1xf32>
    %594 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %595 = stablehlo.broadcast_in_dim %594, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %596 = stablehlo.broadcast_in_dim %593, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %597 = stablehlo.multiply %595, %596 : tensor<33x79x1024xf32>
    %598 = stablehlo.broadcast_in_dim %589, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %599 = stablehlo.subtract %585, %598 : tensor<33x79x1024xf32>
    %600 = stablehlo.multiply %597, %599 : tensor<33x79x1024xf32>
    %601 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %602 = stablehlo.broadcast_in_dim %601, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %603 = stablehlo.add %600, %602 : tensor<33x79x1024xf32>
    %604 = stablehlo.dot_general %603, %arg54, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x128xf32>) -> tensor<33x79x128xf32>
    %605 = stablehlo.broadcast_in_dim %arg53, dims = [2] : (tensor<128xf32>) -> tensor<33x79x128xf32>
    %606 = stablehlo.add %604, %605 : tensor<33x79x128xf32>
    %607 = call @log_softmax(%606) : (tensor<33x79x128xf32>) -> tensor<33x79x128xf32>
    return %607 : tensor<33x79x128xf32>
  }
  func.func private @_var(%arg0: tensor<33x79x1024xf32>, %arg1: tensor<i32>) -> tensor<33x79x1xf32> {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.024000e+03> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %3 = stablehlo.divide %1, %2 : tensor<33x79x1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<33x79x1024xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<33x79x1024xf32>
    %7 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<f32>
    %8 = stablehlo.subtract %cst_0, %7 : tensor<f32>
    %9 = stablehlo.reduce(%6 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %11 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %12 = stablehlo.divide %10, %11 : tensor<33x79x1xf32>
    %13 = stablehlo.compare  GT, %8, %cst_1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %14 = call @_where(%13, %12, %cst) : (tensor<i1>, tensor<33x79x1xf32>, tensor<f32>) -> tensor<33x79x1xf32>
    return %14 : tensor<33x79x1xf32>
  }
  func.func private @_where(%arg0: tensor<i1>, %arg1: tensor<33x79x1xf32>, %arg2: tensor<f32>) -> tensor<33x79x1xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<i1>, tensor<33x79x1xf32>
    return %2 : tensor<33x79x1xf32>
  }
  func.func private @silu(%arg0: tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.negate %arg0 : tensor<33x79x4096xf32>
    %1 = stablehlo.exponential %0 : tensor<33x79x4096xf32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x79x4096xf32>
    %3 = stablehlo.add %2, %1 : tensor<33x79x4096xf32>
    %4 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x79x4096xf32>
    %5 = stablehlo.divide %4, %3 : tensor<33x79x4096xf32>
    %6 = stablehlo.multiply %arg0, %5 : tensor<33x79x4096xf32>
    return %6 : tensor<33x79x4096xf32>
  }
  func.func private @log_softmax(%arg0: tensor<33x79x128xf32>) -> tensor<33x79x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [2] : (tensor<33x79x128xf32>, tensor<f32>) -> tensor<33x79xf32>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<33x79xf32>
    %2 = stablehlo.maximum %1, %0 : tensor<33x79xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x128xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<33x79x128xf32>
    %6 = stablehlo.exponential %5 : tensor<33x79x128xf32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<33x79x128xf32>, tensor<f32>) -> tensor<33x79xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %9 = stablehlo.log %8 : tensor<33x79x1xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x128xf32>
    %11 = stablehlo.subtract %5, %10 : tensor<33x79x128xf32>
    return %11 : tensor<33x79x128xf32>
  }
}
