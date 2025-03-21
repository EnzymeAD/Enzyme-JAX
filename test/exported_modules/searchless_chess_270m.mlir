module @jit_predict_sequence attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1968x1024xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<79x1024xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg4: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg5: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg6: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg7: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg8: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg9: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg10: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg11: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg12: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg13: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg14: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg15: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg16: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg17: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg18: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg19: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg20: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg21: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg22: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg23: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg24: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg25: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg26: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg27: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg28: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg29: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg30: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg31: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg32: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg33: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg34: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg35: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg36: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg37: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg38: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg39: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg40: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg41: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg42: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg43: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg44: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg45: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg46: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg47: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg48: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg49: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg50: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg51: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg52: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg53: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg54: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg55: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg56: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg57: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg58: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg59: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg60: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg61: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg62: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg63: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg64: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg65: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg66: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg67: tensor<1024xf32> {mhlo.sharding = "{replicated}"}, %arg68: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg69: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg70: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg71: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg72: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg73: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg74: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg75: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg76: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg77: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg78: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg79: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg80: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg81: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg82: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg83: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg84: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg85: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg86: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg87: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg88: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg89: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg90: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg91: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg92: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg93: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg94: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg95: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg96: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg97: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg98: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg99: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg100: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg101: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg102: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg103: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg104: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg105: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg106: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg107: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg108: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg109: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg110: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg111: tensor<128xf32> {mhlo.sharding = "{replicated}"}, %arg112: tensor<1024x128xf32> {mhlo.sharding = "{replicated}"}, %arg113: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg114: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg115: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg116: tensor<4096x1024xf32> {mhlo.sharding = "{replicated}"}, %arg117: tensor<1024x4096xf32> {mhlo.sharding = "{replicated}"}, %arg118: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg119: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg120: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg121: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg122: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg123: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg124: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg125: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg126: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg127: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg128: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg129: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg130: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg131: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg132: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg133: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg134: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg135: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg136: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg137: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg138: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg139: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg140: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg141: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg142: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg143: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg144: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg145: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg146: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg147: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg148: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg149: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg150: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg151: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg152: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg153: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg154: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg155: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg156: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg157: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg158: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg159: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg160: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg161: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg162: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg163: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg164: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg165: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg166: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg167: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg168: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg169: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg170: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg171: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg172: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg173: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg174: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg175: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg176: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg177: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg178: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg179: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg180: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg181: tensor<1024x1024xf32> {mhlo.sharding = "{replicated}"}, %arg182: tensor<33x79xi32>) -> (tensor<33x79x128xf32> {jax.result_info = ""}) {
    %0 = call @apply_fn(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %arg146, %arg147, %arg148, %arg149, %arg150, %arg151, %arg152, %arg153, %arg154, %arg155, %arg156, %arg157, %arg158, %arg159, %arg160, %arg161, %arg162, %arg163, %arg164, %arg165, %arg166, %arg167, %arg168, %arg169, %arg170, %arg171, %arg172, %arg173, %arg174, %arg175, %arg176, %arg177, %arg178, %arg179, %arg180, %arg181, %arg182) : (tensor<1968x1024xf32>, tensor<79x1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<128xf32>, tensor<1024x128xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x4096xf32>, tensor<4096x1024xf32>, tensor<1024x4096xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<33x79xi32>) -> tensor<33x79x128xf32>
    return %0 : tensor<33x79x128xf32>
  }
  func.func private @apply_fn(%arg0: tensor<1968x1024xf32>, %arg1: tensor<79x1024xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024xf32>, %arg6: tensor<1024xf32>, %arg7: tensor<1024xf32>, %arg8: tensor<1024xf32>, %arg9: tensor<1024xf32>, %arg10: tensor<1024xf32>, %arg11: tensor<1024xf32>, %arg12: tensor<1024xf32>, %arg13: tensor<1024xf32>, %arg14: tensor<1024xf32>, %arg15: tensor<1024xf32>, %arg16: tensor<1024xf32>, %arg17: tensor<1024xf32>, %arg18: tensor<1024xf32>, %arg19: tensor<1024xf32>, %arg20: tensor<1024xf32>, %arg21: tensor<1024xf32>, %arg22: tensor<1024xf32>, %arg23: tensor<1024xf32>, %arg24: tensor<1024xf32>, %arg25: tensor<1024xf32>, %arg26: tensor<1024xf32>, %arg27: tensor<1024xf32>, %arg28: tensor<1024xf32>, %arg29: tensor<1024xf32>, %arg30: tensor<1024xf32>, %arg31: tensor<1024xf32>, %arg32: tensor<1024xf32>, %arg33: tensor<1024xf32>, %arg34: tensor<1024xf32>, %arg35: tensor<1024xf32>, %arg36: tensor<1024xf32>, %arg37: tensor<1024xf32>, %arg38: tensor<1024xf32>, %arg39: tensor<1024xf32>, %arg40: tensor<1024xf32>, %arg41: tensor<1024xf32>, %arg42: tensor<1024xf32>, %arg43: tensor<1024xf32>, %arg44: tensor<1024xf32>, %arg45: tensor<1024xf32>, %arg46: tensor<1024xf32>, %arg47: tensor<1024xf32>, %arg48: tensor<1024xf32>, %arg49: tensor<1024xf32>, %arg50: tensor<1024xf32>, %arg51: tensor<1024xf32>, %arg52: tensor<1024xf32>, %arg53: tensor<1024xf32>, %arg54: tensor<1024xf32>, %arg55: tensor<1024xf32>, %arg56: tensor<1024xf32>, %arg57: tensor<1024xf32>, %arg58: tensor<1024xf32>, %arg59: tensor<1024xf32>, %arg60: tensor<1024xf32>, %arg61: tensor<1024xf32>, %arg62: tensor<1024xf32>, %arg63: tensor<1024xf32>, %arg64: tensor<1024xf32>, %arg65: tensor<1024xf32>, %arg66: tensor<1024xf32>, %arg67: tensor<1024xf32>, %arg68: tensor<1024x4096xf32>, %arg69: tensor<1024x4096xf32>, %arg70: tensor<1024x4096xf32>, %arg71: tensor<4096x1024xf32>, %arg72: tensor<1024x4096xf32>, %arg73: tensor<1024x4096xf32>, %arg74: tensor<4096x1024xf32>, %arg75: tensor<1024x4096xf32>, %arg76: tensor<1024x4096xf32>, %arg77: tensor<4096x1024xf32>, %arg78: tensor<1024x4096xf32>, %arg79: tensor<1024x4096xf32>, %arg80: tensor<4096x1024xf32>, %arg81: tensor<4096x1024xf32>, %arg82: tensor<1024x4096xf32>, %arg83: tensor<1024x4096xf32>, %arg84: tensor<4096x1024xf32>, %arg85: tensor<1024x4096xf32>, %arg86: tensor<1024x4096xf32>, %arg87: tensor<4096x1024xf32>, %arg88: tensor<1024x4096xf32>, %arg89: tensor<1024x4096xf32>, %arg90: tensor<4096x1024xf32>, %arg91: tensor<1024x4096xf32>, %arg92: tensor<1024x4096xf32>, %arg93: tensor<1024x4096xf32>, %arg94: tensor<4096x1024xf32>, %arg95: tensor<1024x4096xf32>, %arg96: tensor<1024x4096xf32>, %arg97: tensor<4096x1024xf32>, %arg98: tensor<1024x4096xf32>, %arg99: tensor<1024x4096xf32>, %arg100: tensor<4096x1024xf32>, %arg101: tensor<1024x4096xf32>, %arg102: tensor<1024x4096xf32>, %arg103: tensor<1024x4096xf32>, %arg104: tensor<4096x1024xf32>, %arg105: tensor<1024x4096xf32>, %arg106: tensor<1024x4096xf32>, %arg107: tensor<4096x1024xf32>, %arg108: tensor<1024x4096xf32>, %arg109: tensor<1024x4096xf32>, %arg110: tensor<4096x1024xf32>, %arg111: tensor<128xf32>, %arg112: tensor<1024x128xf32>, %arg113: tensor<4096x1024xf32>, %arg114: tensor<1024x4096xf32>, %arg115: tensor<1024x4096xf32>, %arg116: tensor<4096x1024xf32>, %arg117: tensor<1024x4096xf32>, %arg118: tensor<1024x1024xf32>, %arg119: tensor<1024x1024xf32>, %arg120: tensor<1024x1024xf32>, %arg121: tensor<1024x1024xf32>, %arg122: tensor<1024x1024xf32>, %arg123: tensor<1024x1024xf32>, %arg124: tensor<1024x1024xf32>, %arg125: tensor<1024x1024xf32>, %arg126: tensor<1024x1024xf32>, %arg127: tensor<1024x1024xf32>, %arg128: tensor<1024x1024xf32>, %arg129: tensor<1024x1024xf32>, %arg130: tensor<1024x1024xf32>, %arg131: tensor<1024x1024xf32>, %arg132: tensor<1024x1024xf32>, %arg133: tensor<1024x1024xf32>, %arg134: tensor<1024x1024xf32>, %arg135: tensor<1024x1024xf32>, %arg136: tensor<1024x1024xf32>, %arg137: tensor<1024x1024xf32>, %arg138: tensor<1024x1024xf32>, %arg139: tensor<1024x1024xf32>, %arg140: tensor<1024x1024xf32>, %arg141: tensor<1024x1024xf32>, %arg142: tensor<1024x1024xf32>, %arg143: tensor<1024x1024xf32>, %arg144: tensor<1024x1024xf32>, %arg145: tensor<1024x1024xf32>, %arg146: tensor<1024x1024xf32>, %arg147: tensor<1024x1024xf32>, %arg148: tensor<1024x1024xf32>, %arg149: tensor<1024x1024xf32>, %arg150: tensor<1024x1024xf32>, %arg151: tensor<1024x1024xf32>, %arg152: tensor<1024x1024xf32>, %arg153: tensor<1024x1024xf32>, %arg154: tensor<1024x1024xf32>, %arg155: tensor<1024x1024xf32>, %arg156: tensor<1024x1024xf32>, %arg157: tensor<1024x1024xf32>, %arg158: tensor<1024x1024xf32>, %arg159: tensor<1024x1024xf32>, %arg160: tensor<1024x1024xf32>, %arg161: tensor<1024x1024xf32>, %arg162: tensor<1024x1024xf32>, %arg163: tensor<1024x1024xf32>, %arg164: tensor<1024x1024xf32>, %arg165: tensor<1024x1024xf32>, %arg166: tensor<1024x1024xf32>, %arg167: tensor<1024x1024xf32>, %arg168: tensor<1024x1024xf32>, %arg169: tensor<1024x1024xf32>, %arg170: tensor<1024x1024xf32>, %arg171: tensor<1024x1024xf32>, %arg172: tensor<1024x1024xf32>, %arg173: tensor<1024x1024xf32>, %arg174: tensor<1024x1024xf32>, %arg175: tensor<1024x1024xf32>, %arg176: tensor<1024x1024xf32>, %arg177: tensor<1024x1024xf32>, %arg178: tensor<1024x1024xf32>, %arg179: tensor<1024x1024xf32>, %arg180: tensor<1024x1024xf32>, %arg181: tensor<1024x1024xf32>, %arg182: tensor<33x79xi32>) -> tensor<33x79x128xf32> {
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
    %2 = stablehlo.concatenate %1, %arg182, dim = 1 : (tensor<33x1xi32>, tensor<33x79xi32>) -> tensor<33x80xi32>
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
    %44 = stablehlo.dot_general %43, %arg118, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %45 = stablehlo.dot_general %43, %arg119, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %46 = stablehlo.dot_general %43, %arg120, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %70 = stablehlo.dot_general %69, %arg121, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %90 = stablehlo.dot_general %89, %arg68, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %91 = stablehlo.dot_general %89, %arg69, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %92 = call @silu(%90) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %93 = stablehlo.multiply %92, %91 : tensor<33x79x4096xf32>
    %94 = stablehlo.dot_general %93, %arg80, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %95 = stablehlo.add %71, %94 : tensor<33x79x1024xf32>
    %96 = stablehlo.reduce(%95 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %98 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %99 = stablehlo.divide %97, %98 : tensor<33x79x1xf32>
    %100 = call @_var(%95, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %101 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %102 = stablehlo.add %100, %101 : tensor<33x79x1xf32>
    %103 = stablehlo.rsqrt %102 : tensor<33x79x1xf32>
    %104 = stablehlo.broadcast_in_dim %arg27, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %105 = stablehlo.broadcast_in_dim %104, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %106 = stablehlo.broadcast_in_dim %103, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %107 = stablehlo.multiply %105, %106 : tensor<33x79x1024xf32>
    %108 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %109 = stablehlo.subtract %95, %108 : tensor<33x79x1024xf32>
    %110 = stablehlo.multiply %107, %109 : tensor<33x79x1024xf32>
    %111 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %113 = stablehlo.add %110, %112 : tensor<33x79x1024xf32>
    %114 = stablehlo.dot_general %113, %arg122, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %115 = stablehlo.dot_general %113, %arg123, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %116 = stablehlo.dot_general %113, %arg124, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %140 = stablehlo.dot_general %139, %arg125, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %141 = stablehlo.add %95, %140 : tensor<33x79x1024xf32>
    %142 = stablehlo.reduce(%141 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %143 = stablehlo.broadcast_in_dim %142, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %144 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %145 = stablehlo.divide %143, %144 : tensor<33x79x1xf32>
    %146 = call @_var(%141, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %147 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %148 = stablehlo.add %146, %147 : tensor<33x79x1xf32>
    %149 = stablehlo.rsqrt %148 : tensor<33x79x1xf32>
    %150 = stablehlo.broadcast_in_dim %arg49, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %151 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %152 = stablehlo.broadcast_in_dim %149, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %153 = stablehlo.multiply %151, %152 : tensor<33x79x1024xf32>
    %154 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %155 = stablehlo.subtract %141, %154 : tensor<33x79x1024xf32>
    %156 = stablehlo.multiply %153, %155 : tensor<33x79x1024xf32>
    %157 = stablehlo.broadcast_in_dim %arg48, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %159 = stablehlo.add %156, %158 : tensor<33x79x1024xf32>
    %160 = stablehlo.dot_general %159, %arg91, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %161 = stablehlo.dot_general %159, %arg102, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %162 = call @silu(%160) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %163 = stablehlo.multiply %162, %161 : tensor<33x79x4096xf32>
    %164 = stablehlo.dot_general %163, %arg113, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %165 = stablehlo.add %141, %164 : tensor<33x79x1024xf32>
    %166 = stablehlo.reduce(%165 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %168 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %169 = stablehlo.divide %167, %168 : tensor<33x79x1xf32>
    %170 = call @_var(%165, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %171 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %172 = stablehlo.add %170, %171 : tensor<33x79x1xf32>
    %173 = stablehlo.rsqrt %172 : tensor<33x79x1xf32>
    %174 = stablehlo.broadcast_in_dim %arg57, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %175 = stablehlo.broadcast_in_dim %174, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %176 = stablehlo.broadcast_in_dim %173, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %177 = stablehlo.multiply %175, %176 : tensor<33x79x1024xf32>
    %178 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %179 = stablehlo.subtract %165, %178 : tensor<33x79x1024xf32>
    %180 = stablehlo.multiply %177, %179 : tensor<33x79x1024xf32>
    %181 = stablehlo.broadcast_in_dim %arg56, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %182 = stablehlo.broadcast_in_dim %181, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %183 = stablehlo.add %180, %182 : tensor<33x79x1024xf32>
    %184 = stablehlo.dot_general %183, %arg150, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %185 = stablehlo.dot_general %183, %arg151, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %186 = stablehlo.dot_general %183, %arg152, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %210 = stablehlo.dot_general %209, %arg153, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %211 = stablehlo.add %165, %210 : tensor<33x79x1024xf32>
    %212 = stablehlo.reduce(%211 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %214 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %215 = stablehlo.divide %213, %214 : tensor<33x79x1xf32>
    %216 = call @_var(%211, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %217 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %218 = stablehlo.add %216, %217 : tensor<33x79x1xf32>
    %219 = stablehlo.rsqrt %218 : tensor<33x79x1xf32>
    %220 = stablehlo.broadcast_in_dim %arg59, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %221 = stablehlo.broadcast_in_dim %220, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %222 = stablehlo.broadcast_in_dim %219, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %223 = stablehlo.multiply %221, %222 : tensor<33x79x1024xf32>
    %224 = stablehlo.broadcast_in_dim %215, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %225 = stablehlo.subtract %211, %224 : tensor<33x79x1024xf32>
    %226 = stablehlo.multiply %223, %225 : tensor<33x79x1024xf32>
    %227 = stablehlo.broadcast_in_dim %arg58, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %229 = stablehlo.add %226, %228 : tensor<33x79x1024xf32>
    %230 = stablehlo.dot_general %229, %arg114, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %231 = stablehlo.dot_general %229, %arg115, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %232 = call @silu(%230) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %233 = stablehlo.multiply %232, %231 : tensor<33x79x4096xf32>
    %234 = stablehlo.dot_general %233, %arg116, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %235 = stablehlo.add %211, %234 : tensor<33x79x1024xf32>
    %236 = stablehlo.reduce(%235 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %237 = stablehlo.broadcast_in_dim %236, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %238 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %239 = stablehlo.divide %237, %238 : tensor<33x79x1xf32>
    %240 = call @_var(%235, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %241 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %242 = stablehlo.add %240, %241 : tensor<33x79x1xf32>
    %243 = stablehlo.rsqrt %242 : tensor<33x79x1xf32>
    %244 = stablehlo.broadcast_in_dim %arg61, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %245 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %246 = stablehlo.broadcast_in_dim %243, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %247 = stablehlo.multiply %245, %246 : tensor<33x79x1024xf32>
    %248 = stablehlo.broadcast_in_dim %239, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %249 = stablehlo.subtract %235, %248 : tensor<33x79x1024xf32>
    %250 = stablehlo.multiply %247, %249 : tensor<33x79x1024xf32>
    %251 = stablehlo.broadcast_in_dim %arg60, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %252 = stablehlo.broadcast_in_dim %251, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %253 = stablehlo.add %250, %252 : tensor<33x79x1024xf32>
    %254 = stablehlo.dot_general %253, %arg154, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %255 = stablehlo.dot_general %253, %arg155, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %256 = stablehlo.dot_general %253, %arg156, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %280 = stablehlo.dot_general %279, %arg157, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %281 = stablehlo.add %235, %280 : tensor<33x79x1024xf32>
    %282 = stablehlo.reduce(%281 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %283 = stablehlo.broadcast_in_dim %282, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %284 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %285 = stablehlo.divide %283, %284 : tensor<33x79x1xf32>
    %286 = call @_var(%281, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %287 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %288 = stablehlo.add %286, %287 : tensor<33x79x1xf32>
    %289 = stablehlo.rsqrt %288 : tensor<33x79x1xf32>
    %290 = stablehlo.broadcast_in_dim %arg63, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %291 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %292 = stablehlo.broadcast_in_dim %289, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %293 = stablehlo.multiply %291, %292 : tensor<33x79x1024xf32>
    %294 = stablehlo.broadcast_in_dim %285, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %295 = stablehlo.subtract %281, %294 : tensor<33x79x1024xf32>
    %296 = stablehlo.multiply %293, %295 : tensor<33x79x1024xf32>
    %297 = stablehlo.broadcast_in_dim %arg62, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %299 = stablehlo.add %296, %298 : tensor<33x79x1024xf32>
    %300 = stablehlo.dot_general %299, %arg117, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %301 = stablehlo.dot_general %299, %arg70, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %302 = call @silu(%300) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %303 = stablehlo.multiply %302, %301 : tensor<33x79x4096xf32>
    %304 = stablehlo.dot_general %303, %arg71, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %305 = stablehlo.add %281, %304 : tensor<33x79x1024xf32>
    %306 = stablehlo.reduce(%305 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %307 = stablehlo.broadcast_in_dim %306, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %308 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %309 = stablehlo.divide %307, %308 : tensor<33x79x1xf32>
    %310 = call @_var(%305, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %311 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %312 = stablehlo.add %310, %311 : tensor<33x79x1xf32>
    %313 = stablehlo.rsqrt %312 : tensor<33x79x1xf32>
    %314 = stablehlo.broadcast_in_dim %arg65, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %316 = stablehlo.broadcast_in_dim %313, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %317 = stablehlo.multiply %315, %316 : tensor<33x79x1024xf32>
    %318 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %319 = stablehlo.subtract %305, %318 : tensor<33x79x1024xf32>
    %320 = stablehlo.multiply %317, %319 : tensor<33x79x1024xf32>
    %321 = stablehlo.broadcast_in_dim %arg64, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %323 = stablehlo.add %320, %322 : tensor<33x79x1024xf32>
    %324 = stablehlo.dot_general %323, %arg158, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %325 = stablehlo.dot_general %323, %arg159, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %326 = stablehlo.dot_general %323, %arg160, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %350 = stablehlo.dot_general %349, %arg161, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %351 = stablehlo.add %305, %350 : tensor<33x79x1024xf32>
    %352 = stablehlo.reduce(%351 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %353 = stablehlo.broadcast_in_dim %352, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %354 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %355 = stablehlo.divide %353, %354 : tensor<33x79x1xf32>
    %356 = call @_var(%351, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %357 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %358 = stablehlo.add %356, %357 : tensor<33x79x1xf32>
    %359 = stablehlo.rsqrt %358 : tensor<33x79x1xf32>
    %360 = stablehlo.broadcast_in_dim %arg67, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %361 = stablehlo.broadcast_in_dim %360, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %362 = stablehlo.broadcast_in_dim %359, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %363 = stablehlo.multiply %361, %362 : tensor<33x79x1024xf32>
    %364 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %365 = stablehlo.subtract %351, %364 : tensor<33x79x1024xf32>
    %366 = stablehlo.multiply %363, %365 : tensor<33x79x1024xf32>
    %367 = stablehlo.broadcast_in_dim %arg66, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %368 = stablehlo.broadcast_in_dim %367, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %369 = stablehlo.add %366, %368 : tensor<33x79x1024xf32>
    %370 = stablehlo.dot_general %369, %arg72, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %371 = stablehlo.dot_general %369, %arg73, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %372 = call @silu(%370) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %373 = stablehlo.multiply %372, %371 : tensor<33x79x4096xf32>
    %374 = stablehlo.dot_general %373, %arg74, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
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
    %394 = stablehlo.dot_general %393, %arg162, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %395 = stablehlo.dot_general %393, %arg163, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %396 = stablehlo.dot_general %393, %arg164, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %420 = stablehlo.dot_general %419, %arg165, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %440 = stablehlo.dot_general %439, %arg75, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %441 = stablehlo.dot_general %439, %arg76, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %442 = call @silu(%440) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %443 = stablehlo.multiply %442, %441 : tensor<33x79x4096xf32>
    %444 = stablehlo.dot_general %443, %arg77, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
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
    %464 = stablehlo.dot_general %463, %arg166, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %465 = stablehlo.dot_general %463, %arg167, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %466 = stablehlo.dot_general %463, %arg168, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %490 = stablehlo.dot_general %489, %arg169, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %510 = stablehlo.dot_general %509, %arg78, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %511 = stablehlo.dot_general %509, %arg79, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %512 = call @silu(%510) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %513 = stablehlo.multiply %512, %511 : tensor<33x79x4096xf32>
    %514 = stablehlo.dot_general %513, %arg81, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
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
    %534 = stablehlo.dot_general %533, %arg170, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %535 = stablehlo.dot_general %533, %arg171, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %536 = stablehlo.dot_general %533, %arg172, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %560 = stablehlo.dot_general %559, %arg173, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
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
    %580 = stablehlo.dot_general %579, %arg82, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %581 = stablehlo.dot_general %579, %arg83, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %582 = call @silu(%580) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %583 = stablehlo.multiply %582, %581 : tensor<33x79x4096xf32>
    %584 = stablehlo.dot_general %583, %arg84, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
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
    %604 = stablehlo.dot_general %603, %arg174, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %605 = stablehlo.dot_general %603, %arg175, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %606 = stablehlo.dot_general %603, %arg176, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %607 = stablehlo.reshape %604 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %608 = stablehlo.reshape %605 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %609 = stablehlo.reshape %606 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %610 = stablehlo.dot_general %607, %608, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %611 = stablehlo.sqrt %cst_1 : tensor<f32>
    %612 = stablehlo.divide %cst_0, %611 : tensor<f32>
    %613 = stablehlo.convert %612 : tensor<f32>
    %614 = stablehlo.broadcast_in_dim %613, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %615 = stablehlo.multiply %610, %614 : tensor<33x8x79x79xf32>
    %616 = stablehlo.reduce(%615 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %617 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %618 = stablehlo.maximum %617, %616 : tensor<33x8x79xf32>
    %619 = stablehlo.broadcast_in_dim %618, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %620 = stablehlo.broadcast_in_dim %619, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %621 = stablehlo.subtract %615, %620 : tensor<33x8x79x79xf32>
    %622 = stablehlo.exponential %621 : tensor<33x8x79x79xf32>
    %623 = stablehlo.reduce(%622 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %624 = stablehlo.broadcast_in_dim %623, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %625 = stablehlo.broadcast_in_dim %624, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %626 = stablehlo.divide %622, %625 : tensor<33x8x79x79xf32>
    %627 = stablehlo.dot_general %609, %626, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %628 = stablehlo.transpose %627, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %629 = stablehlo.reshape %628 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %630 = stablehlo.dot_general %629, %arg177, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %631 = stablehlo.add %585, %630 : tensor<33x79x1024xf32>
    %632 = stablehlo.reduce(%631 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %633 = stablehlo.broadcast_in_dim %632, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %634 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %635 = stablehlo.divide %633, %634 : tensor<33x79x1xf32>
    %636 = call @_var(%631, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %637 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %638 = stablehlo.add %636, %637 : tensor<33x79x1xf32>
    %639 = stablehlo.rsqrt %638 : tensor<33x79x1xf32>
    %640 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %641 = stablehlo.broadcast_in_dim %640, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %642 = stablehlo.broadcast_in_dim %639, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %643 = stablehlo.multiply %641, %642 : tensor<33x79x1024xf32>
    %644 = stablehlo.broadcast_in_dim %635, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %645 = stablehlo.subtract %631, %644 : tensor<33x79x1024xf32>
    %646 = stablehlo.multiply %643, %645 : tensor<33x79x1024xf32>
    %647 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %648 = stablehlo.broadcast_in_dim %647, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %649 = stablehlo.add %646, %648 : tensor<33x79x1024xf32>
    %650 = stablehlo.dot_general %649, %arg85, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %651 = stablehlo.dot_general %649, %arg86, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %652 = call @silu(%650) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %653 = stablehlo.multiply %652, %651 : tensor<33x79x4096xf32>
    %654 = stablehlo.dot_general %653, %arg87, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %655 = stablehlo.add %631, %654 : tensor<33x79x1024xf32>
    %656 = stablehlo.reduce(%655 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %657 = stablehlo.broadcast_in_dim %656, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %658 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %659 = stablehlo.divide %657, %658 : tensor<33x79x1xf32>
    %660 = call @_var(%655, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %661 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %662 = stablehlo.add %660, %661 : tensor<33x79x1xf32>
    %663 = stablehlo.rsqrt %662 : tensor<33x79x1xf32>
    %664 = stablehlo.broadcast_in_dim %arg23, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %665 = stablehlo.broadcast_in_dim %664, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %666 = stablehlo.broadcast_in_dim %663, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %667 = stablehlo.multiply %665, %666 : tensor<33x79x1024xf32>
    %668 = stablehlo.broadcast_in_dim %659, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %669 = stablehlo.subtract %655, %668 : tensor<33x79x1024xf32>
    %670 = stablehlo.multiply %667, %669 : tensor<33x79x1024xf32>
    %671 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %672 = stablehlo.broadcast_in_dim %671, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %673 = stablehlo.add %670, %672 : tensor<33x79x1024xf32>
    %674 = stablehlo.dot_general %673, %arg178, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %675 = stablehlo.dot_general %673, %arg179, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %676 = stablehlo.dot_general %673, %arg180, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %677 = stablehlo.reshape %674 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %678 = stablehlo.reshape %675 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %679 = stablehlo.reshape %676 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %680 = stablehlo.dot_general %677, %678, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %681 = stablehlo.sqrt %cst_1 : tensor<f32>
    %682 = stablehlo.divide %cst_0, %681 : tensor<f32>
    %683 = stablehlo.convert %682 : tensor<f32>
    %684 = stablehlo.broadcast_in_dim %683, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %685 = stablehlo.multiply %680, %684 : tensor<33x8x79x79xf32>
    %686 = stablehlo.reduce(%685 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %687 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %688 = stablehlo.maximum %687, %686 : tensor<33x8x79xf32>
    %689 = stablehlo.broadcast_in_dim %688, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %690 = stablehlo.broadcast_in_dim %689, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %691 = stablehlo.subtract %685, %690 : tensor<33x8x79x79xf32>
    %692 = stablehlo.exponential %691 : tensor<33x8x79x79xf32>
    %693 = stablehlo.reduce(%692 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %694 = stablehlo.broadcast_in_dim %693, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %695 = stablehlo.broadcast_in_dim %694, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %696 = stablehlo.divide %692, %695 : tensor<33x8x79x79xf32>
    %697 = stablehlo.dot_general %679, %696, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %698 = stablehlo.transpose %697, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %699 = stablehlo.reshape %698 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %700 = stablehlo.dot_general %699, %arg181, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %701 = stablehlo.add %655, %700 : tensor<33x79x1024xf32>
    %702 = stablehlo.reduce(%701 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %703 = stablehlo.broadcast_in_dim %702, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %704 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %705 = stablehlo.divide %703, %704 : tensor<33x79x1xf32>
    %706 = call @_var(%701, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %707 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %708 = stablehlo.add %706, %707 : tensor<33x79x1xf32>
    %709 = stablehlo.rsqrt %708 : tensor<33x79x1xf32>
    %710 = stablehlo.broadcast_in_dim %arg25, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %711 = stablehlo.broadcast_in_dim %710, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %712 = stablehlo.broadcast_in_dim %709, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %713 = stablehlo.multiply %711, %712 : tensor<33x79x1024xf32>
    %714 = stablehlo.broadcast_in_dim %705, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %715 = stablehlo.subtract %701, %714 : tensor<33x79x1024xf32>
    %716 = stablehlo.multiply %713, %715 : tensor<33x79x1024xf32>
    %717 = stablehlo.broadcast_in_dim %arg24, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %718 = stablehlo.broadcast_in_dim %717, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %719 = stablehlo.add %716, %718 : tensor<33x79x1024xf32>
    %720 = stablehlo.dot_general %719, %arg88, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %721 = stablehlo.dot_general %719, %arg89, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %722 = call @silu(%720) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %723 = stablehlo.multiply %722, %721 : tensor<33x79x4096xf32>
    %724 = stablehlo.dot_general %723, %arg90, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %725 = stablehlo.add %701, %724 : tensor<33x79x1024xf32>
    %726 = stablehlo.reduce(%725 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %727 = stablehlo.broadcast_in_dim %726, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %728 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %729 = stablehlo.divide %727, %728 : tensor<33x79x1xf32>
    %730 = call @_var(%725, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %731 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %732 = stablehlo.add %730, %731 : tensor<33x79x1xf32>
    %733 = stablehlo.rsqrt %732 : tensor<33x79x1xf32>
    %734 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %735 = stablehlo.broadcast_in_dim %734, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %736 = stablehlo.broadcast_in_dim %733, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %737 = stablehlo.multiply %735, %736 : tensor<33x79x1024xf32>
    %738 = stablehlo.broadcast_in_dim %729, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %739 = stablehlo.subtract %725, %738 : tensor<33x79x1024xf32>
    %740 = stablehlo.multiply %737, %739 : tensor<33x79x1024xf32>
    %741 = stablehlo.broadcast_in_dim %arg28, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %742 = stablehlo.broadcast_in_dim %741, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %743 = stablehlo.add %740, %742 : tensor<33x79x1024xf32>
    %744 = stablehlo.dot_general %743, %arg126, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %745 = stablehlo.dot_general %743, %arg127, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %746 = stablehlo.dot_general %743, %arg128, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %747 = stablehlo.reshape %744 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %748 = stablehlo.reshape %745 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %749 = stablehlo.reshape %746 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %750 = stablehlo.dot_general %747, %748, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %751 = stablehlo.sqrt %cst_1 : tensor<f32>
    %752 = stablehlo.divide %cst_0, %751 : tensor<f32>
    %753 = stablehlo.convert %752 : tensor<f32>
    %754 = stablehlo.broadcast_in_dim %753, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %755 = stablehlo.multiply %750, %754 : tensor<33x8x79x79xf32>
    %756 = stablehlo.reduce(%755 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %757 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %758 = stablehlo.maximum %757, %756 : tensor<33x8x79xf32>
    %759 = stablehlo.broadcast_in_dim %758, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %760 = stablehlo.broadcast_in_dim %759, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %761 = stablehlo.subtract %755, %760 : tensor<33x8x79x79xf32>
    %762 = stablehlo.exponential %761 : tensor<33x8x79x79xf32>
    %763 = stablehlo.reduce(%762 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %764 = stablehlo.broadcast_in_dim %763, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %765 = stablehlo.broadcast_in_dim %764, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %766 = stablehlo.divide %762, %765 : tensor<33x8x79x79xf32>
    %767 = stablehlo.dot_general %749, %766, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %768 = stablehlo.transpose %767, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %769 = stablehlo.reshape %768 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %770 = stablehlo.dot_general %769, %arg129, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %771 = stablehlo.add %725, %770 : tensor<33x79x1024xf32>
    %772 = stablehlo.reduce(%771 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %773 = stablehlo.broadcast_in_dim %772, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %774 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %775 = stablehlo.divide %773, %774 : tensor<33x79x1xf32>
    %776 = call @_var(%771, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %777 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %778 = stablehlo.add %776, %777 : tensor<33x79x1xf32>
    %779 = stablehlo.rsqrt %778 : tensor<33x79x1xf32>
    %780 = stablehlo.broadcast_in_dim %arg31, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %781 = stablehlo.broadcast_in_dim %780, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %782 = stablehlo.broadcast_in_dim %779, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %783 = stablehlo.multiply %781, %782 : tensor<33x79x1024xf32>
    %784 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %785 = stablehlo.subtract %771, %784 : tensor<33x79x1024xf32>
    %786 = stablehlo.multiply %783, %785 : tensor<33x79x1024xf32>
    %787 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %788 = stablehlo.broadcast_in_dim %787, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %789 = stablehlo.add %786, %788 : tensor<33x79x1024xf32>
    %790 = stablehlo.dot_general %789, %arg92, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %791 = stablehlo.dot_general %789, %arg93, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %792 = call @silu(%790) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %793 = stablehlo.multiply %792, %791 : tensor<33x79x4096xf32>
    %794 = stablehlo.dot_general %793, %arg94, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %795 = stablehlo.add %771, %794 : tensor<33x79x1024xf32>
    %796 = stablehlo.reduce(%795 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %797 = stablehlo.broadcast_in_dim %796, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %798 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %799 = stablehlo.divide %797, %798 : tensor<33x79x1xf32>
    %800 = call @_var(%795, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %801 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %802 = stablehlo.add %800, %801 : tensor<33x79x1xf32>
    %803 = stablehlo.rsqrt %802 : tensor<33x79x1xf32>
    %804 = stablehlo.broadcast_in_dim %arg33, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %805 = stablehlo.broadcast_in_dim %804, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %806 = stablehlo.broadcast_in_dim %803, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %807 = stablehlo.multiply %805, %806 : tensor<33x79x1024xf32>
    %808 = stablehlo.broadcast_in_dim %799, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %809 = stablehlo.subtract %795, %808 : tensor<33x79x1024xf32>
    %810 = stablehlo.multiply %807, %809 : tensor<33x79x1024xf32>
    %811 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %812 = stablehlo.broadcast_in_dim %811, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %813 = stablehlo.add %810, %812 : tensor<33x79x1024xf32>
    %814 = stablehlo.dot_general %813, %arg130, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %815 = stablehlo.dot_general %813, %arg131, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %816 = stablehlo.dot_general %813, %arg132, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %817 = stablehlo.reshape %814 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %818 = stablehlo.reshape %815 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %819 = stablehlo.reshape %816 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %820 = stablehlo.dot_general %817, %818, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %821 = stablehlo.sqrt %cst_1 : tensor<f32>
    %822 = stablehlo.divide %cst_0, %821 : tensor<f32>
    %823 = stablehlo.convert %822 : tensor<f32>
    %824 = stablehlo.broadcast_in_dim %823, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %825 = stablehlo.multiply %820, %824 : tensor<33x8x79x79xf32>
    %826 = stablehlo.reduce(%825 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %827 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %828 = stablehlo.maximum %827, %826 : tensor<33x8x79xf32>
    %829 = stablehlo.broadcast_in_dim %828, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %830 = stablehlo.broadcast_in_dim %829, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %831 = stablehlo.subtract %825, %830 : tensor<33x8x79x79xf32>
    %832 = stablehlo.exponential %831 : tensor<33x8x79x79xf32>
    %833 = stablehlo.reduce(%832 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %834 = stablehlo.broadcast_in_dim %833, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %835 = stablehlo.broadcast_in_dim %834, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %836 = stablehlo.divide %832, %835 : tensor<33x8x79x79xf32>
    %837 = stablehlo.dot_general %819, %836, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %838 = stablehlo.transpose %837, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %839 = stablehlo.reshape %838 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %840 = stablehlo.dot_general %839, %arg133, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %841 = stablehlo.add %795, %840 : tensor<33x79x1024xf32>
    %842 = stablehlo.reduce(%841 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %843 = stablehlo.broadcast_in_dim %842, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %844 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %845 = stablehlo.divide %843, %844 : tensor<33x79x1xf32>
    %846 = call @_var(%841, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %847 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %848 = stablehlo.add %846, %847 : tensor<33x79x1xf32>
    %849 = stablehlo.rsqrt %848 : tensor<33x79x1xf32>
    %850 = stablehlo.broadcast_in_dim %arg35, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %851 = stablehlo.broadcast_in_dim %850, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %852 = stablehlo.broadcast_in_dim %849, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %853 = stablehlo.multiply %851, %852 : tensor<33x79x1024xf32>
    %854 = stablehlo.broadcast_in_dim %845, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %855 = stablehlo.subtract %841, %854 : tensor<33x79x1024xf32>
    %856 = stablehlo.multiply %853, %855 : tensor<33x79x1024xf32>
    %857 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %858 = stablehlo.broadcast_in_dim %857, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %859 = stablehlo.add %856, %858 : tensor<33x79x1024xf32>
    %860 = stablehlo.dot_general %859, %arg95, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %861 = stablehlo.dot_general %859, %arg96, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %862 = call @silu(%860) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %863 = stablehlo.multiply %862, %861 : tensor<33x79x4096xf32>
    %864 = stablehlo.dot_general %863, %arg97, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %865 = stablehlo.add %841, %864 : tensor<33x79x1024xf32>
    %866 = stablehlo.reduce(%865 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %867 = stablehlo.broadcast_in_dim %866, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %868 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %869 = stablehlo.divide %867, %868 : tensor<33x79x1xf32>
    %870 = call @_var(%865, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %871 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %872 = stablehlo.add %870, %871 : tensor<33x79x1xf32>
    %873 = stablehlo.rsqrt %872 : tensor<33x79x1xf32>
    %874 = stablehlo.broadcast_in_dim %arg37, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %875 = stablehlo.broadcast_in_dim %874, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %876 = stablehlo.broadcast_in_dim %873, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %877 = stablehlo.multiply %875, %876 : tensor<33x79x1024xf32>
    %878 = stablehlo.broadcast_in_dim %869, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %879 = stablehlo.subtract %865, %878 : tensor<33x79x1024xf32>
    %880 = stablehlo.multiply %877, %879 : tensor<33x79x1024xf32>
    %881 = stablehlo.broadcast_in_dim %arg36, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %882 = stablehlo.broadcast_in_dim %881, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %883 = stablehlo.add %880, %882 : tensor<33x79x1024xf32>
    %884 = stablehlo.dot_general %883, %arg134, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %885 = stablehlo.dot_general %883, %arg135, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %886 = stablehlo.dot_general %883, %arg136, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %887 = stablehlo.reshape %884 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %888 = stablehlo.reshape %885 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %889 = stablehlo.reshape %886 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %890 = stablehlo.dot_general %887, %888, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %891 = stablehlo.sqrt %cst_1 : tensor<f32>
    %892 = stablehlo.divide %cst_0, %891 : tensor<f32>
    %893 = stablehlo.convert %892 : tensor<f32>
    %894 = stablehlo.broadcast_in_dim %893, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %895 = stablehlo.multiply %890, %894 : tensor<33x8x79x79xf32>
    %896 = stablehlo.reduce(%895 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %897 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %898 = stablehlo.maximum %897, %896 : tensor<33x8x79xf32>
    %899 = stablehlo.broadcast_in_dim %898, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %900 = stablehlo.broadcast_in_dim %899, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %901 = stablehlo.subtract %895, %900 : tensor<33x8x79x79xf32>
    %902 = stablehlo.exponential %901 : tensor<33x8x79x79xf32>
    %903 = stablehlo.reduce(%902 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %904 = stablehlo.broadcast_in_dim %903, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %905 = stablehlo.broadcast_in_dim %904, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %906 = stablehlo.divide %902, %905 : tensor<33x8x79x79xf32>
    %907 = stablehlo.dot_general %889, %906, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %908 = stablehlo.transpose %907, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %909 = stablehlo.reshape %908 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %910 = stablehlo.dot_general %909, %arg137, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %911 = stablehlo.add %865, %910 : tensor<33x79x1024xf32>
    %912 = stablehlo.reduce(%911 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %913 = stablehlo.broadcast_in_dim %912, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %914 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %915 = stablehlo.divide %913, %914 : tensor<33x79x1xf32>
    %916 = call @_var(%911, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %917 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %918 = stablehlo.add %916, %917 : tensor<33x79x1xf32>
    %919 = stablehlo.rsqrt %918 : tensor<33x79x1xf32>
    %920 = stablehlo.broadcast_in_dim %arg39, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %921 = stablehlo.broadcast_in_dim %920, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %922 = stablehlo.broadcast_in_dim %919, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %923 = stablehlo.multiply %921, %922 : tensor<33x79x1024xf32>
    %924 = stablehlo.broadcast_in_dim %915, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %925 = stablehlo.subtract %911, %924 : tensor<33x79x1024xf32>
    %926 = stablehlo.multiply %923, %925 : tensor<33x79x1024xf32>
    %927 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %928 = stablehlo.broadcast_in_dim %927, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %929 = stablehlo.add %926, %928 : tensor<33x79x1024xf32>
    %930 = stablehlo.dot_general %929, %arg98, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %931 = stablehlo.dot_general %929, %arg99, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %932 = call @silu(%930) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %933 = stablehlo.multiply %932, %931 : tensor<33x79x4096xf32>
    %934 = stablehlo.dot_general %933, %arg100, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %935 = stablehlo.add %911, %934 : tensor<33x79x1024xf32>
    %936 = stablehlo.reduce(%935 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %937 = stablehlo.broadcast_in_dim %936, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %938 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %939 = stablehlo.divide %937, %938 : tensor<33x79x1xf32>
    %940 = call @_var(%935, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %941 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %942 = stablehlo.add %940, %941 : tensor<33x79x1xf32>
    %943 = stablehlo.rsqrt %942 : tensor<33x79x1xf32>
    %944 = stablehlo.broadcast_in_dim %arg41, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %945 = stablehlo.broadcast_in_dim %944, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %946 = stablehlo.broadcast_in_dim %943, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %947 = stablehlo.multiply %945, %946 : tensor<33x79x1024xf32>
    %948 = stablehlo.broadcast_in_dim %939, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %949 = stablehlo.subtract %935, %948 : tensor<33x79x1024xf32>
    %950 = stablehlo.multiply %947, %949 : tensor<33x79x1024xf32>
    %951 = stablehlo.broadcast_in_dim %arg40, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %952 = stablehlo.broadcast_in_dim %951, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %953 = stablehlo.add %950, %952 : tensor<33x79x1024xf32>
    %954 = stablehlo.dot_general %953, %arg138, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %955 = stablehlo.dot_general %953, %arg139, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %956 = stablehlo.dot_general %953, %arg140, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %957 = stablehlo.reshape %954 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %958 = stablehlo.reshape %955 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %959 = stablehlo.reshape %956 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %960 = stablehlo.dot_general %957, %958, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %961 = stablehlo.sqrt %cst_1 : tensor<f32>
    %962 = stablehlo.divide %cst_0, %961 : tensor<f32>
    %963 = stablehlo.convert %962 : tensor<f32>
    %964 = stablehlo.broadcast_in_dim %963, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %965 = stablehlo.multiply %960, %964 : tensor<33x8x79x79xf32>
    %966 = stablehlo.reduce(%965 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %967 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %968 = stablehlo.maximum %967, %966 : tensor<33x8x79xf32>
    %969 = stablehlo.broadcast_in_dim %968, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %970 = stablehlo.broadcast_in_dim %969, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %971 = stablehlo.subtract %965, %970 : tensor<33x8x79x79xf32>
    %972 = stablehlo.exponential %971 : tensor<33x8x79x79xf32>
    %973 = stablehlo.reduce(%972 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %974 = stablehlo.broadcast_in_dim %973, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %975 = stablehlo.broadcast_in_dim %974, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %976 = stablehlo.divide %972, %975 : tensor<33x8x79x79xf32>
    %977 = stablehlo.dot_general %959, %976, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %978 = stablehlo.transpose %977, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %979 = stablehlo.reshape %978 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %980 = stablehlo.dot_general %979, %arg141, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %981 = stablehlo.add %935, %980 : tensor<33x79x1024xf32>
    %982 = stablehlo.reduce(%981 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %983 = stablehlo.broadcast_in_dim %982, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %984 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %985 = stablehlo.divide %983, %984 : tensor<33x79x1xf32>
    %986 = call @_var(%981, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %987 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %988 = stablehlo.add %986, %987 : tensor<33x79x1xf32>
    %989 = stablehlo.rsqrt %988 : tensor<33x79x1xf32>
    %990 = stablehlo.broadcast_in_dim %arg43, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %991 = stablehlo.broadcast_in_dim %990, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %992 = stablehlo.broadcast_in_dim %989, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %993 = stablehlo.multiply %991, %992 : tensor<33x79x1024xf32>
    %994 = stablehlo.broadcast_in_dim %985, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %995 = stablehlo.subtract %981, %994 : tensor<33x79x1024xf32>
    %996 = stablehlo.multiply %993, %995 : tensor<33x79x1024xf32>
    %997 = stablehlo.broadcast_in_dim %arg42, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %998 = stablehlo.broadcast_in_dim %997, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %999 = stablehlo.add %996, %998 : tensor<33x79x1024xf32>
    %1000 = stablehlo.dot_general %999, %arg101, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %1001 = stablehlo.dot_general %999, %arg103, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %1002 = call @silu(%1000) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %1003 = stablehlo.multiply %1002, %1001 : tensor<33x79x4096xf32>
    %1004 = stablehlo.dot_general %1003, %arg104, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %1005 = stablehlo.add %981, %1004 : tensor<33x79x1024xf32>
    %1006 = stablehlo.reduce(%1005 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %1007 = stablehlo.broadcast_in_dim %1006, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %1008 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %1009 = stablehlo.divide %1007, %1008 : tensor<33x79x1xf32>
    %1010 = call @_var(%1005, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %1011 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %1012 = stablehlo.add %1010, %1011 : tensor<33x79x1xf32>
    %1013 = stablehlo.rsqrt %1012 : tensor<33x79x1xf32>
    %1014 = stablehlo.broadcast_in_dim %arg45, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1015 = stablehlo.broadcast_in_dim %1014, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %1016 = stablehlo.broadcast_in_dim %1013, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %1017 = stablehlo.multiply %1015, %1016 : tensor<33x79x1024xf32>
    %1018 = stablehlo.broadcast_in_dim %1009, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %1019 = stablehlo.subtract %1005, %1018 : tensor<33x79x1024xf32>
    %1020 = stablehlo.multiply %1017, %1019 : tensor<33x79x1024xf32>
    %1021 = stablehlo.broadcast_in_dim %arg44, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1022 = stablehlo.broadcast_in_dim %1021, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %1023 = stablehlo.add %1020, %1022 : tensor<33x79x1024xf32>
    %1024 = stablehlo.dot_general %1023, %arg142, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %1025 = stablehlo.dot_general %1023, %arg143, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %1026 = stablehlo.dot_general %1023, %arg144, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %1027 = stablehlo.reshape %1024 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %1028 = stablehlo.reshape %1025 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %1029 = stablehlo.reshape %1026 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %1030 = stablehlo.dot_general %1027, %1028, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %1031 = stablehlo.sqrt %cst_1 : tensor<f32>
    %1032 = stablehlo.divide %cst_0, %1031 : tensor<f32>
    %1033 = stablehlo.convert %1032 : tensor<f32>
    %1034 = stablehlo.broadcast_in_dim %1033, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %1035 = stablehlo.multiply %1030, %1034 : tensor<33x8x79x79xf32>
    %1036 = stablehlo.reduce(%1035 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %1037 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %1038 = stablehlo.maximum %1037, %1036 : tensor<33x8x79xf32>
    %1039 = stablehlo.broadcast_in_dim %1038, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %1040 = stablehlo.broadcast_in_dim %1039, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %1041 = stablehlo.subtract %1035, %1040 : tensor<33x8x79x79xf32>
    %1042 = stablehlo.exponential %1041 : tensor<33x8x79x79xf32>
    %1043 = stablehlo.reduce(%1042 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %1044 = stablehlo.broadcast_in_dim %1043, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %1045 = stablehlo.broadcast_in_dim %1044, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %1046 = stablehlo.divide %1042, %1045 : tensor<33x8x79x79xf32>
    %1047 = stablehlo.dot_general %1029, %1046, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %1048 = stablehlo.transpose %1047, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %1049 = stablehlo.reshape %1048 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %1050 = stablehlo.dot_general %1049, %arg145, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %1051 = stablehlo.add %1005, %1050 : tensor<33x79x1024xf32>
    %1052 = stablehlo.reduce(%1051 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %1053 = stablehlo.broadcast_in_dim %1052, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %1054 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %1055 = stablehlo.divide %1053, %1054 : tensor<33x79x1xf32>
    %1056 = call @_var(%1051, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %1057 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %1058 = stablehlo.add %1056, %1057 : tensor<33x79x1xf32>
    %1059 = stablehlo.rsqrt %1058 : tensor<33x79x1xf32>
    %1060 = stablehlo.broadcast_in_dim %arg47, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1061 = stablehlo.broadcast_in_dim %1060, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %1062 = stablehlo.broadcast_in_dim %1059, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %1063 = stablehlo.multiply %1061, %1062 : tensor<33x79x1024xf32>
    %1064 = stablehlo.broadcast_in_dim %1055, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %1065 = stablehlo.subtract %1051, %1064 : tensor<33x79x1024xf32>
    %1066 = stablehlo.multiply %1063, %1065 : tensor<33x79x1024xf32>
    %1067 = stablehlo.broadcast_in_dim %arg46, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1068 = stablehlo.broadcast_in_dim %1067, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %1069 = stablehlo.add %1066, %1068 : tensor<33x79x1024xf32>
    %1070 = stablehlo.dot_general %1069, %arg105, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %1071 = stablehlo.dot_general %1069, %arg106, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %1072 = call @silu(%1070) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %1073 = stablehlo.multiply %1072, %1071 : tensor<33x79x4096xf32>
    %1074 = stablehlo.dot_general %1073, %arg107, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %1075 = stablehlo.add %1051, %1074 : tensor<33x79x1024xf32>
    %1076 = stablehlo.reduce(%1075 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %1077 = stablehlo.broadcast_in_dim %1076, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %1078 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %1079 = stablehlo.divide %1077, %1078 : tensor<33x79x1xf32>
    %1080 = call @_var(%1075, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %1081 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %1082 = stablehlo.add %1080, %1081 : tensor<33x79x1xf32>
    %1083 = stablehlo.rsqrt %1082 : tensor<33x79x1xf32>
    %1084 = stablehlo.broadcast_in_dim %arg51, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1085 = stablehlo.broadcast_in_dim %1084, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %1086 = stablehlo.broadcast_in_dim %1083, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %1087 = stablehlo.multiply %1085, %1086 : tensor<33x79x1024xf32>
    %1088 = stablehlo.broadcast_in_dim %1079, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %1089 = stablehlo.subtract %1075, %1088 : tensor<33x79x1024xf32>
    %1090 = stablehlo.multiply %1087, %1089 : tensor<33x79x1024xf32>
    %1091 = stablehlo.broadcast_in_dim %arg50, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1092 = stablehlo.broadcast_in_dim %1091, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %1093 = stablehlo.add %1090, %1092 : tensor<33x79x1024xf32>
    %1094 = stablehlo.dot_general %1093, %arg146, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %1095 = stablehlo.dot_general %1093, %arg147, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %1096 = stablehlo.dot_general %1093, %arg148, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %1097 = stablehlo.reshape %1094 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %1098 = stablehlo.reshape %1095 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %1099 = stablehlo.reshape %1096 : (tensor<33x79x1024xf32>) -> tensor<33x79x8x128xf32>
    %1100 = stablehlo.dot_general %1097, %1098, batching_dims = [0, 2] x [0, 2], contracting_dims = [3] x [3] : (tensor<33x79x8x128xf32>, tensor<33x79x8x128xf32>) -> tensor<33x8x79x79xf32>
    %1101 = stablehlo.sqrt %cst_1 : tensor<f32>
    %1102 = stablehlo.divide %cst_0, %1101 : tensor<f32>
    %1103 = stablehlo.convert %1102 : tensor<f32>
    %1104 = stablehlo.broadcast_in_dim %1103, dims = [] : (tensor<f32>) -> tensor<33x8x79x79xf32>
    %1105 = stablehlo.multiply %1100, %1104 : tensor<33x8x79x79xf32>
    %1106 = stablehlo.reduce(%1105 init: %cst) applies stablehlo.maximum across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %1107 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<33x8x79xf32>
    %1108 = stablehlo.maximum %1107, %1106 : tensor<33x8x79xf32>
    %1109 = stablehlo.broadcast_in_dim %1108, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %1110 = stablehlo.broadcast_in_dim %1109, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %1111 = stablehlo.subtract %1105, %1110 : tensor<33x8x79x79xf32>
    %1112 = stablehlo.exponential %1111 : tensor<33x8x79x79xf32>
    %1113 = stablehlo.reduce(%1112 init: %cst_3) applies stablehlo.add across dimensions = [3] : (tensor<33x8x79x79xf32>, tensor<f32>) -> tensor<33x8x79xf32>
    %1114 = stablehlo.broadcast_in_dim %1113, dims = [0, 1, 2] : (tensor<33x8x79xf32>) -> tensor<33x8x79x1xf32>
    %1115 = stablehlo.broadcast_in_dim %1114, dims = [0, 1, 2, 3] : (tensor<33x8x79x1xf32>) -> tensor<33x8x79x79xf32>
    %1116 = stablehlo.divide %1112, %1115 : tensor<33x8x79x79xf32>
    %1117 = stablehlo.dot_general %1099, %1116, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [3] : (tensor<33x79x8x128xf32>, tensor<33x8x79x79xf32>) -> tensor<33x8x128x79xf32>
    %1118 = stablehlo.transpose %1117, dims = [0, 3, 1, 2] : (tensor<33x8x128x79xf32>) -> tensor<33x79x8x128xf32>
    %1119 = stablehlo.reshape %1118 : (tensor<33x79x8x128xf32>) -> tensor<33x79x1024xf32>
    %1120 = stablehlo.dot_general %1119, %arg149, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x1024xf32>) -> tensor<33x79x1024xf32>
    %1121 = stablehlo.add %1075, %1120 : tensor<33x79x1024xf32>
    %1122 = stablehlo.reduce(%1121 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %1123 = stablehlo.broadcast_in_dim %1122, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %1124 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %1125 = stablehlo.divide %1123, %1124 : tensor<33x79x1xf32>
    %1126 = call @_var(%1121, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %1127 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %1128 = stablehlo.add %1126, %1127 : tensor<33x79x1xf32>
    %1129 = stablehlo.rsqrt %1128 : tensor<33x79x1xf32>
    %1130 = stablehlo.broadcast_in_dim %arg53, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1131 = stablehlo.broadcast_in_dim %1130, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %1132 = stablehlo.broadcast_in_dim %1129, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %1133 = stablehlo.multiply %1131, %1132 : tensor<33x79x1024xf32>
    %1134 = stablehlo.broadcast_in_dim %1125, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %1135 = stablehlo.subtract %1121, %1134 : tensor<33x79x1024xf32>
    %1136 = stablehlo.multiply %1133, %1135 : tensor<33x79x1024xf32>
    %1137 = stablehlo.broadcast_in_dim %arg52, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1138 = stablehlo.broadcast_in_dim %1137, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %1139 = stablehlo.add %1136, %1138 : tensor<33x79x1024xf32>
    %1140 = stablehlo.dot_general %1139, %arg108, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %1141 = stablehlo.dot_general %1139, %arg109, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x4096xf32>) -> tensor<33x79x4096xf32>
    %1142 = call @silu(%1140) : (tensor<33x79x4096xf32>) -> tensor<33x79x4096xf32>
    %1143 = stablehlo.multiply %1142, %1141 : tensor<33x79x4096xf32>
    %1144 = stablehlo.dot_general %1143, %arg110, contracting_dims = [2] x [0] : (tensor<33x79x4096xf32>, tensor<4096x1024xf32>) -> tensor<33x79x1024xf32>
    %1145 = stablehlo.add %1121, %1144 : tensor<33x79x1024xf32>
    %1146 = stablehlo.reduce(%1145 init: %cst_3) applies stablehlo.add across dimensions = [2] : (tensor<33x79x1024xf32>, tensor<f32>) -> tensor<33x79xf32>
    %1147 = stablehlo.broadcast_in_dim %1146, dims = [0, 1] : (tensor<33x79xf32>) -> tensor<33x79x1xf32>
    %1148 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %1149 = stablehlo.divide %1147, %1148 : tensor<33x79x1xf32>
    %1150 = call @_var(%1145, %c_6) : (tensor<33x79x1024xf32>, tensor<i32>) -> tensor<33x79x1xf32>
    %1151 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<33x79x1xf32>
    %1152 = stablehlo.add %1150, %1151 : tensor<33x79x1xf32>
    %1153 = stablehlo.rsqrt %1152 : tensor<33x79x1xf32>
    %1154 = stablehlo.broadcast_in_dim %arg55, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1155 = stablehlo.broadcast_in_dim %1154, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %1156 = stablehlo.broadcast_in_dim %1153, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %1157 = stablehlo.multiply %1155, %1156 : tensor<33x79x1024xf32>
    %1158 = stablehlo.broadcast_in_dim %1149, dims = [0, 1, 2] : (tensor<33x79x1xf32>) -> tensor<33x79x1024xf32>
    %1159 = stablehlo.subtract %1145, %1158 : tensor<33x79x1024xf32>
    %1160 = stablehlo.multiply %1157, %1159 : tensor<33x79x1024xf32>
    %1161 = stablehlo.broadcast_in_dim %arg54, dims = [2] : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %1162 = stablehlo.broadcast_in_dim %1161, dims = [0, 1, 2] : (tensor<1x1x1024xf32>) -> tensor<33x79x1024xf32>
    %1163 = stablehlo.add %1160, %1162 : tensor<33x79x1024xf32>
    %1164 = stablehlo.dot_general %1163, %arg112, contracting_dims = [2] x [0] : (tensor<33x79x1024xf32>, tensor<1024x128xf32>) -> tensor<33x79x128xf32>
    %1165 = stablehlo.broadcast_in_dim %arg111, dims = [2] : (tensor<128xf32>) -> tensor<33x79x128xf32>
    %1166 = stablehlo.add %1164, %1165 : tensor<33x79x128xf32>
    %1167 = call @log_softmax(%1166) : (tensor<33x79x128xf32>) -> tensor<33x79x128xf32>
    return %1167 : tensor<33x79x128xf32>
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
