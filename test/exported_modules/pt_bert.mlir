module @jit_func attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<30522x768xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<2x768xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<512x768xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg4: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg5: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg6: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg7: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg8: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg9: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg10: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg11: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg12: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg13: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg14: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg15: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg16: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg17: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg18: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg19: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg20: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg21: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg22: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg23: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg24: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg25: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg26: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg27: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg28: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg29: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg30: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg31: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg32: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg33: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg34: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg35: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg36: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg37: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg38: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg39: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg40: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg41: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg42: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg43: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg44: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg45: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg46: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg47: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg48: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg49: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg50: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg51: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg52: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg53: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg54: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg55: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg56: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg57: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg58: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg59: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg60: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg61: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg62: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg63: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg64: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg65: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg66: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg67: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg68: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg69: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg70: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg71: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg72: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg73: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg74: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg75: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg76: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg77: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg78: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg79: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg80: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg81: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg82: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg83: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg84: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg85: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg86: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg87: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg88: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg89: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg90: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg91: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg92: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg93: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg94: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg95: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg96: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg97: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg98: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg99: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg100: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg101: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg102: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg103: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg104: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg105: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg106: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg107: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg108: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg109: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg110: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg111: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg112: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg113: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg114: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg115: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg116: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg117: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg118: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg119: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg120: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg121: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg122: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg123: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg124: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg125: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg126: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg127: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg128: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg129: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg130: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg131: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg132: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg133: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg134: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg135: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg136: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg137: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg138: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg139: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg140: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg141: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg142: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg143: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg144: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg145: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg146: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg147: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg148: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg149: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg150: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg151: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg152: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg153: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg154: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg155: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg156: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg157: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg158: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg159: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg160: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg161: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg162: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg163: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg164: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg165: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg166: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg167: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg168: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg169: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg170: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg171: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg172: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg173: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg174: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg175: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg176: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg177: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg178: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg179: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg180: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg181: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg182: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg183: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg184: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg185: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg186: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg187: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg188: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg189: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg190: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg191: tensor<3072x768xf32> {mhlo.sharding = "{replicated}"}, %arg192: tensor<3072xf32> {mhlo.sharding = "{replicated}"}, %arg193: tensor<768x3072xf32> {mhlo.sharding = "{replicated}"}, %arg194: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg195: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg196: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg197: tensor<768x768xf32> {mhlo.sharding = "{replicated}"}, %arg198: tensor<768xf32> {mhlo.sharding = "{replicated}"}, %arg199: tensor<1x512xi32> {mhlo.sharding = "{replicated}"}, %arg200: tensor<1x7xi32>, %arg201: tensor<1x7xi32>, %arg202: tensor<1x7xi32>) -> (tensor<1x7x768xf32> {jax.result_info = "[0]"}, tensor<1x768xf32> {jax.result_info = "[1]"}) {
    %cst = stablehlo.constant dense<1.12837911> : tensor<1x7x3072xf32>
    %cst_0 = stablehlo.constant dense<-0.37612626> : tensor<1x7x3072xf32>
    %cst_1 = stablehlo.constant dense<0.112835854> : tensor<1x7x3072xf32>
    %cst_2 = stablehlo.constant dense<-0.0268538129> : tensor<1x7x3072xf32>
    %cst_3 = stablehlo.constant dense<0.00518832775> : tensor<1x7x3072xf32>
    %cst_4 = stablehlo.constant dense<-8.0101937E-4> : tensor<1x7x3072xf32>
    %cst_5 = stablehlo.constant dense<7.85386146E-5> : tensor<1x7x3072xf32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<1x7x3072xf32>
    %cst_7 = stablehlo.constant dense<-88.7228394> : tensor<1x7x3072xf32>
    %cst_8 = stablehlo.constant dense<0.564189494> : tensor<1x7x3072xf32>
    %cst_9 = stablehlo.constant dense<-0.282076746> : tensor<1x7x3072xf32>
    %cst_10 = stablehlo.constant dense<0.42184633> : tensor<1x7x3072xf32>
    %cst_11 = stablehlo.constant dense<-1.01526523> : tensor<1x7x3072xf32>
    %cst_12 = stablehlo.constant dense<2.92101908> : tensor<1x7x3072xf32>
    %cst_13 = stablehlo.constant dense<-7.49551868> : tensor<1x7x3072xf32>
    %cst_14 = stablehlo.constant dense<1.297720e+01> : tensor<1x7x3072xf32>
    %cst_15 = stablehlo.constant dense<-10.477664> : tensor<1x7x3072xf32>
    %cst_16 = stablehlo.constant dense<0.563825965> : tensor<1x7x3072xf32>
    %cst_17 = stablehlo.constant dense<-0.274112701> : tensor<1x7x3072xf32>
    %cst_18 = stablehlo.constant dense<3.404880e-01> : tensor<1x7x3072xf32>
    %cst_19 = stablehlo.constant dense<-0.494451523> : tensor<1x7x3072xf32>
    %cst_20 = stablehlo.constant dense<0.621000468> : tensor<1x7x3072xf32>
    %cst_21 = stablehlo.constant dense<-0.582473278> : tensor<1x7x3072xf32>
    %cst_22 = stablehlo.constant dense<0.368742466> : tensor<1x7x3072xf32>
    %cst_23 = stablehlo.constant dense<-0.138703942> : tensor<1x7x3072xf32>
    %cst_24 = stablehlo.constant dense<2.326820e-02> : tensor<1x7x3072xf32>
    %cst_25 = stablehlo.constant dense<2.000000e+00> : tensor<1x7x3072xf32>
    %cst_26 = stablehlo.constant dense<1.000000e+00> : tensor<1x7x3072xf32>
    %cst_27 = stablehlo.constant dense<0.707106769> : tensor<f32>
    %cst_28 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %c = stablehlo.constant dense<false> : tensor<i1>
    %cst_29 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_30 = stablehlo.constant dense<0.353553385> : tensor<f32>
    %cst_31 = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %cst_32 = stablehlo.constant dense<9.99999996E-13> : tensor<f32>
    %c_33 = stablehlo.constant dense<0> : tensor<i32>
    %cst_34 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %cst_35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_36 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.slice %arg199 [0:1, 0:7] : (tensor<1x512xi32>) -> tensor<1x7xi32>
    %1 = call @_take(%arg0, %arg200) : (tensor<30522x768xf32>, tensor<1x7xi32>) -> tensor<1x7x768xf32>
    %2 = call @_take_0(%arg1, %arg201) : (tensor<2x768xf32>, tensor<1x7xi32>) -> tensor<1x7x768xf32>
    %3 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %4 = stablehlo.multiply %2, %3 : tensor<1x7x768xf32>
    %5 = stablehlo.add %1, %4 : tensor<1x7x768xf32>
    %6 = call @_take_1(%arg2, %0) : (tensor<512x768xf32>, tensor<1x7xi32>) -> tensor<1x7x768xf32>
    %7 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x7x768xf32>
    %9 = stablehlo.add %5, %8 : tensor<1x7x768xf32>
    %10 = stablehlo.reduce(%9 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %12 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %13 = stablehlo.divide %11, %12 : tensor<1x7x1xf32>
    %14 = call @_var(%9, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %15 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %16 = stablehlo.add %14, %15 : tensor<1x7x1xf32>
    %17 = stablehlo.rsqrt %16 : tensor<1x7x1xf32>
    %18 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %19 = stablehlo.subtract %9, %18 : tensor<1x7x768xf32>
    %20 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<1x7x768xf32>
    %22 = stablehlo.broadcast_in_dim %arg3, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %24 = stablehlo.multiply %21, %23 : tensor<1x7x768xf32>
    %25 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %27 = stablehlo.add %24, %26 : tensor<1x7x768xf32>
    %28 = stablehlo.broadcast_in_dim %arg202, dims = [0, 2] : (tensor<1x7xi32>) -> tensor<1x1x7xi32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 3] : (tensor<1x1x7xi32>) -> tensor<1x1x1x7xi32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2, 3] : (tensor<1x1x1x7xi32>) -> tensor<1x1x7x7xi32>
    %31 = stablehlo.convert %30 : (tensor<1x1x7x7xi32>) -> tensor<1x1x7x7xf32>
    %32 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<1x1x7x7xf32>
    %34 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %35 = stablehlo.subtract %34, %33 : tensor<1x1x7x7xf32>
    %36 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %37 = stablehlo.compare  NE, %35, %36,  FLOAT : (tensor<1x1x7x7xf32>, tensor<1x1x7x7xf32>) -> tensor<1x1x7x7xi1>
    %38 = stablehlo.convert %37 : tensor<1x1x7x7xi1>
    %39 = call @_where_3(%38, %cst_31, %35) : (tensor<1x1x7x7xi1>, tensor<f32>, tensor<1x1x7x7xf32>) -> tensor<1x1x7x7xf32>
    %40 = stablehlo.reshape %27 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %41 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %42 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %43 = stablehlo.multiply %arg6, %42 : tensor<768xf32>
    %44 = stablehlo.dot_general %40, %41, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %45 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %46 = stablehlo.multiply %45, %44 : tensor<7x768xf32>
    %47 = stablehlo.broadcast_in_dim %43, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %49 = stablehlo.add %48, %46 : tensor<7x768xf32>
    %50 = stablehlo.reshape %49 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %51 = stablehlo.reshape %50 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %52 = stablehlo.transpose %51, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %53 = stablehlo.reshape %27 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %54 = stablehlo.transpose %arg7, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %55 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %56 = stablehlo.multiply %arg8, %55 : tensor<768xf32>
    %57 = stablehlo.dot_general %53, %54, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %58 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %59 = stablehlo.multiply %58, %57 : tensor<7x768xf32>
    %60 = stablehlo.broadcast_in_dim %56, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %62 = stablehlo.add %61, %59 : tensor<7x768xf32>
    %63 = stablehlo.reshape %62 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %64 = stablehlo.reshape %63 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %65 = stablehlo.transpose %64, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %66 = stablehlo.reshape %27 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %67 = stablehlo.transpose %arg9, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %68 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %69 = stablehlo.multiply %arg10, %68 : tensor<768xf32>
    %70 = stablehlo.dot_general %66, %67, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %71 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %72 = stablehlo.multiply %71, %70 : tensor<7x768xf32>
    %73 = stablehlo.broadcast_in_dim %69, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %75 = stablehlo.add %74, %72 : tensor<7x768xf32>
    %76 = stablehlo.reshape %75 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %77 = stablehlo.reshape %76 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %78 = stablehlo.transpose %77, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %79 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %80 = stablehlo.multiply %52, %79 : tensor<1x12x7x64xf32>
    %81 = stablehlo.transpose %65, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %82 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %83 = stablehlo.multiply %81, %82 : tensor<1x12x64x7xf32>
    %84 = stablehlo.reshape %80 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %85 = stablehlo.reshape %83 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %86 = stablehlo.dot_general %84, %85, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %87 = stablehlo.reshape %86 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %88 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %89 = stablehlo.multiply %39, %88 : tensor<1x1x7x7xf32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %91 = stablehlo.add %87, %90 : tensor<1x12x7x7xf32>
    %92 = stablehlo.reduce(%91 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %93 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %94 = stablehlo.maximum %93, %92 : tensor<1x12x7xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %97 = stablehlo.subtract %91, %96 : tensor<1x12x7x7xf32>
    %98 = stablehlo.exponential %97 : tensor<1x12x7x7xf32>
    %99 = stablehlo.reduce(%98 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %102 = stablehlo.divide %98, %101 : tensor<1x12x7x7xf32>
    %103 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %104 = stablehlo.compare  EQ, %91, %103,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %105 = stablehlo.not %104 : tensor<1x12x7x7xi1>
    %106 = stablehlo.reduce(%105 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %107 = stablehlo.broadcast_in_dim %106, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %108 = stablehlo.not %107 : tensor<1x12x7x1xi1>
    %109 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %110 = call @_where_4(%108, %109, %102) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %111 = stablehlo.reshape %110 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %112 = stablehlo.reshape %78 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %113 = stablehlo.dot_general %111, %112, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %114 = stablehlo.reshape %113 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %115 = stablehlo.transpose %114, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %116 = stablehlo.transpose %115, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %117 = stablehlo.transpose %116, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %118 = stablehlo.reshape %117 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %119 = stablehlo.reshape %118 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %120 = stablehlo.transpose %arg11, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %121 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %122 = stablehlo.multiply %arg12, %121 : tensor<768xf32>
    %123 = stablehlo.dot_general %119, %120, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %124 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %125 = stablehlo.multiply %124, %123 : tensor<7x768xf32>
    %126 = stablehlo.broadcast_in_dim %122, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %127 = stablehlo.broadcast_in_dim %126, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %128 = stablehlo.add %127, %125 : tensor<7x768xf32>
    %129 = stablehlo.reshape %128 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %130 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %131 = stablehlo.multiply %27, %130 : tensor<1x7x768xf32>
    %132 = stablehlo.add %129, %131 : tensor<1x7x768xf32>
    %133 = stablehlo.reduce(%132 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %134 = stablehlo.broadcast_in_dim %133, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %135 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %136 = stablehlo.divide %134, %135 : tensor<1x7x1xf32>
    %137 = call @_var(%132, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %138 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %139 = stablehlo.add %137, %138 : tensor<1x7x1xf32>
    %140 = stablehlo.rsqrt %139 : tensor<1x7x1xf32>
    %141 = stablehlo.broadcast_in_dim %136, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %142 = stablehlo.subtract %132, %141 : tensor<1x7x768xf32>
    %143 = stablehlo.broadcast_in_dim %140, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %144 = stablehlo.multiply %142, %143 : tensor<1x7x768xf32>
    %145 = stablehlo.broadcast_in_dim %arg13, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %146 = stablehlo.broadcast_in_dim %145, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %147 = stablehlo.multiply %144, %146 : tensor<1x7x768xf32>
    %148 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %150 = stablehlo.add %147, %149 : tensor<1x7x768xf32>
    %151 = stablehlo.reshape %150 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %152 = stablehlo.transpose %arg15, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %153 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %154 = stablehlo.multiply %arg16, %153 : tensor<3072xf32>
    %155 = stablehlo.dot_general %151, %152, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %156 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %157 = stablehlo.multiply %156, %155 : tensor<7x3072xf32>
    %158 = stablehlo.broadcast_in_dim %154, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %159 = stablehlo.broadcast_in_dim %158, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %160 = stablehlo.add %159, %157 : tensor<7x3072xf32>
    %161 = stablehlo.reshape %160 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %162 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %163 = stablehlo.multiply %162, %161 : tensor<1x7x3072xf32>
    %164 = stablehlo.negate %161 : tensor<1x7x3072xf32>
    %165 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %166 = stablehlo.multiply %164, %165 : tensor<1x7x3072xf32>
    %167 = stablehlo.multiply %166, %166 : tensor<1x7x3072xf32>
    %168 = stablehlo.negate %167 : tensor<1x7x3072xf32>
    %169 = stablehlo.abs %166 : tensor<1x7x3072xf32>
    %170 = stablehlo.divide %cst_26, %167 : tensor<1x7x3072xf32>
    %171 = stablehlo.exponential %168 : tensor<1x7x3072xf32>
    %172 = stablehlo.divide %cst_26, %169 : tensor<1x7x3072xf32>
    %173 = stablehlo.multiply %171, %172 : tensor<1x7x3072xf32>
    %174 = stablehlo.compare  LT, %169, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %175 = stablehlo.multiply %cst_24, %170 : tensor<1x7x3072xf32>
    %176 = stablehlo.add %175, %cst_23 : tensor<1x7x3072xf32>
    %177 = stablehlo.multiply %176, %170 : tensor<1x7x3072xf32>
    %178 = stablehlo.add %177, %cst_22 : tensor<1x7x3072xf32>
    %179 = stablehlo.multiply %178, %170 : tensor<1x7x3072xf32>
    %180 = stablehlo.add %179, %cst_21 : tensor<1x7x3072xf32>
    %181 = stablehlo.multiply %180, %170 : tensor<1x7x3072xf32>
    %182 = stablehlo.add %181, %cst_20 : tensor<1x7x3072xf32>
    %183 = stablehlo.multiply %182, %170 : tensor<1x7x3072xf32>
    %184 = stablehlo.add %183, %cst_19 : tensor<1x7x3072xf32>
    %185 = stablehlo.multiply %184, %170 : tensor<1x7x3072xf32>
    %186 = stablehlo.add %185, %cst_18 : tensor<1x7x3072xf32>
    %187 = stablehlo.multiply %186, %170 : tensor<1x7x3072xf32>
    %188 = stablehlo.add %187, %cst_17 : tensor<1x7x3072xf32>
    %189 = stablehlo.multiply %188, %170 : tensor<1x7x3072xf32>
    %190 = stablehlo.add %189, %cst_16 : tensor<1x7x3072xf32>
    %191 = stablehlo.multiply %cst_15, %170 : tensor<1x7x3072xf32>
    %192 = stablehlo.add %191, %cst_14 : tensor<1x7x3072xf32>
    %193 = stablehlo.multiply %192, %170 : tensor<1x7x3072xf32>
    %194 = stablehlo.add %193, %cst_13 : tensor<1x7x3072xf32>
    %195 = stablehlo.multiply %194, %170 : tensor<1x7x3072xf32>
    %196 = stablehlo.add %195, %cst_12 : tensor<1x7x3072xf32>
    %197 = stablehlo.multiply %196, %170 : tensor<1x7x3072xf32>
    %198 = stablehlo.add %197, %cst_11 : tensor<1x7x3072xf32>
    %199 = stablehlo.multiply %198, %170 : tensor<1x7x3072xf32>
    %200 = stablehlo.add %199, %cst_10 : tensor<1x7x3072xf32>
    %201 = stablehlo.multiply %200, %170 : tensor<1x7x3072xf32>
    %202 = stablehlo.add %201, %cst_9 : tensor<1x7x3072xf32>
    %203 = stablehlo.multiply %202, %170 : tensor<1x7x3072xf32>
    %204 = stablehlo.add %203, %cst_8 : tensor<1x7x3072xf32>
    %205 = stablehlo.select %174, %190, %204 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %206 = stablehlo.multiply %173, %205 : tensor<1x7x3072xf32>
    %207 = stablehlo.compare  LT, %168, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %208 = stablehlo.select %207, %cst_6, %206 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %209 = stablehlo.compare  LT, %166, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %210 = stablehlo.subtract %cst_25, %208 : tensor<1x7x3072xf32>
    %211 = stablehlo.select %209, %210, %208 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %212 = stablehlo.multiply %166, %166 : tensor<1x7x3072xf32>
    %213 = stablehlo.multiply %cst_5, %212 : tensor<1x7x3072xf32>
    %214 = stablehlo.add %213, %cst_4 : tensor<1x7x3072xf32>
    %215 = stablehlo.multiply %214, %212 : tensor<1x7x3072xf32>
    %216 = stablehlo.add %215, %cst_3 : tensor<1x7x3072xf32>
    %217 = stablehlo.multiply %216, %212 : tensor<1x7x3072xf32>
    %218 = stablehlo.add %217, %cst_2 : tensor<1x7x3072xf32>
    %219 = stablehlo.multiply %218, %212 : tensor<1x7x3072xf32>
    %220 = stablehlo.add %219, %cst_1 : tensor<1x7x3072xf32>
    %221 = stablehlo.multiply %220, %212 : tensor<1x7x3072xf32>
    %222 = stablehlo.add %221, %cst_0 : tensor<1x7x3072xf32>
    %223 = stablehlo.multiply %222, %212 : tensor<1x7x3072xf32>
    %224 = stablehlo.add %223, %cst : tensor<1x7x3072xf32>
    %225 = stablehlo.multiply %166, %224 : tensor<1x7x3072xf32>
    %226 = stablehlo.subtract %cst_26, %225 : tensor<1x7x3072xf32>
    %227 = stablehlo.abs %166 : tensor<1x7x3072xf32>
    %228 = stablehlo.compare  LT, %227, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %229 = stablehlo.select %228, %226, %211 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %230 = stablehlo.multiply %163, %229 : tensor<1x7x3072xf32>
    %231 = stablehlo.reshape %230 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %232 = stablehlo.transpose %arg17, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %233 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %234 = stablehlo.multiply %arg18, %233 : tensor<768xf32>
    %235 = stablehlo.dot_general %231, %232, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %236 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %237 = stablehlo.multiply %236, %235 : tensor<7x768xf32>
    %238 = stablehlo.broadcast_in_dim %234, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %239 = stablehlo.broadcast_in_dim %238, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %240 = stablehlo.add %239, %237 : tensor<7x768xf32>
    %241 = stablehlo.reshape %240 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %242 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %243 = stablehlo.multiply %150, %242 : tensor<1x7x768xf32>
    %244 = stablehlo.add %241, %243 : tensor<1x7x768xf32>
    %245 = stablehlo.reduce(%244 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %247 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %248 = stablehlo.divide %246, %247 : tensor<1x7x1xf32>
    %249 = call @_var(%244, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %250 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %251 = stablehlo.add %249, %250 : tensor<1x7x1xf32>
    %252 = stablehlo.rsqrt %251 : tensor<1x7x1xf32>
    %253 = stablehlo.broadcast_in_dim %248, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %254 = stablehlo.subtract %244, %253 : tensor<1x7x768xf32>
    %255 = stablehlo.broadcast_in_dim %252, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %256 = stablehlo.multiply %254, %255 : tensor<1x7x768xf32>
    %257 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %258 = stablehlo.broadcast_in_dim %257, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %259 = stablehlo.multiply %256, %258 : tensor<1x7x768xf32>
    %260 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %261 = stablehlo.broadcast_in_dim %260, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %262 = stablehlo.add %259, %261 : tensor<1x7x768xf32>
    %263 = stablehlo.reshape %262 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %264 = stablehlo.transpose %arg21, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %265 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %266 = stablehlo.multiply %arg22, %265 : tensor<768xf32>
    %267 = stablehlo.dot_general %263, %264, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %268 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %269 = stablehlo.multiply %268, %267 : tensor<7x768xf32>
    %270 = stablehlo.broadcast_in_dim %266, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %271 = stablehlo.broadcast_in_dim %270, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %272 = stablehlo.add %271, %269 : tensor<7x768xf32>
    %273 = stablehlo.reshape %272 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %274 = stablehlo.reshape %273 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %275 = stablehlo.transpose %274, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %276 = stablehlo.reshape %262 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %277 = stablehlo.transpose %arg23, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %278 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %279 = stablehlo.multiply %arg24, %278 : tensor<768xf32>
    %280 = stablehlo.dot_general %276, %277, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %281 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %282 = stablehlo.multiply %281, %280 : tensor<7x768xf32>
    %283 = stablehlo.broadcast_in_dim %279, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %284 = stablehlo.broadcast_in_dim %283, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %285 = stablehlo.add %284, %282 : tensor<7x768xf32>
    %286 = stablehlo.reshape %285 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %287 = stablehlo.reshape %286 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %288 = stablehlo.transpose %287, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %289 = stablehlo.reshape %262 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %290 = stablehlo.transpose %arg25, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %291 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %292 = stablehlo.multiply %arg26, %291 : tensor<768xf32>
    %293 = stablehlo.dot_general %289, %290, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %294 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %295 = stablehlo.multiply %294, %293 : tensor<7x768xf32>
    %296 = stablehlo.broadcast_in_dim %292, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %297 = stablehlo.broadcast_in_dim %296, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %298 = stablehlo.add %297, %295 : tensor<7x768xf32>
    %299 = stablehlo.reshape %298 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %300 = stablehlo.reshape %299 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %301 = stablehlo.transpose %300, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %302 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %303 = stablehlo.multiply %275, %302 : tensor<1x12x7x64xf32>
    %304 = stablehlo.transpose %288, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %305 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %306 = stablehlo.multiply %304, %305 : tensor<1x12x64x7xf32>
    %307 = stablehlo.reshape %303 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %308 = stablehlo.reshape %306 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %309 = stablehlo.dot_general %307, %308, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %310 = stablehlo.reshape %309 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %311 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %312 = stablehlo.multiply %39, %311 : tensor<1x1x7x7xf32>
    %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %314 = stablehlo.add %310, %313 : tensor<1x12x7x7xf32>
    %315 = stablehlo.reduce(%314 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %316 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %317 = stablehlo.maximum %316, %315 : tensor<1x12x7xf32>
    %318 = stablehlo.broadcast_in_dim %317, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %319 = stablehlo.broadcast_in_dim %318, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %320 = stablehlo.subtract %314, %319 : tensor<1x12x7x7xf32>
    %321 = stablehlo.exponential %320 : tensor<1x12x7x7xf32>
    %322 = stablehlo.reduce(%321 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %323 = stablehlo.broadcast_in_dim %322, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %324 = stablehlo.broadcast_in_dim %323, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %325 = stablehlo.divide %321, %324 : tensor<1x12x7x7xf32>
    %326 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %327 = stablehlo.compare  EQ, %314, %326,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %328 = stablehlo.not %327 : tensor<1x12x7x7xi1>
    %329 = stablehlo.reduce(%328 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %330 = stablehlo.broadcast_in_dim %329, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %331 = stablehlo.not %330 : tensor<1x12x7x1xi1>
    %332 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %333 = call @_where_4(%331, %332, %325) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %334 = stablehlo.reshape %333 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %335 = stablehlo.reshape %301 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %336 = stablehlo.dot_general %334, %335, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %337 = stablehlo.reshape %336 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %338 = stablehlo.transpose %337, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %339 = stablehlo.transpose %338, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %340 = stablehlo.transpose %339, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %341 = stablehlo.reshape %340 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %342 = stablehlo.reshape %341 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %343 = stablehlo.transpose %arg27, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %344 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %345 = stablehlo.multiply %arg28, %344 : tensor<768xf32>
    %346 = stablehlo.dot_general %342, %343, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %347 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %348 = stablehlo.multiply %347, %346 : tensor<7x768xf32>
    %349 = stablehlo.broadcast_in_dim %345, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %350 = stablehlo.broadcast_in_dim %349, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %351 = stablehlo.add %350, %348 : tensor<7x768xf32>
    %352 = stablehlo.reshape %351 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %353 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %354 = stablehlo.multiply %262, %353 : tensor<1x7x768xf32>
    %355 = stablehlo.add %352, %354 : tensor<1x7x768xf32>
    %356 = stablehlo.reduce(%355 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %357 = stablehlo.broadcast_in_dim %356, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %358 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %359 = stablehlo.divide %357, %358 : tensor<1x7x1xf32>
    %360 = call @_var(%355, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %361 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %362 = stablehlo.add %360, %361 : tensor<1x7x1xf32>
    %363 = stablehlo.rsqrt %362 : tensor<1x7x1xf32>
    %364 = stablehlo.broadcast_in_dim %359, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %365 = stablehlo.subtract %355, %364 : tensor<1x7x768xf32>
    %366 = stablehlo.broadcast_in_dim %363, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %367 = stablehlo.multiply %365, %366 : tensor<1x7x768xf32>
    %368 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %369 = stablehlo.broadcast_in_dim %368, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %370 = stablehlo.multiply %367, %369 : tensor<1x7x768xf32>
    %371 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %372 = stablehlo.broadcast_in_dim %371, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %373 = stablehlo.add %370, %372 : tensor<1x7x768xf32>
    %374 = stablehlo.reshape %373 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %375 = stablehlo.transpose %arg31, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %376 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %377 = stablehlo.multiply %arg32, %376 : tensor<3072xf32>
    %378 = stablehlo.dot_general %374, %375, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %379 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %380 = stablehlo.multiply %379, %378 : tensor<7x3072xf32>
    %381 = stablehlo.broadcast_in_dim %377, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %382 = stablehlo.broadcast_in_dim %381, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %383 = stablehlo.add %382, %380 : tensor<7x3072xf32>
    %384 = stablehlo.reshape %383 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %385 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %386 = stablehlo.multiply %385, %384 : tensor<1x7x3072xf32>
    %387 = stablehlo.negate %384 : tensor<1x7x3072xf32>
    %388 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %389 = stablehlo.multiply %387, %388 : tensor<1x7x3072xf32>
    %390 = stablehlo.multiply %389, %389 : tensor<1x7x3072xf32>
    %391 = stablehlo.negate %390 : tensor<1x7x3072xf32>
    %392 = stablehlo.abs %389 : tensor<1x7x3072xf32>
    %393 = stablehlo.divide %cst_26, %390 : tensor<1x7x3072xf32>
    %394 = stablehlo.exponential %391 : tensor<1x7x3072xf32>
    %395 = stablehlo.divide %cst_26, %392 : tensor<1x7x3072xf32>
    %396 = stablehlo.multiply %394, %395 : tensor<1x7x3072xf32>
    %397 = stablehlo.compare  LT, %392, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %398 = stablehlo.multiply %cst_24, %393 : tensor<1x7x3072xf32>
    %399 = stablehlo.add %398, %cst_23 : tensor<1x7x3072xf32>
    %400 = stablehlo.multiply %399, %393 : tensor<1x7x3072xf32>
    %401 = stablehlo.add %400, %cst_22 : tensor<1x7x3072xf32>
    %402 = stablehlo.multiply %401, %393 : tensor<1x7x3072xf32>
    %403 = stablehlo.add %402, %cst_21 : tensor<1x7x3072xf32>
    %404 = stablehlo.multiply %403, %393 : tensor<1x7x3072xf32>
    %405 = stablehlo.add %404, %cst_20 : tensor<1x7x3072xf32>
    %406 = stablehlo.multiply %405, %393 : tensor<1x7x3072xf32>
    %407 = stablehlo.add %406, %cst_19 : tensor<1x7x3072xf32>
    %408 = stablehlo.multiply %407, %393 : tensor<1x7x3072xf32>
    %409 = stablehlo.add %408, %cst_18 : tensor<1x7x3072xf32>
    %410 = stablehlo.multiply %409, %393 : tensor<1x7x3072xf32>
    %411 = stablehlo.add %410, %cst_17 : tensor<1x7x3072xf32>
    %412 = stablehlo.multiply %411, %393 : tensor<1x7x3072xf32>
    %413 = stablehlo.add %412, %cst_16 : tensor<1x7x3072xf32>
    %414 = stablehlo.multiply %cst_15, %393 : tensor<1x7x3072xf32>
    %415 = stablehlo.add %414, %cst_14 : tensor<1x7x3072xf32>
    %416 = stablehlo.multiply %415, %393 : tensor<1x7x3072xf32>
    %417 = stablehlo.add %416, %cst_13 : tensor<1x7x3072xf32>
    %418 = stablehlo.multiply %417, %393 : tensor<1x7x3072xf32>
    %419 = stablehlo.add %418, %cst_12 : tensor<1x7x3072xf32>
    %420 = stablehlo.multiply %419, %393 : tensor<1x7x3072xf32>
    %421 = stablehlo.add %420, %cst_11 : tensor<1x7x3072xf32>
    %422 = stablehlo.multiply %421, %393 : tensor<1x7x3072xf32>
    %423 = stablehlo.add %422, %cst_10 : tensor<1x7x3072xf32>
    %424 = stablehlo.multiply %423, %393 : tensor<1x7x3072xf32>
    %425 = stablehlo.add %424, %cst_9 : tensor<1x7x3072xf32>
    %426 = stablehlo.multiply %425, %393 : tensor<1x7x3072xf32>
    %427 = stablehlo.add %426, %cst_8 : tensor<1x7x3072xf32>
    %428 = stablehlo.select %397, %413, %427 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %429 = stablehlo.multiply %396, %428 : tensor<1x7x3072xf32>
    %430 = stablehlo.compare  LT, %391, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %431 = stablehlo.select %430, %cst_6, %429 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %432 = stablehlo.compare  LT, %389, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %433 = stablehlo.subtract %cst_25, %431 : tensor<1x7x3072xf32>
    %434 = stablehlo.select %432, %433, %431 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %435 = stablehlo.multiply %389, %389 : tensor<1x7x3072xf32>
    %436 = stablehlo.multiply %cst_5, %435 : tensor<1x7x3072xf32>
    %437 = stablehlo.add %436, %cst_4 : tensor<1x7x3072xf32>
    %438 = stablehlo.multiply %437, %435 : tensor<1x7x3072xf32>
    %439 = stablehlo.add %438, %cst_3 : tensor<1x7x3072xf32>
    %440 = stablehlo.multiply %439, %435 : tensor<1x7x3072xf32>
    %441 = stablehlo.add %440, %cst_2 : tensor<1x7x3072xf32>
    %442 = stablehlo.multiply %441, %435 : tensor<1x7x3072xf32>
    %443 = stablehlo.add %442, %cst_1 : tensor<1x7x3072xf32>
    %444 = stablehlo.multiply %443, %435 : tensor<1x7x3072xf32>
    %445 = stablehlo.add %444, %cst_0 : tensor<1x7x3072xf32>
    %446 = stablehlo.multiply %445, %435 : tensor<1x7x3072xf32>
    %447 = stablehlo.add %446, %cst : tensor<1x7x3072xf32>
    %448 = stablehlo.multiply %389, %447 : tensor<1x7x3072xf32>
    %449 = stablehlo.subtract %cst_26, %448 : tensor<1x7x3072xf32>
    %450 = stablehlo.abs %389 : tensor<1x7x3072xf32>
    %451 = stablehlo.compare  LT, %450, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %452 = stablehlo.select %451, %449, %434 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %453 = stablehlo.multiply %386, %452 : tensor<1x7x3072xf32>
    %454 = stablehlo.reshape %453 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %455 = stablehlo.transpose %arg33, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %456 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %457 = stablehlo.multiply %arg34, %456 : tensor<768xf32>
    %458 = stablehlo.dot_general %454, %455, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %459 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %460 = stablehlo.multiply %459, %458 : tensor<7x768xf32>
    %461 = stablehlo.broadcast_in_dim %457, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %462 = stablehlo.broadcast_in_dim %461, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %463 = stablehlo.add %462, %460 : tensor<7x768xf32>
    %464 = stablehlo.reshape %463 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %465 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %466 = stablehlo.multiply %373, %465 : tensor<1x7x768xf32>
    %467 = stablehlo.add %464, %466 : tensor<1x7x768xf32>
    %468 = stablehlo.reduce(%467 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %469 = stablehlo.broadcast_in_dim %468, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %470 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %471 = stablehlo.divide %469, %470 : tensor<1x7x1xf32>
    %472 = call @_var(%467, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %473 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %474 = stablehlo.add %472, %473 : tensor<1x7x1xf32>
    %475 = stablehlo.rsqrt %474 : tensor<1x7x1xf32>
    %476 = stablehlo.broadcast_in_dim %471, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %477 = stablehlo.subtract %467, %476 : tensor<1x7x768xf32>
    %478 = stablehlo.broadcast_in_dim %475, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %479 = stablehlo.multiply %477, %478 : tensor<1x7x768xf32>
    %480 = stablehlo.broadcast_in_dim %arg35, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %481 = stablehlo.broadcast_in_dim %480, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %482 = stablehlo.multiply %479, %481 : tensor<1x7x768xf32>
    %483 = stablehlo.broadcast_in_dim %arg36, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %484 = stablehlo.broadcast_in_dim %483, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %485 = stablehlo.add %482, %484 : tensor<1x7x768xf32>
    %486 = stablehlo.reshape %485 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %487 = stablehlo.transpose %arg37, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %488 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %489 = stablehlo.multiply %arg38, %488 : tensor<768xf32>
    %490 = stablehlo.dot_general %486, %487, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %491 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %492 = stablehlo.multiply %491, %490 : tensor<7x768xf32>
    %493 = stablehlo.broadcast_in_dim %489, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %494 = stablehlo.broadcast_in_dim %493, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %495 = stablehlo.add %494, %492 : tensor<7x768xf32>
    %496 = stablehlo.reshape %495 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %497 = stablehlo.reshape %496 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %498 = stablehlo.transpose %497, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %499 = stablehlo.reshape %485 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %500 = stablehlo.transpose %arg39, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %501 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %502 = stablehlo.multiply %arg40, %501 : tensor<768xf32>
    %503 = stablehlo.dot_general %499, %500, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %504 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %505 = stablehlo.multiply %504, %503 : tensor<7x768xf32>
    %506 = stablehlo.broadcast_in_dim %502, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %507 = stablehlo.broadcast_in_dim %506, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %508 = stablehlo.add %507, %505 : tensor<7x768xf32>
    %509 = stablehlo.reshape %508 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %510 = stablehlo.reshape %509 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %511 = stablehlo.transpose %510, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %512 = stablehlo.reshape %485 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %513 = stablehlo.transpose %arg41, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %514 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %515 = stablehlo.multiply %arg42, %514 : tensor<768xf32>
    %516 = stablehlo.dot_general %512, %513, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %517 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %518 = stablehlo.multiply %517, %516 : tensor<7x768xf32>
    %519 = stablehlo.broadcast_in_dim %515, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %520 = stablehlo.broadcast_in_dim %519, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %521 = stablehlo.add %520, %518 : tensor<7x768xf32>
    %522 = stablehlo.reshape %521 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %523 = stablehlo.reshape %522 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %524 = stablehlo.transpose %523, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %525 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %526 = stablehlo.multiply %498, %525 : tensor<1x12x7x64xf32>
    %527 = stablehlo.transpose %511, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %528 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %529 = stablehlo.multiply %527, %528 : tensor<1x12x64x7xf32>
    %530 = stablehlo.reshape %526 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %531 = stablehlo.reshape %529 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %532 = stablehlo.dot_general %530, %531, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %533 = stablehlo.reshape %532 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %534 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %535 = stablehlo.multiply %39, %534 : tensor<1x1x7x7xf32>
    %536 = stablehlo.broadcast_in_dim %535, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %537 = stablehlo.add %533, %536 : tensor<1x12x7x7xf32>
    %538 = stablehlo.reduce(%537 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %539 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %540 = stablehlo.maximum %539, %538 : tensor<1x12x7xf32>
    %541 = stablehlo.broadcast_in_dim %540, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %542 = stablehlo.broadcast_in_dim %541, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %543 = stablehlo.subtract %537, %542 : tensor<1x12x7x7xf32>
    %544 = stablehlo.exponential %543 : tensor<1x12x7x7xf32>
    %545 = stablehlo.reduce(%544 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %546 = stablehlo.broadcast_in_dim %545, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %547 = stablehlo.broadcast_in_dim %546, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %548 = stablehlo.divide %544, %547 : tensor<1x12x7x7xf32>
    %549 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %550 = stablehlo.compare  EQ, %537, %549,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %551 = stablehlo.not %550 : tensor<1x12x7x7xi1>
    %552 = stablehlo.reduce(%551 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %553 = stablehlo.broadcast_in_dim %552, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %554 = stablehlo.not %553 : tensor<1x12x7x1xi1>
    %555 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %556 = call @_where_4(%554, %555, %548) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %557 = stablehlo.reshape %556 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %558 = stablehlo.reshape %524 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %559 = stablehlo.dot_general %557, %558, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %560 = stablehlo.reshape %559 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %561 = stablehlo.transpose %560, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %562 = stablehlo.transpose %561, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %563 = stablehlo.transpose %562, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %564 = stablehlo.reshape %563 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %565 = stablehlo.reshape %564 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %566 = stablehlo.transpose %arg43, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %567 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %568 = stablehlo.multiply %arg44, %567 : tensor<768xf32>
    %569 = stablehlo.dot_general %565, %566, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %570 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %571 = stablehlo.multiply %570, %569 : tensor<7x768xf32>
    %572 = stablehlo.broadcast_in_dim %568, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %573 = stablehlo.broadcast_in_dim %572, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %574 = stablehlo.add %573, %571 : tensor<7x768xf32>
    %575 = stablehlo.reshape %574 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %576 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %577 = stablehlo.multiply %485, %576 : tensor<1x7x768xf32>
    %578 = stablehlo.add %575, %577 : tensor<1x7x768xf32>
    %579 = stablehlo.reduce(%578 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %580 = stablehlo.broadcast_in_dim %579, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %581 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %582 = stablehlo.divide %580, %581 : tensor<1x7x1xf32>
    %583 = call @_var(%578, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %584 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %585 = stablehlo.add %583, %584 : tensor<1x7x1xf32>
    %586 = stablehlo.rsqrt %585 : tensor<1x7x1xf32>
    %587 = stablehlo.broadcast_in_dim %582, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %588 = stablehlo.subtract %578, %587 : tensor<1x7x768xf32>
    %589 = stablehlo.broadcast_in_dim %586, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %590 = stablehlo.multiply %588, %589 : tensor<1x7x768xf32>
    %591 = stablehlo.broadcast_in_dim %arg45, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %592 = stablehlo.broadcast_in_dim %591, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %593 = stablehlo.multiply %590, %592 : tensor<1x7x768xf32>
    %594 = stablehlo.broadcast_in_dim %arg46, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %595 = stablehlo.broadcast_in_dim %594, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %596 = stablehlo.add %593, %595 : tensor<1x7x768xf32>
    %597 = stablehlo.reshape %596 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %598 = stablehlo.transpose %arg47, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %599 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %600 = stablehlo.multiply %arg48, %599 : tensor<3072xf32>
    %601 = stablehlo.dot_general %597, %598, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %602 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %603 = stablehlo.multiply %602, %601 : tensor<7x3072xf32>
    %604 = stablehlo.broadcast_in_dim %600, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %605 = stablehlo.broadcast_in_dim %604, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %606 = stablehlo.add %605, %603 : tensor<7x3072xf32>
    %607 = stablehlo.reshape %606 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %608 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %609 = stablehlo.multiply %608, %607 : tensor<1x7x3072xf32>
    %610 = stablehlo.negate %607 : tensor<1x7x3072xf32>
    %611 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %612 = stablehlo.multiply %610, %611 : tensor<1x7x3072xf32>
    %613 = stablehlo.multiply %612, %612 : tensor<1x7x3072xf32>
    %614 = stablehlo.negate %613 : tensor<1x7x3072xf32>
    %615 = stablehlo.abs %612 : tensor<1x7x3072xf32>
    %616 = stablehlo.divide %cst_26, %613 : tensor<1x7x3072xf32>
    %617 = stablehlo.exponential %614 : tensor<1x7x3072xf32>
    %618 = stablehlo.divide %cst_26, %615 : tensor<1x7x3072xf32>
    %619 = stablehlo.multiply %617, %618 : tensor<1x7x3072xf32>
    %620 = stablehlo.compare  LT, %615, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %621 = stablehlo.multiply %cst_24, %616 : tensor<1x7x3072xf32>
    %622 = stablehlo.add %621, %cst_23 : tensor<1x7x3072xf32>
    %623 = stablehlo.multiply %622, %616 : tensor<1x7x3072xf32>
    %624 = stablehlo.add %623, %cst_22 : tensor<1x7x3072xf32>
    %625 = stablehlo.multiply %624, %616 : tensor<1x7x3072xf32>
    %626 = stablehlo.add %625, %cst_21 : tensor<1x7x3072xf32>
    %627 = stablehlo.multiply %626, %616 : tensor<1x7x3072xf32>
    %628 = stablehlo.add %627, %cst_20 : tensor<1x7x3072xf32>
    %629 = stablehlo.multiply %628, %616 : tensor<1x7x3072xf32>
    %630 = stablehlo.add %629, %cst_19 : tensor<1x7x3072xf32>
    %631 = stablehlo.multiply %630, %616 : tensor<1x7x3072xf32>
    %632 = stablehlo.add %631, %cst_18 : tensor<1x7x3072xf32>
    %633 = stablehlo.multiply %632, %616 : tensor<1x7x3072xf32>
    %634 = stablehlo.add %633, %cst_17 : tensor<1x7x3072xf32>
    %635 = stablehlo.multiply %634, %616 : tensor<1x7x3072xf32>
    %636 = stablehlo.add %635, %cst_16 : tensor<1x7x3072xf32>
    %637 = stablehlo.multiply %cst_15, %616 : tensor<1x7x3072xf32>
    %638 = stablehlo.add %637, %cst_14 : tensor<1x7x3072xf32>
    %639 = stablehlo.multiply %638, %616 : tensor<1x7x3072xf32>
    %640 = stablehlo.add %639, %cst_13 : tensor<1x7x3072xf32>
    %641 = stablehlo.multiply %640, %616 : tensor<1x7x3072xf32>
    %642 = stablehlo.add %641, %cst_12 : tensor<1x7x3072xf32>
    %643 = stablehlo.multiply %642, %616 : tensor<1x7x3072xf32>
    %644 = stablehlo.add %643, %cst_11 : tensor<1x7x3072xf32>
    %645 = stablehlo.multiply %644, %616 : tensor<1x7x3072xf32>
    %646 = stablehlo.add %645, %cst_10 : tensor<1x7x3072xf32>
    %647 = stablehlo.multiply %646, %616 : tensor<1x7x3072xf32>
    %648 = stablehlo.add %647, %cst_9 : tensor<1x7x3072xf32>
    %649 = stablehlo.multiply %648, %616 : tensor<1x7x3072xf32>
    %650 = stablehlo.add %649, %cst_8 : tensor<1x7x3072xf32>
    %651 = stablehlo.select %620, %636, %650 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %652 = stablehlo.multiply %619, %651 : tensor<1x7x3072xf32>
    %653 = stablehlo.compare  LT, %614, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %654 = stablehlo.select %653, %cst_6, %652 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %655 = stablehlo.compare  LT, %612, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %656 = stablehlo.subtract %cst_25, %654 : tensor<1x7x3072xf32>
    %657 = stablehlo.select %655, %656, %654 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %658 = stablehlo.multiply %612, %612 : tensor<1x7x3072xf32>
    %659 = stablehlo.multiply %cst_5, %658 : tensor<1x7x3072xf32>
    %660 = stablehlo.add %659, %cst_4 : tensor<1x7x3072xf32>
    %661 = stablehlo.multiply %660, %658 : tensor<1x7x3072xf32>
    %662 = stablehlo.add %661, %cst_3 : tensor<1x7x3072xf32>
    %663 = stablehlo.multiply %662, %658 : tensor<1x7x3072xf32>
    %664 = stablehlo.add %663, %cst_2 : tensor<1x7x3072xf32>
    %665 = stablehlo.multiply %664, %658 : tensor<1x7x3072xf32>
    %666 = stablehlo.add %665, %cst_1 : tensor<1x7x3072xf32>
    %667 = stablehlo.multiply %666, %658 : tensor<1x7x3072xf32>
    %668 = stablehlo.add %667, %cst_0 : tensor<1x7x3072xf32>
    %669 = stablehlo.multiply %668, %658 : tensor<1x7x3072xf32>
    %670 = stablehlo.add %669, %cst : tensor<1x7x3072xf32>
    %671 = stablehlo.multiply %612, %670 : tensor<1x7x3072xf32>
    %672 = stablehlo.subtract %cst_26, %671 : tensor<1x7x3072xf32>
    %673 = stablehlo.abs %612 : tensor<1x7x3072xf32>
    %674 = stablehlo.compare  LT, %673, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %675 = stablehlo.select %674, %672, %657 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %676 = stablehlo.multiply %609, %675 : tensor<1x7x3072xf32>
    %677 = stablehlo.reshape %676 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %678 = stablehlo.transpose %arg49, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %679 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %680 = stablehlo.multiply %arg50, %679 : tensor<768xf32>
    %681 = stablehlo.dot_general %677, %678, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %682 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %683 = stablehlo.multiply %682, %681 : tensor<7x768xf32>
    %684 = stablehlo.broadcast_in_dim %680, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %685 = stablehlo.broadcast_in_dim %684, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %686 = stablehlo.add %685, %683 : tensor<7x768xf32>
    %687 = stablehlo.reshape %686 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %688 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %689 = stablehlo.multiply %596, %688 : tensor<1x7x768xf32>
    %690 = stablehlo.add %687, %689 : tensor<1x7x768xf32>
    %691 = stablehlo.reduce(%690 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %692 = stablehlo.broadcast_in_dim %691, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %693 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %694 = stablehlo.divide %692, %693 : tensor<1x7x1xf32>
    %695 = call @_var(%690, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %696 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %697 = stablehlo.add %695, %696 : tensor<1x7x1xf32>
    %698 = stablehlo.rsqrt %697 : tensor<1x7x1xf32>
    %699 = stablehlo.broadcast_in_dim %694, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %700 = stablehlo.subtract %690, %699 : tensor<1x7x768xf32>
    %701 = stablehlo.broadcast_in_dim %698, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %702 = stablehlo.multiply %700, %701 : tensor<1x7x768xf32>
    %703 = stablehlo.broadcast_in_dim %arg51, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %704 = stablehlo.broadcast_in_dim %703, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %705 = stablehlo.multiply %702, %704 : tensor<1x7x768xf32>
    %706 = stablehlo.broadcast_in_dim %arg52, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %707 = stablehlo.broadcast_in_dim %706, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %708 = stablehlo.add %705, %707 : tensor<1x7x768xf32>
    %709 = stablehlo.reshape %708 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %710 = stablehlo.transpose %arg53, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %711 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %712 = stablehlo.multiply %arg54, %711 : tensor<768xf32>
    %713 = stablehlo.dot_general %709, %710, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %714 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %715 = stablehlo.multiply %714, %713 : tensor<7x768xf32>
    %716 = stablehlo.broadcast_in_dim %712, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %717 = stablehlo.broadcast_in_dim %716, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %718 = stablehlo.add %717, %715 : tensor<7x768xf32>
    %719 = stablehlo.reshape %718 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %720 = stablehlo.reshape %719 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %721 = stablehlo.transpose %720, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %722 = stablehlo.reshape %708 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %723 = stablehlo.transpose %arg55, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %724 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %725 = stablehlo.multiply %arg56, %724 : tensor<768xf32>
    %726 = stablehlo.dot_general %722, %723, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %727 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %728 = stablehlo.multiply %727, %726 : tensor<7x768xf32>
    %729 = stablehlo.broadcast_in_dim %725, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %730 = stablehlo.broadcast_in_dim %729, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %731 = stablehlo.add %730, %728 : tensor<7x768xf32>
    %732 = stablehlo.reshape %731 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %733 = stablehlo.reshape %732 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %734 = stablehlo.transpose %733, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %735 = stablehlo.reshape %708 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %736 = stablehlo.transpose %arg57, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %737 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %738 = stablehlo.multiply %arg58, %737 : tensor<768xf32>
    %739 = stablehlo.dot_general %735, %736, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %740 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %741 = stablehlo.multiply %740, %739 : tensor<7x768xf32>
    %742 = stablehlo.broadcast_in_dim %738, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %743 = stablehlo.broadcast_in_dim %742, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %744 = stablehlo.add %743, %741 : tensor<7x768xf32>
    %745 = stablehlo.reshape %744 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %746 = stablehlo.reshape %745 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %747 = stablehlo.transpose %746, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %748 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %749 = stablehlo.multiply %721, %748 : tensor<1x12x7x64xf32>
    %750 = stablehlo.transpose %734, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %751 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %752 = stablehlo.multiply %750, %751 : tensor<1x12x64x7xf32>
    %753 = stablehlo.reshape %749 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %754 = stablehlo.reshape %752 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %755 = stablehlo.dot_general %753, %754, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %756 = stablehlo.reshape %755 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %757 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %758 = stablehlo.multiply %39, %757 : tensor<1x1x7x7xf32>
    %759 = stablehlo.broadcast_in_dim %758, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %760 = stablehlo.add %756, %759 : tensor<1x12x7x7xf32>
    %761 = stablehlo.reduce(%760 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %762 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %763 = stablehlo.maximum %762, %761 : tensor<1x12x7xf32>
    %764 = stablehlo.broadcast_in_dim %763, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %765 = stablehlo.broadcast_in_dim %764, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %766 = stablehlo.subtract %760, %765 : tensor<1x12x7x7xf32>
    %767 = stablehlo.exponential %766 : tensor<1x12x7x7xf32>
    %768 = stablehlo.reduce(%767 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %769 = stablehlo.broadcast_in_dim %768, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %770 = stablehlo.broadcast_in_dim %769, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %771 = stablehlo.divide %767, %770 : tensor<1x12x7x7xf32>
    %772 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %773 = stablehlo.compare  EQ, %760, %772,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %774 = stablehlo.not %773 : tensor<1x12x7x7xi1>
    %775 = stablehlo.reduce(%774 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %776 = stablehlo.broadcast_in_dim %775, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %777 = stablehlo.not %776 : tensor<1x12x7x1xi1>
    %778 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %779 = call @_where_4(%777, %778, %771) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %780 = stablehlo.reshape %779 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %781 = stablehlo.reshape %747 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %782 = stablehlo.dot_general %780, %781, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %783 = stablehlo.reshape %782 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %784 = stablehlo.transpose %783, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %785 = stablehlo.transpose %784, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %786 = stablehlo.transpose %785, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %787 = stablehlo.reshape %786 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %788 = stablehlo.reshape %787 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %789 = stablehlo.transpose %arg59, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %790 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %791 = stablehlo.multiply %arg60, %790 : tensor<768xf32>
    %792 = stablehlo.dot_general %788, %789, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %793 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %794 = stablehlo.multiply %793, %792 : tensor<7x768xf32>
    %795 = stablehlo.broadcast_in_dim %791, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %796 = stablehlo.broadcast_in_dim %795, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %797 = stablehlo.add %796, %794 : tensor<7x768xf32>
    %798 = stablehlo.reshape %797 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %799 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %800 = stablehlo.multiply %708, %799 : tensor<1x7x768xf32>
    %801 = stablehlo.add %798, %800 : tensor<1x7x768xf32>
    %802 = stablehlo.reduce(%801 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %803 = stablehlo.broadcast_in_dim %802, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %804 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %805 = stablehlo.divide %803, %804 : tensor<1x7x1xf32>
    %806 = call @_var(%801, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %807 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %808 = stablehlo.add %806, %807 : tensor<1x7x1xf32>
    %809 = stablehlo.rsqrt %808 : tensor<1x7x1xf32>
    %810 = stablehlo.broadcast_in_dim %805, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %811 = stablehlo.subtract %801, %810 : tensor<1x7x768xf32>
    %812 = stablehlo.broadcast_in_dim %809, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %813 = stablehlo.multiply %811, %812 : tensor<1x7x768xf32>
    %814 = stablehlo.broadcast_in_dim %arg61, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %815 = stablehlo.broadcast_in_dim %814, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %816 = stablehlo.multiply %813, %815 : tensor<1x7x768xf32>
    %817 = stablehlo.broadcast_in_dim %arg62, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %818 = stablehlo.broadcast_in_dim %817, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %819 = stablehlo.add %816, %818 : tensor<1x7x768xf32>
    %820 = stablehlo.reshape %819 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %821 = stablehlo.transpose %arg63, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %822 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %823 = stablehlo.multiply %arg64, %822 : tensor<3072xf32>
    %824 = stablehlo.dot_general %820, %821, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %825 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %826 = stablehlo.multiply %825, %824 : tensor<7x3072xf32>
    %827 = stablehlo.broadcast_in_dim %823, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %828 = stablehlo.broadcast_in_dim %827, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %829 = stablehlo.add %828, %826 : tensor<7x3072xf32>
    %830 = stablehlo.reshape %829 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %831 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %832 = stablehlo.multiply %831, %830 : tensor<1x7x3072xf32>
    %833 = stablehlo.negate %830 : tensor<1x7x3072xf32>
    %834 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %835 = stablehlo.multiply %833, %834 : tensor<1x7x3072xf32>
    %836 = stablehlo.multiply %835, %835 : tensor<1x7x3072xf32>
    %837 = stablehlo.negate %836 : tensor<1x7x3072xf32>
    %838 = stablehlo.abs %835 : tensor<1x7x3072xf32>
    %839 = stablehlo.divide %cst_26, %836 : tensor<1x7x3072xf32>
    %840 = stablehlo.exponential %837 : tensor<1x7x3072xf32>
    %841 = stablehlo.divide %cst_26, %838 : tensor<1x7x3072xf32>
    %842 = stablehlo.multiply %840, %841 : tensor<1x7x3072xf32>
    %843 = stablehlo.compare  LT, %838, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %844 = stablehlo.multiply %cst_24, %839 : tensor<1x7x3072xf32>
    %845 = stablehlo.add %844, %cst_23 : tensor<1x7x3072xf32>
    %846 = stablehlo.multiply %845, %839 : tensor<1x7x3072xf32>
    %847 = stablehlo.add %846, %cst_22 : tensor<1x7x3072xf32>
    %848 = stablehlo.multiply %847, %839 : tensor<1x7x3072xf32>
    %849 = stablehlo.add %848, %cst_21 : tensor<1x7x3072xf32>
    %850 = stablehlo.multiply %849, %839 : tensor<1x7x3072xf32>
    %851 = stablehlo.add %850, %cst_20 : tensor<1x7x3072xf32>
    %852 = stablehlo.multiply %851, %839 : tensor<1x7x3072xf32>
    %853 = stablehlo.add %852, %cst_19 : tensor<1x7x3072xf32>
    %854 = stablehlo.multiply %853, %839 : tensor<1x7x3072xf32>
    %855 = stablehlo.add %854, %cst_18 : tensor<1x7x3072xf32>
    %856 = stablehlo.multiply %855, %839 : tensor<1x7x3072xf32>
    %857 = stablehlo.add %856, %cst_17 : tensor<1x7x3072xf32>
    %858 = stablehlo.multiply %857, %839 : tensor<1x7x3072xf32>
    %859 = stablehlo.add %858, %cst_16 : tensor<1x7x3072xf32>
    %860 = stablehlo.multiply %cst_15, %839 : tensor<1x7x3072xf32>
    %861 = stablehlo.add %860, %cst_14 : tensor<1x7x3072xf32>
    %862 = stablehlo.multiply %861, %839 : tensor<1x7x3072xf32>
    %863 = stablehlo.add %862, %cst_13 : tensor<1x7x3072xf32>
    %864 = stablehlo.multiply %863, %839 : tensor<1x7x3072xf32>
    %865 = stablehlo.add %864, %cst_12 : tensor<1x7x3072xf32>
    %866 = stablehlo.multiply %865, %839 : tensor<1x7x3072xf32>
    %867 = stablehlo.add %866, %cst_11 : tensor<1x7x3072xf32>
    %868 = stablehlo.multiply %867, %839 : tensor<1x7x3072xf32>
    %869 = stablehlo.add %868, %cst_10 : tensor<1x7x3072xf32>
    %870 = stablehlo.multiply %869, %839 : tensor<1x7x3072xf32>
    %871 = stablehlo.add %870, %cst_9 : tensor<1x7x3072xf32>
    %872 = stablehlo.multiply %871, %839 : tensor<1x7x3072xf32>
    %873 = stablehlo.add %872, %cst_8 : tensor<1x7x3072xf32>
    %874 = stablehlo.select %843, %859, %873 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %875 = stablehlo.multiply %842, %874 : tensor<1x7x3072xf32>
    %876 = stablehlo.compare  LT, %837, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %877 = stablehlo.select %876, %cst_6, %875 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %878 = stablehlo.compare  LT, %835, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %879 = stablehlo.subtract %cst_25, %877 : tensor<1x7x3072xf32>
    %880 = stablehlo.select %878, %879, %877 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %881 = stablehlo.multiply %835, %835 : tensor<1x7x3072xf32>
    %882 = stablehlo.multiply %cst_5, %881 : tensor<1x7x3072xf32>
    %883 = stablehlo.add %882, %cst_4 : tensor<1x7x3072xf32>
    %884 = stablehlo.multiply %883, %881 : tensor<1x7x3072xf32>
    %885 = stablehlo.add %884, %cst_3 : tensor<1x7x3072xf32>
    %886 = stablehlo.multiply %885, %881 : tensor<1x7x3072xf32>
    %887 = stablehlo.add %886, %cst_2 : tensor<1x7x3072xf32>
    %888 = stablehlo.multiply %887, %881 : tensor<1x7x3072xf32>
    %889 = stablehlo.add %888, %cst_1 : tensor<1x7x3072xf32>
    %890 = stablehlo.multiply %889, %881 : tensor<1x7x3072xf32>
    %891 = stablehlo.add %890, %cst_0 : tensor<1x7x3072xf32>
    %892 = stablehlo.multiply %891, %881 : tensor<1x7x3072xf32>
    %893 = stablehlo.add %892, %cst : tensor<1x7x3072xf32>
    %894 = stablehlo.multiply %835, %893 : tensor<1x7x3072xf32>
    %895 = stablehlo.subtract %cst_26, %894 : tensor<1x7x3072xf32>
    %896 = stablehlo.abs %835 : tensor<1x7x3072xf32>
    %897 = stablehlo.compare  LT, %896, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %898 = stablehlo.select %897, %895, %880 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %899 = stablehlo.multiply %832, %898 : tensor<1x7x3072xf32>
    %900 = stablehlo.reshape %899 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %901 = stablehlo.transpose %arg65, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %902 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %903 = stablehlo.multiply %arg66, %902 : tensor<768xf32>
    %904 = stablehlo.dot_general %900, %901, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %905 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %906 = stablehlo.multiply %905, %904 : tensor<7x768xf32>
    %907 = stablehlo.broadcast_in_dim %903, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %908 = stablehlo.broadcast_in_dim %907, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %909 = stablehlo.add %908, %906 : tensor<7x768xf32>
    %910 = stablehlo.reshape %909 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %911 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %912 = stablehlo.multiply %819, %911 : tensor<1x7x768xf32>
    %913 = stablehlo.add %910, %912 : tensor<1x7x768xf32>
    %914 = stablehlo.reduce(%913 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %915 = stablehlo.broadcast_in_dim %914, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %916 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %917 = stablehlo.divide %915, %916 : tensor<1x7x1xf32>
    %918 = call @_var(%913, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %919 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %920 = stablehlo.add %918, %919 : tensor<1x7x1xf32>
    %921 = stablehlo.rsqrt %920 : tensor<1x7x1xf32>
    %922 = stablehlo.broadcast_in_dim %917, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %923 = stablehlo.subtract %913, %922 : tensor<1x7x768xf32>
    %924 = stablehlo.broadcast_in_dim %921, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %925 = stablehlo.multiply %923, %924 : tensor<1x7x768xf32>
    %926 = stablehlo.broadcast_in_dim %arg67, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %927 = stablehlo.broadcast_in_dim %926, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %928 = stablehlo.multiply %925, %927 : tensor<1x7x768xf32>
    %929 = stablehlo.broadcast_in_dim %arg68, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %930 = stablehlo.broadcast_in_dim %929, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %931 = stablehlo.add %928, %930 : tensor<1x7x768xf32>
    %932 = stablehlo.reshape %931 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %933 = stablehlo.transpose %arg69, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %934 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %935 = stablehlo.multiply %arg70, %934 : tensor<768xf32>
    %936 = stablehlo.dot_general %932, %933, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %937 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %938 = stablehlo.multiply %937, %936 : tensor<7x768xf32>
    %939 = stablehlo.broadcast_in_dim %935, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %940 = stablehlo.broadcast_in_dim %939, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %941 = stablehlo.add %940, %938 : tensor<7x768xf32>
    %942 = stablehlo.reshape %941 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %943 = stablehlo.reshape %942 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %944 = stablehlo.transpose %943, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %945 = stablehlo.reshape %931 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %946 = stablehlo.transpose %arg71, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %947 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %948 = stablehlo.multiply %arg72, %947 : tensor<768xf32>
    %949 = stablehlo.dot_general %945, %946, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %950 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %951 = stablehlo.multiply %950, %949 : tensor<7x768xf32>
    %952 = stablehlo.broadcast_in_dim %948, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %953 = stablehlo.broadcast_in_dim %952, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %954 = stablehlo.add %953, %951 : tensor<7x768xf32>
    %955 = stablehlo.reshape %954 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %956 = stablehlo.reshape %955 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %957 = stablehlo.transpose %956, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %958 = stablehlo.reshape %931 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %959 = stablehlo.transpose %arg73, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %960 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %961 = stablehlo.multiply %arg74, %960 : tensor<768xf32>
    %962 = stablehlo.dot_general %958, %959, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %963 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %964 = stablehlo.multiply %963, %962 : tensor<7x768xf32>
    %965 = stablehlo.broadcast_in_dim %961, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %966 = stablehlo.broadcast_in_dim %965, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %967 = stablehlo.add %966, %964 : tensor<7x768xf32>
    %968 = stablehlo.reshape %967 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %969 = stablehlo.reshape %968 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %970 = stablehlo.transpose %969, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %971 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %972 = stablehlo.multiply %944, %971 : tensor<1x12x7x64xf32>
    %973 = stablehlo.transpose %957, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %974 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %975 = stablehlo.multiply %973, %974 : tensor<1x12x64x7xf32>
    %976 = stablehlo.reshape %972 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %977 = stablehlo.reshape %975 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %978 = stablehlo.dot_general %976, %977, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %979 = stablehlo.reshape %978 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %980 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %981 = stablehlo.multiply %39, %980 : tensor<1x1x7x7xf32>
    %982 = stablehlo.broadcast_in_dim %981, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %983 = stablehlo.add %979, %982 : tensor<1x12x7x7xf32>
    %984 = stablehlo.reduce(%983 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %985 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %986 = stablehlo.maximum %985, %984 : tensor<1x12x7xf32>
    %987 = stablehlo.broadcast_in_dim %986, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %988 = stablehlo.broadcast_in_dim %987, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %989 = stablehlo.subtract %983, %988 : tensor<1x12x7x7xf32>
    %990 = stablehlo.exponential %989 : tensor<1x12x7x7xf32>
    %991 = stablehlo.reduce(%990 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %992 = stablehlo.broadcast_in_dim %991, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %993 = stablehlo.broadcast_in_dim %992, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %994 = stablehlo.divide %990, %993 : tensor<1x12x7x7xf32>
    %995 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %996 = stablehlo.compare  EQ, %983, %995,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %997 = stablehlo.not %996 : tensor<1x12x7x7xi1>
    %998 = stablehlo.reduce(%997 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %999 = stablehlo.broadcast_in_dim %998, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %1000 = stablehlo.not %999 : tensor<1x12x7x1xi1>
    %1001 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %1002 = call @_where_4(%1000, %1001, %994) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1003 = stablehlo.reshape %1002 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %1004 = stablehlo.reshape %970 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %1005 = stablehlo.dot_general %1003, %1004, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %1006 = stablehlo.reshape %1005 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %1007 = stablehlo.transpose %1006, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %1008 = stablehlo.transpose %1007, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1009 = stablehlo.transpose %1008, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %1010 = stablehlo.reshape %1009 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %1011 = stablehlo.reshape %1010 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1012 = stablehlo.transpose %arg75, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1013 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1014 = stablehlo.multiply %arg76, %1013 : tensor<768xf32>
    %1015 = stablehlo.dot_general %1011, %1012, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1016 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1017 = stablehlo.multiply %1016, %1015 : tensor<7x768xf32>
    %1018 = stablehlo.broadcast_in_dim %1014, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1019 = stablehlo.broadcast_in_dim %1018, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1020 = stablehlo.add %1019, %1017 : tensor<7x768xf32>
    %1021 = stablehlo.reshape %1020 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1022 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %1023 = stablehlo.multiply %931, %1022 : tensor<1x7x768xf32>
    %1024 = stablehlo.add %1021, %1023 : tensor<1x7x768xf32>
    %1025 = stablehlo.reduce(%1024 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1026 = stablehlo.broadcast_in_dim %1025, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1027 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1028 = stablehlo.divide %1026, %1027 : tensor<1x7x1xf32>
    %1029 = call @_var(%1024, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %1030 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1031 = stablehlo.add %1029, %1030 : tensor<1x7x1xf32>
    %1032 = stablehlo.rsqrt %1031 : tensor<1x7x1xf32>
    %1033 = stablehlo.broadcast_in_dim %1028, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1034 = stablehlo.subtract %1024, %1033 : tensor<1x7x768xf32>
    %1035 = stablehlo.broadcast_in_dim %1032, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1036 = stablehlo.multiply %1034, %1035 : tensor<1x7x768xf32>
    %1037 = stablehlo.broadcast_in_dim %arg77, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1038 = stablehlo.broadcast_in_dim %1037, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1039 = stablehlo.multiply %1036, %1038 : tensor<1x7x768xf32>
    %1040 = stablehlo.broadcast_in_dim %arg78, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1041 = stablehlo.broadcast_in_dim %1040, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1042 = stablehlo.add %1039, %1041 : tensor<1x7x768xf32>
    %1043 = stablehlo.reshape %1042 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1044 = stablehlo.transpose %arg79, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1045 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %1046 = stablehlo.multiply %arg80, %1045 : tensor<3072xf32>
    %1047 = stablehlo.dot_general %1043, %1044, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %1048 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %1049 = stablehlo.multiply %1048, %1047 : tensor<7x3072xf32>
    %1050 = stablehlo.broadcast_in_dim %1046, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1051 = stablehlo.broadcast_in_dim %1050, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %1052 = stablehlo.add %1051, %1049 : tensor<7x3072xf32>
    %1053 = stablehlo.reshape %1052 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %1054 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1055 = stablehlo.multiply %1054, %1053 : tensor<1x7x3072xf32>
    %1056 = stablehlo.negate %1053 : tensor<1x7x3072xf32>
    %1057 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1058 = stablehlo.multiply %1056, %1057 : tensor<1x7x3072xf32>
    %1059 = stablehlo.multiply %1058, %1058 : tensor<1x7x3072xf32>
    %1060 = stablehlo.negate %1059 : tensor<1x7x3072xf32>
    %1061 = stablehlo.abs %1058 : tensor<1x7x3072xf32>
    %1062 = stablehlo.divide %cst_26, %1059 : tensor<1x7x3072xf32>
    %1063 = stablehlo.exponential %1060 : tensor<1x7x3072xf32>
    %1064 = stablehlo.divide %cst_26, %1061 : tensor<1x7x3072xf32>
    %1065 = stablehlo.multiply %1063, %1064 : tensor<1x7x3072xf32>
    %1066 = stablehlo.compare  LT, %1061, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1067 = stablehlo.multiply %cst_24, %1062 : tensor<1x7x3072xf32>
    %1068 = stablehlo.add %1067, %cst_23 : tensor<1x7x3072xf32>
    %1069 = stablehlo.multiply %1068, %1062 : tensor<1x7x3072xf32>
    %1070 = stablehlo.add %1069, %cst_22 : tensor<1x7x3072xf32>
    %1071 = stablehlo.multiply %1070, %1062 : tensor<1x7x3072xf32>
    %1072 = stablehlo.add %1071, %cst_21 : tensor<1x7x3072xf32>
    %1073 = stablehlo.multiply %1072, %1062 : tensor<1x7x3072xf32>
    %1074 = stablehlo.add %1073, %cst_20 : tensor<1x7x3072xf32>
    %1075 = stablehlo.multiply %1074, %1062 : tensor<1x7x3072xf32>
    %1076 = stablehlo.add %1075, %cst_19 : tensor<1x7x3072xf32>
    %1077 = stablehlo.multiply %1076, %1062 : tensor<1x7x3072xf32>
    %1078 = stablehlo.add %1077, %cst_18 : tensor<1x7x3072xf32>
    %1079 = stablehlo.multiply %1078, %1062 : tensor<1x7x3072xf32>
    %1080 = stablehlo.add %1079, %cst_17 : tensor<1x7x3072xf32>
    %1081 = stablehlo.multiply %1080, %1062 : tensor<1x7x3072xf32>
    %1082 = stablehlo.add %1081, %cst_16 : tensor<1x7x3072xf32>
    %1083 = stablehlo.multiply %cst_15, %1062 : tensor<1x7x3072xf32>
    %1084 = stablehlo.add %1083, %cst_14 : tensor<1x7x3072xf32>
    %1085 = stablehlo.multiply %1084, %1062 : tensor<1x7x3072xf32>
    %1086 = stablehlo.add %1085, %cst_13 : tensor<1x7x3072xf32>
    %1087 = stablehlo.multiply %1086, %1062 : tensor<1x7x3072xf32>
    %1088 = stablehlo.add %1087, %cst_12 : tensor<1x7x3072xf32>
    %1089 = stablehlo.multiply %1088, %1062 : tensor<1x7x3072xf32>
    %1090 = stablehlo.add %1089, %cst_11 : tensor<1x7x3072xf32>
    %1091 = stablehlo.multiply %1090, %1062 : tensor<1x7x3072xf32>
    %1092 = stablehlo.add %1091, %cst_10 : tensor<1x7x3072xf32>
    %1093 = stablehlo.multiply %1092, %1062 : tensor<1x7x3072xf32>
    %1094 = stablehlo.add %1093, %cst_9 : tensor<1x7x3072xf32>
    %1095 = stablehlo.multiply %1094, %1062 : tensor<1x7x3072xf32>
    %1096 = stablehlo.add %1095, %cst_8 : tensor<1x7x3072xf32>
    %1097 = stablehlo.select %1066, %1082, %1096 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1098 = stablehlo.multiply %1065, %1097 : tensor<1x7x3072xf32>
    %1099 = stablehlo.compare  LT, %1060, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1100 = stablehlo.select %1099, %cst_6, %1098 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1101 = stablehlo.compare  LT, %1058, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1102 = stablehlo.subtract %cst_25, %1100 : tensor<1x7x3072xf32>
    %1103 = stablehlo.select %1101, %1102, %1100 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1104 = stablehlo.multiply %1058, %1058 : tensor<1x7x3072xf32>
    %1105 = stablehlo.multiply %cst_5, %1104 : tensor<1x7x3072xf32>
    %1106 = stablehlo.add %1105, %cst_4 : tensor<1x7x3072xf32>
    %1107 = stablehlo.multiply %1106, %1104 : tensor<1x7x3072xf32>
    %1108 = stablehlo.add %1107, %cst_3 : tensor<1x7x3072xf32>
    %1109 = stablehlo.multiply %1108, %1104 : tensor<1x7x3072xf32>
    %1110 = stablehlo.add %1109, %cst_2 : tensor<1x7x3072xf32>
    %1111 = stablehlo.multiply %1110, %1104 : tensor<1x7x3072xf32>
    %1112 = stablehlo.add %1111, %cst_1 : tensor<1x7x3072xf32>
    %1113 = stablehlo.multiply %1112, %1104 : tensor<1x7x3072xf32>
    %1114 = stablehlo.add %1113, %cst_0 : tensor<1x7x3072xf32>
    %1115 = stablehlo.multiply %1114, %1104 : tensor<1x7x3072xf32>
    %1116 = stablehlo.add %1115, %cst : tensor<1x7x3072xf32>
    %1117 = stablehlo.multiply %1058, %1116 : tensor<1x7x3072xf32>
    %1118 = stablehlo.subtract %cst_26, %1117 : tensor<1x7x3072xf32>
    %1119 = stablehlo.abs %1058 : tensor<1x7x3072xf32>
    %1120 = stablehlo.compare  LT, %1119, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1121 = stablehlo.select %1120, %1118, %1103 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1122 = stablehlo.multiply %1055, %1121 : tensor<1x7x3072xf32>
    %1123 = stablehlo.reshape %1122 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %1124 = stablehlo.transpose %arg81, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1125 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1126 = stablehlo.multiply %arg82, %1125 : tensor<768xf32>
    %1127 = stablehlo.dot_general %1123, %1124, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %1128 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1129 = stablehlo.multiply %1128, %1127 : tensor<7x768xf32>
    %1130 = stablehlo.broadcast_in_dim %1126, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1131 = stablehlo.broadcast_in_dim %1130, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1132 = stablehlo.add %1131, %1129 : tensor<7x768xf32>
    %1133 = stablehlo.reshape %1132 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1134 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %1135 = stablehlo.multiply %1042, %1134 : tensor<1x7x768xf32>
    %1136 = stablehlo.add %1133, %1135 : tensor<1x7x768xf32>
    %1137 = stablehlo.reduce(%1136 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1138 = stablehlo.broadcast_in_dim %1137, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1139 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1140 = stablehlo.divide %1138, %1139 : tensor<1x7x1xf32>
    %1141 = call @_var(%1136, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %1142 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1143 = stablehlo.add %1141, %1142 : tensor<1x7x1xf32>
    %1144 = stablehlo.rsqrt %1143 : tensor<1x7x1xf32>
    %1145 = stablehlo.broadcast_in_dim %1140, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1146 = stablehlo.subtract %1136, %1145 : tensor<1x7x768xf32>
    %1147 = stablehlo.broadcast_in_dim %1144, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1148 = stablehlo.multiply %1146, %1147 : tensor<1x7x768xf32>
    %1149 = stablehlo.broadcast_in_dim %arg83, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1150 = stablehlo.broadcast_in_dim %1149, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1151 = stablehlo.multiply %1148, %1150 : tensor<1x7x768xf32>
    %1152 = stablehlo.broadcast_in_dim %arg84, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1153 = stablehlo.broadcast_in_dim %1152, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1154 = stablehlo.add %1151, %1153 : tensor<1x7x768xf32>
    %1155 = stablehlo.reshape %1154 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1156 = stablehlo.transpose %arg85, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1157 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1158 = stablehlo.multiply %arg86, %1157 : tensor<768xf32>
    %1159 = stablehlo.dot_general %1155, %1156, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1160 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1161 = stablehlo.multiply %1160, %1159 : tensor<7x768xf32>
    %1162 = stablehlo.broadcast_in_dim %1158, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1163 = stablehlo.broadcast_in_dim %1162, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1164 = stablehlo.add %1163, %1161 : tensor<7x768xf32>
    %1165 = stablehlo.reshape %1164 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1166 = stablehlo.reshape %1165 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1167 = stablehlo.transpose %1166, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1168 = stablehlo.reshape %1154 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1169 = stablehlo.transpose %arg87, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1170 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1171 = stablehlo.multiply %arg88, %1170 : tensor<768xf32>
    %1172 = stablehlo.dot_general %1168, %1169, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1173 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1174 = stablehlo.multiply %1173, %1172 : tensor<7x768xf32>
    %1175 = stablehlo.broadcast_in_dim %1171, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1176 = stablehlo.broadcast_in_dim %1175, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1177 = stablehlo.add %1176, %1174 : tensor<7x768xf32>
    %1178 = stablehlo.reshape %1177 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1179 = stablehlo.reshape %1178 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1180 = stablehlo.transpose %1179, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1181 = stablehlo.reshape %1154 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1182 = stablehlo.transpose %arg89, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1183 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1184 = stablehlo.multiply %arg90, %1183 : tensor<768xf32>
    %1185 = stablehlo.dot_general %1181, %1182, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1186 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1187 = stablehlo.multiply %1186, %1185 : tensor<7x768xf32>
    %1188 = stablehlo.broadcast_in_dim %1184, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1189 = stablehlo.broadcast_in_dim %1188, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1190 = stablehlo.add %1189, %1187 : tensor<7x768xf32>
    %1191 = stablehlo.reshape %1190 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1192 = stablehlo.reshape %1191 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1193 = stablehlo.transpose %1192, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1194 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %1195 = stablehlo.multiply %1167, %1194 : tensor<1x12x7x64xf32>
    %1196 = stablehlo.transpose %1180, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %1197 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %1198 = stablehlo.multiply %1196, %1197 : tensor<1x12x64x7xf32>
    %1199 = stablehlo.reshape %1195 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %1200 = stablehlo.reshape %1198 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %1201 = stablehlo.dot_general %1199, %1200, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %1202 = stablehlo.reshape %1201 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1203 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %1204 = stablehlo.multiply %39, %1203 : tensor<1x1x7x7xf32>
    %1205 = stablehlo.broadcast_in_dim %1204, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1206 = stablehlo.add %1202, %1205 : tensor<1x12x7x7xf32>
    %1207 = stablehlo.reduce(%1206 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1208 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %1209 = stablehlo.maximum %1208, %1207 : tensor<1x12x7xf32>
    %1210 = stablehlo.broadcast_in_dim %1209, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1211 = stablehlo.broadcast_in_dim %1210, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %1212 = stablehlo.subtract %1206, %1211 : tensor<1x12x7x7xf32>
    %1213 = stablehlo.exponential %1212 : tensor<1x12x7x7xf32>
    %1214 = stablehlo.reduce(%1213 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1215 = stablehlo.broadcast_in_dim %1214, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1216 = stablehlo.broadcast_in_dim %1215, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %1217 = stablehlo.divide %1213, %1216 : tensor<1x12x7x7xf32>
    %1218 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %1219 = stablehlo.compare  EQ, %1206, %1218,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %1220 = stablehlo.not %1219 : tensor<1x12x7x7xi1>
    %1221 = stablehlo.reduce(%1220 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %1222 = stablehlo.broadcast_in_dim %1221, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %1223 = stablehlo.not %1222 : tensor<1x12x7x1xi1>
    %1224 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %1225 = call @_where_4(%1223, %1224, %1217) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1226 = stablehlo.reshape %1225 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %1227 = stablehlo.reshape %1193 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %1228 = stablehlo.dot_general %1226, %1227, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %1229 = stablehlo.reshape %1228 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %1230 = stablehlo.transpose %1229, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %1231 = stablehlo.transpose %1230, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1232 = stablehlo.transpose %1231, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %1233 = stablehlo.reshape %1232 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %1234 = stablehlo.reshape %1233 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1235 = stablehlo.transpose %arg91, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1236 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1237 = stablehlo.multiply %arg92, %1236 : tensor<768xf32>
    %1238 = stablehlo.dot_general %1234, %1235, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1239 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1240 = stablehlo.multiply %1239, %1238 : tensor<7x768xf32>
    %1241 = stablehlo.broadcast_in_dim %1237, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1242 = stablehlo.broadcast_in_dim %1241, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1243 = stablehlo.add %1242, %1240 : tensor<7x768xf32>
    %1244 = stablehlo.reshape %1243 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1245 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %1246 = stablehlo.multiply %1154, %1245 : tensor<1x7x768xf32>
    %1247 = stablehlo.add %1244, %1246 : tensor<1x7x768xf32>
    %1248 = stablehlo.reduce(%1247 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1249 = stablehlo.broadcast_in_dim %1248, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1250 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1251 = stablehlo.divide %1249, %1250 : tensor<1x7x1xf32>
    %1252 = call @_var(%1247, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %1253 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1254 = stablehlo.add %1252, %1253 : tensor<1x7x1xf32>
    %1255 = stablehlo.rsqrt %1254 : tensor<1x7x1xf32>
    %1256 = stablehlo.broadcast_in_dim %1251, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1257 = stablehlo.subtract %1247, %1256 : tensor<1x7x768xf32>
    %1258 = stablehlo.broadcast_in_dim %1255, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1259 = stablehlo.multiply %1257, %1258 : tensor<1x7x768xf32>
    %1260 = stablehlo.broadcast_in_dim %arg93, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1261 = stablehlo.broadcast_in_dim %1260, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1262 = stablehlo.multiply %1259, %1261 : tensor<1x7x768xf32>
    %1263 = stablehlo.broadcast_in_dim %arg94, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1264 = stablehlo.broadcast_in_dim %1263, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1265 = stablehlo.add %1262, %1264 : tensor<1x7x768xf32>
    %1266 = stablehlo.reshape %1265 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1267 = stablehlo.transpose %arg95, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1268 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %1269 = stablehlo.multiply %arg96, %1268 : tensor<3072xf32>
    %1270 = stablehlo.dot_general %1266, %1267, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %1271 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %1272 = stablehlo.multiply %1271, %1270 : tensor<7x3072xf32>
    %1273 = stablehlo.broadcast_in_dim %1269, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1274 = stablehlo.broadcast_in_dim %1273, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %1275 = stablehlo.add %1274, %1272 : tensor<7x3072xf32>
    %1276 = stablehlo.reshape %1275 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %1277 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1278 = stablehlo.multiply %1277, %1276 : tensor<1x7x3072xf32>
    %1279 = stablehlo.negate %1276 : tensor<1x7x3072xf32>
    %1280 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1281 = stablehlo.multiply %1279, %1280 : tensor<1x7x3072xf32>
    %1282 = stablehlo.multiply %1281, %1281 : tensor<1x7x3072xf32>
    %1283 = stablehlo.negate %1282 : tensor<1x7x3072xf32>
    %1284 = stablehlo.abs %1281 : tensor<1x7x3072xf32>
    %1285 = stablehlo.divide %cst_26, %1282 : tensor<1x7x3072xf32>
    %1286 = stablehlo.exponential %1283 : tensor<1x7x3072xf32>
    %1287 = stablehlo.divide %cst_26, %1284 : tensor<1x7x3072xf32>
    %1288 = stablehlo.multiply %1286, %1287 : tensor<1x7x3072xf32>
    %1289 = stablehlo.compare  LT, %1284, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1290 = stablehlo.multiply %cst_24, %1285 : tensor<1x7x3072xf32>
    %1291 = stablehlo.add %1290, %cst_23 : tensor<1x7x3072xf32>
    %1292 = stablehlo.multiply %1291, %1285 : tensor<1x7x3072xf32>
    %1293 = stablehlo.add %1292, %cst_22 : tensor<1x7x3072xf32>
    %1294 = stablehlo.multiply %1293, %1285 : tensor<1x7x3072xf32>
    %1295 = stablehlo.add %1294, %cst_21 : tensor<1x7x3072xf32>
    %1296 = stablehlo.multiply %1295, %1285 : tensor<1x7x3072xf32>
    %1297 = stablehlo.add %1296, %cst_20 : tensor<1x7x3072xf32>
    %1298 = stablehlo.multiply %1297, %1285 : tensor<1x7x3072xf32>
    %1299 = stablehlo.add %1298, %cst_19 : tensor<1x7x3072xf32>
    %1300 = stablehlo.multiply %1299, %1285 : tensor<1x7x3072xf32>
    %1301 = stablehlo.add %1300, %cst_18 : tensor<1x7x3072xf32>
    %1302 = stablehlo.multiply %1301, %1285 : tensor<1x7x3072xf32>
    %1303 = stablehlo.add %1302, %cst_17 : tensor<1x7x3072xf32>
    %1304 = stablehlo.multiply %1303, %1285 : tensor<1x7x3072xf32>
    %1305 = stablehlo.add %1304, %cst_16 : tensor<1x7x3072xf32>
    %1306 = stablehlo.multiply %cst_15, %1285 : tensor<1x7x3072xf32>
    %1307 = stablehlo.add %1306, %cst_14 : tensor<1x7x3072xf32>
    %1308 = stablehlo.multiply %1307, %1285 : tensor<1x7x3072xf32>
    %1309 = stablehlo.add %1308, %cst_13 : tensor<1x7x3072xf32>
    %1310 = stablehlo.multiply %1309, %1285 : tensor<1x7x3072xf32>
    %1311 = stablehlo.add %1310, %cst_12 : tensor<1x7x3072xf32>
    %1312 = stablehlo.multiply %1311, %1285 : tensor<1x7x3072xf32>
    %1313 = stablehlo.add %1312, %cst_11 : tensor<1x7x3072xf32>
    %1314 = stablehlo.multiply %1313, %1285 : tensor<1x7x3072xf32>
    %1315 = stablehlo.add %1314, %cst_10 : tensor<1x7x3072xf32>
    %1316 = stablehlo.multiply %1315, %1285 : tensor<1x7x3072xf32>
    %1317 = stablehlo.add %1316, %cst_9 : tensor<1x7x3072xf32>
    %1318 = stablehlo.multiply %1317, %1285 : tensor<1x7x3072xf32>
    %1319 = stablehlo.add %1318, %cst_8 : tensor<1x7x3072xf32>
    %1320 = stablehlo.select %1289, %1305, %1319 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1321 = stablehlo.multiply %1288, %1320 : tensor<1x7x3072xf32>
    %1322 = stablehlo.compare  LT, %1283, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1323 = stablehlo.select %1322, %cst_6, %1321 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1324 = stablehlo.compare  LT, %1281, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1325 = stablehlo.subtract %cst_25, %1323 : tensor<1x7x3072xf32>
    %1326 = stablehlo.select %1324, %1325, %1323 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1327 = stablehlo.multiply %1281, %1281 : tensor<1x7x3072xf32>
    %1328 = stablehlo.multiply %cst_5, %1327 : tensor<1x7x3072xf32>
    %1329 = stablehlo.add %1328, %cst_4 : tensor<1x7x3072xf32>
    %1330 = stablehlo.multiply %1329, %1327 : tensor<1x7x3072xf32>
    %1331 = stablehlo.add %1330, %cst_3 : tensor<1x7x3072xf32>
    %1332 = stablehlo.multiply %1331, %1327 : tensor<1x7x3072xf32>
    %1333 = stablehlo.add %1332, %cst_2 : tensor<1x7x3072xf32>
    %1334 = stablehlo.multiply %1333, %1327 : tensor<1x7x3072xf32>
    %1335 = stablehlo.add %1334, %cst_1 : tensor<1x7x3072xf32>
    %1336 = stablehlo.multiply %1335, %1327 : tensor<1x7x3072xf32>
    %1337 = stablehlo.add %1336, %cst_0 : tensor<1x7x3072xf32>
    %1338 = stablehlo.multiply %1337, %1327 : tensor<1x7x3072xf32>
    %1339 = stablehlo.add %1338, %cst : tensor<1x7x3072xf32>
    %1340 = stablehlo.multiply %1281, %1339 : tensor<1x7x3072xf32>
    %1341 = stablehlo.subtract %cst_26, %1340 : tensor<1x7x3072xf32>
    %1342 = stablehlo.abs %1281 : tensor<1x7x3072xf32>
    %1343 = stablehlo.compare  LT, %1342, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1344 = stablehlo.select %1343, %1341, %1326 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1345 = stablehlo.multiply %1278, %1344 : tensor<1x7x3072xf32>
    %1346 = stablehlo.reshape %1345 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %1347 = stablehlo.transpose %arg97, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1348 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1349 = stablehlo.multiply %arg98, %1348 : tensor<768xf32>
    %1350 = stablehlo.dot_general %1346, %1347, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %1351 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1352 = stablehlo.multiply %1351, %1350 : tensor<7x768xf32>
    %1353 = stablehlo.broadcast_in_dim %1349, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1354 = stablehlo.broadcast_in_dim %1353, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1355 = stablehlo.add %1354, %1352 : tensor<7x768xf32>
    %1356 = stablehlo.reshape %1355 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1357 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %1358 = stablehlo.multiply %1265, %1357 : tensor<1x7x768xf32>
    %1359 = stablehlo.add %1356, %1358 : tensor<1x7x768xf32>
    %1360 = stablehlo.reduce(%1359 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1361 = stablehlo.broadcast_in_dim %1360, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1362 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1363 = stablehlo.divide %1361, %1362 : tensor<1x7x1xf32>
    %1364 = call @_var(%1359, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %1365 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1366 = stablehlo.add %1364, %1365 : tensor<1x7x1xf32>
    %1367 = stablehlo.rsqrt %1366 : tensor<1x7x1xf32>
    %1368 = stablehlo.broadcast_in_dim %1363, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1369 = stablehlo.subtract %1359, %1368 : tensor<1x7x768xf32>
    %1370 = stablehlo.broadcast_in_dim %1367, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1371 = stablehlo.multiply %1369, %1370 : tensor<1x7x768xf32>
    %1372 = stablehlo.broadcast_in_dim %arg99, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1373 = stablehlo.broadcast_in_dim %1372, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1374 = stablehlo.multiply %1371, %1373 : tensor<1x7x768xf32>
    %1375 = stablehlo.broadcast_in_dim %arg100, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1376 = stablehlo.broadcast_in_dim %1375, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1377 = stablehlo.add %1374, %1376 : tensor<1x7x768xf32>
    %1378 = stablehlo.reshape %1377 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1379 = stablehlo.transpose %arg101, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1380 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1381 = stablehlo.multiply %arg102, %1380 : tensor<768xf32>
    %1382 = stablehlo.dot_general %1378, %1379, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1383 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1384 = stablehlo.multiply %1383, %1382 : tensor<7x768xf32>
    %1385 = stablehlo.broadcast_in_dim %1381, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1386 = stablehlo.broadcast_in_dim %1385, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1387 = stablehlo.add %1386, %1384 : tensor<7x768xf32>
    %1388 = stablehlo.reshape %1387 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1389 = stablehlo.reshape %1388 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1390 = stablehlo.transpose %1389, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1391 = stablehlo.reshape %1377 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1392 = stablehlo.transpose %arg103, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1393 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1394 = stablehlo.multiply %arg104, %1393 : tensor<768xf32>
    %1395 = stablehlo.dot_general %1391, %1392, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1396 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1397 = stablehlo.multiply %1396, %1395 : tensor<7x768xf32>
    %1398 = stablehlo.broadcast_in_dim %1394, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1399 = stablehlo.broadcast_in_dim %1398, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1400 = stablehlo.add %1399, %1397 : tensor<7x768xf32>
    %1401 = stablehlo.reshape %1400 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1402 = stablehlo.reshape %1401 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1403 = stablehlo.transpose %1402, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1404 = stablehlo.reshape %1377 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1405 = stablehlo.transpose %arg105, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1406 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1407 = stablehlo.multiply %arg106, %1406 : tensor<768xf32>
    %1408 = stablehlo.dot_general %1404, %1405, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1409 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1410 = stablehlo.multiply %1409, %1408 : tensor<7x768xf32>
    %1411 = stablehlo.broadcast_in_dim %1407, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1412 = stablehlo.broadcast_in_dim %1411, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1413 = stablehlo.add %1412, %1410 : tensor<7x768xf32>
    %1414 = stablehlo.reshape %1413 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1415 = stablehlo.reshape %1414 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1416 = stablehlo.transpose %1415, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1417 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %1418 = stablehlo.multiply %1390, %1417 : tensor<1x12x7x64xf32>
    %1419 = stablehlo.transpose %1403, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %1420 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %1421 = stablehlo.multiply %1419, %1420 : tensor<1x12x64x7xf32>
    %1422 = stablehlo.reshape %1418 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %1423 = stablehlo.reshape %1421 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %1424 = stablehlo.dot_general %1422, %1423, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %1425 = stablehlo.reshape %1424 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1426 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %1427 = stablehlo.multiply %39, %1426 : tensor<1x1x7x7xf32>
    %1428 = stablehlo.broadcast_in_dim %1427, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1429 = stablehlo.add %1425, %1428 : tensor<1x12x7x7xf32>
    %1430 = stablehlo.reduce(%1429 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1431 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %1432 = stablehlo.maximum %1431, %1430 : tensor<1x12x7xf32>
    %1433 = stablehlo.broadcast_in_dim %1432, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1434 = stablehlo.broadcast_in_dim %1433, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %1435 = stablehlo.subtract %1429, %1434 : tensor<1x12x7x7xf32>
    %1436 = stablehlo.exponential %1435 : tensor<1x12x7x7xf32>
    %1437 = stablehlo.reduce(%1436 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1438 = stablehlo.broadcast_in_dim %1437, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1439 = stablehlo.broadcast_in_dim %1438, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %1440 = stablehlo.divide %1436, %1439 : tensor<1x12x7x7xf32>
    %1441 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %1442 = stablehlo.compare  EQ, %1429, %1441,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %1443 = stablehlo.not %1442 : tensor<1x12x7x7xi1>
    %1444 = stablehlo.reduce(%1443 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %1445 = stablehlo.broadcast_in_dim %1444, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %1446 = stablehlo.not %1445 : tensor<1x12x7x1xi1>
    %1447 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %1448 = call @_where_4(%1446, %1447, %1440) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1449 = stablehlo.reshape %1448 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %1450 = stablehlo.reshape %1416 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %1451 = stablehlo.dot_general %1449, %1450, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %1452 = stablehlo.reshape %1451 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %1453 = stablehlo.transpose %1452, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %1454 = stablehlo.transpose %1453, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1455 = stablehlo.transpose %1454, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %1456 = stablehlo.reshape %1455 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %1457 = stablehlo.reshape %1456 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1458 = stablehlo.transpose %arg107, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1459 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1460 = stablehlo.multiply %arg108, %1459 : tensor<768xf32>
    %1461 = stablehlo.dot_general %1457, %1458, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1462 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1463 = stablehlo.multiply %1462, %1461 : tensor<7x768xf32>
    %1464 = stablehlo.broadcast_in_dim %1460, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1465 = stablehlo.broadcast_in_dim %1464, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1466 = stablehlo.add %1465, %1463 : tensor<7x768xf32>
    %1467 = stablehlo.reshape %1466 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1468 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %1469 = stablehlo.multiply %1377, %1468 : tensor<1x7x768xf32>
    %1470 = stablehlo.add %1467, %1469 : tensor<1x7x768xf32>
    %1471 = stablehlo.reduce(%1470 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1472 = stablehlo.broadcast_in_dim %1471, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1473 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1474 = stablehlo.divide %1472, %1473 : tensor<1x7x1xf32>
    %1475 = call @_var(%1470, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %1476 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1477 = stablehlo.add %1475, %1476 : tensor<1x7x1xf32>
    %1478 = stablehlo.rsqrt %1477 : tensor<1x7x1xf32>
    %1479 = stablehlo.broadcast_in_dim %1474, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1480 = stablehlo.subtract %1470, %1479 : tensor<1x7x768xf32>
    %1481 = stablehlo.broadcast_in_dim %1478, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1482 = stablehlo.multiply %1480, %1481 : tensor<1x7x768xf32>
    %1483 = stablehlo.broadcast_in_dim %arg109, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1484 = stablehlo.broadcast_in_dim %1483, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1485 = stablehlo.multiply %1482, %1484 : tensor<1x7x768xf32>
    %1486 = stablehlo.broadcast_in_dim %arg110, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1487 = stablehlo.broadcast_in_dim %1486, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1488 = stablehlo.add %1485, %1487 : tensor<1x7x768xf32>
    %1489 = stablehlo.reshape %1488 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1490 = stablehlo.transpose %arg111, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1491 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %1492 = stablehlo.multiply %arg112, %1491 : tensor<3072xf32>
    %1493 = stablehlo.dot_general %1489, %1490, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %1494 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %1495 = stablehlo.multiply %1494, %1493 : tensor<7x3072xf32>
    %1496 = stablehlo.broadcast_in_dim %1492, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1497 = stablehlo.broadcast_in_dim %1496, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %1498 = stablehlo.add %1497, %1495 : tensor<7x3072xf32>
    %1499 = stablehlo.reshape %1498 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %1500 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1501 = stablehlo.multiply %1500, %1499 : tensor<1x7x3072xf32>
    %1502 = stablehlo.negate %1499 : tensor<1x7x3072xf32>
    %1503 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1504 = stablehlo.multiply %1502, %1503 : tensor<1x7x3072xf32>
    %1505 = stablehlo.multiply %1504, %1504 : tensor<1x7x3072xf32>
    %1506 = stablehlo.negate %1505 : tensor<1x7x3072xf32>
    %1507 = stablehlo.abs %1504 : tensor<1x7x3072xf32>
    %1508 = stablehlo.divide %cst_26, %1505 : tensor<1x7x3072xf32>
    %1509 = stablehlo.exponential %1506 : tensor<1x7x3072xf32>
    %1510 = stablehlo.divide %cst_26, %1507 : tensor<1x7x3072xf32>
    %1511 = stablehlo.multiply %1509, %1510 : tensor<1x7x3072xf32>
    %1512 = stablehlo.compare  LT, %1507, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1513 = stablehlo.multiply %cst_24, %1508 : tensor<1x7x3072xf32>
    %1514 = stablehlo.add %1513, %cst_23 : tensor<1x7x3072xf32>
    %1515 = stablehlo.multiply %1514, %1508 : tensor<1x7x3072xf32>
    %1516 = stablehlo.add %1515, %cst_22 : tensor<1x7x3072xf32>
    %1517 = stablehlo.multiply %1516, %1508 : tensor<1x7x3072xf32>
    %1518 = stablehlo.add %1517, %cst_21 : tensor<1x7x3072xf32>
    %1519 = stablehlo.multiply %1518, %1508 : tensor<1x7x3072xf32>
    %1520 = stablehlo.add %1519, %cst_20 : tensor<1x7x3072xf32>
    %1521 = stablehlo.multiply %1520, %1508 : tensor<1x7x3072xf32>
    %1522 = stablehlo.add %1521, %cst_19 : tensor<1x7x3072xf32>
    %1523 = stablehlo.multiply %1522, %1508 : tensor<1x7x3072xf32>
    %1524 = stablehlo.add %1523, %cst_18 : tensor<1x7x3072xf32>
    %1525 = stablehlo.multiply %1524, %1508 : tensor<1x7x3072xf32>
    %1526 = stablehlo.add %1525, %cst_17 : tensor<1x7x3072xf32>
    %1527 = stablehlo.multiply %1526, %1508 : tensor<1x7x3072xf32>
    %1528 = stablehlo.add %1527, %cst_16 : tensor<1x7x3072xf32>
    %1529 = stablehlo.multiply %cst_15, %1508 : tensor<1x7x3072xf32>
    %1530 = stablehlo.add %1529, %cst_14 : tensor<1x7x3072xf32>
    %1531 = stablehlo.multiply %1530, %1508 : tensor<1x7x3072xf32>
    %1532 = stablehlo.add %1531, %cst_13 : tensor<1x7x3072xf32>
    %1533 = stablehlo.multiply %1532, %1508 : tensor<1x7x3072xf32>
    %1534 = stablehlo.add %1533, %cst_12 : tensor<1x7x3072xf32>
    %1535 = stablehlo.multiply %1534, %1508 : tensor<1x7x3072xf32>
    %1536 = stablehlo.add %1535, %cst_11 : tensor<1x7x3072xf32>
    %1537 = stablehlo.multiply %1536, %1508 : tensor<1x7x3072xf32>
    %1538 = stablehlo.add %1537, %cst_10 : tensor<1x7x3072xf32>
    %1539 = stablehlo.multiply %1538, %1508 : tensor<1x7x3072xf32>
    %1540 = stablehlo.add %1539, %cst_9 : tensor<1x7x3072xf32>
    %1541 = stablehlo.multiply %1540, %1508 : tensor<1x7x3072xf32>
    %1542 = stablehlo.add %1541, %cst_8 : tensor<1x7x3072xf32>
    %1543 = stablehlo.select %1512, %1528, %1542 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1544 = stablehlo.multiply %1511, %1543 : tensor<1x7x3072xf32>
    %1545 = stablehlo.compare  LT, %1506, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1546 = stablehlo.select %1545, %cst_6, %1544 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1547 = stablehlo.compare  LT, %1504, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1548 = stablehlo.subtract %cst_25, %1546 : tensor<1x7x3072xf32>
    %1549 = stablehlo.select %1547, %1548, %1546 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1550 = stablehlo.multiply %1504, %1504 : tensor<1x7x3072xf32>
    %1551 = stablehlo.multiply %cst_5, %1550 : tensor<1x7x3072xf32>
    %1552 = stablehlo.add %1551, %cst_4 : tensor<1x7x3072xf32>
    %1553 = stablehlo.multiply %1552, %1550 : tensor<1x7x3072xf32>
    %1554 = stablehlo.add %1553, %cst_3 : tensor<1x7x3072xf32>
    %1555 = stablehlo.multiply %1554, %1550 : tensor<1x7x3072xf32>
    %1556 = stablehlo.add %1555, %cst_2 : tensor<1x7x3072xf32>
    %1557 = stablehlo.multiply %1556, %1550 : tensor<1x7x3072xf32>
    %1558 = stablehlo.add %1557, %cst_1 : tensor<1x7x3072xf32>
    %1559 = stablehlo.multiply %1558, %1550 : tensor<1x7x3072xf32>
    %1560 = stablehlo.add %1559, %cst_0 : tensor<1x7x3072xf32>
    %1561 = stablehlo.multiply %1560, %1550 : tensor<1x7x3072xf32>
    %1562 = stablehlo.add %1561, %cst : tensor<1x7x3072xf32>
    %1563 = stablehlo.multiply %1504, %1562 : tensor<1x7x3072xf32>
    %1564 = stablehlo.subtract %cst_26, %1563 : tensor<1x7x3072xf32>
    %1565 = stablehlo.abs %1504 : tensor<1x7x3072xf32>
    %1566 = stablehlo.compare  LT, %1565, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1567 = stablehlo.select %1566, %1564, %1549 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1568 = stablehlo.multiply %1501, %1567 : tensor<1x7x3072xf32>
    %1569 = stablehlo.reshape %1568 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %1570 = stablehlo.transpose %arg113, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1571 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1572 = stablehlo.multiply %arg114, %1571 : tensor<768xf32>
    %1573 = stablehlo.dot_general %1569, %1570, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %1574 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1575 = stablehlo.multiply %1574, %1573 : tensor<7x768xf32>
    %1576 = stablehlo.broadcast_in_dim %1572, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1577 = stablehlo.broadcast_in_dim %1576, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1578 = stablehlo.add %1577, %1575 : tensor<7x768xf32>
    %1579 = stablehlo.reshape %1578 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1580 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %1581 = stablehlo.multiply %1488, %1580 : tensor<1x7x768xf32>
    %1582 = stablehlo.add %1579, %1581 : tensor<1x7x768xf32>
    %1583 = stablehlo.reduce(%1582 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1584 = stablehlo.broadcast_in_dim %1583, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1585 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1586 = stablehlo.divide %1584, %1585 : tensor<1x7x1xf32>
    %1587 = call @_var(%1582, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %1588 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1589 = stablehlo.add %1587, %1588 : tensor<1x7x1xf32>
    %1590 = stablehlo.rsqrt %1589 : tensor<1x7x1xf32>
    %1591 = stablehlo.broadcast_in_dim %1586, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1592 = stablehlo.subtract %1582, %1591 : tensor<1x7x768xf32>
    %1593 = stablehlo.broadcast_in_dim %1590, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1594 = stablehlo.multiply %1592, %1593 : tensor<1x7x768xf32>
    %1595 = stablehlo.broadcast_in_dim %arg115, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1596 = stablehlo.broadcast_in_dim %1595, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1597 = stablehlo.multiply %1594, %1596 : tensor<1x7x768xf32>
    %1598 = stablehlo.broadcast_in_dim %arg116, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1599 = stablehlo.broadcast_in_dim %1598, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1600 = stablehlo.add %1597, %1599 : tensor<1x7x768xf32>
    %1601 = stablehlo.reshape %1600 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1602 = stablehlo.transpose %arg117, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1603 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1604 = stablehlo.multiply %arg118, %1603 : tensor<768xf32>
    %1605 = stablehlo.dot_general %1601, %1602, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1606 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1607 = stablehlo.multiply %1606, %1605 : tensor<7x768xf32>
    %1608 = stablehlo.broadcast_in_dim %1604, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1609 = stablehlo.broadcast_in_dim %1608, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1610 = stablehlo.add %1609, %1607 : tensor<7x768xf32>
    %1611 = stablehlo.reshape %1610 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1612 = stablehlo.reshape %1611 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1613 = stablehlo.transpose %1612, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1614 = stablehlo.reshape %1600 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1615 = stablehlo.transpose %arg119, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1616 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1617 = stablehlo.multiply %arg120, %1616 : tensor<768xf32>
    %1618 = stablehlo.dot_general %1614, %1615, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1619 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1620 = stablehlo.multiply %1619, %1618 : tensor<7x768xf32>
    %1621 = stablehlo.broadcast_in_dim %1617, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1622 = stablehlo.broadcast_in_dim %1621, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1623 = stablehlo.add %1622, %1620 : tensor<7x768xf32>
    %1624 = stablehlo.reshape %1623 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1625 = stablehlo.reshape %1624 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1626 = stablehlo.transpose %1625, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1627 = stablehlo.reshape %1600 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1628 = stablehlo.transpose %arg121, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1629 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1630 = stablehlo.multiply %arg122, %1629 : tensor<768xf32>
    %1631 = stablehlo.dot_general %1627, %1628, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1632 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1633 = stablehlo.multiply %1632, %1631 : tensor<7x768xf32>
    %1634 = stablehlo.broadcast_in_dim %1630, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1635 = stablehlo.broadcast_in_dim %1634, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1636 = stablehlo.add %1635, %1633 : tensor<7x768xf32>
    %1637 = stablehlo.reshape %1636 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1638 = stablehlo.reshape %1637 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1639 = stablehlo.transpose %1638, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1640 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %1641 = stablehlo.multiply %1613, %1640 : tensor<1x12x7x64xf32>
    %1642 = stablehlo.transpose %1626, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %1643 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %1644 = stablehlo.multiply %1642, %1643 : tensor<1x12x64x7xf32>
    %1645 = stablehlo.reshape %1641 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %1646 = stablehlo.reshape %1644 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %1647 = stablehlo.dot_general %1645, %1646, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %1648 = stablehlo.reshape %1647 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1649 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %1650 = stablehlo.multiply %39, %1649 : tensor<1x1x7x7xf32>
    %1651 = stablehlo.broadcast_in_dim %1650, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1652 = stablehlo.add %1648, %1651 : tensor<1x12x7x7xf32>
    %1653 = stablehlo.reduce(%1652 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1654 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %1655 = stablehlo.maximum %1654, %1653 : tensor<1x12x7xf32>
    %1656 = stablehlo.broadcast_in_dim %1655, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1657 = stablehlo.broadcast_in_dim %1656, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %1658 = stablehlo.subtract %1652, %1657 : tensor<1x12x7x7xf32>
    %1659 = stablehlo.exponential %1658 : tensor<1x12x7x7xf32>
    %1660 = stablehlo.reduce(%1659 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1661 = stablehlo.broadcast_in_dim %1660, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1662 = stablehlo.broadcast_in_dim %1661, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %1663 = stablehlo.divide %1659, %1662 : tensor<1x12x7x7xf32>
    %1664 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %1665 = stablehlo.compare  EQ, %1652, %1664,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %1666 = stablehlo.not %1665 : tensor<1x12x7x7xi1>
    %1667 = stablehlo.reduce(%1666 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %1668 = stablehlo.broadcast_in_dim %1667, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %1669 = stablehlo.not %1668 : tensor<1x12x7x1xi1>
    %1670 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %1671 = call @_where_4(%1669, %1670, %1663) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1672 = stablehlo.reshape %1671 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %1673 = stablehlo.reshape %1639 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %1674 = stablehlo.dot_general %1672, %1673, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %1675 = stablehlo.reshape %1674 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %1676 = stablehlo.transpose %1675, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %1677 = stablehlo.transpose %1676, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1678 = stablehlo.transpose %1677, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %1679 = stablehlo.reshape %1678 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %1680 = stablehlo.reshape %1679 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1681 = stablehlo.transpose %arg123, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1682 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1683 = stablehlo.multiply %arg124, %1682 : tensor<768xf32>
    %1684 = stablehlo.dot_general %1680, %1681, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1685 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1686 = stablehlo.multiply %1685, %1684 : tensor<7x768xf32>
    %1687 = stablehlo.broadcast_in_dim %1683, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1688 = stablehlo.broadcast_in_dim %1687, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1689 = stablehlo.add %1688, %1686 : tensor<7x768xf32>
    %1690 = stablehlo.reshape %1689 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1691 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %1692 = stablehlo.multiply %1600, %1691 : tensor<1x7x768xf32>
    %1693 = stablehlo.add %1690, %1692 : tensor<1x7x768xf32>
    %1694 = stablehlo.reduce(%1693 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1695 = stablehlo.broadcast_in_dim %1694, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1696 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1697 = stablehlo.divide %1695, %1696 : tensor<1x7x1xf32>
    %1698 = call @_var(%1693, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %1699 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1700 = stablehlo.add %1698, %1699 : tensor<1x7x1xf32>
    %1701 = stablehlo.rsqrt %1700 : tensor<1x7x1xf32>
    %1702 = stablehlo.broadcast_in_dim %1697, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1703 = stablehlo.subtract %1693, %1702 : tensor<1x7x768xf32>
    %1704 = stablehlo.broadcast_in_dim %1701, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1705 = stablehlo.multiply %1703, %1704 : tensor<1x7x768xf32>
    %1706 = stablehlo.broadcast_in_dim %arg125, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1707 = stablehlo.broadcast_in_dim %1706, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1708 = stablehlo.multiply %1705, %1707 : tensor<1x7x768xf32>
    %1709 = stablehlo.broadcast_in_dim %arg126, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1710 = stablehlo.broadcast_in_dim %1709, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1711 = stablehlo.add %1708, %1710 : tensor<1x7x768xf32>
    %1712 = stablehlo.reshape %1711 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1713 = stablehlo.transpose %arg127, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1714 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %1715 = stablehlo.multiply %arg128, %1714 : tensor<3072xf32>
    %1716 = stablehlo.dot_general %1712, %1713, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %1717 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %1718 = stablehlo.multiply %1717, %1716 : tensor<7x3072xf32>
    %1719 = stablehlo.broadcast_in_dim %1715, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1720 = stablehlo.broadcast_in_dim %1719, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %1721 = stablehlo.add %1720, %1718 : tensor<7x3072xf32>
    %1722 = stablehlo.reshape %1721 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %1723 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1724 = stablehlo.multiply %1723, %1722 : tensor<1x7x3072xf32>
    %1725 = stablehlo.negate %1722 : tensor<1x7x3072xf32>
    %1726 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1727 = stablehlo.multiply %1725, %1726 : tensor<1x7x3072xf32>
    %1728 = stablehlo.multiply %1727, %1727 : tensor<1x7x3072xf32>
    %1729 = stablehlo.negate %1728 : tensor<1x7x3072xf32>
    %1730 = stablehlo.abs %1727 : tensor<1x7x3072xf32>
    %1731 = stablehlo.divide %cst_26, %1728 : tensor<1x7x3072xf32>
    %1732 = stablehlo.exponential %1729 : tensor<1x7x3072xf32>
    %1733 = stablehlo.divide %cst_26, %1730 : tensor<1x7x3072xf32>
    %1734 = stablehlo.multiply %1732, %1733 : tensor<1x7x3072xf32>
    %1735 = stablehlo.compare  LT, %1730, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1736 = stablehlo.multiply %cst_24, %1731 : tensor<1x7x3072xf32>
    %1737 = stablehlo.add %1736, %cst_23 : tensor<1x7x3072xf32>
    %1738 = stablehlo.multiply %1737, %1731 : tensor<1x7x3072xf32>
    %1739 = stablehlo.add %1738, %cst_22 : tensor<1x7x3072xf32>
    %1740 = stablehlo.multiply %1739, %1731 : tensor<1x7x3072xf32>
    %1741 = stablehlo.add %1740, %cst_21 : tensor<1x7x3072xf32>
    %1742 = stablehlo.multiply %1741, %1731 : tensor<1x7x3072xf32>
    %1743 = stablehlo.add %1742, %cst_20 : tensor<1x7x3072xf32>
    %1744 = stablehlo.multiply %1743, %1731 : tensor<1x7x3072xf32>
    %1745 = stablehlo.add %1744, %cst_19 : tensor<1x7x3072xf32>
    %1746 = stablehlo.multiply %1745, %1731 : tensor<1x7x3072xf32>
    %1747 = stablehlo.add %1746, %cst_18 : tensor<1x7x3072xf32>
    %1748 = stablehlo.multiply %1747, %1731 : tensor<1x7x3072xf32>
    %1749 = stablehlo.add %1748, %cst_17 : tensor<1x7x3072xf32>
    %1750 = stablehlo.multiply %1749, %1731 : tensor<1x7x3072xf32>
    %1751 = stablehlo.add %1750, %cst_16 : tensor<1x7x3072xf32>
    %1752 = stablehlo.multiply %cst_15, %1731 : tensor<1x7x3072xf32>
    %1753 = stablehlo.add %1752, %cst_14 : tensor<1x7x3072xf32>
    %1754 = stablehlo.multiply %1753, %1731 : tensor<1x7x3072xf32>
    %1755 = stablehlo.add %1754, %cst_13 : tensor<1x7x3072xf32>
    %1756 = stablehlo.multiply %1755, %1731 : tensor<1x7x3072xf32>
    %1757 = stablehlo.add %1756, %cst_12 : tensor<1x7x3072xf32>
    %1758 = stablehlo.multiply %1757, %1731 : tensor<1x7x3072xf32>
    %1759 = stablehlo.add %1758, %cst_11 : tensor<1x7x3072xf32>
    %1760 = stablehlo.multiply %1759, %1731 : tensor<1x7x3072xf32>
    %1761 = stablehlo.add %1760, %cst_10 : tensor<1x7x3072xf32>
    %1762 = stablehlo.multiply %1761, %1731 : tensor<1x7x3072xf32>
    %1763 = stablehlo.add %1762, %cst_9 : tensor<1x7x3072xf32>
    %1764 = stablehlo.multiply %1763, %1731 : tensor<1x7x3072xf32>
    %1765 = stablehlo.add %1764, %cst_8 : tensor<1x7x3072xf32>
    %1766 = stablehlo.select %1735, %1751, %1765 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1767 = stablehlo.multiply %1734, %1766 : tensor<1x7x3072xf32>
    %1768 = stablehlo.compare  LT, %1729, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1769 = stablehlo.select %1768, %cst_6, %1767 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1770 = stablehlo.compare  LT, %1727, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1771 = stablehlo.subtract %cst_25, %1769 : tensor<1x7x3072xf32>
    %1772 = stablehlo.select %1770, %1771, %1769 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1773 = stablehlo.multiply %1727, %1727 : tensor<1x7x3072xf32>
    %1774 = stablehlo.multiply %cst_5, %1773 : tensor<1x7x3072xf32>
    %1775 = stablehlo.add %1774, %cst_4 : tensor<1x7x3072xf32>
    %1776 = stablehlo.multiply %1775, %1773 : tensor<1x7x3072xf32>
    %1777 = stablehlo.add %1776, %cst_3 : tensor<1x7x3072xf32>
    %1778 = stablehlo.multiply %1777, %1773 : tensor<1x7x3072xf32>
    %1779 = stablehlo.add %1778, %cst_2 : tensor<1x7x3072xf32>
    %1780 = stablehlo.multiply %1779, %1773 : tensor<1x7x3072xf32>
    %1781 = stablehlo.add %1780, %cst_1 : tensor<1x7x3072xf32>
    %1782 = stablehlo.multiply %1781, %1773 : tensor<1x7x3072xf32>
    %1783 = stablehlo.add %1782, %cst_0 : tensor<1x7x3072xf32>
    %1784 = stablehlo.multiply %1783, %1773 : tensor<1x7x3072xf32>
    %1785 = stablehlo.add %1784, %cst : tensor<1x7x3072xf32>
    %1786 = stablehlo.multiply %1727, %1785 : tensor<1x7x3072xf32>
    %1787 = stablehlo.subtract %cst_26, %1786 : tensor<1x7x3072xf32>
    %1788 = stablehlo.abs %1727 : tensor<1x7x3072xf32>
    %1789 = stablehlo.compare  LT, %1788, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1790 = stablehlo.select %1789, %1787, %1772 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1791 = stablehlo.multiply %1724, %1790 : tensor<1x7x3072xf32>
    %1792 = stablehlo.reshape %1791 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %1793 = stablehlo.transpose %arg129, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %1794 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1795 = stablehlo.multiply %arg130, %1794 : tensor<768xf32>
    %1796 = stablehlo.dot_general %1792, %1793, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %1797 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1798 = stablehlo.multiply %1797, %1796 : tensor<7x768xf32>
    %1799 = stablehlo.broadcast_in_dim %1795, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1800 = stablehlo.broadcast_in_dim %1799, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1801 = stablehlo.add %1800, %1798 : tensor<7x768xf32>
    %1802 = stablehlo.reshape %1801 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1803 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %1804 = stablehlo.multiply %1711, %1803 : tensor<1x7x768xf32>
    %1805 = stablehlo.add %1802, %1804 : tensor<1x7x768xf32>
    %1806 = stablehlo.reduce(%1805 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1807 = stablehlo.broadcast_in_dim %1806, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1808 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1809 = stablehlo.divide %1807, %1808 : tensor<1x7x1xf32>
    %1810 = call @_var(%1805, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %1811 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1812 = stablehlo.add %1810, %1811 : tensor<1x7x1xf32>
    %1813 = stablehlo.rsqrt %1812 : tensor<1x7x1xf32>
    %1814 = stablehlo.broadcast_in_dim %1809, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1815 = stablehlo.subtract %1805, %1814 : tensor<1x7x768xf32>
    %1816 = stablehlo.broadcast_in_dim %1813, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1817 = stablehlo.multiply %1815, %1816 : tensor<1x7x768xf32>
    %1818 = stablehlo.broadcast_in_dim %arg131, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1819 = stablehlo.broadcast_in_dim %1818, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1820 = stablehlo.multiply %1817, %1819 : tensor<1x7x768xf32>
    %1821 = stablehlo.broadcast_in_dim %arg132, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1822 = stablehlo.broadcast_in_dim %1821, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1823 = stablehlo.add %1820, %1822 : tensor<1x7x768xf32>
    %1824 = stablehlo.reshape %1823 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1825 = stablehlo.transpose %arg133, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1826 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1827 = stablehlo.multiply %arg134, %1826 : tensor<768xf32>
    %1828 = stablehlo.dot_general %1824, %1825, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1829 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1830 = stablehlo.multiply %1829, %1828 : tensor<7x768xf32>
    %1831 = stablehlo.broadcast_in_dim %1827, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1832 = stablehlo.broadcast_in_dim %1831, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1833 = stablehlo.add %1832, %1830 : tensor<7x768xf32>
    %1834 = stablehlo.reshape %1833 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1835 = stablehlo.reshape %1834 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1836 = stablehlo.transpose %1835, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1837 = stablehlo.reshape %1823 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1838 = stablehlo.transpose %arg135, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1839 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1840 = stablehlo.multiply %arg136, %1839 : tensor<768xf32>
    %1841 = stablehlo.dot_general %1837, %1838, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1842 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1843 = stablehlo.multiply %1842, %1841 : tensor<7x768xf32>
    %1844 = stablehlo.broadcast_in_dim %1840, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1845 = stablehlo.broadcast_in_dim %1844, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1846 = stablehlo.add %1845, %1843 : tensor<7x768xf32>
    %1847 = stablehlo.reshape %1846 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1848 = stablehlo.reshape %1847 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1849 = stablehlo.transpose %1848, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1850 = stablehlo.reshape %1823 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1851 = stablehlo.transpose %arg137, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1852 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1853 = stablehlo.multiply %arg138, %1852 : tensor<768xf32>
    %1854 = stablehlo.dot_general %1850, %1851, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1855 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1856 = stablehlo.multiply %1855, %1854 : tensor<7x768xf32>
    %1857 = stablehlo.broadcast_in_dim %1853, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1858 = stablehlo.broadcast_in_dim %1857, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1859 = stablehlo.add %1858, %1856 : tensor<7x768xf32>
    %1860 = stablehlo.reshape %1859 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1861 = stablehlo.reshape %1860 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %1862 = stablehlo.transpose %1861, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1863 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %1864 = stablehlo.multiply %1836, %1863 : tensor<1x12x7x64xf32>
    %1865 = stablehlo.transpose %1849, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %1866 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %1867 = stablehlo.multiply %1865, %1866 : tensor<1x12x64x7xf32>
    %1868 = stablehlo.reshape %1864 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %1869 = stablehlo.reshape %1867 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %1870 = stablehlo.dot_general %1868, %1869, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %1871 = stablehlo.reshape %1870 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1872 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %1873 = stablehlo.multiply %39, %1872 : tensor<1x1x7x7xf32>
    %1874 = stablehlo.broadcast_in_dim %1873, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1875 = stablehlo.add %1871, %1874 : tensor<1x12x7x7xf32>
    %1876 = stablehlo.reduce(%1875 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1877 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %1878 = stablehlo.maximum %1877, %1876 : tensor<1x12x7xf32>
    %1879 = stablehlo.broadcast_in_dim %1878, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1880 = stablehlo.broadcast_in_dim %1879, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %1881 = stablehlo.subtract %1875, %1880 : tensor<1x12x7x7xf32>
    %1882 = stablehlo.exponential %1881 : tensor<1x12x7x7xf32>
    %1883 = stablehlo.reduce(%1882 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %1884 = stablehlo.broadcast_in_dim %1883, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %1885 = stablehlo.broadcast_in_dim %1884, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %1886 = stablehlo.divide %1882, %1885 : tensor<1x12x7x7xf32>
    %1887 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %1888 = stablehlo.compare  EQ, %1875, %1887,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %1889 = stablehlo.not %1888 : tensor<1x12x7x7xi1>
    %1890 = stablehlo.reduce(%1889 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %1891 = stablehlo.broadcast_in_dim %1890, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %1892 = stablehlo.not %1891 : tensor<1x12x7x1xi1>
    %1893 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %1894 = call @_where_4(%1892, %1893, %1886) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %1895 = stablehlo.reshape %1894 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %1896 = stablehlo.reshape %1862 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %1897 = stablehlo.dot_general %1895, %1896, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %1898 = stablehlo.reshape %1897 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %1899 = stablehlo.transpose %1898, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %1900 = stablehlo.transpose %1899, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %1901 = stablehlo.transpose %1900, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %1902 = stablehlo.reshape %1901 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %1903 = stablehlo.reshape %1902 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1904 = stablehlo.transpose %arg139, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %1905 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %1906 = stablehlo.multiply %arg140, %1905 : tensor<768xf32>
    %1907 = stablehlo.dot_general %1903, %1904, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %1908 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %1909 = stablehlo.multiply %1908, %1907 : tensor<7x768xf32>
    %1910 = stablehlo.broadcast_in_dim %1906, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %1911 = stablehlo.broadcast_in_dim %1910, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %1912 = stablehlo.add %1911, %1909 : tensor<7x768xf32>
    %1913 = stablehlo.reshape %1912 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %1914 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %1915 = stablehlo.multiply %1823, %1914 : tensor<1x7x768xf32>
    %1916 = stablehlo.add %1913, %1915 : tensor<1x7x768xf32>
    %1917 = stablehlo.reduce(%1916 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1918 = stablehlo.broadcast_in_dim %1917, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %1919 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1920 = stablehlo.divide %1918, %1919 : tensor<1x7x1xf32>
    %1921 = call @_var(%1916, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %1922 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %1923 = stablehlo.add %1921, %1922 : tensor<1x7x1xf32>
    %1924 = stablehlo.rsqrt %1923 : tensor<1x7x1xf32>
    %1925 = stablehlo.broadcast_in_dim %1920, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1926 = stablehlo.subtract %1916, %1925 : tensor<1x7x768xf32>
    %1927 = stablehlo.broadcast_in_dim %1924, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %1928 = stablehlo.multiply %1926, %1927 : tensor<1x7x768xf32>
    %1929 = stablehlo.broadcast_in_dim %arg141, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1930 = stablehlo.broadcast_in_dim %1929, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1931 = stablehlo.multiply %1928, %1930 : tensor<1x7x768xf32>
    %1932 = stablehlo.broadcast_in_dim %arg142, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %1933 = stablehlo.broadcast_in_dim %1932, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %1934 = stablehlo.add %1931, %1933 : tensor<1x7x768xf32>
    %1935 = stablehlo.reshape %1934 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %1936 = stablehlo.transpose %arg143, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %1937 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %1938 = stablehlo.multiply %arg144, %1937 : tensor<3072xf32>
    %1939 = stablehlo.dot_general %1935, %1936, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %1940 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %1941 = stablehlo.multiply %1940, %1939 : tensor<7x3072xf32>
    %1942 = stablehlo.broadcast_in_dim %1938, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %1943 = stablehlo.broadcast_in_dim %1942, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %1944 = stablehlo.add %1943, %1941 : tensor<7x3072xf32>
    %1945 = stablehlo.reshape %1944 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %1946 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1947 = stablehlo.multiply %1946, %1945 : tensor<1x7x3072xf32>
    %1948 = stablehlo.negate %1945 : tensor<1x7x3072xf32>
    %1949 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %1950 = stablehlo.multiply %1948, %1949 : tensor<1x7x3072xf32>
    %1951 = stablehlo.multiply %1950, %1950 : tensor<1x7x3072xf32>
    %1952 = stablehlo.negate %1951 : tensor<1x7x3072xf32>
    %1953 = stablehlo.abs %1950 : tensor<1x7x3072xf32>
    %1954 = stablehlo.divide %cst_26, %1951 : tensor<1x7x3072xf32>
    %1955 = stablehlo.exponential %1952 : tensor<1x7x3072xf32>
    %1956 = stablehlo.divide %cst_26, %1953 : tensor<1x7x3072xf32>
    %1957 = stablehlo.multiply %1955, %1956 : tensor<1x7x3072xf32>
    %1958 = stablehlo.compare  LT, %1953, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1959 = stablehlo.multiply %cst_24, %1954 : tensor<1x7x3072xf32>
    %1960 = stablehlo.add %1959, %cst_23 : tensor<1x7x3072xf32>
    %1961 = stablehlo.multiply %1960, %1954 : tensor<1x7x3072xf32>
    %1962 = stablehlo.add %1961, %cst_22 : tensor<1x7x3072xf32>
    %1963 = stablehlo.multiply %1962, %1954 : tensor<1x7x3072xf32>
    %1964 = stablehlo.add %1963, %cst_21 : tensor<1x7x3072xf32>
    %1965 = stablehlo.multiply %1964, %1954 : tensor<1x7x3072xf32>
    %1966 = stablehlo.add %1965, %cst_20 : tensor<1x7x3072xf32>
    %1967 = stablehlo.multiply %1966, %1954 : tensor<1x7x3072xf32>
    %1968 = stablehlo.add %1967, %cst_19 : tensor<1x7x3072xf32>
    %1969 = stablehlo.multiply %1968, %1954 : tensor<1x7x3072xf32>
    %1970 = stablehlo.add %1969, %cst_18 : tensor<1x7x3072xf32>
    %1971 = stablehlo.multiply %1970, %1954 : tensor<1x7x3072xf32>
    %1972 = stablehlo.add %1971, %cst_17 : tensor<1x7x3072xf32>
    %1973 = stablehlo.multiply %1972, %1954 : tensor<1x7x3072xf32>
    %1974 = stablehlo.add %1973, %cst_16 : tensor<1x7x3072xf32>
    %1975 = stablehlo.multiply %cst_15, %1954 : tensor<1x7x3072xf32>
    %1976 = stablehlo.add %1975, %cst_14 : tensor<1x7x3072xf32>
    %1977 = stablehlo.multiply %1976, %1954 : tensor<1x7x3072xf32>
    %1978 = stablehlo.add %1977, %cst_13 : tensor<1x7x3072xf32>
    %1979 = stablehlo.multiply %1978, %1954 : tensor<1x7x3072xf32>
    %1980 = stablehlo.add %1979, %cst_12 : tensor<1x7x3072xf32>
    %1981 = stablehlo.multiply %1980, %1954 : tensor<1x7x3072xf32>
    %1982 = stablehlo.add %1981, %cst_11 : tensor<1x7x3072xf32>
    %1983 = stablehlo.multiply %1982, %1954 : tensor<1x7x3072xf32>
    %1984 = stablehlo.add %1983, %cst_10 : tensor<1x7x3072xf32>
    %1985 = stablehlo.multiply %1984, %1954 : tensor<1x7x3072xf32>
    %1986 = stablehlo.add %1985, %cst_9 : tensor<1x7x3072xf32>
    %1987 = stablehlo.multiply %1986, %1954 : tensor<1x7x3072xf32>
    %1988 = stablehlo.add %1987, %cst_8 : tensor<1x7x3072xf32>
    %1989 = stablehlo.select %1958, %1974, %1988 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1990 = stablehlo.multiply %1957, %1989 : tensor<1x7x3072xf32>
    %1991 = stablehlo.compare  LT, %1952, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1992 = stablehlo.select %1991, %cst_6, %1990 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1993 = stablehlo.compare  LT, %1950, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %1994 = stablehlo.subtract %cst_25, %1992 : tensor<1x7x3072xf32>
    %1995 = stablehlo.select %1993, %1994, %1992 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %1996 = stablehlo.multiply %1950, %1950 : tensor<1x7x3072xf32>
    %1997 = stablehlo.multiply %cst_5, %1996 : tensor<1x7x3072xf32>
    %1998 = stablehlo.add %1997, %cst_4 : tensor<1x7x3072xf32>
    %1999 = stablehlo.multiply %1998, %1996 : tensor<1x7x3072xf32>
    %2000 = stablehlo.add %1999, %cst_3 : tensor<1x7x3072xf32>
    %2001 = stablehlo.multiply %2000, %1996 : tensor<1x7x3072xf32>
    %2002 = stablehlo.add %2001, %cst_2 : tensor<1x7x3072xf32>
    %2003 = stablehlo.multiply %2002, %1996 : tensor<1x7x3072xf32>
    %2004 = stablehlo.add %2003, %cst_1 : tensor<1x7x3072xf32>
    %2005 = stablehlo.multiply %2004, %1996 : tensor<1x7x3072xf32>
    %2006 = stablehlo.add %2005, %cst_0 : tensor<1x7x3072xf32>
    %2007 = stablehlo.multiply %2006, %1996 : tensor<1x7x3072xf32>
    %2008 = stablehlo.add %2007, %cst : tensor<1x7x3072xf32>
    %2009 = stablehlo.multiply %1950, %2008 : tensor<1x7x3072xf32>
    %2010 = stablehlo.subtract %cst_26, %2009 : tensor<1x7x3072xf32>
    %2011 = stablehlo.abs %1950 : tensor<1x7x3072xf32>
    %2012 = stablehlo.compare  LT, %2011, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2013 = stablehlo.select %2012, %2010, %1995 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2014 = stablehlo.multiply %1947, %2013 : tensor<1x7x3072xf32>
    %2015 = stablehlo.reshape %2014 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %2016 = stablehlo.transpose %arg145, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %2017 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2018 = stablehlo.multiply %arg146, %2017 : tensor<768xf32>
    %2019 = stablehlo.dot_general %2015, %2016, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %2020 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2021 = stablehlo.multiply %2020, %2019 : tensor<7x768xf32>
    %2022 = stablehlo.broadcast_in_dim %2018, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2023 = stablehlo.broadcast_in_dim %2022, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2024 = stablehlo.add %2023, %2021 : tensor<7x768xf32>
    %2025 = stablehlo.reshape %2024 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2026 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %2027 = stablehlo.multiply %1934, %2026 : tensor<1x7x768xf32>
    %2028 = stablehlo.add %2025, %2027 : tensor<1x7x768xf32>
    %2029 = stablehlo.reduce(%2028 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2030 = stablehlo.broadcast_in_dim %2029, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2031 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2032 = stablehlo.divide %2030, %2031 : tensor<1x7x1xf32>
    %2033 = call @_var(%2028, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %2034 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2035 = stablehlo.add %2033, %2034 : tensor<1x7x1xf32>
    %2036 = stablehlo.rsqrt %2035 : tensor<1x7x1xf32>
    %2037 = stablehlo.broadcast_in_dim %2032, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2038 = stablehlo.subtract %2028, %2037 : tensor<1x7x768xf32>
    %2039 = stablehlo.broadcast_in_dim %2036, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2040 = stablehlo.multiply %2038, %2039 : tensor<1x7x768xf32>
    %2041 = stablehlo.broadcast_in_dim %arg147, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2042 = stablehlo.broadcast_in_dim %2041, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2043 = stablehlo.multiply %2040, %2042 : tensor<1x7x768xf32>
    %2044 = stablehlo.broadcast_in_dim %arg148, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2045 = stablehlo.broadcast_in_dim %2044, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2046 = stablehlo.add %2043, %2045 : tensor<1x7x768xf32>
    %2047 = stablehlo.reshape %2046 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2048 = stablehlo.transpose %arg149, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2049 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2050 = stablehlo.multiply %arg150, %2049 : tensor<768xf32>
    %2051 = stablehlo.dot_general %2047, %2048, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2052 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2053 = stablehlo.multiply %2052, %2051 : tensor<7x768xf32>
    %2054 = stablehlo.broadcast_in_dim %2050, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2055 = stablehlo.broadcast_in_dim %2054, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2056 = stablehlo.add %2055, %2053 : tensor<7x768xf32>
    %2057 = stablehlo.reshape %2056 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2058 = stablehlo.reshape %2057 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2059 = stablehlo.transpose %2058, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2060 = stablehlo.reshape %2046 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2061 = stablehlo.transpose %arg151, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2062 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2063 = stablehlo.multiply %arg152, %2062 : tensor<768xf32>
    %2064 = stablehlo.dot_general %2060, %2061, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2065 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2066 = stablehlo.multiply %2065, %2064 : tensor<7x768xf32>
    %2067 = stablehlo.broadcast_in_dim %2063, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2068 = stablehlo.broadcast_in_dim %2067, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2069 = stablehlo.add %2068, %2066 : tensor<7x768xf32>
    %2070 = stablehlo.reshape %2069 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2071 = stablehlo.reshape %2070 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2072 = stablehlo.transpose %2071, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2073 = stablehlo.reshape %2046 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2074 = stablehlo.transpose %arg153, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2075 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2076 = stablehlo.multiply %arg154, %2075 : tensor<768xf32>
    %2077 = stablehlo.dot_general %2073, %2074, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2078 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2079 = stablehlo.multiply %2078, %2077 : tensor<7x768xf32>
    %2080 = stablehlo.broadcast_in_dim %2076, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2081 = stablehlo.broadcast_in_dim %2080, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2082 = stablehlo.add %2081, %2079 : tensor<7x768xf32>
    %2083 = stablehlo.reshape %2082 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2084 = stablehlo.reshape %2083 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2085 = stablehlo.transpose %2084, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2086 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %2087 = stablehlo.multiply %2059, %2086 : tensor<1x12x7x64xf32>
    %2088 = stablehlo.transpose %2072, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %2089 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %2090 = stablehlo.multiply %2088, %2089 : tensor<1x12x64x7xf32>
    %2091 = stablehlo.reshape %2087 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %2092 = stablehlo.reshape %2090 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %2093 = stablehlo.dot_general %2091, %2092, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %2094 = stablehlo.reshape %2093 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %2095 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %2096 = stablehlo.multiply %39, %2095 : tensor<1x1x7x7xf32>
    %2097 = stablehlo.broadcast_in_dim %2096, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %2098 = stablehlo.add %2094, %2097 : tensor<1x12x7x7xf32>
    %2099 = stablehlo.reduce(%2098 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2100 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %2101 = stablehlo.maximum %2100, %2099 : tensor<1x12x7xf32>
    %2102 = stablehlo.broadcast_in_dim %2101, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2103 = stablehlo.broadcast_in_dim %2102, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %2104 = stablehlo.subtract %2098, %2103 : tensor<1x12x7x7xf32>
    %2105 = stablehlo.exponential %2104 : tensor<1x12x7x7xf32>
    %2106 = stablehlo.reduce(%2105 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2107 = stablehlo.broadcast_in_dim %2106, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2108 = stablehlo.broadcast_in_dim %2107, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %2109 = stablehlo.divide %2105, %2108 : tensor<1x12x7x7xf32>
    %2110 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %2111 = stablehlo.compare  EQ, %2098, %2110,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %2112 = stablehlo.not %2111 : tensor<1x12x7x7xi1>
    %2113 = stablehlo.reduce(%2112 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %2114 = stablehlo.broadcast_in_dim %2113, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %2115 = stablehlo.not %2114 : tensor<1x12x7x1xi1>
    %2116 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %2117 = call @_where_4(%2115, %2116, %2109) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %2118 = stablehlo.reshape %2117 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %2119 = stablehlo.reshape %2085 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %2120 = stablehlo.dot_general %2118, %2119, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %2121 = stablehlo.reshape %2120 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %2122 = stablehlo.transpose %2121, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %2123 = stablehlo.transpose %2122, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2124 = stablehlo.transpose %2123, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %2125 = stablehlo.reshape %2124 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %2126 = stablehlo.reshape %2125 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2127 = stablehlo.transpose %arg155, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2128 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2129 = stablehlo.multiply %arg156, %2128 : tensor<768xf32>
    %2130 = stablehlo.dot_general %2126, %2127, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2131 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2132 = stablehlo.multiply %2131, %2130 : tensor<7x768xf32>
    %2133 = stablehlo.broadcast_in_dim %2129, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2134 = stablehlo.broadcast_in_dim %2133, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2135 = stablehlo.add %2134, %2132 : tensor<7x768xf32>
    %2136 = stablehlo.reshape %2135 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2137 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %2138 = stablehlo.multiply %2046, %2137 : tensor<1x7x768xf32>
    %2139 = stablehlo.add %2136, %2138 : tensor<1x7x768xf32>
    %2140 = stablehlo.reduce(%2139 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2141 = stablehlo.broadcast_in_dim %2140, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2142 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2143 = stablehlo.divide %2141, %2142 : tensor<1x7x1xf32>
    %2144 = call @_var(%2139, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %2145 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2146 = stablehlo.add %2144, %2145 : tensor<1x7x1xf32>
    %2147 = stablehlo.rsqrt %2146 : tensor<1x7x1xf32>
    %2148 = stablehlo.broadcast_in_dim %2143, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2149 = stablehlo.subtract %2139, %2148 : tensor<1x7x768xf32>
    %2150 = stablehlo.broadcast_in_dim %2147, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2151 = stablehlo.multiply %2149, %2150 : tensor<1x7x768xf32>
    %2152 = stablehlo.broadcast_in_dim %arg157, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2153 = stablehlo.broadcast_in_dim %2152, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2154 = stablehlo.multiply %2151, %2153 : tensor<1x7x768xf32>
    %2155 = stablehlo.broadcast_in_dim %arg158, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2156 = stablehlo.broadcast_in_dim %2155, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2157 = stablehlo.add %2154, %2156 : tensor<1x7x768xf32>
    %2158 = stablehlo.reshape %2157 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2159 = stablehlo.transpose %arg159, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %2160 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %2161 = stablehlo.multiply %arg160, %2160 : tensor<3072xf32>
    %2162 = stablehlo.dot_general %2158, %2159, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %2163 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %2164 = stablehlo.multiply %2163, %2162 : tensor<7x3072xf32>
    %2165 = stablehlo.broadcast_in_dim %2161, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %2166 = stablehlo.broadcast_in_dim %2165, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %2167 = stablehlo.add %2166, %2164 : tensor<7x3072xf32>
    %2168 = stablehlo.reshape %2167 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %2169 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2170 = stablehlo.multiply %2169, %2168 : tensor<1x7x3072xf32>
    %2171 = stablehlo.negate %2168 : tensor<1x7x3072xf32>
    %2172 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2173 = stablehlo.multiply %2171, %2172 : tensor<1x7x3072xf32>
    %2174 = stablehlo.multiply %2173, %2173 : tensor<1x7x3072xf32>
    %2175 = stablehlo.negate %2174 : tensor<1x7x3072xf32>
    %2176 = stablehlo.abs %2173 : tensor<1x7x3072xf32>
    %2177 = stablehlo.divide %cst_26, %2174 : tensor<1x7x3072xf32>
    %2178 = stablehlo.exponential %2175 : tensor<1x7x3072xf32>
    %2179 = stablehlo.divide %cst_26, %2176 : tensor<1x7x3072xf32>
    %2180 = stablehlo.multiply %2178, %2179 : tensor<1x7x3072xf32>
    %2181 = stablehlo.compare  LT, %2176, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2182 = stablehlo.multiply %cst_24, %2177 : tensor<1x7x3072xf32>
    %2183 = stablehlo.add %2182, %cst_23 : tensor<1x7x3072xf32>
    %2184 = stablehlo.multiply %2183, %2177 : tensor<1x7x3072xf32>
    %2185 = stablehlo.add %2184, %cst_22 : tensor<1x7x3072xf32>
    %2186 = stablehlo.multiply %2185, %2177 : tensor<1x7x3072xf32>
    %2187 = stablehlo.add %2186, %cst_21 : tensor<1x7x3072xf32>
    %2188 = stablehlo.multiply %2187, %2177 : tensor<1x7x3072xf32>
    %2189 = stablehlo.add %2188, %cst_20 : tensor<1x7x3072xf32>
    %2190 = stablehlo.multiply %2189, %2177 : tensor<1x7x3072xf32>
    %2191 = stablehlo.add %2190, %cst_19 : tensor<1x7x3072xf32>
    %2192 = stablehlo.multiply %2191, %2177 : tensor<1x7x3072xf32>
    %2193 = stablehlo.add %2192, %cst_18 : tensor<1x7x3072xf32>
    %2194 = stablehlo.multiply %2193, %2177 : tensor<1x7x3072xf32>
    %2195 = stablehlo.add %2194, %cst_17 : tensor<1x7x3072xf32>
    %2196 = stablehlo.multiply %2195, %2177 : tensor<1x7x3072xf32>
    %2197 = stablehlo.add %2196, %cst_16 : tensor<1x7x3072xf32>
    %2198 = stablehlo.multiply %cst_15, %2177 : tensor<1x7x3072xf32>
    %2199 = stablehlo.add %2198, %cst_14 : tensor<1x7x3072xf32>
    %2200 = stablehlo.multiply %2199, %2177 : tensor<1x7x3072xf32>
    %2201 = stablehlo.add %2200, %cst_13 : tensor<1x7x3072xf32>
    %2202 = stablehlo.multiply %2201, %2177 : tensor<1x7x3072xf32>
    %2203 = stablehlo.add %2202, %cst_12 : tensor<1x7x3072xf32>
    %2204 = stablehlo.multiply %2203, %2177 : tensor<1x7x3072xf32>
    %2205 = stablehlo.add %2204, %cst_11 : tensor<1x7x3072xf32>
    %2206 = stablehlo.multiply %2205, %2177 : tensor<1x7x3072xf32>
    %2207 = stablehlo.add %2206, %cst_10 : tensor<1x7x3072xf32>
    %2208 = stablehlo.multiply %2207, %2177 : tensor<1x7x3072xf32>
    %2209 = stablehlo.add %2208, %cst_9 : tensor<1x7x3072xf32>
    %2210 = stablehlo.multiply %2209, %2177 : tensor<1x7x3072xf32>
    %2211 = stablehlo.add %2210, %cst_8 : tensor<1x7x3072xf32>
    %2212 = stablehlo.select %2181, %2197, %2211 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2213 = stablehlo.multiply %2180, %2212 : tensor<1x7x3072xf32>
    %2214 = stablehlo.compare  LT, %2175, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2215 = stablehlo.select %2214, %cst_6, %2213 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2216 = stablehlo.compare  LT, %2173, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2217 = stablehlo.subtract %cst_25, %2215 : tensor<1x7x3072xf32>
    %2218 = stablehlo.select %2216, %2217, %2215 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2219 = stablehlo.multiply %2173, %2173 : tensor<1x7x3072xf32>
    %2220 = stablehlo.multiply %cst_5, %2219 : tensor<1x7x3072xf32>
    %2221 = stablehlo.add %2220, %cst_4 : tensor<1x7x3072xf32>
    %2222 = stablehlo.multiply %2221, %2219 : tensor<1x7x3072xf32>
    %2223 = stablehlo.add %2222, %cst_3 : tensor<1x7x3072xf32>
    %2224 = stablehlo.multiply %2223, %2219 : tensor<1x7x3072xf32>
    %2225 = stablehlo.add %2224, %cst_2 : tensor<1x7x3072xf32>
    %2226 = stablehlo.multiply %2225, %2219 : tensor<1x7x3072xf32>
    %2227 = stablehlo.add %2226, %cst_1 : tensor<1x7x3072xf32>
    %2228 = stablehlo.multiply %2227, %2219 : tensor<1x7x3072xf32>
    %2229 = stablehlo.add %2228, %cst_0 : tensor<1x7x3072xf32>
    %2230 = stablehlo.multiply %2229, %2219 : tensor<1x7x3072xf32>
    %2231 = stablehlo.add %2230, %cst : tensor<1x7x3072xf32>
    %2232 = stablehlo.multiply %2173, %2231 : tensor<1x7x3072xf32>
    %2233 = stablehlo.subtract %cst_26, %2232 : tensor<1x7x3072xf32>
    %2234 = stablehlo.abs %2173 : tensor<1x7x3072xf32>
    %2235 = stablehlo.compare  LT, %2234, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2236 = stablehlo.select %2235, %2233, %2218 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2237 = stablehlo.multiply %2170, %2236 : tensor<1x7x3072xf32>
    %2238 = stablehlo.reshape %2237 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %2239 = stablehlo.transpose %arg161, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %2240 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2241 = stablehlo.multiply %arg162, %2240 : tensor<768xf32>
    %2242 = stablehlo.dot_general %2238, %2239, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %2243 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2244 = stablehlo.multiply %2243, %2242 : tensor<7x768xf32>
    %2245 = stablehlo.broadcast_in_dim %2241, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2246 = stablehlo.broadcast_in_dim %2245, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2247 = stablehlo.add %2246, %2244 : tensor<7x768xf32>
    %2248 = stablehlo.reshape %2247 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2249 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %2250 = stablehlo.multiply %2157, %2249 : tensor<1x7x768xf32>
    %2251 = stablehlo.add %2248, %2250 : tensor<1x7x768xf32>
    %2252 = stablehlo.reduce(%2251 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2253 = stablehlo.broadcast_in_dim %2252, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2254 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2255 = stablehlo.divide %2253, %2254 : tensor<1x7x1xf32>
    %2256 = call @_var(%2251, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %2257 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2258 = stablehlo.add %2256, %2257 : tensor<1x7x1xf32>
    %2259 = stablehlo.rsqrt %2258 : tensor<1x7x1xf32>
    %2260 = stablehlo.broadcast_in_dim %2255, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2261 = stablehlo.subtract %2251, %2260 : tensor<1x7x768xf32>
    %2262 = stablehlo.broadcast_in_dim %2259, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2263 = stablehlo.multiply %2261, %2262 : tensor<1x7x768xf32>
    %2264 = stablehlo.broadcast_in_dim %arg163, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2265 = stablehlo.broadcast_in_dim %2264, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2266 = stablehlo.multiply %2263, %2265 : tensor<1x7x768xf32>
    %2267 = stablehlo.broadcast_in_dim %arg164, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2268 = stablehlo.broadcast_in_dim %2267, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2269 = stablehlo.add %2266, %2268 : tensor<1x7x768xf32>
    %2270 = stablehlo.reshape %2269 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2271 = stablehlo.transpose %arg165, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2272 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2273 = stablehlo.multiply %arg166, %2272 : tensor<768xf32>
    %2274 = stablehlo.dot_general %2270, %2271, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2275 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2276 = stablehlo.multiply %2275, %2274 : tensor<7x768xf32>
    %2277 = stablehlo.broadcast_in_dim %2273, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2278 = stablehlo.broadcast_in_dim %2277, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2279 = stablehlo.add %2278, %2276 : tensor<7x768xf32>
    %2280 = stablehlo.reshape %2279 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2281 = stablehlo.reshape %2280 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2282 = stablehlo.transpose %2281, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2283 = stablehlo.reshape %2269 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2284 = stablehlo.transpose %arg167, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2285 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2286 = stablehlo.multiply %arg168, %2285 : tensor<768xf32>
    %2287 = stablehlo.dot_general %2283, %2284, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2288 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2289 = stablehlo.multiply %2288, %2287 : tensor<7x768xf32>
    %2290 = stablehlo.broadcast_in_dim %2286, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2291 = stablehlo.broadcast_in_dim %2290, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2292 = stablehlo.add %2291, %2289 : tensor<7x768xf32>
    %2293 = stablehlo.reshape %2292 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2294 = stablehlo.reshape %2293 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2295 = stablehlo.transpose %2294, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2296 = stablehlo.reshape %2269 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2297 = stablehlo.transpose %arg169, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2298 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2299 = stablehlo.multiply %arg170, %2298 : tensor<768xf32>
    %2300 = stablehlo.dot_general %2296, %2297, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2301 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2302 = stablehlo.multiply %2301, %2300 : tensor<7x768xf32>
    %2303 = stablehlo.broadcast_in_dim %2299, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2304 = stablehlo.broadcast_in_dim %2303, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2305 = stablehlo.add %2304, %2302 : tensor<7x768xf32>
    %2306 = stablehlo.reshape %2305 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2307 = stablehlo.reshape %2306 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2308 = stablehlo.transpose %2307, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2309 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %2310 = stablehlo.multiply %2282, %2309 : tensor<1x12x7x64xf32>
    %2311 = stablehlo.transpose %2295, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %2312 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %2313 = stablehlo.multiply %2311, %2312 : tensor<1x12x64x7xf32>
    %2314 = stablehlo.reshape %2310 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %2315 = stablehlo.reshape %2313 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %2316 = stablehlo.dot_general %2314, %2315, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %2317 = stablehlo.reshape %2316 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %2318 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %2319 = stablehlo.multiply %39, %2318 : tensor<1x1x7x7xf32>
    %2320 = stablehlo.broadcast_in_dim %2319, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %2321 = stablehlo.add %2317, %2320 : tensor<1x12x7x7xf32>
    %2322 = stablehlo.reduce(%2321 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2323 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %2324 = stablehlo.maximum %2323, %2322 : tensor<1x12x7xf32>
    %2325 = stablehlo.broadcast_in_dim %2324, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2326 = stablehlo.broadcast_in_dim %2325, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %2327 = stablehlo.subtract %2321, %2326 : tensor<1x12x7x7xf32>
    %2328 = stablehlo.exponential %2327 : tensor<1x12x7x7xf32>
    %2329 = stablehlo.reduce(%2328 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2330 = stablehlo.broadcast_in_dim %2329, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2331 = stablehlo.broadcast_in_dim %2330, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %2332 = stablehlo.divide %2328, %2331 : tensor<1x12x7x7xf32>
    %2333 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %2334 = stablehlo.compare  EQ, %2321, %2333,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %2335 = stablehlo.not %2334 : tensor<1x12x7x7xi1>
    %2336 = stablehlo.reduce(%2335 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %2337 = stablehlo.broadcast_in_dim %2336, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %2338 = stablehlo.not %2337 : tensor<1x12x7x1xi1>
    %2339 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %2340 = call @_where_4(%2338, %2339, %2332) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %2341 = stablehlo.reshape %2340 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %2342 = stablehlo.reshape %2308 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %2343 = stablehlo.dot_general %2341, %2342, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %2344 = stablehlo.reshape %2343 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %2345 = stablehlo.transpose %2344, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %2346 = stablehlo.transpose %2345, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2347 = stablehlo.transpose %2346, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %2348 = stablehlo.reshape %2347 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %2349 = stablehlo.reshape %2348 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2350 = stablehlo.transpose %arg171, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2351 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2352 = stablehlo.multiply %arg172, %2351 : tensor<768xf32>
    %2353 = stablehlo.dot_general %2349, %2350, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2354 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2355 = stablehlo.multiply %2354, %2353 : tensor<7x768xf32>
    %2356 = stablehlo.broadcast_in_dim %2352, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2357 = stablehlo.broadcast_in_dim %2356, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2358 = stablehlo.add %2357, %2355 : tensor<7x768xf32>
    %2359 = stablehlo.reshape %2358 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2360 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %2361 = stablehlo.multiply %2269, %2360 : tensor<1x7x768xf32>
    %2362 = stablehlo.add %2359, %2361 : tensor<1x7x768xf32>
    %2363 = stablehlo.reduce(%2362 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2364 = stablehlo.broadcast_in_dim %2363, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2365 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2366 = stablehlo.divide %2364, %2365 : tensor<1x7x1xf32>
    %2367 = call @_var(%2362, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %2368 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2369 = stablehlo.add %2367, %2368 : tensor<1x7x1xf32>
    %2370 = stablehlo.rsqrt %2369 : tensor<1x7x1xf32>
    %2371 = stablehlo.broadcast_in_dim %2366, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2372 = stablehlo.subtract %2362, %2371 : tensor<1x7x768xf32>
    %2373 = stablehlo.broadcast_in_dim %2370, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2374 = stablehlo.multiply %2372, %2373 : tensor<1x7x768xf32>
    %2375 = stablehlo.broadcast_in_dim %arg173, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2376 = stablehlo.broadcast_in_dim %2375, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2377 = stablehlo.multiply %2374, %2376 : tensor<1x7x768xf32>
    %2378 = stablehlo.broadcast_in_dim %arg174, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2379 = stablehlo.broadcast_in_dim %2378, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2380 = stablehlo.add %2377, %2379 : tensor<1x7x768xf32>
    %2381 = stablehlo.reshape %2380 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2382 = stablehlo.transpose %arg175, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %2383 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %2384 = stablehlo.multiply %arg176, %2383 : tensor<3072xf32>
    %2385 = stablehlo.dot_general %2381, %2382, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %2386 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %2387 = stablehlo.multiply %2386, %2385 : tensor<7x3072xf32>
    %2388 = stablehlo.broadcast_in_dim %2384, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %2389 = stablehlo.broadcast_in_dim %2388, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %2390 = stablehlo.add %2389, %2387 : tensor<7x3072xf32>
    %2391 = stablehlo.reshape %2390 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %2392 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2393 = stablehlo.multiply %2392, %2391 : tensor<1x7x3072xf32>
    %2394 = stablehlo.negate %2391 : tensor<1x7x3072xf32>
    %2395 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2396 = stablehlo.multiply %2394, %2395 : tensor<1x7x3072xf32>
    %2397 = stablehlo.multiply %2396, %2396 : tensor<1x7x3072xf32>
    %2398 = stablehlo.negate %2397 : tensor<1x7x3072xf32>
    %2399 = stablehlo.abs %2396 : tensor<1x7x3072xf32>
    %2400 = stablehlo.divide %cst_26, %2397 : tensor<1x7x3072xf32>
    %2401 = stablehlo.exponential %2398 : tensor<1x7x3072xf32>
    %2402 = stablehlo.divide %cst_26, %2399 : tensor<1x7x3072xf32>
    %2403 = stablehlo.multiply %2401, %2402 : tensor<1x7x3072xf32>
    %2404 = stablehlo.compare  LT, %2399, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2405 = stablehlo.multiply %cst_24, %2400 : tensor<1x7x3072xf32>
    %2406 = stablehlo.add %2405, %cst_23 : tensor<1x7x3072xf32>
    %2407 = stablehlo.multiply %2406, %2400 : tensor<1x7x3072xf32>
    %2408 = stablehlo.add %2407, %cst_22 : tensor<1x7x3072xf32>
    %2409 = stablehlo.multiply %2408, %2400 : tensor<1x7x3072xf32>
    %2410 = stablehlo.add %2409, %cst_21 : tensor<1x7x3072xf32>
    %2411 = stablehlo.multiply %2410, %2400 : tensor<1x7x3072xf32>
    %2412 = stablehlo.add %2411, %cst_20 : tensor<1x7x3072xf32>
    %2413 = stablehlo.multiply %2412, %2400 : tensor<1x7x3072xf32>
    %2414 = stablehlo.add %2413, %cst_19 : tensor<1x7x3072xf32>
    %2415 = stablehlo.multiply %2414, %2400 : tensor<1x7x3072xf32>
    %2416 = stablehlo.add %2415, %cst_18 : tensor<1x7x3072xf32>
    %2417 = stablehlo.multiply %2416, %2400 : tensor<1x7x3072xf32>
    %2418 = stablehlo.add %2417, %cst_17 : tensor<1x7x3072xf32>
    %2419 = stablehlo.multiply %2418, %2400 : tensor<1x7x3072xf32>
    %2420 = stablehlo.add %2419, %cst_16 : tensor<1x7x3072xf32>
    %2421 = stablehlo.multiply %cst_15, %2400 : tensor<1x7x3072xf32>
    %2422 = stablehlo.add %2421, %cst_14 : tensor<1x7x3072xf32>
    %2423 = stablehlo.multiply %2422, %2400 : tensor<1x7x3072xf32>
    %2424 = stablehlo.add %2423, %cst_13 : tensor<1x7x3072xf32>
    %2425 = stablehlo.multiply %2424, %2400 : tensor<1x7x3072xf32>
    %2426 = stablehlo.add %2425, %cst_12 : tensor<1x7x3072xf32>
    %2427 = stablehlo.multiply %2426, %2400 : tensor<1x7x3072xf32>
    %2428 = stablehlo.add %2427, %cst_11 : tensor<1x7x3072xf32>
    %2429 = stablehlo.multiply %2428, %2400 : tensor<1x7x3072xf32>
    %2430 = stablehlo.add %2429, %cst_10 : tensor<1x7x3072xf32>
    %2431 = stablehlo.multiply %2430, %2400 : tensor<1x7x3072xf32>
    %2432 = stablehlo.add %2431, %cst_9 : tensor<1x7x3072xf32>
    %2433 = stablehlo.multiply %2432, %2400 : tensor<1x7x3072xf32>
    %2434 = stablehlo.add %2433, %cst_8 : tensor<1x7x3072xf32>
    %2435 = stablehlo.select %2404, %2420, %2434 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2436 = stablehlo.multiply %2403, %2435 : tensor<1x7x3072xf32>
    %2437 = stablehlo.compare  LT, %2398, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2438 = stablehlo.select %2437, %cst_6, %2436 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2439 = stablehlo.compare  LT, %2396, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2440 = stablehlo.subtract %cst_25, %2438 : tensor<1x7x3072xf32>
    %2441 = stablehlo.select %2439, %2440, %2438 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2442 = stablehlo.multiply %2396, %2396 : tensor<1x7x3072xf32>
    %2443 = stablehlo.multiply %cst_5, %2442 : tensor<1x7x3072xf32>
    %2444 = stablehlo.add %2443, %cst_4 : tensor<1x7x3072xf32>
    %2445 = stablehlo.multiply %2444, %2442 : tensor<1x7x3072xf32>
    %2446 = stablehlo.add %2445, %cst_3 : tensor<1x7x3072xf32>
    %2447 = stablehlo.multiply %2446, %2442 : tensor<1x7x3072xf32>
    %2448 = stablehlo.add %2447, %cst_2 : tensor<1x7x3072xf32>
    %2449 = stablehlo.multiply %2448, %2442 : tensor<1x7x3072xf32>
    %2450 = stablehlo.add %2449, %cst_1 : tensor<1x7x3072xf32>
    %2451 = stablehlo.multiply %2450, %2442 : tensor<1x7x3072xf32>
    %2452 = stablehlo.add %2451, %cst_0 : tensor<1x7x3072xf32>
    %2453 = stablehlo.multiply %2452, %2442 : tensor<1x7x3072xf32>
    %2454 = stablehlo.add %2453, %cst : tensor<1x7x3072xf32>
    %2455 = stablehlo.multiply %2396, %2454 : tensor<1x7x3072xf32>
    %2456 = stablehlo.subtract %cst_26, %2455 : tensor<1x7x3072xf32>
    %2457 = stablehlo.abs %2396 : tensor<1x7x3072xf32>
    %2458 = stablehlo.compare  LT, %2457, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2459 = stablehlo.select %2458, %2456, %2441 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2460 = stablehlo.multiply %2393, %2459 : tensor<1x7x3072xf32>
    %2461 = stablehlo.reshape %2460 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %2462 = stablehlo.transpose %arg177, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %2463 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2464 = stablehlo.multiply %arg178, %2463 : tensor<768xf32>
    %2465 = stablehlo.dot_general %2461, %2462, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %2466 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2467 = stablehlo.multiply %2466, %2465 : tensor<7x768xf32>
    %2468 = stablehlo.broadcast_in_dim %2464, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2469 = stablehlo.broadcast_in_dim %2468, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2470 = stablehlo.add %2469, %2467 : tensor<7x768xf32>
    %2471 = stablehlo.reshape %2470 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2472 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %2473 = stablehlo.multiply %2380, %2472 : tensor<1x7x768xf32>
    %2474 = stablehlo.add %2471, %2473 : tensor<1x7x768xf32>
    %2475 = stablehlo.reduce(%2474 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2476 = stablehlo.broadcast_in_dim %2475, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2477 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2478 = stablehlo.divide %2476, %2477 : tensor<1x7x1xf32>
    %2479 = call @_var(%2474, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %2480 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2481 = stablehlo.add %2479, %2480 : tensor<1x7x1xf32>
    %2482 = stablehlo.rsqrt %2481 : tensor<1x7x1xf32>
    %2483 = stablehlo.broadcast_in_dim %2478, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2484 = stablehlo.subtract %2474, %2483 : tensor<1x7x768xf32>
    %2485 = stablehlo.broadcast_in_dim %2482, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2486 = stablehlo.multiply %2484, %2485 : tensor<1x7x768xf32>
    %2487 = stablehlo.broadcast_in_dim %arg179, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2488 = stablehlo.broadcast_in_dim %2487, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2489 = stablehlo.multiply %2486, %2488 : tensor<1x7x768xf32>
    %2490 = stablehlo.broadcast_in_dim %arg180, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2491 = stablehlo.broadcast_in_dim %2490, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2492 = stablehlo.add %2489, %2491 : tensor<1x7x768xf32>
    %2493 = stablehlo.reshape %2492 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2494 = stablehlo.transpose %arg181, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2495 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2496 = stablehlo.multiply %arg182, %2495 : tensor<768xf32>
    %2497 = stablehlo.dot_general %2493, %2494, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2498 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2499 = stablehlo.multiply %2498, %2497 : tensor<7x768xf32>
    %2500 = stablehlo.broadcast_in_dim %2496, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2501 = stablehlo.broadcast_in_dim %2500, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2502 = stablehlo.add %2501, %2499 : tensor<7x768xf32>
    %2503 = stablehlo.reshape %2502 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2504 = stablehlo.reshape %2503 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2505 = stablehlo.transpose %2504, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2506 = stablehlo.reshape %2492 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2507 = stablehlo.transpose %arg183, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2508 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2509 = stablehlo.multiply %arg184, %2508 : tensor<768xf32>
    %2510 = stablehlo.dot_general %2506, %2507, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2511 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2512 = stablehlo.multiply %2511, %2510 : tensor<7x768xf32>
    %2513 = stablehlo.broadcast_in_dim %2509, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2514 = stablehlo.broadcast_in_dim %2513, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2515 = stablehlo.add %2514, %2512 : tensor<7x768xf32>
    %2516 = stablehlo.reshape %2515 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2517 = stablehlo.reshape %2516 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2518 = stablehlo.transpose %2517, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2519 = stablehlo.reshape %2492 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2520 = stablehlo.transpose %arg185, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2521 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2522 = stablehlo.multiply %arg186, %2521 : tensor<768xf32>
    %2523 = stablehlo.dot_general %2519, %2520, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2524 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2525 = stablehlo.multiply %2524, %2523 : tensor<7x768xf32>
    %2526 = stablehlo.broadcast_in_dim %2522, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2527 = stablehlo.broadcast_in_dim %2526, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2528 = stablehlo.add %2527, %2525 : tensor<7x768xf32>
    %2529 = stablehlo.reshape %2528 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2530 = stablehlo.reshape %2529 : (tensor<1x7x768xf32>) -> tensor<1x7x12x64xf32>
    %2531 = stablehlo.transpose %2530, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2532 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x7x64xf32>
    %2533 = stablehlo.multiply %2505, %2532 : tensor<1x12x7x64xf32>
    %2534 = stablehlo.transpose %2518, dims = [0, 1, 3, 2] : (tensor<1x12x7x64xf32>) -> tensor<1x12x64x7xf32>
    %2535 = stablehlo.broadcast_in_dim %cst_30, dims = [] : (tensor<f32>) -> tensor<1x12x64x7xf32>
    %2536 = stablehlo.multiply %2534, %2535 : tensor<1x12x64x7xf32>
    %2537 = stablehlo.reshape %2533 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %2538 = stablehlo.reshape %2536 : (tensor<1x12x64x7xf32>) -> tensor<12x64x7xf32>
    %2539 = stablehlo.dot_general %2537, %2538, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x64xf32>, tensor<12x64x7xf32>) -> tensor<12x7x7xf32>
    %2540 = stablehlo.reshape %2539 : (tensor<12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %2541 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %2542 = stablehlo.multiply %39, %2541 : tensor<1x1x7x7xf32>
    %2543 = stablehlo.broadcast_in_dim %2542, dims = [0, 1, 2, 3] : (tensor<1x1x7x7xf32>) -> tensor<1x12x7x7xf32>
    %2544 = stablehlo.add %2540, %2543 : tensor<1x12x7x7xf32>
    %2545 = stablehlo.reduce(%2544 init: %cst_29) applies stablehlo.maximum across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2546 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7xf32>
    %2547 = stablehlo.maximum %2546, %2545 : tensor<1x12x7xf32>
    %2548 = stablehlo.broadcast_in_dim %2547, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2549 = stablehlo.broadcast_in_dim %2548, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %2550 = stablehlo.subtract %2544, %2549 : tensor<1x12x7x7xf32>
    %2551 = stablehlo.exponential %2550 : tensor<1x12x7x7xf32>
    %2552 = stablehlo.reduce(%2551 init: %cst_35) applies stablehlo.add across dimensions = [3] : (tensor<1x12x7x7xf32>, tensor<f32>) -> tensor<1x12x7xf32>
    %2553 = stablehlo.broadcast_in_dim %2552, dims = [0, 1, 2] : (tensor<1x12x7xf32>) -> tensor<1x12x7x1xf32>
    %2554 = stablehlo.broadcast_in_dim %2553, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xf32>) -> tensor<1x12x7x7xf32>
    %2555 = stablehlo.divide %2551, %2554 : tensor<1x12x7x7xf32>
    %2556 = stablehlo.broadcast_in_dim %cst_29, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %2557 = stablehlo.compare  EQ, %2544, %2556,  FLOAT : (tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xi1>
    %2558 = stablehlo.not %2557 : tensor<1x12x7x7xi1>
    %2559 = stablehlo.reduce(%2558 init: %c) applies stablehlo.or across dimensions = [3] : (tensor<1x12x7x7xi1>, tensor<i1>) -> tensor<1x12x7xi1>
    %2560 = stablehlo.broadcast_in_dim %2559, dims = [0, 1, 2] : (tensor<1x12x7xi1>) -> tensor<1x12x7x1xi1>
    %2561 = stablehlo.not %2560 : tensor<1x12x7x1xi1>
    %2562 = stablehlo.broadcast_in_dim %cst_35, dims = [] : (tensor<f32>) -> tensor<1x12x7x7xf32>
    %2563 = call @_where_4(%2561, %2562, %2555) : (tensor<1x12x7x1xi1>, tensor<1x12x7x7xf32>, tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32>
    %2564 = stablehlo.reshape %2563 : (tensor<1x12x7x7xf32>) -> tensor<12x7x7xf32>
    %2565 = stablehlo.reshape %2531 : (tensor<1x12x7x64xf32>) -> tensor<12x7x64xf32>
    %2566 = stablehlo.dot_general %2564, %2565, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<12x7x7xf32>, tensor<12x7x64xf32>) -> tensor<12x7x64xf32>
    %2567 = stablehlo.reshape %2566 : (tensor<12x7x64xf32>) -> tensor<1x12x7x64xf32>
    %2568 = stablehlo.transpose %2567, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %2569 = stablehlo.transpose %2568, dims = [0, 2, 1, 3] : (tensor<1x7x12x64xf32>) -> tensor<1x12x7x64xf32>
    %2570 = stablehlo.transpose %2569, dims = [0, 2, 1, 3] : (tensor<1x12x7x64xf32>) -> tensor<1x7x12x64xf32>
    %2571 = stablehlo.reshape %2570 : (tensor<1x7x12x64xf32>) -> tensor<1x7x768xf32>
    %2572 = stablehlo.reshape %2571 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2573 = stablehlo.transpose %arg187, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2574 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2575 = stablehlo.multiply %arg188, %2574 : tensor<768xf32>
    %2576 = stablehlo.dot_general %2572, %2573, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x768xf32>) -> tensor<7x768xf32>
    %2577 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2578 = stablehlo.multiply %2577, %2576 : tensor<7x768xf32>
    %2579 = stablehlo.broadcast_in_dim %2575, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2580 = stablehlo.broadcast_in_dim %2579, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2581 = stablehlo.add %2580, %2578 : tensor<7x768xf32>
    %2582 = stablehlo.reshape %2581 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2583 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %2584 = stablehlo.multiply %2492, %2583 : tensor<1x7x768xf32>
    %2585 = stablehlo.add %2582, %2584 : tensor<1x7x768xf32>
    %2586 = stablehlo.reduce(%2585 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2587 = stablehlo.broadcast_in_dim %2586, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2588 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2589 = stablehlo.divide %2587, %2588 : tensor<1x7x1xf32>
    %2590 = call @_var(%2585, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %2591 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2592 = stablehlo.add %2590, %2591 : tensor<1x7x1xf32>
    %2593 = stablehlo.rsqrt %2592 : tensor<1x7x1xf32>
    %2594 = stablehlo.broadcast_in_dim %2589, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2595 = stablehlo.subtract %2585, %2594 : tensor<1x7x768xf32>
    %2596 = stablehlo.broadcast_in_dim %2593, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2597 = stablehlo.multiply %2595, %2596 : tensor<1x7x768xf32>
    %2598 = stablehlo.broadcast_in_dim %arg189, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2599 = stablehlo.broadcast_in_dim %2598, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2600 = stablehlo.multiply %2597, %2599 : tensor<1x7x768xf32>
    %2601 = stablehlo.broadcast_in_dim %arg190, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2602 = stablehlo.broadcast_in_dim %2601, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2603 = stablehlo.add %2600, %2602 : tensor<1x7x768xf32>
    %2604 = stablehlo.reshape %2603 : (tensor<1x7x768xf32>) -> tensor<7x768xf32>
    %2605 = stablehlo.transpose %arg191, dims = [1, 0] : (tensor<3072x768xf32>) -> tensor<768x3072xf32>
    %2606 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %2607 = stablehlo.multiply %arg192, %2606 : tensor<3072xf32>
    %2608 = stablehlo.dot_general %2604, %2605, contracting_dims = [1] x [0] : (tensor<7x768xf32>, tensor<768x3072xf32>) -> tensor<7x3072xf32>
    %2609 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x3072xf32>
    %2610 = stablehlo.multiply %2609, %2608 : tensor<7x3072xf32>
    %2611 = stablehlo.broadcast_in_dim %2607, dims = [1] : (tensor<3072xf32>) -> tensor<1x3072xf32>
    %2612 = stablehlo.broadcast_in_dim %2611, dims = [0, 1] : (tensor<1x3072xf32>) -> tensor<7x3072xf32>
    %2613 = stablehlo.add %2612, %2610 : tensor<7x3072xf32>
    %2614 = stablehlo.reshape %2613 : (tensor<7x3072xf32>) -> tensor<1x7x3072xf32>
    %2615 = stablehlo.broadcast_in_dim %cst_28, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2616 = stablehlo.multiply %2615, %2614 : tensor<1x7x3072xf32>
    %2617 = stablehlo.negate %2614 : tensor<1x7x3072xf32>
    %2618 = stablehlo.broadcast_in_dim %cst_27, dims = [] : (tensor<f32>) -> tensor<1x7x3072xf32>
    %2619 = stablehlo.multiply %2617, %2618 : tensor<1x7x3072xf32>
    %2620 = stablehlo.multiply %2619, %2619 : tensor<1x7x3072xf32>
    %2621 = stablehlo.negate %2620 : tensor<1x7x3072xf32>
    %2622 = stablehlo.abs %2619 : tensor<1x7x3072xf32>
    %2623 = stablehlo.divide %cst_26, %2620 : tensor<1x7x3072xf32>
    %2624 = stablehlo.exponential %2621 : tensor<1x7x3072xf32>
    %2625 = stablehlo.divide %cst_26, %2622 : tensor<1x7x3072xf32>
    %2626 = stablehlo.multiply %2624, %2625 : tensor<1x7x3072xf32>
    %2627 = stablehlo.compare  LT, %2622, %cst_25 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2628 = stablehlo.multiply %cst_24, %2623 : tensor<1x7x3072xf32>
    %2629 = stablehlo.add %2628, %cst_23 : tensor<1x7x3072xf32>
    %2630 = stablehlo.multiply %2629, %2623 : tensor<1x7x3072xf32>
    %2631 = stablehlo.add %2630, %cst_22 : tensor<1x7x3072xf32>
    %2632 = stablehlo.multiply %2631, %2623 : tensor<1x7x3072xf32>
    %2633 = stablehlo.add %2632, %cst_21 : tensor<1x7x3072xf32>
    %2634 = stablehlo.multiply %2633, %2623 : tensor<1x7x3072xf32>
    %2635 = stablehlo.add %2634, %cst_20 : tensor<1x7x3072xf32>
    %2636 = stablehlo.multiply %2635, %2623 : tensor<1x7x3072xf32>
    %2637 = stablehlo.add %2636, %cst_19 : tensor<1x7x3072xf32>
    %2638 = stablehlo.multiply %2637, %2623 : tensor<1x7x3072xf32>
    %2639 = stablehlo.add %2638, %cst_18 : tensor<1x7x3072xf32>
    %2640 = stablehlo.multiply %2639, %2623 : tensor<1x7x3072xf32>
    %2641 = stablehlo.add %2640, %cst_17 : tensor<1x7x3072xf32>
    %2642 = stablehlo.multiply %2641, %2623 : tensor<1x7x3072xf32>
    %2643 = stablehlo.add %2642, %cst_16 : tensor<1x7x3072xf32>
    %2644 = stablehlo.multiply %cst_15, %2623 : tensor<1x7x3072xf32>
    %2645 = stablehlo.add %2644, %cst_14 : tensor<1x7x3072xf32>
    %2646 = stablehlo.multiply %2645, %2623 : tensor<1x7x3072xf32>
    %2647 = stablehlo.add %2646, %cst_13 : tensor<1x7x3072xf32>
    %2648 = stablehlo.multiply %2647, %2623 : tensor<1x7x3072xf32>
    %2649 = stablehlo.add %2648, %cst_12 : tensor<1x7x3072xf32>
    %2650 = stablehlo.multiply %2649, %2623 : tensor<1x7x3072xf32>
    %2651 = stablehlo.add %2650, %cst_11 : tensor<1x7x3072xf32>
    %2652 = stablehlo.multiply %2651, %2623 : tensor<1x7x3072xf32>
    %2653 = stablehlo.add %2652, %cst_10 : tensor<1x7x3072xf32>
    %2654 = stablehlo.multiply %2653, %2623 : tensor<1x7x3072xf32>
    %2655 = stablehlo.add %2654, %cst_9 : tensor<1x7x3072xf32>
    %2656 = stablehlo.multiply %2655, %2623 : tensor<1x7x3072xf32>
    %2657 = stablehlo.add %2656, %cst_8 : tensor<1x7x3072xf32>
    %2658 = stablehlo.select %2627, %2643, %2657 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2659 = stablehlo.multiply %2626, %2658 : tensor<1x7x3072xf32>
    %2660 = stablehlo.compare  LT, %2621, %cst_7 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2661 = stablehlo.select %2660, %cst_6, %2659 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2662 = stablehlo.compare  LT, %2619, %cst_6 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2663 = stablehlo.subtract %cst_25, %2661 : tensor<1x7x3072xf32>
    %2664 = stablehlo.select %2662, %2663, %2661 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2665 = stablehlo.multiply %2619, %2619 : tensor<1x7x3072xf32>
    %2666 = stablehlo.multiply %cst_5, %2665 : tensor<1x7x3072xf32>
    %2667 = stablehlo.add %2666, %cst_4 : tensor<1x7x3072xf32>
    %2668 = stablehlo.multiply %2667, %2665 : tensor<1x7x3072xf32>
    %2669 = stablehlo.add %2668, %cst_3 : tensor<1x7x3072xf32>
    %2670 = stablehlo.multiply %2669, %2665 : tensor<1x7x3072xf32>
    %2671 = stablehlo.add %2670, %cst_2 : tensor<1x7x3072xf32>
    %2672 = stablehlo.multiply %2671, %2665 : tensor<1x7x3072xf32>
    %2673 = stablehlo.add %2672, %cst_1 : tensor<1x7x3072xf32>
    %2674 = stablehlo.multiply %2673, %2665 : tensor<1x7x3072xf32>
    %2675 = stablehlo.add %2674, %cst_0 : tensor<1x7x3072xf32>
    %2676 = stablehlo.multiply %2675, %2665 : tensor<1x7x3072xf32>
    %2677 = stablehlo.add %2676, %cst : tensor<1x7x3072xf32>
    %2678 = stablehlo.multiply %2619, %2677 : tensor<1x7x3072xf32>
    %2679 = stablehlo.subtract %cst_26, %2678 : tensor<1x7x3072xf32>
    %2680 = stablehlo.abs %2619 : tensor<1x7x3072xf32>
    %2681 = stablehlo.compare  LT, %2680, %cst_26 : (tensor<1x7x3072xf32>, tensor<1x7x3072xf32>) -> tensor<1x7x3072xi1>
    %2682 = stablehlo.select %2681, %2679, %2664 : tensor<1x7x3072xi1>, tensor<1x7x3072xf32>
    %2683 = stablehlo.multiply %2616, %2682 : tensor<1x7x3072xf32>
    %2684 = stablehlo.reshape %2683 : (tensor<1x7x3072xf32>) -> tensor<7x3072xf32>
    %2685 = stablehlo.transpose %arg193, dims = [1, 0] : (tensor<768x3072xf32>) -> tensor<3072x768xf32>
    %2686 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2687 = stablehlo.multiply %arg194, %2686 : tensor<768xf32>
    %2688 = stablehlo.dot_general %2684, %2685, contracting_dims = [1] x [0] : (tensor<7x3072xf32>, tensor<3072x768xf32>) -> tensor<7x768xf32>
    %2689 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<7x768xf32>
    %2690 = stablehlo.multiply %2689, %2688 : tensor<7x768xf32>
    %2691 = stablehlo.broadcast_in_dim %2687, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2692 = stablehlo.broadcast_in_dim %2691, dims = [0, 1] : (tensor<1x768xf32>) -> tensor<7x768xf32>
    %2693 = stablehlo.add %2692, %2690 : tensor<7x768xf32>
    %2694 = stablehlo.reshape %2693 : (tensor<7x768xf32>) -> tensor<1x7x768xf32>
    %2695 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %2696 = stablehlo.multiply %2603, %2695 : tensor<1x7x768xf32>
    %2697 = stablehlo.add %2694, %2696 : tensor<1x7x768xf32>
    %2698 = stablehlo.reduce(%2697 init: %cst_35) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %2699 = stablehlo.broadcast_in_dim %2698, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2700 = stablehlo.broadcast_in_dim %cst_34, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2701 = stablehlo.divide %2699, %2700 : tensor<1x7x1xf32>
    %2702 = call @_var(%2697, %c_33) : (tensor<1x7x768xf32>, tensor<i32>) -> tensor<1x7x1xf32>
    %2703 = stablehlo.broadcast_in_dim %cst_32, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2704 = stablehlo.add %2702, %2703 : tensor<1x7x1xf32>
    %2705 = stablehlo.rsqrt %2704 : tensor<1x7x1xf32>
    %2706 = stablehlo.broadcast_in_dim %2701, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2707 = stablehlo.subtract %2697, %2706 : tensor<1x7x768xf32>
    %2708 = stablehlo.broadcast_in_dim %2705, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %2709 = stablehlo.multiply %2707, %2708 : tensor<1x7x768xf32>
    %2710 = stablehlo.broadcast_in_dim %arg195, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2711 = stablehlo.broadcast_in_dim %2710, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2712 = stablehlo.multiply %2709, %2711 : tensor<1x7x768xf32>
    %2713 = stablehlo.broadcast_in_dim %arg196, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %2714 = stablehlo.broadcast_in_dim %2713, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x7x768xf32>
    %2715 = stablehlo.add %2712, %2714 : tensor<1x7x768xf32>
    %2716 = stablehlo.slice %2715 [0:1, 0:1, 0:768] : (tensor<1x7x768xf32>) -> tensor<1x1x768xf32>
    %2717 = stablehlo.reshape %2716 : (tensor<1x1x768xf32>) -> tensor<1x768xf32>
    %2718 = stablehlo.transpose %arg197, dims = [1, 0] : (tensor<768x768xf32>) -> tensor<768x768xf32>
    %2719 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %2720 = stablehlo.multiply %arg198, %2719 : tensor<768xf32>
    %2721 = stablehlo.dot_general %2717, %2718, contracting_dims = [1] x [0] : (tensor<1x768xf32>, tensor<768x768xf32>) -> tensor<1x768xf32>
    %2722 = stablehlo.broadcast_in_dim %cst_36, dims = [] : (tensor<f32>) -> tensor<1x768xf32>
    %2723 = stablehlo.multiply %2722, %2721 : tensor<1x768xf32>
    %2724 = stablehlo.broadcast_in_dim %2720, dims = [1] : (tensor<768xf32>) -> tensor<1x768xf32>
    %2725 = stablehlo.add %2724, %2723 : tensor<1x768xf32>
    %2726 = stablehlo.tanh %2725 : tensor<1x768xf32>
    return %2715, %2726 : tensor<1x7x768xf32>, tensor<1x768xf32>
  }
  func.func private @_take(%arg0: tensor<30522x768xf32>, %arg1: tensor<1x7xi32>) -> tensor<1x7x768xf32> {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %c = stablehlo.constant dense<true> : tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i32>
    %c_2 = stablehlo.constant dense<768> : tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_4 = stablehlo.constant dense<30522> : tensor<i32>
    %c_5 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1x7xi32>, tensor<1x7xi32>) -> tensor<1x7xi1>
    %2 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1x7xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1x7xi1>, tensor<1x7xi32>, tensor<1x7xi32>) -> tensor<1x7xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x7xi32>) -> tensor<1x7x1xi32>
    %6 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.compare  LT, %c_3, %9,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %11 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.add %c_3, %11 : tensor<1xi32>
    %13 = stablehlo.select %10, %12, %c_3 : tensor<1xi1>, tensor<1xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %15 = "stablehlo.gather"(%8, %14) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %16 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.concatenate %16, %17, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %19 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %20 = stablehlo.compare  LT, %c_3, %19,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %21 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %22 = stablehlo.add %c_3, %21 : tensor<1xi32>
    %23 = stablehlo.select %20, %22, %c_3 : tensor<1xi1>, tensor<1xi32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %25 = "stablehlo.gather"(%18, %24) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %26 = stablehlo.subtract %15, %25 : tensor<1xi32>
    %27 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1x7x1xi32>
    %28 = stablehlo.compare  GE, %5, %27,  SIGNED : (tensor<1x7x1xi32>, tensor<1x7x1xi32>) -> tensor<1x7x1xi1>
    %29 = stablehlo.broadcast_in_dim %26, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x7x1xi32>
    %31 = stablehlo.compare  LE, %5, %30,  SIGNED : (tensor<1x7x1xi32>, tensor<1x7x1xi32>) -> tensor<1x7x1xi1>
    %32 = stablehlo.and %28, %31 : tensor<1x7x1xi1>
    %33 = stablehlo.reduce(%32 init: %c) applies stablehlo.and across dimensions = [2] : (tensor<1x7x1xi1>, tensor<i1>) -> tensor<1x7xi1>
    %34 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 768>}> : (tensor<30522x768xf32>, tensor<1x7x1xi32>) -> tensor<1x7x768xf32>
    %35 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<1x7xi1>) -> tensor<1x7x768xi1>
    %36 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %37 = stablehlo.select %35, %34, %36 : tensor<1x7x768xi1>, tensor<1x7x768xf32>
    return %37 : tensor<1x7x768xf32>
  }
  func.func private @_where(%arg0: tensor<1x7xi1>, %arg1: tensor<1x7xi32>, %arg2: tensor<1x7xi32>) -> tensor<1x7xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<1x7xi1>, tensor<1x7xi32>
    return %0 : tensor<1x7xi32>
  }
  func.func private @_take_0(%arg0: tensor<2x768xf32>, %arg1: tensor<1x7xi32>) -> tensor<1x7x768xf32> {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %c = stablehlo.constant dense<true> : tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<768> : tensor<i32>
    %c_2 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_3 = stablehlo.constant dense<2> : tensor<i32>
    %c_4 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1x7xi32>, tensor<1x7xi32>) -> tensor<1x7xi1>
    %2 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1x7xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1x7xi1>, tensor<1x7xi32>, tensor<1x7xi32>) -> tensor<1x7xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x7xi32>) -> tensor<1x7x1xi32>
    %6 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.compare  LT, %c_2, %9,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %11 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.add %c_2, %11 : tensor<1xi32>
    %13 = stablehlo.select %10, %12, %c_2 : tensor<1xi1>, tensor<1xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %15 = "stablehlo.gather"(%8, %14) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %16 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.concatenate %16, %17, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %19 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %20 = stablehlo.compare  LT, %c_2, %19,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %21 = stablehlo.broadcast_in_dim %c_3, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %22 = stablehlo.add %c_2, %21 : tensor<1xi32>
    %23 = stablehlo.select %20, %22, %c_2 : tensor<1xi1>, tensor<1xi32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %25 = "stablehlo.gather"(%18, %24) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %26 = stablehlo.subtract %15, %25 : tensor<1xi32>
    %27 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1x7x1xi32>
    %28 = stablehlo.compare  GE, %5, %27,  SIGNED : (tensor<1x7x1xi32>, tensor<1x7x1xi32>) -> tensor<1x7x1xi1>
    %29 = stablehlo.broadcast_in_dim %26, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x7x1xi32>
    %31 = stablehlo.compare  LE, %5, %30,  SIGNED : (tensor<1x7x1xi32>, tensor<1x7x1xi32>) -> tensor<1x7x1xi1>
    %32 = stablehlo.and %28, %31 : tensor<1x7x1xi1>
    %33 = stablehlo.reduce(%32 init: %c) applies stablehlo.and across dimensions = [2] : (tensor<1x7x1xi1>, tensor<i1>) -> tensor<1x7xi1>
    %34 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 768>}> : (tensor<2x768xf32>, tensor<1x7x1xi32>) -> tensor<1x7x768xf32>
    %35 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<1x7xi1>) -> tensor<1x7x768xi1>
    %36 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %37 = stablehlo.select %35, %34, %36 : tensor<1x7x768xi1>, tensor<1x7x768xf32>
    return %37 : tensor<1x7x768xf32>
  }
  func.func private @_take_1(%arg0: tensor<512x768xf32>, %arg1: tensor<1x7xi32>) -> tensor<1x7x768xf32> {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %c = stablehlo.constant dense<true> : tensor<i1>
    %c_0 = stablehlo.constant dense<1> : tensor<i32>
    %c_1 = stablehlo.constant dense<2> : tensor<i32>
    %c_2 = stablehlo.constant dense<768> : tensor<i32>
    %c_3 = stablehlo.constant dense<0> : tensor<1xi32>
    %c_4 = stablehlo.constant dense<512> : tensor<i32>
    %c_5 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %1 = stablehlo.compare  LT, %arg1, %0,  SIGNED : (tensor<1x7xi32>, tensor<1x7xi32>) -> tensor<1x7xi1>
    %2 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1x7xi32>
    %3 = stablehlo.add %arg1, %2 : tensor<1x7xi32>
    %4 = call @_where(%1, %3, %arg1) : (tensor<1x7xi1>, tensor<1x7xi32>, tensor<1x7xi32>) -> tensor<1x7xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x7xi32>) -> tensor<1x7x1xi32>
    %6 = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.compare  LT, %c_3, %9,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %11 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.add %c_3, %11 : tensor<1xi32>
    %13 = stablehlo.select %10, %12, %c_3 : tensor<1xi1>, tensor<1xi32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %15 = "stablehlo.gather"(%8, %14) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %16 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.broadcast_in_dim %c_2, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.concatenate %16, %17, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %19 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %20 = stablehlo.compare  LT, %c_3, %19,  SIGNED : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %21 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<i32>) -> tensor<1xi32>
    %22 = stablehlo.add %c_3, %21 : tensor<1xi32>
    %23 = stablehlo.select %20, %22, %c_3 : tensor<1xi1>, tensor<1xi32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [0] : (tensor<1xi32>) -> tensor<1x1xi32>
    %25 = "stablehlo.gather"(%18, %24) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<2xi32>, tensor<1x1xi32>) -> tensor<1xi32>
    %26 = stablehlo.subtract %15, %25 : tensor<1xi32>
    %27 = stablehlo.broadcast_in_dim %c_5, dims = [] : (tensor<i32>) -> tensor<1x7x1xi32>
    %28 = stablehlo.compare  GE, %5, %27,  SIGNED : (tensor<1x7x1xi32>, tensor<1x7x1xi32>) -> tensor<1x7x1xi1>
    %29 = stablehlo.broadcast_in_dim %26, dims = [2] : (tensor<1xi32>) -> tensor<1x1x1xi32>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2] : (tensor<1x1x1xi32>) -> tensor<1x7x1xi32>
    %31 = stablehlo.compare  LE, %5, %30,  SIGNED : (tensor<1x7x1xi32>, tensor<1x7x1xi32>) -> tensor<1x7x1xi1>
    %32 = stablehlo.and %28, %31 : tensor<1x7x1xi1>
    %33 = stablehlo.reduce(%32 init: %c) applies stablehlo.and across dimensions = [2] : (tensor<1x7x1xi1>, tensor<i1>) -> tensor<1x7xi1>
    %34 = "stablehlo.gather"(%arg0, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 768>}> : (tensor<512x768xf32>, tensor<1x7x1xi32>) -> tensor<1x7x768xf32>
    %35 = stablehlo.broadcast_in_dim %33, dims = [0, 1] : (tensor<1x7xi1>) -> tensor<1x7x768xi1>
    %36 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<1x7x768xf32>
    %37 = stablehlo.select %35, %34, %36 : tensor<1x7x768xi1>, tensor<1x7x768xf32>
    return %37 : tensor<1x7x768xf32>
  }
  func.func private @_var(%arg0: tensor<1x7x768xf32>, %arg1: tensor<i32>) -> tensor<1x7x1xf32> {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<7.680000e+02> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %3 = stablehlo.divide %1, %2 : tensor<1x7x1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1, 2] : (tensor<1x7x1xf32>) -> tensor<1x7x768xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<1x7x768xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<1x7x768xf32>
    %7 = stablehlo.convert %arg1 : (tensor<i32>) -> tensor<f32>
    %8 = stablehlo.subtract %cst_0, %7 : tensor<f32>
    %9 = stablehlo.reduce(%6 init: %cst_1) applies stablehlo.add across dimensions = [2] : (tensor<1x7x768xf32>, tensor<f32>) -> tensor<1x7xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1] : (tensor<1x7xf32>) -> tensor<1x7x1xf32>
    %11 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %12 = stablehlo.divide %10, %11 : tensor<1x7x1xf32>
    %13 = stablehlo.compare  GT, %8, %cst_1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %14 = call @_where_2(%13, %12, %cst) : (tensor<i1>, tensor<1x7x1xf32>, tensor<f32>) -> tensor<1x7x1xf32>
    return %14 : tensor<1x7x1xf32>
  }
  func.func private @_where_2(%arg0: tensor<i1>, %arg1: tensor<1x7x1xf32>, %arg2: tensor<f32>) -> tensor<1x7x1xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<1x7x1xf32>
    %2 = stablehlo.select %arg0, %arg1, %1 : tensor<i1>, tensor<1x7x1xf32>
    return %2 : tensor<1x7x1xf32>
  }
  func.func private @_where_3(%arg0: tensor<1x1x7x7xi1>, %arg1: tensor<f32>, %arg2: tensor<1x1x7x7xf32>) -> tensor<1x1x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [] : (tensor<f32>) -> tensor<1x1x7x7xf32>
    %1 = stablehlo.select %arg0, %0, %arg2 : tensor<1x1x7x7xi1>, tensor<1x1x7x7xf32>
    return %1 : tensor<1x1x7x7xf32>
  }
  func.func private @_where_4(%arg0: tensor<1x12x7x1xi1>, %arg1: tensor<1x12x7x7xf32>, %arg2: tensor<1x12x7x7xf32>) -> tensor<1x12x7x7xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x12x7x1xi1>) -> tensor<1x12x7x7xi1>
    %1 = stablehlo.select %0, %arg1, %arg2 : tensor<1x12x7x7xi1>, tensor<1x12x7x7xf32>
    return %1 : tensor<1x12x7x7xf32>
  }
}
