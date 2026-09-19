// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s -o /dev/null

// A shrunk llama forward pass whose layers are calls of one function (24 wide,
// 6 heads, 2 layers, exported from test/distributed_llama.py). Ops with no
// shardable axes, such as the calls and the rotary matrix computation, share a
// kernel bucket. Grouping them into kernels must not make two kernels depend on
// each other, even when that only shows up once each kernel is treated as a
// single node.
module @jit_forward_batched attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["tp"=4]>
  func.func public @main(%arg0: tensor<2x24xf32>, %arg1: tensor<2x24xf32>, %arg2: tensor<2x24xf32>, %arg3: tensor<24xf32>, %arg4: tensor<2x32x24xf32>, %arg5: tensor<2x24x32xf32>, %arg6: tensor<2x32x24xf32>, %arg7: tensor<2x24x24xf32>, %arg8: tensor<2x24x24xf32>, %arg9: tensor<2x24x24xf32>, %arg10: tensor<2x24x24xf32>, %arg11: tensor<2x2x4x24xf32>, %arg12: tensor<2x2x4x24xf32>) -> (tensor<2x24xf32> {jax.result_info = "result"}) {
    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}]> : tensor<2x24xf32>
    %1 = sdy.sharding_constraint %arg9 <@mesh, [{}, {"tp"}, {}]> : tensor<2x24x24xf32>
    %2 = sdy.sharding_constraint %arg7 <@mesh, [{}, {"tp"}, {}]> : tensor<2x24x24xf32>
    %3 = sdy.sharding_constraint %arg10 <@mesh, [{}, {"tp"}, {}]> : tensor<2x24x24xf32>
    %4 = sdy.sharding_constraint %arg8 <@mesh, [{}, {}, {"tp"}]> : tensor<2x24x24xf32>
    %5 = sdy.sharding_constraint %arg4 <@mesh, [{}, {"tp"}, {}]> : tensor<2x32x24xf32>
    %6 = sdy.sharding_constraint %arg5 <@mesh, [{}, {}, {"tp"}]> : tensor<2x24x32xf32>
    %7 = sdy.sharding_constraint %arg6 <@mesh, [{}, {"tp"}, {}]> : tensor<2x32x24xf32>
    %cst = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.power %cst, %cst_0 : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %9 = stablehlo.divide %cst_1, %8 : tensor<f32>
    %cst_2 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %10 = stablehlo.multiply %cst_2, %9 : tensor<f32>
    %11 = stablehlo.cosine %10 : tensor<f32>
    %12 = stablehlo.sine %10 : tensor<f32>
    %13 = stablehlo.negate %12 : tensor<f32>
    %14 = stablehlo.convert %11 : tensor<f32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %16 = stablehlo.convert %13 : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %18 = stablehlo.concatenate %15, %17, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %20 = stablehlo.convert %12 : tensor<f32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %22 = stablehlo.convert %11 : tensor<f32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %24 = stablehlo.concatenate %21, %23, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %26 = stablehlo.concatenate %19, %25, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %27 = stablehlo.power %cst_3, %cst_4 : tensor<f32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %28 = stablehlo.divide %cst_5, %27 : tensor<f32>
    %cst_6 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %29 = stablehlo.multiply %cst_6, %28 : tensor<f32>
    %30 = stablehlo.cosine %29 : tensor<f32>
    %31 = stablehlo.sine %29 : tensor<f32>
    %32 = stablehlo.negate %31 : tensor<f32>
    %33 = stablehlo.convert %30 : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %35 = stablehlo.convert %32 : tensor<f32>
    %36 = stablehlo.broadcast_in_dim %35, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %37 = stablehlo.concatenate %34, %36, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %39 = stablehlo.convert %31 : tensor<f32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %41 = stablehlo.convert %30 : tensor<f32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %43 = stablehlo.concatenate %40, %42, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %44 = stablehlo.broadcast_in_dim %43, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %45 = stablehlo.concatenate %38, %44, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %46 = stablehlo.power %cst_7, %cst_8 : tensor<f32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %47 = stablehlo.divide %cst_9, %46 : tensor<f32>
    %cst_10 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %48 = stablehlo.multiply %cst_10, %47 : tensor<f32>
    %49 = stablehlo.cosine %48 : tensor<f32>
    %50 = stablehlo.sine %48 : tensor<f32>
    %51 = stablehlo.negate %50 : tensor<f32>
    %52 = stablehlo.convert %49 : tensor<f32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %54 = stablehlo.convert %51 : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %56 = stablehlo.concatenate %53, %55, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %58 = stablehlo.convert %50 : tensor<f32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %60 = stablehlo.convert %49 : tensor<f32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %62 = stablehlo.concatenate %59, %61, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %64 = stablehlo.concatenate %57, %63, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_12 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %65 = stablehlo.power %cst_11, %cst_12 : tensor<f32>
    %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %66 = stablehlo.divide %cst_13, %65 : tensor<f32>
    %cst_14 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %67 = stablehlo.multiply %cst_14, %66 : tensor<f32>
    %68 = stablehlo.cosine %67 : tensor<f32>
    %69 = stablehlo.sine %67 : tensor<f32>
    %70 = stablehlo.negate %69 : tensor<f32>
    %71 = stablehlo.convert %68 : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %73 = stablehlo.convert %70 : tensor<f32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %75 = stablehlo.concatenate %72, %74, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %77 = stablehlo.convert %69 : tensor<f32>
    %78 = stablehlo.broadcast_in_dim %77, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %79 = stablehlo.convert %68 : tensor<f32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %81 = stablehlo.concatenate %78, %80, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %82 = stablehlo.broadcast_in_dim %81, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %83 = stablehlo.concatenate %76, %82, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_15 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %84 = stablehlo.power %cst_15, %cst_16 : tensor<f32>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %85 = stablehlo.divide %cst_17, %84 : tensor<f32>
    %cst_18 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %86 = stablehlo.multiply %cst_18, %85 : tensor<f32>
    %87 = stablehlo.cosine %86 : tensor<f32>
    %88 = stablehlo.sine %86 : tensor<f32>
    %89 = stablehlo.negate %88 : tensor<f32>
    %90 = stablehlo.convert %87 : tensor<f32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %92 = stablehlo.convert %89 : tensor<f32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %94 = stablehlo.concatenate %91, %93, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %96 = stablehlo.convert %88 : tensor<f32>
    %97 = stablehlo.broadcast_in_dim %96, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %98 = stablehlo.convert %87 : tensor<f32>
    %99 = stablehlo.broadcast_in_dim %98, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %100 = stablehlo.concatenate %97, %99, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %102 = stablehlo.concatenate %95, %101, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_19 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_20 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %103 = stablehlo.power %cst_19, %cst_20 : tensor<f32>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %104 = stablehlo.divide %cst_21, %103 : tensor<f32>
    %cst_22 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %105 = stablehlo.multiply %cst_22, %104 : tensor<f32>
    %106 = stablehlo.cosine %105 : tensor<f32>
    %107 = stablehlo.sine %105 : tensor<f32>
    %108 = stablehlo.negate %107 : tensor<f32>
    %109 = stablehlo.convert %106 : tensor<f32>
    %110 = stablehlo.broadcast_in_dim %109, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %111 = stablehlo.convert %108 : tensor<f32>
    %112 = stablehlo.broadcast_in_dim %111, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %113 = stablehlo.concatenate %110, %112, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %114 = stablehlo.broadcast_in_dim %113, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %115 = stablehlo.convert %107 : tensor<f32>
    %116 = stablehlo.broadcast_in_dim %115, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %117 = stablehlo.convert %106 : tensor<f32>
    %118 = stablehlo.broadcast_in_dim %117, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %119 = stablehlo.concatenate %116, %118, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %120 = stablehlo.broadcast_in_dim %119, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %121 = stablehlo.concatenate %114, %120, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_23 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %122 = stablehlo.power %cst_23, %cst_24 : tensor<f32>
    %cst_25 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %123 = stablehlo.divide %cst_25, %122 : tensor<f32>
    %cst_26 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %124 = stablehlo.multiply %cst_26, %123 : tensor<f32>
    %125 = stablehlo.cosine %124 : tensor<f32>
    %126 = stablehlo.sine %124 : tensor<f32>
    %127 = stablehlo.negate %126 : tensor<f32>
    %128 = stablehlo.convert %125 : tensor<f32>
    %129 = stablehlo.broadcast_in_dim %128, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %130 = stablehlo.convert %127 : tensor<f32>
    %131 = stablehlo.broadcast_in_dim %130, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %132 = stablehlo.concatenate %129, %131, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %133 = stablehlo.broadcast_in_dim %132, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %134 = stablehlo.convert %126 : tensor<f32>
    %135 = stablehlo.broadcast_in_dim %134, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %136 = stablehlo.convert %125 : tensor<f32>
    %137 = stablehlo.broadcast_in_dim %136, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %138 = stablehlo.concatenate %135, %137, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %139 = stablehlo.broadcast_in_dim %138, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %140 = stablehlo.concatenate %133, %139, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_27 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_28 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %141 = stablehlo.power %cst_27, %cst_28 : tensor<f32>
    %cst_29 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %142 = stablehlo.divide %cst_29, %141 : tensor<f32>
    %cst_30 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %143 = stablehlo.multiply %cst_30, %142 : tensor<f32>
    %144 = stablehlo.cosine %143 : tensor<f32>
    %145 = stablehlo.sine %143 : tensor<f32>
    %146 = stablehlo.negate %145 : tensor<f32>
    %147 = stablehlo.convert %144 : tensor<f32>
    %148 = stablehlo.broadcast_in_dim %147, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %149 = stablehlo.convert %146 : tensor<f32>
    %150 = stablehlo.broadcast_in_dim %149, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %151 = stablehlo.concatenate %148, %150, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %152 = stablehlo.broadcast_in_dim %151, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %153 = stablehlo.convert %145 : tensor<f32>
    %154 = stablehlo.broadcast_in_dim %153, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %155 = stablehlo.convert %144 : tensor<f32>
    %156 = stablehlo.broadcast_in_dim %155, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %157 = stablehlo.concatenate %154, %156, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %158 = stablehlo.broadcast_in_dim %157, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %159 = stablehlo.concatenate %152, %158, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_31 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %160 = stablehlo.power %cst_31, %cst_32 : tensor<f32>
    %cst_33 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %161 = stablehlo.divide %cst_33, %160 : tensor<f32>
    %cst_34 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %162 = stablehlo.multiply %cst_34, %161 : tensor<f32>
    %163 = stablehlo.cosine %162 : tensor<f32>
    %164 = stablehlo.sine %162 : tensor<f32>
    %165 = stablehlo.negate %164 : tensor<f32>
    %166 = stablehlo.convert %163 : tensor<f32>
    %167 = stablehlo.broadcast_in_dim %166, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %168 = stablehlo.convert %165 : tensor<f32>
    %169 = stablehlo.broadcast_in_dim %168, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %170 = stablehlo.concatenate %167, %169, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %171 = stablehlo.broadcast_in_dim %170, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %172 = stablehlo.convert %164 : tensor<f32>
    %173 = stablehlo.broadcast_in_dim %172, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %174 = stablehlo.convert %163 : tensor<f32>
    %175 = stablehlo.broadcast_in_dim %174, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %176 = stablehlo.concatenate %173, %175, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %177 = stablehlo.broadcast_in_dim %176, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %178 = stablehlo.concatenate %171, %177, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_35 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_36 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %179 = stablehlo.power %cst_35, %cst_36 : tensor<f32>
    %cst_37 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %180 = stablehlo.divide %cst_37, %179 : tensor<f32>
    %cst_38 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %181 = stablehlo.multiply %cst_38, %180 : tensor<f32>
    %182 = stablehlo.cosine %181 : tensor<f32>
    %183 = stablehlo.sine %181 : tensor<f32>
    %184 = stablehlo.negate %183 : tensor<f32>
    %185 = stablehlo.convert %182 : tensor<f32>
    %186 = stablehlo.broadcast_in_dim %185, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %187 = stablehlo.convert %184 : tensor<f32>
    %188 = stablehlo.broadcast_in_dim %187, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %189 = stablehlo.concatenate %186, %188, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %190 = stablehlo.broadcast_in_dim %189, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %191 = stablehlo.convert %183 : tensor<f32>
    %192 = stablehlo.broadcast_in_dim %191, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %193 = stablehlo.convert %182 : tensor<f32>
    %194 = stablehlo.broadcast_in_dim %193, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %195 = stablehlo.concatenate %192, %194, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %196 = stablehlo.broadcast_in_dim %195, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %197 = stablehlo.concatenate %190, %196, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_39 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %198 = stablehlo.power %cst_39, %cst_40 : tensor<f32>
    %cst_41 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %199 = stablehlo.divide %cst_41, %198 : tensor<f32>
    %cst_42 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %200 = stablehlo.multiply %cst_42, %199 : tensor<f32>
    %201 = stablehlo.cosine %200 : tensor<f32>
    %202 = stablehlo.sine %200 : tensor<f32>
    %203 = stablehlo.negate %202 : tensor<f32>
    %204 = stablehlo.convert %201 : tensor<f32>
    %205 = stablehlo.broadcast_in_dim %204, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %206 = stablehlo.convert %203 : tensor<f32>
    %207 = stablehlo.broadcast_in_dim %206, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %208 = stablehlo.concatenate %205, %207, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %210 = stablehlo.convert %202 : tensor<f32>
    %211 = stablehlo.broadcast_in_dim %210, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %212 = stablehlo.convert %201 : tensor<f32>
    %213 = stablehlo.broadcast_in_dim %212, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %214 = stablehlo.concatenate %211, %213, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %215 = stablehlo.broadcast_in_dim %214, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %216 = stablehlo.concatenate %209, %215, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_43 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_44 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %217 = stablehlo.power %cst_43, %cst_44 : tensor<f32>
    %cst_45 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %218 = stablehlo.divide %cst_45, %217 : tensor<f32>
    %cst_46 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %219 = stablehlo.multiply %cst_46, %218 : tensor<f32>
    %220 = stablehlo.cosine %219 : tensor<f32>
    %221 = stablehlo.sine %219 : tensor<f32>
    %222 = stablehlo.negate %221 : tensor<f32>
    %223 = stablehlo.convert %220 : tensor<f32>
    %224 = stablehlo.broadcast_in_dim %223, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %225 = stablehlo.convert %222 : tensor<f32>
    %226 = stablehlo.broadcast_in_dim %225, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %227 = stablehlo.concatenate %224, %226, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %228 = stablehlo.broadcast_in_dim %227, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %229 = stablehlo.convert %221 : tensor<f32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %231 = stablehlo.convert %220 : tensor<f32>
    %232 = stablehlo.broadcast_in_dim %231, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %233 = stablehlo.concatenate %230, %232, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %234 = stablehlo.broadcast_in_dim %233, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %235 = stablehlo.concatenate %228, %234, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %236 = stablehlo.broadcast_in_dim %26, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %237 = stablehlo.broadcast_in_dim %45, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %238 = stablehlo.broadcast_in_dim %64, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %239 = stablehlo.broadcast_in_dim %83, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %240 = stablehlo.broadcast_in_dim %102, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %241 = stablehlo.broadcast_in_dim %121, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %242 = stablehlo.broadcast_in_dim %140, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %243 = stablehlo.broadcast_in_dim %159, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %244 = stablehlo.broadcast_in_dim %178, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %245 = stablehlo.broadcast_in_dim %197, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %246 = stablehlo.broadcast_in_dim %216, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %247 = stablehlo.broadcast_in_dim %235, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %248 = stablehlo.concatenate %236, %237, %238, %239, %240, %241, %242, %243, %244, %245, %246, %247, dim = 0 : (tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<12x2x2xf32>
    %249 = stablehlo.broadcast_in_dim %26, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %250 = stablehlo.broadcast_in_dim %45, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %251 = stablehlo.broadcast_in_dim %64, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %252 = stablehlo.broadcast_in_dim %83, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %253 = stablehlo.broadcast_in_dim %102, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %254 = stablehlo.broadcast_in_dim %121, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %255 = stablehlo.broadcast_in_dim %140, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %256 = stablehlo.broadcast_in_dim %159, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %257 = stablehlo.broadcast_in_dim %178, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %258 = stablehlo.broadcast_in_dim %197, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %259 = stablehlo.broadcast_in_dim %216, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %260 = stablehlo.broadcast_in_dim %235, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %261 = stablehlo.concatenate %249, %250, %251, %252, %253, %254, %255, %256, %257, %258, %259, %260, dim = 0 : (tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<12x2x2xf32>
    %262 = stablehlo.slice %1 [0:1, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %263 = stablehlo.reshape %262 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %264 = stablehlo.slice %2 [0:1, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %265 = stablehlo.reshape %264 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %266 = stablehlo.slice %3 [0:1, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %267 = stablehlo.reshape %266 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %268 = stablehlo.slice %4 [0:1, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %269 = stablehlo.reshape %268 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %270 = stablehlo.slice %5 [0:1, 0:32, 0:24] : (tensor<2x32x24xf32>) -> tensor<1x32x24xf32>
    %271 = stablehlo.reshape %270 : (tensor<1x32x24xf32>) -> tensor<32x24xf32>
    %272 = stablehlo.slice %6 [0:1, 0:24, 0:32] : (tensor<2x24x32xf32>) -> tensor<1x24x32xf32>
    %273 = stablehlo.reshape %272 : (tensor<1x24x32xf32>) -> tensor<24x32xf32>
    %274 = stablehlo.slice %7 [0:1, 0:32, 0:24] : (tensor<2x32x24xf32>) -> tensor<1x32x24xf32>
    %275 = stablehlo.reshape %274 : (tensor<1x32x24xf32>) -> tensor<32x24xf32>
    %276 = stablehlo.slice %arg1 [0:1, 0:24] : (tensor<2x24xf32>) -> tensor<1x24xf32>
    %277 = stablehlo.reshape %276 : (tensor<1x24xf32>) -> tensor<24xf32>
    %278 = stablehlo.slice %arg2 [0:1, 0:24] : (tensor<2x24xf32>) -> tensor<1x24xf32>
    %279 = stablehlo.reshape %278 : (tensor<1x24xf32>) -> tensor<24xf32>
    %280 = stablehlo.slice %arg11 [0:2, 0:1, 0:4, 0:24] : (tensor<2x2x4x24xf32>) -> tensor<2x1x4x24xf32>
    %281 = stablehlo.reshape %280 : (tensor<2x1x4x24xf32>) -> tensor<2x4x24xf32>
    %282 = stablehlo.slice %arg12 [0:2, 0:1, 0:4, 0:24] : (tensor<2x2x4x24xf32>) -> tensor<2x1x4x24xf32>
    %283 = stablehlo.reshape %282 : (tensor<2x1x4x24xf32>) -> tensor<2x4x24xf32>
    %284 = call @transformer_layer(%0, %263, %265, %267, %269, %271, %273, %275, %277, %279, %281, %283, %248, %261) : (tensor<2x24xf32>, tensor<24x24xf32>, tensor<24x24xf32>, tensor<24x24xf32>, tensor<24x24xf32>, tensor<32x24xf32>, tensor<24x32xf32>, tensor<32x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<2x4x24xf32>, tensor<2x4x24xf32>, tensor<12x2x2xf32>, tensor<12x2x2xf32>) -> tensor<2x24xf32>
    %285 = stablehlo.slice %1 [1:2, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %286 = stablehlo.reshape %285 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %287 = stablehlo.slice %2 [1:2, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %288 = stablehlo.reshape %287 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %289 = stablehlo.slice %3 [1:2, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %290 = stablehlo.reshape %289 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %291 = stablehlo.slice %4 [1:2, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %292 = stablehlo.reshape %291 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %293 = stablehlo.slice %5 [1:2, 0:32, 0:24] : (tensor<2x32x24xf32>) -> tensor<1x32x24xf32>
    %294 = stablehlo.reshape %293 : (tensor<1x32x24xf32>) -> tensor<32x24xf32>
    %295 = stablehlo.slice %6 [1:2, 0:24, 0:32] : (tensor<2x24x32xf32>) -> tensor<1x24x32xf32>
    %296 = stablehlo.reshape %295 : (tensor<1x24x32xf32>) -> tensor<24x32xf32>
    %297 = stablehlo.slice %7 [1:2, 0:32, 0:24] : (tensor<2x32x24xf32>) -> tensor<1x32x24xf32>
    %298 = stablehlo.reshape %297 : (tensor<1x32x24xf32>) -> tensor<32x24xf32>
    %299 = stablehlo.slice %arg1 [1:2, 0:24] : (tensor<2x24xf32>) -> tensor<1x24xf32>
    %300 = stablehlo.reshape %299 : (tensor<1x24xf32>) -> tensor<24xf32>
    %301 = stablehlo.slice %arg2 [1:2, 0:24] : (tensor<2x24xf32>) -> tensor<1x24xf32>
    %302 = stablehlo.reshape %301 : (tensor<1x24xf32>) -> tensor<24xf32>
    %303 = stablehlo.slice %arg11 [0:2, 1:2, 0:4, 0:24] : (tensor<2x2x4x24xf32>) -> tensor<2x1x4x24xf32>
    %304 = stablehlo.reshape %303 : (tensor<2x1x4x24xf32>) -> tensor<2x4x24xf32>
    %305 = stablehlo.slice %arg12 [0:2, 1:2, 0:4, 0:24] : (tensor<2x2x4x24xf32>) -> tensor<2x1x4x24xf32>
    %306 = stablehlo.reshape %305 : (tensor<2x1x4x24xf32>) -> tensor<2x4x24xf32>
    %307 = call @transformer_layer(%284, %286, %288, %290, %292, %294, %296, %298, %300, %302, %304, %306, %248, %261) : (tensor<2x24xf32>, tensor<24x24xf32>, tensor<24x24xf32>, tensor<24x24xf32>, tensor<24x24xf32>, tensor<32x24xf32>, tensor<24x32xf32>, tensor<32x24xf32>, tensor<24xf32>, tensor<24xf32>, tensor<2x4x24xf32>, tensor<2x4x24xf32>, tensor<12x2x2xf32>, tensor<12x2x2xf32>) -> tensor<2x24xf32>
    %308 = stablehlo.dot_general %307, %307, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24xf32>, tensor<2x24xf32>) -> tensor<2xf32>
    %cst_47 = stablehlo.constant dense<2.400000e+01> : tensor<f32>
    %309 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %310 = stablehlo.divide %308, %309 : tensor<2xf32>
    %cst_48 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %311 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %312 = stablehlo.add %310, %311 : tensor<2xf32>
    %313 = stablehlo.sqrt %312 : tensor<2xf32>
    %cst_49 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %314 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %315 = stablehlo.divide %314, %313 : tensor<2xf32>
    %316 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<24xf32>) -> tensor<1x24xf32>
    %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1] : (tensor<1x24xf32>) -> tensor<2x24xf32>
    %318 = stablehlo.multiply %317, %307 : tensor<2x24xf32>
    %319 = stablehlo.broadcast_in_dim %315, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x24xf32>
    %321 = stablehlo.multiply %318, %320 : tensor<2x24xf32>
    return %321 : tensor<2x24xf32>
  }
  func.func private @transformer_layer(%arg0: tensor<2x24xf32>, %arg1: tensor<24x24xf32>, %arg2: tensor<24x24xf32>, %arg3: tensor<24x24xf32>, %arg4: tensor<24x24xf32>, %arg5: tensor<32x24xf32>, %arg6: tensor<24x32xf32>, %arg7: tensor<32x24xf32>, %arg8: tensor<24xf32>, %arg9: tensor<24xf32>, %arg10: tensor<2x4x24xf32>, %arg11: tensor<2x4x24xf32>, %arg12: tensor<12x2x2xf32>, %arg13: tensor<12x2x2xf32>) -> tensor<2x24xf32> {
    %0 = stablehlo.dot_general %arg0, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24xf32>, tensor<2x24xf32>) -> tensor<2xf32>
    %cst = stablehlo.constant dense<2.400000e+01> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %2 = stablehlo.divide %0, %1 : tensor<2xf32>
    %cst_0 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %4 = stablehlo.add %2, %3 : tensor<2xf32>
    %5 = stablehlo.sqrt %4 : tensor<2xf32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %7 = stablehlo.divide %6, %5 : tensor<2xf32>
    %8 = stablehlo.broadcast_in_dim %arg8, dims = [1] : (tensor<24xf32>) -> tensor<1x24xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<1x24xf32>) -> tensor<2x24xf32>
    %10 = stablehlo.multiply %9, %arg0 : tensor<2x24xf32>
    %11 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x24xf32>
    %13 = stablehlo.multiply %10, %12 : tensor<2x24xf32>
    %14 = stablehlo.dot_general %arg1, %13, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %15 = stablehlo.dot_general %arg2, %13, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %16 = stablehlo.dot_general %arg3, %13, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %17 = stablehlo.transpose %14, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %18 = stablehlo.reshape %17 : (tensor<2x24xf32>) -> tensor<2x12x2xf32>
    %19 = stablehlo.transpose %15, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %20 = stablehlo.reshape %19 : (tensor<2x24xf32>) -> tensor<2x12x2xf32>
    %21 = stablehlo.dot_general %arg13, %20, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<12x2x2xf32>, tensor<2x12x2xf32>) -> tensor<12x2x2xf32>
    %22 = stablehlo.transpose %21, dims = [2, 0, 1] : (tensor<12x2x2xf32>) -> tensor<2x12x2xf32>
    %23 = stablehlo.reshape %22 : (tensor<2x12x2xf32>) -> tensor<2x24xf32>
    %24 = stablehlo.dot_general %arg12, %18, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<12x2x2xf32>, tensor<2x12x2xf32>) -> tensor<12x2x2xf32>
    %25 = stablehlo.transpose %24, dims = [2, 0, 1] : (tensor<12x2x2xf32>) -> tensor<2x12x2xf32>
    %26 = stablehlo.reshape %25 : (tensor<2x12x2xf32>) -> tensor<2x24xf32>
    %27 = stablehlo.reshape %23 : (tensor<2x24xf32>) -> tensor<2x1x24xf32>
    %28 = stablehlo.concatenate %arg10, %27, dim = 1 : (tensor<2x4x24xf32>, tensor<2x1x24xf32>) -> tensor<2x5x24xf32>
    %29 = stablehlo.transpose %16, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %30 = stablehlo.reshape %29 : (tensor<2x24xf32>) -> tensor<2x1x24xf32>
    %31 = stablehlo.concatenate %arg11, %30, dim = 1 : (tensor<2x4x24xf32>, tensor<2x1x24xf32>) -> tensor<2x5x24xf32>
    %32 = stablehlo.reshape %26 : (tensor<2x24xf32>) -> tensor<2x6x4xf32>
    %33 = stablehlo.reshape %28 : (tensor<2x5x24xf32>) -> tensor<2x5x6x4xf32>
    %34 = stablehlo.reshape %31 : (tensor<2x5x24xf32>) -> tensor<2x5x6x4xf32>
    %35 = stablehlo.dot_general %33, %32, batching_dims = [0, 2] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x5x6x4xf32>, tensor<2x6x4xf32>) -> tensor<2x6x5xf32>
    %cst_2 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %36 = stablehlo.sqrt %cst_2 : tensor<f32>
    %37 = stablehlo.convert %36 : tensor<f32>
    %38 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<2x6x5xf32>
    %39 = stablehlo.divide %35, %38 : tensor<2x6x5xf32>
    %cst_3 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %40 = stablehlo.reduce(%39 init: %cst_3) applies stablehlo.maximum across dimensions = [2] : (tensor<2x6x5xf32>, tensor<f32>) -> tensor<2x6xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x5xf32>
    %43 = stablehlo.subtract %39, %42 : tensor<2x6x5xf32>
    %44 = stablehlo.exponential %43 : tensor<2x6x5xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = stablehlo.reduce(%44 init: %cst_4) applies stablehlo.add across dimensions = [2] : (tensor<2x6x5xf32>, tensor<f32>) -> tensor<2x6xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x5xf32>
    %48 = stablehlo.divide %44, %47 : tensor<2x6x5xf32>
    %49 = stablehlo.dot_general %34, %48, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x5x6x4xf32>, tensor<2x6x5xf32>) -> tensor<2x6x4xf32>
    %50 = stablehlo.reshape %49 : (tensor<2x6x4xf32>) -> tensor<2x24xf32>
    %51 = stablehlo.dot_general %arg4, %50, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %52 = stablehlo.transpose %51, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %53 = stablehlo.add %arg0, %52 : tensor<2x24xf32>
    %54 = stablehlo.dot_general %53, %53, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24xf32>, tensor<2x24xf32>) -> tensor<2xf32>
    %cst_5 = stablehlo.constant dense<2.400000e+01> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_5, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %56 = stablehlo.divide %54, %55 : tensor<2xf32>
    %cst_6 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %57 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %58 = stablehlo.add %56, %57 : tensor<2xf32>
    %59 = stablehlo.sqrt %58 : tensor<2xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %cst_7, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %61 = stablehlo.divide %60, %59 : tensor<2xf32>
    %62 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<24xf32>) -> tensor<1x24xf32>
    %63 = stablehlo.broadcast_in_dim %62, dims = [0, 1] : (tensor<1x24xf32>) -> tensor<2x24xf32>
    %64 = stablehlo.multiply %63, %53 : tensor<2x24xf32>
    %65 = stablehlo.broadcast_in_dim %61, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x24xf32>
    %67 = stablehlo.multiply %64, %66 : tensor<2x24xf32>
    %68 = stablehlo.dot_general %arg5, %67, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x24xf32>, tensor<2x24xf32>) -> tensor<32x2xf32>
    %69 = stablehlo.dot_general %arg7, %67, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x24xf32>, tensor<2x24xf32>) -> tensor<32x2xf32>
    %70 = stablehlo.negate %68 : tensor<32x2xf32>
    %71 = stablehlo.exponential %70 : tensor<32x2xf32>
    %cst_8 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %72 = stablehlo.broadcast_in_dim %cst_8, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %73 = stablehlo.add %72, %71 : tensor<32x2xf32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %74 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %75 = stablehlo.divide %74, %73 : tensor<32x2xf32>
    %76 = stablehlo.multiply %68, %75 : tensor<32x2xf32>
    %77 = stablehlo.multiply %76, %69 : tensor<32x2xf32>
    %78 = stablehlo.dot_general %arg6, %77, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<24x32xf32>, tensor<32x2xf32>) -> tensor<24x2xf32>
    %79 = stablehlo.transpose %78, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %80 = stablehlo.add %53, %79 : tensor<2x24xf32>
    return %80 : tensor<2x24xf32>
  }
}
