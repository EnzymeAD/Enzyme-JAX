// RUN: enzymexlamlir-opt --sdy-propagation-pipeline --shardy-to-distributed-pipeline %s -o /dev/null

// A shrunk llama forward pass (24 wide, 6 heads, 2 layers, exported from
// test/distributed_llama.py). Some dimensions mix shardable and unshardable
// factors, such as the head count and the rotary pairs. The unshardable ones
// are materialized as device-local axes in each cast's axes, so a cast covers
// its whole dimension and the axis analysis ClusterDistributedKernels rebuilds
// can reconcile it with the producer's symbol for the dimension.
module @jit_forward_batched attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["tp"=4]>
  func.func public @main(%arg0: tensor<2x24xf32>, %arg1: tensor<2x24xf32>, %arg2: tensor<2x24xf32>, %arg3: tensor<24xf32>, %arg4: tensor<2x32x24xf32>, %arg5: tensor<2x24x32xf32>, %arg6: tensor<2x32x24xf32>, %arg7: tensor<2x24x24xf32>, %arg8: tensor<2x24x24xf32>, %arg9: tensor<2x24x24xf32>, %arg10: tensor<2x24x24xf32>, %arg11: tensor<2x2x4x24xf32>, %arg12: tensor<2x2x4x24xf32>) -> (tensor<2x24xf32> {jax.result_info = "result"}) {
    %0 = sdy.sharding_constraint %arg9 <@mesh, [{}, {"tp"}, {}]> : tensor<2x24x24xf32>
    %1 = sdy.sharding_constraint %arg7 <@mesh, [{}, {"tp"}, {}]> : tensor<2x24x24xf32>
    %2 = sdy.sharding_constraint %arg10 <@mesh, [{}, {"tp"}, {}]> : tensor<2x24x24xf32>
    %3 = sdy.sharding_constraint %arg8 <@mesh, [{}, {}, {"tp"}]> : tensor<2x24x24xf32>
    %4 = sdy.sharding_constraint %arg4 <@mesh, [{}, {"tp"}, {}]> : tensor<2x32x24xf32>
    %5 = sdy.sharding_constraint %arg5 <@mesh, [{}, {}, {"tp"}]> : tensor<2x24x32xf32>
    %6 = sdy.sharding_constraint %arg6 <@mesh, [{}, {"tp"}, {}]> : tensor<2x32x24xf32>
    %cst = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.power %cst, %cst_0 : tensor<f32>
    %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %8 = stablehlo.divide %cst_1, %7 : tensor<f32>
    %cst_2 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %9 = stablehlo.multiply %cst_2, %8 : tensor<f32>
    %10 = stablehlo.cosine %9 : tensor<f32>
    %11 = stablehlo.sine %9 : tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.convert %10 : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %15 = stablehlo.convert %12 : tensor<f32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %17 = stablehlo.concatenate %14, %16, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %19 = stablehlo.convert %11 : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %21 = stablehlo.convert %10 : tensor<f32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %23 = stablehlo.concatenate %20, %22, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %24 = stablehlo.broadcast_in_dim %23, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %25 = stablehlo.concatenate %18, %24, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_3 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_4 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %26 = stablehlo.power %cst_3, %cst_4 : tensor<f32>
    %cst_5 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %27 = stablehlo.divide %cst_5, %26 : tensor<f32>
    %cst_6 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %28 = stablehlo.multiply %cst_6, %27 : tensor<f32>
    %29 = stablehlo.cosine %28 : tensor<f32>
    %30 = stablehlo.sine %28 : tensor<f32>
    %31 = stablehlo.negate %30 : tensor<f32>
    %32 = stablehlo.convert %29 : tensor<f32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %34 = stablehlo.convert %31 : tensor<f32>
    %35 = stablehlo.broadcast_in_dim %34, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %36 = stablehlo.concatenate %33, %35, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %38 = stablehlo.convert %30 : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %40 = stablehlo.convert %29 : tensor<f32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %42 = stablehlo.concatenate %39, %41, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %44 = stablehlo.concatenate %37, %43, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_7 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %45 = stablehlo.power %cst_7, %cst_8 : tensor<f32>
    %cst_9 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %46 = stablehlo.divide %cst_9, %45 : tensor<f32>
    %cst_10 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %47 = stablehlo.multiply %cst_10, %46 : tensor<f32>
    %48 = stablehlo.cosine %47 : tensor<f32>
    %49 = stablehlo.sine %47 : tensor<f32>
    %50 = stablehlo.negate %49 : tensor<f32>
    %51 = stablehlo.convert %48 : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %53 = stablehlo.convert %50 : tensor<f32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %55 = stablehlo.concatenate %52, %54, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %57 = stablehlo.convert %49 : tensor<f32>
    %58 = stablehlo.broadcast_in_dim %57, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %59 = stablehlo.convert %48 : tensor<f32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %61 = stablehlo.concatenate %58, %60, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %62 = stablehlo.broadcast_in_dim %61, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %63 = stablehlo.concatenate %56, %62, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_11 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_12 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %64 = stablehlo.power %cst_11, %cst_12 : tensor<f32>
    %cst_13 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %65 = stablehlo.divide %cst_13, %64 : tensor<f32>
    %cst_14 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %66 = stablehlo.multiply %cst_14, %65 : tensor<f32>
    %67 = stablehlo.cosine %66 : tensor<f32>
    %68 = stablehlo.sine %66 : tensor<f32>
    %69 = stablehlo.negate %68 : tensor<f32>
    %70 = stablehlo.convert %67 : tensor<f32>
    %71 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %72 = stablehlo.convert %69 : tensor<f32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %74 = stablehlo.concatenate %71, %73, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %76 = stablehlo.convert %68 : tensor<f32>
    %77 = stablehlo.broadcast_in_dim %76, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %78 = stablehlo.convert %67 : tensor<f32>
    %79 = stablehlo.broadcast_in_dim %78, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %80 = stablehlo.concatenate %77, %79, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %81 = stablehlo.broadcast_in_dim %80, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %82 = stablehlo.concatenate %75, %81, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_15 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %83 = stablehlo.power %cst_15, %cst_16 : tensor<f32>
    %cst_17 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %84 = stablehlo.divide %cst_17, %83 : tensor<f32>
    %cst_18 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %85 = stablehlo.multiply %cst_18, %84 : tensor<f32>
    %86 = stablehlo.cosine %85 : tensor<f32>
    %87 = stablehlo.sine %85 : tensor<f32>
    %88 = stablehlo.negate %87 : tensor<f32>
    %89 = stablehlo.convert %86 : tensor<f32>
    %90 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %91 = stablehlo.convert %88 : tensor<f32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %93 = stablehlo.concatenate %90, %92, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %94 = stablehlo.broadcast_in_dim %93, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %95 = stablehlo.convert %87 : tensor<f32>
    %96 = stablehlo.broadcast_in_dim %95, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %97 = stablehlo.convert %86 : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %99 = stablehlo.concatenate %96, %98, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %100 = stablehlo.broadcast_in_dim %99, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %101 = stablehlo.concatenate %94, %100, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_19 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_20 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %102 = stablehlo.power %cst_19, %cst_20 : tensor<f32>
    %cst_21 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %103 = stablehlo.divide %cst_21, %102 : tensor<f32>
    %cst_22 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %104 = stablehlo.multiply %cst_22, %103 : tensor<f32>
    %105 = stablehlo.cosine %104 : tensor<f32>
    %106 = stablehlo.sine %104 : tensor<f32>
    %107 = stablehlo.negate %106 : tensor<f32>
    %108 = stablehlo.convert %105 : tensor<f32>
    %109 = stablehlo.broadcast_in_dim %108, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %110 = stablehlo.convert %107 : tensor<f32>
    %111 = stablehlo.broadcast_in_dim %110, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %112 = stablehlo.concatenate %109, %111, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %113 = stablehlo.broadcast_in_dim %112, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %114 = stablehlo.convert %106 : tensor<f32>
    %115 = stablehlo.broadcast_in_dim %114, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %116 = stablehlo.convert %105 : tensor<f32>
    %117 = stablehlo.broadcast_in_dim %116, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %118 = stablehlo.concatenate %115, %117, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %119 = stablehlo.broadcast_in_dim %118, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %120 = stablehlo.concatenate %113, %119, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_23 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %121 = stablehlo.power %cst_23, %cst_24 : tensor<f32>
    %cst_25 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %122 = stablehlo.divide %cst_25, %121 : tensor<f32>
    %cst_26 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %123 = stablehlo.multiply %cst_26, %122 : tensor<f32>
    %124 = stablehlo.cosine %123 : tensor<f32>
    %125 = stablehlo.sine %123 : tensor<f32>
    %126 = stablehlo.negate %125 : tensor<f32>
    %127 = stablehlo.convert %124 : tensor<f32>
    %128 = stablehlo.broadcast_in_dim %127, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %129 = stablehlo.convert %126 : tensor<f32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %131 = stablehlo.concatenate %128, %130, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %132 = stablehlo.broadcast_in_dim %131, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %133 = stablehlo.convert %125 : tensor<f32>
    %134 = stablehlo.broadcast_in_dim %133, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %135 = stablehlo.convert %124 : tensor<f32>
    %136 = stablehlo.broadcast_in_dim %135, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %137 = stablehlo.concatenate %134, %136, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %138 = stablehlo.broadcast_in_dim %137, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %139 = stablehlo.concatenate %132, %138, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_27 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_28 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %140 = stablehlo.power %cst_27, %cst_28 : tensor<f32>
    %cst_29 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %141 = stablehlo.divide %cst_29, %140 : tensor<f32>
    %cst_30 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %142 = stablehlo.multiply %cst_30, %141 : tensor<f32>
    %143 = stablehlo.cosine %142 : tensor<f32>
    %144 = stablehlo.sine %142 : tensor<f32>
    %145 = stablehlo.negate %144 : tensor<f32>
    %146 = stablehlo.convert %143 : tensor<f32>
    %147 = stablehlo.broadcast_in_dim %146, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %148 = stablehlo.convert %145 : tensor<f32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %150 = stablehlo.concatenate %147, %149, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %151 = stablehlo.broadcast_in_dim %150, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %152 = stablehlo.convert %144 : tensor<f32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %154 = stablehlo.convert %143 : tensor<f32>
    %155 = stablehlo.broadcast_in_dim %154, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %156 = stablehlo.concatenate %153, %155, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %157 = stablehlo.broadcast_in_dim %156, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %158 = stablehlo.concatenate %151, %157, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_31 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_32 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %159 = stablehlo.power %cst_31, %cst_32 : tensor<f32>
    %cst_33 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %160 = stablehlo.divide %cst_33, %159 : tensor<f32>
    %cst_34 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %161 = stablehlo.multiply %cst_34, %160 : tensor<f32>
    %162 = stablehlo.cosine %161 : tensor<f32>
    %163 = stablehlo.sine %161 : tensor<f32>
    %164 = stablehlo.negate %163 : tensor<f32>
    %165 = stablehlo.convert %162 : tensor<f32>
    %166 = stablehlo.broadcast_in_dim %165, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %167 = stablehlo.convert %164 : tensor<f32>
    %168 = stablehlo.broadcast_in_dim %167, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %169 = stablehlo.concatenate %166, %168, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %170 = stablehlo.broadcast_in_dim %169, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %171 = stablehlo.convert %163 : tensor<f32>
    %172 = stablehlo.broadcast_in_dim %171, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %173 = stablehlo.convert %162 : tensor<f32>
    %174 = stablehlo.broadcast_in_dim %173, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %175 = stablehlo.concatenate %172, %174, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %176 = stablehlo.broadcast_in_dim %175, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %177 = stablehlo.concatenate %170, %176, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_35 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_36 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %178 = stablehlo.power %cst_35, %cst_36 : tensor<f32>
    %cst_37 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %179 = stablehlo.divide %cst_37, %178 : tensor<f32>
    %cst_38 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %180 = stablehlo.multiply %cst_38, %179 : tensor<f32>
    %181 = stablehlo.cosine %180 : tensor<f32>
    %182 = stablehlo.sine %180 : tensor<f32>
    %183 = stablehlo.negate %182 : tensor<f32>
    %184 = stablehlo.convert %181 : tensor<f32>
    %185 = stablehlo.broadcast_in_dim %184, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %186 = stablehlo.convert %183 : tensor<f32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %188 = stablehlo.concatenate %185, %187, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %189 = stablehlo.broadcast_in_dim %188, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %190 = stablehlo.convert %182 : tensor<f32>
    %191 = stablehlo.broadcast_in_dim %190, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %192 = stablehlo.convert %181 : tensor<f32>
    %193 = stablehlo.broadcast_in_dim %192, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %194 = stablehlo.concatenate %191, %193, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %195 = stablehlo.broadcast_in_dim %194, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %196 = stablehlo.concatenate %189, %195, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_39 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %197 = stablehlo.power %cst_39, %cst_40 : tensor<f32>
    %cst_41 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %198 = stablehlo.divide %cst_41, %197 : tensor<f32>
    %cst_42 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %199 = stablehlo.multiply %cst_42, %198 : tensor<f32>
    %200 = stablehlo.cosine %199 : tensor<f32>
    %201 = stablehlo.sine %199 : tensor<f32>
    %202 = stablehlo.negate %201 : tensor<f32>
    %203 = stablehlo.convert %200 : tensor<f32>
    %204 = stablehlo.broadcast_in_dim %203, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %205 = stablehlo.convert %202 : tensor<f32>
    %206 = stablehlo.broadcast_in_dim %205, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %207 = stablehlo.concatenate %204, %206, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %208 = stablehlo.broadcast_in_dim %207, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %209 = stablehlo.convert %201 : tensor<f32>
    %210 = stablehlo.broadcast_in_dim %209, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %211 = stablehlo.convert %200 : tensor<f32>
    %212 = stablehlo.broadcast_in_dim %211, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %213 = stablehlo.concatenate %210, %212, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %214 = stablehlo.broadcast_in_dim %213, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %215 = stablehlo.concatenate %208, %214, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %cst_43 = stablehlo.constant dense<1.000000e+04> : tensor<f32>
    %cst_44 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %216 = stablehlo.power %cst_43, %cst_44 : tensor<f32>
    %cst_45 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %217 = stablehlo.divide %cst_45, %216 : tensor<f32>
    %cst_46 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %218 = stablehlo.multiply %cst_46, %217 : tensor<f32>
    %219 = stablehlo.cosine %218 : tensor<f32>
    %220 = stablehlo.sine %218 : tensor<f32>
    %221 = stablehlo.negate %220 : tensor<f32>
    %222 = stablehlo.convert %219 : tensor<f32>
    %223 = stablehlo.broadcast_in_dim %222, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %224 = stablehlo.convert %221 : tensor<f32>
    %225 = stablehlo.broadcast_in_dim %224, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %226 = stablehlo.concatenate %223, %225, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %227 = stablehlo.broadcast_in_dim %226, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %228 = stablehlo.convert %220 : tensor<f32>
    %229 = stablehlo.broadcast_in_dim %228, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %230 = stablehlo.convert %219 : tensor<f32>
    %231 = stablehlo.broadcast_in_dim %230, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %232 = stablehlo.concatenate %229, %231, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %233 = stablehlo.broadcast_in_dim %232, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %234 = stablehlo.concatenate %227, %233, dim = 0 : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    %235 = stablehlo.broadcast_in_dim %25, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %236 = stablehlo.broadcast_in_dim %44, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %237 = stablehlo.broadcast_in_dim %63, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %238 = stablehlo.broadcast_in_dim %82, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %239 = stablehlo.broadcast_in_dim %101, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %240 = stablehlo.broadcast_in_dim %120, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %241 = stablehlo.broadcast_in_dim %139, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %242 = stablehlo.broadcast_in_dim %158, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %243 = stablehlo.broadcast_in_dim %177, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %244 = stablehlo.broadcast_in_dim %196, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %245 = stablehlo.broadcast_in_dim %215, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %246 = stablehlo.broadcast_in_dim %234, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %247 = stablehlo.concatenate %235, %236, %237, %238, %239, %240, %241, %242, %243, %244, %245, %246, dim = 0 : (tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<12x2x2xf32>
    %248 = stablehlo.broadcast_in_dim %25, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %249 = stablehlo.broadcast_in_dim %44, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %250 = stablehlo.broadcast_in_dim %63, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %251 = stablehlo.broadcast_in_dim %82, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %252 = stablehlo.broadcast_in_dim %101, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %253 = stablehlo.broadcast_in_dim %120, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %254 = stablehlo.broadcast_in_dim %139, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %255 = stablehlo.broadcast_in_dim %158, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %256 = stablehlo.broadcast_in_dim %177, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %257 = stablehlo.broadcast_in_dim %196, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %258 = stablehlo.broadcast_in_dim %215, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %259 = stablehlo.broadcast_in_dim %234, dims = [1, 2] : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %260 = stablehlo.concatenate %248, %249, %250, %251, %252, %253, %254, %255, %256, %257, %258, %259, dim = 0 : (tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<12x2x2xf32>
    %261 = stablehlo.slice %arg1 [0:1, 0:24] : (tensor<2x24xf32>) -> tensor<1x24xf32>
    %262 = stablehlo.reshape %261 : (tensor<1x24xf32>) -> tensor<24xf32>
    %263 = stablehlo.dot_general %arg0, %arg0, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24xf32>, tensor<2x24xf32>) -> tensor<2xf32>
    %cst_47 = stablehlo.constant dense<2.400000e+01> : tensor<f32>
    %264 = stablehlo.broadcast_in_dim %cst_47, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %265 = stablehlo.divide %263, %264 : tensor<2xf32>
    %cst_48 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %266 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %267 = stablehlo.add %265, %266 : tensor<2xf32>
    %268 = stablehlo.sqrt %267 : tensor<2xf32>
    %cst_49 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %269 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %270 = stablehlo.divide %269, %268 : tensor<2xf32>
    %271 = stablehlo.broadcast_in_dim %262, dims = [1] : (tensor<24xf32>) -> tensor<1x24xf32>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1] : (tensor<1x24xf32>) -> tensor<2x24xf32>
    %273 = stablehlo.multiply %272, %arg0 : tensor<2x24xf32>
    %274 = stablehlo.broadcast_in_dim %270, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x24xf32>
    %276 = stablehlo.multiply %273, %275 : tensor<2x24xf32>
    %277 = stablehlo.slice %0 [0:1, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %278 = stablehlo.reshape %277 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %279 = stablehlo.dot_general %278, %276, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %280 = stablehlo.slice %1 [0:1, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %281 = stablehlo.reshape %280 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %282 = stablehlo.dot_general %281, %276, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %283 = stablehlo.slice %2 [0:1, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %284 = stablehlo.reshape %283 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %285 = stablehlo.dot_general %284, %276, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %286 = stablehlo.transpose %279, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %287 = stablehlo.reshape %286 : (tensor<2x24xf32>) -> tensor<2x12x2xf32>
    %288 = stablehlo.transpose %282, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %289 = stablehlo.reshape %288 : (tensor<2x24xf32>) -> tensor<2x12x2xf32>
    %290 = stablehlo.dot_general %260, %289, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<12x2x2xf32>, tensor<2x12x2xf32>) -> tensor<12x2x2xf32>
    %291 = stablehlo.transpose %290, dims = [2, 0, 1] : (tensor<12x2x2xf32>) -> tensor<2x12x2xf32>
    %292 = stablehlo.reshape %291 : (tensor<2x12x2xf32>) -> tensor<2x24xf32>
    %293 = stablehlo.dot_general %247, %287, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<12x2x2xf32>, tensor<2x12x2xf32>) -> tensor<12x2x2xf32>
    %294 = stablehlo.transpose %293, dims = [2, 0, 1] : (tensor<12x2x2xf32>) -> tensor<2x12x2xf32>
    %295 = stablehlo.reshape %294 : (tensor<2x12x2xf32>) -> tensor<2x24xf32>
    %296 = stablehlo.slice %arg11 [0:2, 0:1, 0:4, 0:24] : (tensor<2x2x4x24xf32>) -> tensor<2x1x4x24xf32>
    %297 = stablehlo.reshape %296 : (tensor<2x1x4x24xf32>) -> tensor<2x4x24xf32>
    %298 = stablehlo.reshape %292 : (tensor<2x24xf32>) -> tensor<2x1x24xf32>
    %299 = stablehlo.concatenate %297, %298, dim = 1 : (tensor<2x4x24xf32>, tensor<2x1x24xf32>) -> tensor<2x5x24xf32>
    %300 = stablehlo.slice %arg12 [0:2, 0:1, 0:4, 0:24] : (tensor<2x2x4x24xf32>) -> tensor<2x1x4x24xf32>
    %301 = stablehlo.reshape %300 : (tensor<2x1x4x24xf32>) -> tensor<2x4x24xf32>
    %302 = stablehlo.transpose %285, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %303 = stablehlo.reshape %302 : (tensor<2x24xf32>) -> tensor<2x1x24xf32>
    %304 = stablehlo.concatenate %301, %303, dim = 1 : (tensor<2x4x24xf32>, tensor<2x1x24xf32>) -> tensor<2x5x24xf32>
    %305 = stablehlo.reshape %295 : (tensor<2x24xf32>) -> tensor<2x6x4xf32>
    %306 = stablehlo.reshape %299 : (tensor<2x5x24xf32>) -> tensor<2x5x6x4xf32>
    %307 = stablehlo.reshape %304 : (tensor<2x5x24xf32>) -> tensor<2x5x6x4xf32>
    %308 = stablehlo.dot_general %306, %305, batching_dims = [0, 2] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x5x6x4xf32>, tensor<2x6x4xf32>) -> tensor<2x6x5xf32>
    %cst_50 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %309 = stablehlo.sqrt %cst_50 : tensor<f32>
    %310 = stablehlo.convert %309 : tensor<f32>
    %311 = stablehlo.broadcast_in_dim %310, dims = [] : (tensor<f32>) -> tensor<2x6x5xf32>
    %312 = stablehlo.divide %308, %311 : tensor<2x6x5xf32>
    %cst_51 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %313 = stablehlo.reduce(%312 init: %cst_51) applies stablehlo.maximum across dimensions = [2] : (tensor<2x6x5xf32>, tensor<f32>) -> tensor<2x6xf32>
    %314 = stablehlo.broadcast_in_dim %313, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %315 = stablehlo.broadcast_in_dim %314, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x5xf32>
    %316 = stablehlo.subtract %312, %315 : tensor<2x6x5xf32>
    %317 = stablehlo.exponential %316 : tensor<2x6x5xf32>
    %cst_52 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %318 = stablehlo.reduce(%317 init: %cst_52) applies stablehlo.add across dimensions = [2] : (tensor<2x6x5xf32>, tensor<f32>) -> tensor<2x6xf32>
    %319 = stablehlo.broadcast_in_dim %318, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x5xf32>
    %321 = stablehlo.divide %317, %320 : tensor<2x6x5xf32>
    %322 = stablehlo.dot_general %307, %321, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x5x6x4xf32>, tensor<2x6x5xf32>) -> tensor<2x6x4xf32>
    %323 = stablehlo.reshape %322 : (tensor<2x6x4xf32>) -> tensor<2x24xf32>
    %324 = stablehlo.slice %3 [0:1, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %325 = stablehlo.reshape %324 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %326 = stablehlo.dot_general %325, %323, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %327 = stablehlo.transpose %326, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %328 = stablehlo.add %arg0, %327 : tensor<2x24xf32>
    %329 = stablehlo.slice %arg2 [0:1, 0:24] : (tensor<2x24xf32>) -> tensor<1x24xf32>
    %330 = stablehlo.reshape %329 : (tensor<1x24xf32>) -> tensor<24xf32>
    %331 = stablehlo.dot_general %328, %328, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24xf32>, tensor<2x24xf32>) -> tensor<2xf32>
    %cst_53 = stablehlo.constant dense<2.400000e+01> : tensor<f32>
    %332 = stablehlo.broadcast_in_dim %cst_53, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %333 = stablehlo.divide %331, %332 : tensor<2xf32>
    %cst_54 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %334 = stablehlo.broadcast_in_dim %cst_54, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %335 = stablehlo.add %333, %334 : tensor<2xf32>
    %336 = stablehlo.sqrt %335 : tensor<2xf32>
    %cst_55 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %337 = stablehlo.broadcast_in_dim %cst_55, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %338 = stablehlo.divide %337, %336 : tensor<2xf32>
    %339 = stablehlo.broadcast_in_dim %330, dims = [1] : (tensor<24xf32>) -> tensor<1x24xf32>
    %340 = stablehlo.broadcast_in_dim %339, dims = [0, 1] : (tensor<1x24xf32>) -> tensor<2x24xf32>
    %341 = stablehlo.multiply %340, %328 : tensor<2x24xf32>
    %342 = stablehlo.broadcast_in_dim %338, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %343 = stablehlo.broadcast_in_dim %342, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x24xf32>
    %344 = stablehlo.multiply %341, %343 : tensor<2x24xf32>
    %345 = stablehlo.slice %4 [0:1, 0:32, 0:24] : (tensor<2x32x24xf32>) -> tensor<1x32x24xf32>
    %346 = stablehlo.reshape %345 : (tensor<1x32x24xf32>) -> tensor<32x24xf32>
    %347 = stablehlo.dot_general %346, %344, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x24xf32>, tensor<2x24xf32>) -> tensor<32x2xf32>
    %348 = stablehlo.slice %6 [0:1, 0:32, 0:24] : (tensor<2x32x24xf32>) -> tensor<1x32x24xf32>
    %349 = stablehlo.reshape %348 : (tensor<1x32x24xf32>) -> tensor<32x24xf32>
    %350 = stablehlo.dot_general %349, %344, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x24xf32>, tensor<2x24xf32>) -> tensor<32x2xf32>
    %351 = stablehlo.negate %347 : tensor<32x2xf32>
    %352 = stablehlo.exponential %351 : tensor<32x2xf32>
    %cst_56 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %353 = stablehlo.broadcast_in_dim %cst_56, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %354 = stablehlo.add %353, %352 : tensor<32x2xf32>
    %cst_57 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %355 = stablehlo.broadcast_in_dim %cst_57, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %356 = stablehlo.divide %355, %354 : tensor<32x2xf32>
    %357 = stablehlo.multiply %347, %356 : tensor<32x2xf32>
    %358 = stablehlo.multiply %357, %350 : tensor<32x2xf32>
    %359 = stablehlo.slice %5 [0:1, 0:24, 0:32] : (tensor<2x24x32xf32>) -> tensor<1x24x32xf32>
    %360 = stablehlo.reshape %359 : (tensor<1x24x32xf32>) -> tensor<24x32xf32>
    %361 = stablehlo.dot_general %360, %358, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<24x32xf32>, tensor<32x2xf32>) -> tensor<24x2xf32>
    %362 = stablehlo.transpose %361, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %363 = stablehlo.add %328, %362 : tensor<2x24xf32>
    %364 = stablehlo.slice %arg1 [1:2, 0:24] : (tensor<2x24xf32>) -> tensor<1x24xf32>
    %365 = stablehlo.reshape %364 : (tensor<1x24xf32>) -> tensor<24xf32>
    %366 = stablehlo.dot_general %363, %363, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24xf32>, tensor<2x24xf32>) -> tensor<2xf32>
    %cst_58 = stablehlo.constant dense<2.400000e+01> : tensor<f32>
    %367 = stablehlo.broadcast_in_dim %cst_58, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %368 = stablehlo.divide %366, %367 : tensor<2xf32>
    %cst_59 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %369 = stablehlo.broadcast_in_dim %cst_59, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %370 = stablehlo.add %368, %369 : tensor<2xf32>
    %371 = stablehlo.sqrt %370 : tensor<2xf32>
    %cst_60 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %372 = stablehlo.broadcast_in_dim %cst_60, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %373 = stablehlo.divide %372, %371 : tensor<2xf32>
    %374 = stablehlo.broadcast_in_dim %365, dims = [1] : (tensor<24xf32>) -> tensor<1x24xf32>
    %375 = stablehlo.broadcast_in_dim %374, dims = [0, 1] : (tensor<1x24xf32>) -> tensor<2x24xf32>
    %376 = stablehlo.multiply %375, %363 : tensor<2x24xf32>
    %377 = stablehlo.broadcast_in_dim %373, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %378 = stablehlo.broadcast_in_dim %377, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x24xf32>
    %379 = stablehlo.multiply %376, %378 : tensor<2x24xf32>
    %380 = stablehlo.slice %0 [1:2, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %381 = stablehlo.reshape %380 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %382 = stablehlo.dot_general %381, %379, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %383 = stablehlo.slice %1 [1:2, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %384 = stablehlo.reshape %383 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %385 = stablehlo.dot_general %384, %379, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %386 = stablehlo.slice %2 [1:2, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %387 = stablehlo.reshape %386 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %388 = stablehlo.dot_general %387, %379, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %389 = stablehlo.transpose %382, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %390 = stablehlo.reshape %389 : (tensor<2x24xf32>) -> tensor<2x12x2xf32>
    %391 = stablehlo.transpose %385, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %392 = stablehlo.reshape %391 : (tensor<2x24xf32>) -> tensor<2x12x2xf32>
    %393 = stablehlo.dot_general %260, %392, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<12x2x2xf32>, tensor<2x12x2xf32>) -> tensor<12x2x2xf32>
    %394 = stablehlo.transpose %393, dims = [2, 0, 1] : (tensor<12x2x2xf32>) -> tensor<2x12x2xf32>
    %395 = stablehlo.reshape %394 : (tensor<2x12x2xf32>) -> tensor<2x24xf32>
    %396 = stablehlo.dot_general %247, %390, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<12x2x2xf32>, tensor<2x12x2xf32>) -> tensor<12x2x2xf32>
    %397 = stablehlo.transpose %396, dims = [2, 0, 1] : (tensor<12x2x2xf32>) -> tensor<2x12x2xf32>
    %398 = stablehlo.reshape %397 : (tensor<2x12x2xf32>) -> tensor<2x24xf32>
    %399 = stablehlo.slice %arg11 [0:2, 1:2, 0:4, 0:24] : (tensor<2x2x4x24xf32>) -> tensor<2x1x4x24xf32>
    %400 = stablehlo.reshape %399 : (tensor<2x1x4x24xf32>) -> tensor<2x4x24xf32>
    %401 = stablehlo.reshape %395 : (tensor<2x24xf32>) -> tensor<2x1x24xf32>
    %402 = stablehlo.concatenate %400, %401, dim = 1 : (tensor<2x4x24xf32>, tensor<2x1x24xf32>) -> tensor<2x5x24xf32>
    %403 = stablehlo.slice %arg12 [0:2, 1:2, 0:4, 0:24] : (tensor<2x2x4x24xf32>) -> tensor<2x1x4x24xf32>
    %404 = stablehlo.reshape %403 : (tensor<2x1x4x24xf32>) -> tensor<2x4x24xf32>
    %405 = stablehlo.transpose %388, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %406 = stablehlo.reshape %405 : (tensor<2x24xf32>) -> tensor<2x1x24xf32>
    %407 = stablehlo.concatenate %404, %406, dim = 1 : (tensor<2x4x24xf32>, tensor<2x1x24xf32>) -> tensor<2x5x24xf32>
    %408 = stablehlo.reshape %398 : (tensor<2x24xf32>) -> tensor<2x6x4xf32>
    %409 = stablehlo.reshape %402 : (tensor<2x5x24xf32>) -> tensor<2x5x6x4xf32>
    %410 = stablehlo.reshape %407 : (tensor<2x5x24xf32>) -> tensor<2x5x6x4xf32>
    %411 = stablehlo.dot_general %409, %408, batching_dims = [0, 2] x [0, 1], contracting_dims = [3] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x5x6x4xf32>, tensor<2x6x4xf32>) -> tensor<2x6x5xf32>
    %cst_61 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %412 = stablehlo.sqrt %cst_61 : tensor<f32>
    %413 = stablehlo.convert %412 : tensor<f32>
    %414 = stablehlo.broadcast_in_dim %413, dims = [] : (tensor<f32>) -> tensor<2x6x5xf32>
    %415 = stablehlo.divide %411, %414 : tensor<2x6x5xf32>
    %cst_62 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %416 = stablehlo.reduce(%415 init: %cst_62) applies stablehlo.maximum across dimensions = [2] : (tensor<2x6x5xf32>, tensor<f32>) -> tensor<2x6xf32>
    %417 = stablehlo.broadcast_in_dim %416, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %418 = stablehlo.broadcast_in_dim %417, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x5xf32>
    %419 = stablehlo.subtract %415, %418 : tensor<2x6x5xf32>
    %420 = stablehlo.exponential %419 : tensor<2x6x5xf32>
    %cst_63 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %421 = stablehlo.reduce(%420 init: %cst_63) applies stablehlo.add across dimensions = [2] : (tensor<2x6x5xf32>, tensor<f32>) -> tensor<2x6xf32>
    %422 = stablehlo.broadcast_in_dim %421, dims = [0, 1] : (tensor<2x6xf32>) -> tensor<2x6x1xf32>
    %423 = stablehlo.broadcast_in_dim %422, dims = [0, 1, 2] : (tensor<2x6x1xf32>) -> tensor<2x6x5xf32>
    %424 = stablehlo.divide %420, %423 : tensor<2x6x5xf32>
    %425 = stablehlo.dot_general %410, %424, batching_dims = [0, 2] x [0, 1], contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2x5x6x4xf32>, tensor<2x6x5xf32>) -> tensor<2x6x4xf32>
    %426 = stablehlo.reshape %425 : (tensor<2x6x4xf32>) -> tensor<2x24xf32>
    %427 = stablehlo.slice %3 [1:2, 0:24, 0:24] : (tensor<2x24x24xf32>) -> tensor<1x24x24xf32>
    %428 = stablehlo.reshape %427 : (tensor<1x24x24xf32>) -> tensor<24x24xf32>
    %429 = stablehlo.dot_general %428, %426, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<24x24xf32>, tensor<2x24xf32>) -> tensor<24x2xf32>
    %430 = stablehlo.transpose %429, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %431 = stablehlo.add %363, %430 : tensor<2x24xf32>
    %432 = stablehlo.slice %arg2 [1:2, 0:24] : (tensor<2x24xf32>) -> tensor<1x24xf32>
    %433 = stablehlo.reshape %432 : (tensor<1x24xf32>) -> tensor<24xf32>
    %434 = stablehlo.dot_general %431, %431, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24xf32>, tensor<2x24xf32>) -> tensor<2xf32>
    %cst_64 = stablehlo.constant dense<2.400000e+01> : tensor<f32>
    %435 = stablehlo.broadcast_in_dim %cst_64, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %436 = stablehlo.divide %434, %435 : tensor<2xf32>
    %cst_65 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %437 = stablehlo.broadcast_in_dim %cst_65, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %438 = stablehlo.add %436, %437 : tensor<2xf32>
    %439 = stablehlo.sqrt %438 : tensor<2xf32>
    %cst_66 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %440 = stablehlo.broadcast_in_dim %cst_66, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %441 = stablehlo.divide %440, %439 : tensor<2xf32>
    %442 = stablehlo.broadcast_in_dim %433, dims = [1] : (tensor<24xf32>) -> tensor<1x24xf32>
    %443 = stablehlo.broadcast_in_dim %442, dims = [0, 1] : (tensor<1x24xf32>) -> tensor<2x24xf32>
    %444 = stablehlo.multiply %443, %431 : tensor<2x24xf32>
    %445 = stablehlo.broadcast_in_dim %441, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %446 = stablehlo.broadcast_in_dim %445, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x24xf32>
    %447 = stablehlo.multiply %444, %446 : tensor<2x24xf32>
    %448 = stablehlo.slice %4 [1:2, 0:32, 0:24] : (tensor<2x32x24xf32>) -> tensor<1x32x24xf32>
    %449 = stablehlo.reshape %448 : (tensor<1x32x24xf32>) -> tensor<32x24xf32>
    %450 = stablehlo.dot_general %449, %447, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x24xf32>, tensor<2x24xf32>) -> tensor<32x2xf32>
    %451 = stablehlo.slice %6 [1:2, 0:32, 0:24] : (tensor<2x32x24xf32>) -> tensor<1x32x24xf32>
    %452 = stablehlo.reshape %451 : (tensor<1x32x24xf32>) -> tensor<32x24xf32>
    %453 = stablehlo.dot_general %452, %447, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x24xf32>, tensor<2x24xf32>) -> tensor<32x2xf32>
    %454 = stablehlo.negate %450 : tensor<32x2xf32>
    %455 = stablehlo.exponential %454 : tensor<32x2xf32>
    %cst_67 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %456 = stablehlo.broadcast_in_dim %cst_67, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %457 = stablehlo.add %456, %455 : tensor<32x2xf32>
    %cst_68 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %458 = stablehlo.broadcast_in_dim %cst_68, dims = [] : (tensor<f32>) -> tensor<32x2xf32>
    %459 = stablehlo.divide %458, %457 : tensor<32x2xf32>
    %460 = stablehlo.multiply %450, %459 : tensor<32x2xf32>
    %461 = stablehlo.multiply %460, %453 : tensor<32x2xf32>
    %462 = stablehlo.slice %5 [1:2, 0:24, 0:32] : (tensor<2x24x32xf32>) -> tensor<1x24x32xf32>
    %463 = stablehlo.reshape %462 : (tensor<1x24x32xf32>) -> tensor<24x32xf32>
    %464 = stablehlo.dot_general %463, %461, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<24x32xf32>, tensor<32x2xf32>) -> tensor<24x2xf32>
    %465 = stablehlo.transpose %464, dims = [1, 0] : (tensor<24x2xf32>) -> tensor<2x24xf32>
    %466 = stablehlo.add %431, %465 : tensor<2x24xf32>
    %467 = stablehlo.dot_general %466, %466, batching_dims = [0] x [0], contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x24xf32>, tensor<2x24xf32>) -> tensor<2xf32>
    %cst_69 = stablehlo.constant dense<2.400000e+01> : tensor<f32>
    %468 = stablehlo.broadcast_in_dim %cst_69, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %469 = stablehlo.divide %467, %468 : tensor<2xf32>
    %cst_70 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %470 = stablehlo.broadcast_in_dim %cst_70, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %471 = stablehlo.add %469, %470 : tensor<2xf32>
    %472 = stablehlo.sqrt %471 : tensor<2xf32>
    %cst_71 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %473 = stablehlo.broadcast_in_dim %cst_71, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %474 = stablehlo.divide %473, %472 : tensor<2xf32>
    %475 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<24xf32>) -> tensor<1x24xf32>
    %476 = stablehlo.broadcast_in_dim %475, dims = [0, 1] : (tensor<1x24xf32>) -> tensor<2x24xf32>
    %477 = stablehlo.multiply %476, %466 : tensor<2x24xf32>
    %478 = stablehlo.broadcast_in_dim %474, dims = [0] : (tensor<2xf32>) -> tensor<2x1xf32>
    %479 = stablehlo.broadcast_in_dim %478, dims = [0, 1] : (tensor<2x1xf32>) -> tensor<2x24xf32>
    %480 = stablehlo.multiply %477, %479 : tensor<2x24xf32>
    return %480 : tensor<2x24xf32>
  }
}
