// RUN: enzymexlamlir-opt --transform-interpreter %s | FileCheck %s

module @reactant_f2 attributes {mhlo.num_partitions = 1 : i64, mhlo.num_replicas = 1 : i64, transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.enzyme_hlo.concat_concat_axis_swap
    } : !transform.any_op
    transform.yield 
  }
  func.func @main(%arg0: tensor<6x5xf32> {enzymexla.memory_effects = ["read", "write", "allocate", "free"], tf.aliasing_output = 1 : i32}, %arg1: tensor<3x3xf32> {enzymexla.memory_effects = [], tf.aliasing_output = 2 : i32}) -> (tensor<f32>, tensor<6x5xf32>, tensor<3x3xf32>) attributes {enzymexla.memory_effects = ["read", "write", "allocate", "free"]} {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<6x5xf32>) -> tensor<5x6xf32>
    %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<5x6xf32>
    %3 = stablehlo.slice %0 [0:1, 0:1] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %4 = stablehlo.slice %1 [0:1, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<1x1xf32>
    %6 = stablehlo.slice %1 [1:2, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %7 = stablehlo.multiply %3, %6 : tensor<1x1xf32>
    %8 = stablehlo.slice %0 [1:2, 0:1] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %9 = stablehlo.slice %1 [2:3, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %10 = stablehlo.multiply %8, %9 : tensor<1x1xf32>
    %11 = stablehlo.slice %1 [0:1, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %12 = stablehlo.slice %1 [1:2, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %13 = stablehlo.multiply %3, %12 : tensor<1x1xf32>
    %14 = stablehlo.slice %1 [2:3, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %15 = stablehlo.multiply %8, %14 : tensor<1x1xf32>
    %16 = stablehlo.slice %0 [0:1, 1:2] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %17 = stablehlo.slice %1 [0:1, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %18 = stablehlo.slice %1 [1:2, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %19 = stablehlo.multiply %16, %18 : tensor<1x1xf32>
    %20 = stablehlo.slice %0 [1:2, 1:2] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %21 = stablehlo.slice %1 [2:3, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<1x1xf32>
    %23 = stablehlo.multiply %8, %6 : tensor<1x1xf32>
    %24 = stablehlo.slice %0 [2:3, 0:1] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %25 = stablehlo.multiply %24, %9 : tensor<1x1xf32>
    %26 = stablehlo.multiply %8, %4 : tensor<1x1xf32>
    %27 = stablehlo.multiply %24, %6 : tensor<1x1xf32>
    %28 = stablehlo.slice %0 [3:4, 0:1] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %29 = stablehlo.multiply %28, %9 : tensor<1x1xf32>
    %30 = stablehlo.slice %0 [2:4, 1:2] : (tensor<5x6xf32>) -> tensor<2x1xf32>
    %31 = stablehlo.concatenate %30, dim = 0 : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %32 = "enzymexla.extend"(%21) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<1x1xf32>) -> tensor<2x1xf32>
    %33 = stablehlo.multiply %31, %32 : tensor<2x1xf32>
    %34 = stablehlo.multiply %24, %4 : tensor<1x1xf32>
    %35 = stablehlo.multiply %28, %6 : tensor<1x1xf32>
    %36 = stablehlo.slice %0 [4:5, 0:1] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %37 = stablehlo.multiply %36, %9 : tensor<1x1xf32>
    %38 = stablehlo.slice %0 [4:5, 1:2] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %39 = stablehlo.multiply %38, %21 : tensor<1x1xf32>
    %40 = stablehlo.multiply %28, %4 : tensor<1x1xf32>
    %41 = stablehlo.multiply %36, %6 : tensor<1x1xf32>
    %42 = stablehlo.slice %0 [2:5, 0:1] : (tensor<5x6xf32>) -> tensor<3x1xf32>
    %43 = stablehlo.slice %0 [1:5, 0:1] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %44 = stablehlo.concatenate %43, dim = 0 : (tensor<4x1xf32>) -> tensor<4x1xf32>
    %45 = "enzymexla.extend"(%12) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x1xf32>) -> tensor<3x1xf32>
    %46 = stablehlo.concatenate %45, %12, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %47 = stablehlo.multiply %44, %46 : tensor<4x1xf32>
    %48 = "enzymexla.extend"(%42) <{dimension = 0 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<3x1xf32>) -> tensor<4x1xf32>
    %49 = "enzymexla.extend"(%14) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x1xf32>) -> tensor<3x1xf32>
    %50 = stablehlo.concatenate %49, %14, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %51 = stablehlo.multiply %48, %50 : tensor<4x1xf32>
    %52 = stablehlo.slice %0 [1:5, 1:2] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %53 = stablehlo.concatenate %52, dim = 0 : (tensor<4x1xf32>) -> tensor<4x1xf32>
    %54 = "enzymexla.extend"(%18) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x1xf32>) -> tensor<3x1xf32>
    %55 = stablehlo.concatenate %54, %18, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %56 = stablehlo.multiply %53, %55 : tensor<4x1xf32>
    %57 = stablehlo.slice %2 [0:5, 0:1] : (tensor<5x6xf32>) -> tensor<5x1xf32>
    %58 = stablehlo.concatenate %57, dim = 0 : (tensor<5x1xf32>) -> tensor<5x1xf32>
    %59 = stablehlo.concatenate %5, %5, %26, %34, %40, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %60 = stablehlo.concatenate %7, %23, %27, %35, %41, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %61 = stablehlo.concatenate %10, %25, %29, %37, %37, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %62 = stablehlo.slice %0 [0:4, 0:1] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %63 = "enzymexla.extend"(%62) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1xf32>) -> tensor<5x1xf32>
    %64 = "enzymexla.extend"(%11) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x1xf32>) -> tensor<3x1xf32>
    %65 = stablehlo.concatenate %64, %11, %11, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %66 = stablehlo.concatenate %13, %47, dim = 0 : (tensor<1x1xf32>, tensor<4x1xf32>) -> tensor<5x1xf32>
    %67 = stablehlo.concatenate %15, %51, dim = 0 : (tensor<1x1xf32>, tensor<4x1xf32>) -> tensor<5x1xf32>
    %68 = stablehlo.slice %0 [0:4, 1:2] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %69 = "enzymexla.extend"(%68) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1xf32>) -> tensor<5x1xf32>
    %70 = "enzymexla.extend"(%17) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x1xf32>) -> tensor<3x1xf32>
    %71 = stablehlo.concatenate %70, %17, %17, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %72 = stablehlo.multiply %69, %71 : tensor<5x1xf32>
    %73 = stablehlo.concatenate %19, %56, dim = 0 : (tensor<1x1xf32>, tensor<4x1xf32>) -> tensor<5x1xf32>
    %74 = stablehlo.concatenate %22, %33, %39, %39, dim = 0 : (tensor<1x1xf32>, tensor<2x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %75 = stablehlo.slice %2 [0:5, 1:6] : (tensor<5x6xf32>) -> tensor<5x5xf32>
    %76 = stablehlo.slice %2 [0:1, 1:2] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %77 = stablehlo.slice %0 [0:1, 2:3] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %78 = stablehlo.slice %0 [1:2, 2:3] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %79 = stablehlo.multiply %78, %21 : tensor<1x1xf32>
    %80 = stablehlo.slice %75 [1:2, 0:1] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %81 = stablehlo.concatenate %80, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %82 = stablehlo.slice %75 [2:3, 0:1] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %83 = stablehlo.concatenate %82, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %84 = stablehlo.slice %0 [2:4, 2:3] : (tensor<5x6xf32>) -> tensor<2x1xf32>
    %85 = stablehlo.concatenate %84, dim = 0 : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %86 = stablehlo.multiply %85, %32 : tensor<2x1xf32>
    %87 = stablehlo.slice %75 [3:4, 0:1] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %88 = stablehlo.concatenate %87, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %89 = stablehlo.slice %0 [4:5, 2:3] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %90 = stablehlo.multiply %89, %21 : tensor<1x1xf32>
    %91 = stablehlo.slice %75 [4:5, 0:1] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %92 = stablehlo.concatenate %91, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %93 = stablehlo.concatenate %76, %81, %83, %88, %92, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %94 = stablehlo.concatenate %5, %5, %26, %34, %40, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %95 = stablehlo.concatenate %7, %23, %27, %35, %41, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %96 = stablehlo.concatenate %10, %25, %29, %37, %37, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %97 = stablehlo.concatenate %64, %11, %11, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %98 = stablehlo.slice %0 [0:5, 1:2] : (tensor<5x6xf32>) -> tensor<5x1xf32>
    %99 = stablehlo.concatenate %98, dim = 0 : (tensor<5x1xf32>) -> tensor<5x1xf32>
    %100 = stablehlo.concatenate %45, %12, %12, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %101 = stablehlo.multiply %99, %100 : tensor<5x1xf32>
    %102 = "enzymexla.extend"(%52) <{dimension = 0 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<4x1xf32>) -> tensor<5x1xf32>
    %103 = stablehlo.concatenate %49, %14, %14, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %104 = stablehlo.multiply %102, %103 : tensor<5x1xf32>
    %105 = stablehlo.slice %0 [1:4, 2:3] : (tensor<5x6xf32>) -> tensor<3x1xf32>
    %106 = stablehlo.slice %0 [0:4, 2:3] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %107 = "enzymexla.extend"(%106) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1xf32>) -> tensor<5x1xf32>
    %108 = stablehlo.concatenate %70, %17, %17, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %109 = stablehlo.multiply %107, %108 : tensor<5x1xf32>
    %110 = stablehlo.slice %0 [2:5, 2:3] : (tensor<5x6xf32>) -> tensor<3x1xf32>
    %111 = stablehlo.slice %0 [1:5, 2:3] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %112 = stablehlo.slice %0 [0:5, 2:3] : (tensor<5x6xf32>) -> tensor<5x1xf32>
    %113 = stablehlo.concatenate %112, dim = 0 : (tensor<5x1xf32>) -> tensor<5x1xf32>
    %114 = stablehlo.concatenate %54, %18, %18, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %115 = stablehlo.multiply %113, %114 : tensor<5x1xf32>
    %116 = stablehlo.concatenate %79, %86, %90, %90, dim = 0 : (tensor<1x1xf32>, tensor<2x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %117 = stablehlo.slice %2 [0:1, 2:3] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %118 = stablehlo.slice %0 [0:1, 3:4] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %119 = stablehlo.slice %0 [1:2, 3:4] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %120 = stablehlo.multiply %119, %21 : tensor<1x1xf32>
    %121 = stablehlo.slice %75 [1:2, 1:2] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %122 = stablehlo.concatenate %121, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %123 = stablehlo.slice %75 [2:3, 1:2] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %124 = stablehlo.concatenate %123, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %125 = stablehlo.slice %0 [2:4, 3:4] : (tensor<5x6xf32>) -> tensor<2x1xf32>
    %126 = stablehlo.concatenate %125, dim = 0 : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %127 = stablehlo.multiply %126, %32 : tensor<2x1xf32>
    %128 = stablehlo.slice %75 [3:4, 1:2] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %129 = stablehlo.concatenate %128, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %130 = stablehlo.slice %0 [4:5, 3:4] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %131 = stablehlo.multiply %130, %21 : tensor<1x1xf32>
    %132 = stablehlo.slice %75 [4:5, 1:2] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %133 = stablehlo.concatenate %132, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %134 = stablehlo.concatenate %117, %122, %124, %129, %133, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %135 = "enzymexla.extend"(%4) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x1xf32>) -> tensor<3x1xf32>
    %136 = stablehlo.concatenate %135, %4, %4, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %137 = stablehlo.multiply %69, %136 : tensor<5x1xf32>
    %138 = stablehlo.concatenate %98, dim = 0 : (tensor<5x1xf32>) -> tensor<5x1xf32>
    %139 = "enzymexla.extend"(%6) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x1xf32>) -> tensor<3x1xf32>
    %140 = stablehlo.concatenate %139, %6, %6, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %141 = stablehlo.multiply %138, %140 : tensor<5x1xf32>
    %142 = "enzymexla.extend"(%9) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 1 : i64}> : (tensor<1x1xf32>) -> tensor<3x1xf32>
    %143 = stablehlo.concatenate %142, %9, %9, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %144 = stablehlo.multiply %102, %143 : tensor<5x1xf32>
    %145 = stablehlo.concatenate %64, %11, %11, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %146 = stablehlo.concatenate %112, dim = 0 : (tensor<5x1xf32>) -> tensor<5x1xf32>
    %147 = stablehlo.concatenate %45, %12, %12, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %148 = stablehlo.multiply %146, %147 : tensor<5x1xf32>
    %149 = "enzymexla.extend"(%111) <{dimension = 0 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<4x1xf32>) -> tensor<5x1xf32>
    %150 = stablehlo.concatenate %49, %14, %14, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %151 = stablehlo.multiply %149, %150 : tensor<5x1xf32>
    %152 = stablehlo.slice %0 [1:4, 3:4] : (tensor<5x6xf32>) -> tensor<3x1xf32>
    %153 = stablehlo.slice %0 [0:4, 3:4] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %154 = "enzymexla.extend"(%153) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1xf32>) -> tensor<5x1xf32>
    %155 = stablehlo.concatenate %70, %17, %17, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %156 = stablehlo.multiply %154, %155 : tensor<5x1xf32>
    %157 = stablehlo.slice %0 [2:5, 3:4] : (tensor<5x6xf32>) -> tensor<3x1xf32>
    %158 = stablehlo.slice %0 [0:5, 3:4] : (tensor<5x6xf32>) -> tensor<5x1xf32>
    %159 = stablehlo.concatenate %158, dim = 0 : (tensor<5x1xf32>) -> tensor<5x1xf32>
    %160 = stablehlo.concatenate %54, %18, %18, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %161 = stablehlo.multiply %159, %160 : tensor<5x1xf32>
    %162 = stablehlo.concatenate %120, %127, %131, %131, dim = 0 : (tensor<1x1xf32>, tensor<2x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %163 = stablehlo.slice %2 [0:1, 3:4] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %164 = stablehlo.multiply %77, %6 : tensor<1x1xf32>
    %165 = stablehlo.multiply %78, %9 : tensor<1x1xf32>
    %166 = stablehlo.multiply %118, %12 : tensor<1x1xf32>
    %167 = stablehlo.multiply %119, %14 : tensor<1x1xf32>
    %168 = stablehlo.slice %0 [0:1, 4:5] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %169 = stablehlo.multiply %168, %18 : tensor<1x1xf32>
    %170 = stablehlo.slice %0 [1:2, 4:5] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %171 = stablehlo.multiply %170, %21 : tensor<1x1xf32>
    %172 = stablehlo.slice %75 [1:2, 2:3] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %173 = stablehlo.concatenate %172, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %174 = stablehlo.slice %75 [2:3, 2:3] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %175 = stablehlo.concatenate %174, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %176 = stablehlo.slice %0 [2:4, 4:5] : (tensor<5x6xf32>) -> tensor<2x1xf32>
    %177 = stablehlo.concatenate %176, dim = 0 : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %178 = stablehlo.multiply %177, %32 : tensor<2x1xf32>
    %179 = stablehlo.slice %75 [3:4, 2:3] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %180 = stablehlo.concatenate %179, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %181 = stablehlo.slice %0 [4:5, 4:5] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %182 = stablehlo.multiply %181, %21 : tensor<1x1xf32>
    %183 = stablehlo.concatenate %105, dim = 0 : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %184 = stablehlo.multiply %183, %139 : tensor<3x1xf32>
    %185 = stablehlo.concatenate %152, dim = 0 : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %186 = stablehlo.multiply %185, %45 : tensor<3x1xf32>
    %187 = stablehlo.slice %0 [1:4, 4:5] : (tensor<5x6xf32>) -> tensor<3x1xf32>
    %188 = stablehlo.concatenate %187, dim = 0 : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %189 = stablehlo.multiply %188, %54 : tensor<3x1xf32>
    %190 = stablehlo.slice %75 [4:5, 2:3] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %191 = stablehlo.concatenate %190, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %192 = stablehlo.multiply %89, %6 : tensor<1x1xf32>
    %193 = stablehlo.multiply %130, %12 : tensor<1x1xf32>
    %194 = stablehlo.multiply %181, %18 : tensor<1x1xf32>
    %195 = "enzymexla.extend"(%110) <{dimension = 0 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<3x1xf32>) -> tensor<4x1xf32>
    %196 = stablehlo.concatenate %142, %9, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %197 = stablehlo.multiply %195, %196 : tensor<4x1xf32>
    %198 = "enzymexla.extend"(%157) <{dimension = 0 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<3x1xf32>) -> tensor<4x1xf32>
    %199 = stablehlo.concatenate %49, %14, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %200 = stablehlo.multiply %198, %199 : tensor<4x1xf32>
    %201 = stablehlo.concatenate %163, %173, %175, %180, %191, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %202 = stablehlo.concatenate %135, %4, %4, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %203 = stablehlo.multiply %107, %202 : tensor<5x1xf32>
    %204 = stablehlo.concatenate %164, %184, %192, dim = 0 : (tensor<1x1xf32>, tensor<3x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %205 = stablehlo.concatenate %165, %197, dim = 0 : (tensor<1x1xf32>, tensor<4x1xf32>) -> tensor<5x1xf32>
    %206 = stablehlo.concatenate %64, %11, %11, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %207 = stablehlo.concatenate %166, %186, %193, dim = 0 : (tensor<1x1xf32>, tensor<3x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %208 = stablehlo.concatenate %167, %200, dim = 0 : (tensor<1x1xf32>, tensor<4x1xf32>) -> tensor<5x1xf32>
    %209 = stablehlo.slice %0 [0:4, 4:5] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %210 = "enzymexla.extend"(%209) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1xf32>) -> tensor<5x1xf32>
    %211 = stablehlo.concatenate %70, %17, %17, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %212 = stablehlo.multiply %210, %211 : tensor<5x1xf32>
    %213 = stablehlo.concatenate %169, %189, %194, dim = 0 : (tensor<1x1xf32>, tensor<3x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %214 = stablehlo.concatenate %171, %178, %182, %182, dim = 0 : (tensor<1x1xf32>, tensor<2x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %215 = stablehlo.slice %2 [0:1, 4:5] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %216 = stablehlo.multiply %119, %9 : tensor<1x1xf32>
    %217 = stablehlo.multiply %170, %14 : tensor<1x1xf32>
    %218 = stablehlo.slice %0 [0:1, 5:6] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %219 = stablehlo.multiply %218, %17 : tensor<1x1xf32>
    %220 = stablehlo.multiply %218, %18 : tensor<1x1xf32>
    %221 = stablehlo.slice %0 [1:2, 5:6] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %222 = stablehlo.multiply %221, %21 : tensor<1x1xf32>
    %223 = stablehlo.slice %75 [1:2, 3:4] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %224 = stablehlo.concatenate %223, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %225 = stablehlo.multiply %221, %18 : tensor<1x1xf32>
    %226 = stablehlo.slice %0 [2:3, 5:6] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %227 = stablehlo.multiply %226, %21 : tensor<1x1xf32>
    %228 = stablehlo.slice %75 [2:3, 3:4] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %229 = stablehlo.concatenate %228, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %230 = stablehlo.multiply %221, %17 : tensor<1x1xf32>
    %231 = stablehlo.multiply %226, %18 : tensor<1x1xf32>
    %232 = stablehlo.slice %0 [3:4, 5:6] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %233 = stablehlo.multiply %232, %21 : tensor<1x1xf32>
    %234 = stablehlo.slice %75 [3:4, 3:4] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %235 = stablehlo.concatenate %234, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %236 = stablehlo.multiply %226, %17 : tensor<1x1xf32>
    %237 = stablehlo.multiply %232, %18 : tensor<1x1xf32>
    %238 = stablehlo.slice %0 [4:5, 5:6] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %239 = stablehlo.multiply %238, %21 : tensor<1x1xf32>
    %240 = stablehlo.slice %75 [4:5, 3:4] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %241 = stablehlo.concatenate %240, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %242 = stablehlo.multiply %232, %17 : tensor<1x1xf32>
    %243 = stablehlo.multiply %238, %18 : tensor<1x1xf32>
    %244 = stablehlo.concatenate %142, %9, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %245 = stablehlo.multiply %198, %244 : tensor<4x1xf32>
    %246 = stablehlo.slice %0 [2:5, 4:5] : (tensor<5x6xf32>) -> tensor<3x1xf32>
    %247 = "enzymexla.extend"(%246) <{dimension = 0 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<3x1xf32>) -> tensor<4x1xf32>
    %248 = stablehlo.concatenate %49, %14, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>) -> tensor<4x1xf32>
    %249 = stablehlo.multiply %247, %248 : tensor<4x1xf32>
    %250 = stablehlo.concatenate %215, %224, %229, %235, %241, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %251 = stablehlo.concatenate %135, %4, %4, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %252 = stablehlo.multiply %154, %251 : tensor<5x1xf32>
    %253 = stablehlo.concatenate %158, dim = 0 : (tensor<5x1xf32>) -> tensor<5x1xf32>
    %254 = stablehlo.concatenate %139, %6, %6, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %255 = stablehlo.multiply %253, %254 : tensor<5x1xf32>
    %256 = stablehlo.concatenate %216, %245, dim = 0 : (tensor<1x1xf32>, tensor<4x1xf32>) -> tensor<5x1xf32>
    %257 = stablehlo.concatenate %64, %11, %11, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %258 = stablehlo.slice %0 [1:5, 4:5] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %259 = stablehlo.slice %0 [0:5, 4:5] : (tensor<5x6xf32>) -> tensor<5x1xf32>
    %260 = stablehlo.concatenate %259, dim = 0 : (tensor<5x1xf32>) -> tensor<5x1xf32>
    %261 = stablehlo.concatenate %45, %12, %12, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %262 = stablehlo.multiply %260, %261 : tensor<5x1xf32>
    %263 = stablehlo.concatenate %217, %249, dim = 0 : (tensor<1x1xf32>, tensor<4x1xf32>) -> tensor<5x1xf32>
    %264 = stablehlo.concatenate %219, %219, %230, %236, %242, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %265 = stablehlo.concatenate %220, %225, %231, %237, %243, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %266 = stablehlo.concatenate %222, %227, %233, %239, %239, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %267 = stablehlo.slice %2 [0:1, 5:6] : (tensor<5x6xf32>) -> tensor<1x1xf32>
    %268 = stablehlo.slice %75 [1:2, 4:5] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %269 = stablehlo.concatenate %268, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %270 = stablehlo.slice %75 [2:3, 4:5] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %271 = stablehlo.concatenate %270, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %272 = stablehlo.slice %75 [3:4, 4:5] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %273 = stablehlo.concatenate %272, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %274 = stablehlo.slice %75 [4:5, 4:5] : (tensor<5x5xf32>) -> tensor<1x1xf32>
    %275 = stablehlo.concatenate %274, dim = 1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %276 = stablehlo.concatenate %267, %269, %271, %273, %275, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %277 = stablehlo.concatenate %135, %4, %4, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %278 = stablehlo.multiply %210, %277 : tensor<5x1xf32>
    %279 = stablehlo.concatenate %259, dim = 0 : (tensor<5x1xf32>) -> tensor<5x1xf32>
    %280 = stablehlo.concatenate %139, %6, %6, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %281 = stablehlo.multiply %279, %280 : tensor<5x1xf32>
    %282 = "enzymexla.extend"(%258) <{dimension = 0 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<4x1xf32>) -> tensor<5x1xf32>
    %283 = stablehlo.concatenate %142, %9, %9, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %284 = stablehlo.multiply %282, %283 : tensor<5x1xf32>
    %285 = stablehlo.slice %0 [0:4, 5:6] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %286 = "enzymexla.extend"(%285) <{dimension = 0 : i64, lhs = 1 : i64, rhs = 0 : i64}> : (tensor<4x1xf32>) -> tensor<5x1xf32>
    %287 = stablehlo.concatenate %64, %11, %11, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %288 = stablehlo.slice %0 [1:5, 5:6] : (tensor<5x6xf32>) -> tensor<4x1xf32>
    %289 = stablehlo.slice %0 [0:5, 5:6] : (tensor<5x6xf32>) -> tensor<5x1xf32>
    %290 = stablehlo.concatenate %289, dim = 0 : (tensor<5x1xf32>) -> tensor<5x1xf32>
    %291 = stablehlo.concatenate %45, %12, %12, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %292 = stablehlo.multiply %290, %291 : tensor<5x1xf32>
    %293 = "enzymexla.extend"(%288) <{dimension = 0 : i64, lhs = 0 : i64, rhs = 1 : i64}> : (tensor<4x1xf32>) -> tensor<5x1xf32>
    %294 = stablehlo.concatenate %49, %14, %14, dim = 0 : (tensor<3x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %295 = stablehlo.multiply %293, %294 : tensor<5x1xf32>
    %296 = stablehlo.concatenate %219, %219, %230, %236, %242, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %297 = stablehlo.concatenate %220, %225, %231, %237, %243, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %298 = stablehlo.concatenate %222, %227, %233, %239, %239, dim = 0 : (tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<5x1xf32>
    %299 = stablehlo.concatenate %58, %93, %134, %201, %250, %276, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %300 = stablehlo.concatenate %59, %94, %137, %203, %252, %278, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %301 = stablehlo.add %299, %300 : tensor<5x6xf32>
    %302 = stablehlo.concatenate %60, %95, %141, %204, %255, %281, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %303 = stablehlo.add %301, %302 : tensor<5x6xf32>
    %304 = stablehlo.concatenate %61, %96, %144, %205, %256, %284, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %305 = stablehlo.add %303, %304 : tensor<5x6xf32>
    %306 = stablehlo.concatenate %63, %69, %107, %154, %210, %286, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %307 = stablehlo.concatenate %65, %97, %145, %206, %257, %287, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %308 = stablehlo.multiply %306, %307 : tensor<5x6xf32>
    %309 = stablehlo.add %305, %308 : tensor<5x6xf32>
    %310 = stablehlo.concatenate %66, %101, %148, %207, %262, %292, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %311 = stablehlo.add %309, %310 : tensor<5x6xf32>
    %312 = stablehlo.concatenate %67, %104, %151, %208, %263, %295, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %313 = stablehlo.add %311, %312 : tensor<5x6xf32>
    %314 = stablehlo.concatenate %72, %109, %156, %212, %264, %296, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %315 = stablehlo.add %313, %314 : tensor<5x6xf32>
    %316 = stablehlo.concatenate %73, %115, %161, %213, %265, %297, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %317 = stablehlo.add %315, %316 : tensor<5x6xf32>
    %318 = stablehlo.concatenate %74, %116, %162, %214, %266, %298, dim = 1 : (tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>, tensor<5x1xf32>) -> tensor<5x6xf32>
    %319 = stablehlo.add %317, %318 : tensor<5x6xf32>
    %320 = stablehlo.multiply %319, %319 : tensor<5x6xf32>
    %321 = stablehlo.reduce(%320 init: %cst) applies stablehlo.add across dimensions = [0, 1] : (tensor<5x6xf32>, tensor<f32>) -> tensor<f32>
    %322 = stablehlo.transpose %0, dims = [1, 0] : (tensor<5x6xf32>) -> tensor<6x5xf32>
    %323 = stablehlo.transpose %1, dims = [1, 0] : (tensor<3x3xf32>) -> tensor<3x3xf32>
    return %321, %322, %323 : tensor<f32>, tensor<6x5xf32>, tensor<3x3xf32>
  }
}

// CHECK: return
