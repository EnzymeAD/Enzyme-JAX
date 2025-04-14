// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// CHECK-LABEL: @shuffle_and_compare
func.func @shuffle_and_compare(%3241 : tensor<4x6128x12272xi64>,
               %3242 : tensor<4x6128x12272xi64>,
               %3243 : tensor<4x6128x12272xi64>,
               %3248 : tensor<4x6128x12272xi64>,
               %3247 : tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1> {
      // CHECK-NOT: dense<306>
      // CHECK: %[[v0:.+]] = stablehlo.compare GE
      // CHECK: %[[v1:.+]] = stablehlo.compare LE
      // CHECK: %[[v2:.+]] = stablehlo.and %[[v1]], %[[v0]]
      // CHECK: return %[[v2]]
      %c_304 = stablehlo.constant dense<304> : tensor<4x6128x12272xi64>
      %c_306 = stablehlo.constant dense<306> : tensor<4x6128x12272xi64>
      %c_308 = stablehlo.constant dense<308> : tensor<4x6128x12272xi64>
      %3249 = stablehlo.compare  LE, %3241, %c_304 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3250 = stablehlo.compare  LE, %3242, %c_304 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3251 = stablehlo.compare  LE, %3243, %c_304 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3252 = stablehlo.compare  LE, %3248, %c_304 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3253 = stablehlo.compare  GE, %3241, %c_308 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3254 = stablehlo.compare  GE, %3247, %c_308 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3255 = stablehlo.and %3249, %3253 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : tensor<4x6128x12272xi1>
      %3256 = stablehlo.compare  LE, %3241, %c_306 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3257 = stablehlo.compare  LE, %3242, %c_306 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3258 = stablehlo.compare  LE, %3243, %c_306 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3259 = stablehlo.compare  LE, %3247, %c_306 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3260 = stablehlo.compare  LE, %3248, %c_306 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : (tensor<4x6128x12272xi64>, tensor<4x6128x12272xi64>) -> tensor<4x6128x12272xi1>
      %3261 = stablehlo.and %3255, %3256 {mhlo.sharding = "{devices=[1,4,4]<=[4,4]T(1,0)}"} : tensor<4x6128x12272xi1>
     return %3261 : tensor<4x6128x12272xi1>
}
