%26 = stablehlo.slice %arg11 [8:520, 1:1023, 2034:2040] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x6xf64> loc(#loc1191)
%27 = stablehlo.slice %arg11 [8:520, 1:1023, 8:2040] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64> loc(#loc1191)
%23 = stablehlo.slice %arg11 [8:520, 1:1023, 8:16] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64> loc(#loc1190)
%36 = stablehlo.concatenate %26, %27, %23, dim = 2 {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<512x1022x6xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x8xf64>) -> tensor<512x1022x2046xf64> loc(#loc1262

