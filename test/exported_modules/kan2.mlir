module {
  func.func private @softsign_broadcast_scalar(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.abs %0 : tensor<f32>
    %2 = stablehlo.add %cst, %1 : tensor<f32>
    %3 = stablehlo.divide %0, %2 : tensor<f32>
    %4 = stablehlo.transpose %3, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %4, %5 : tensor<f32>, tensor<f32>
  }
  func.func private @"-_broadcast_scalar"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %arg1, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.subtract %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func private @"*_broadcast_scalar"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %arg1, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.multiply %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func private @literal_pow_broadcast_scalar(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.multiply %0, %0 : tensor<f32>
    %2 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    %3 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<f32>, tensor<f32>
  }
  func.func private @"-_broadcast_scalar1"(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.negate %0 : tensor<f32>
    %2 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    %3 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<f32>, tensor<f32>
  }
  func.func private @exp_broadcast_scalar(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.exponential %0 : tensor<f32>
    %2 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    %3 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<f32>, tensor<f32>
  }
  func.func private @softsign_broadcast_scalar1(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.abs %0 : tensor<f32>
    %2 = stablehlo.add %cst, %1 : tensor<f32>
    %3 = stablehlo.divide %0, %2 : tensor<f32>
    %4 = stablehlo.transpose %3, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %4, %5 : tensor<f32>, tensor<f32>
  }
  func.func private @"-_broadcast_scalar2"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %arg1, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.subtract %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func private @"*_broadcast_scalar1"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %arg1, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.multiply %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func private @literal_pow_broadcast_scalar1(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.multiply %0, %0 : tensor<f32>
    %2 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    %3 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<f32>, tensor<f32>
  }
  func.func private @"-_broadcast_scalar3"(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.negate %0 : tensor<f32>
    %2 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    %3 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<f32>, tensor<f32>
  }
  func.func private @exp_broadcast_scalar1(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.exponential %0 : tensor<f32>
    %2 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    %3 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<f32>, tensor<f32>
  }
  func.func private @softsign_broadcast_scalar2(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.abs %0 : tensor<f32>
    %2 = stablehlo.add %cst, %1 : tensor<f32>
    %3 = stablehlo.divide %0, %2 : tensor<f32>
    %4 = stablehlo.transpose %3, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %4, %5 : tensor<f32>, tensor<f32>
  }
  func.func private @"-_broadcast_scalar4"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %arg1, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.subtract %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func private @"*_broadcast_scalar2"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %arg1, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.multiply %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func private @literal_pow_broadcast_scalar2(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.multiply %0, %0 : tensor<f32>
    %2 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    %3 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<f32>, tensor<f32>
  }
  func.func private @"-_broadcast_scalar5"(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.negate %0 : tensor<f32>
    %2 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    %3 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<f32>, tensor<f32>
  }
  func.func private @exp_broadcast_scalar2(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.exponential %0 : tensor<f32>
    %2 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    %3 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<f32>, tensor<f32>
  }
  func.func private @batched_softsign_broadcast_scalar(%arg0: tensor<1x100xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<1x100xf32>
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %1 = stablehlo.abs %0 : tensor<1x100xf32>
    %2 = stablehlo.add %cst, %1 : tensor<1x100xf32>
    %3 = stablehlo.divide %0, %2 : tensor<1x100xf32>
    %4 = stablehlo.transpose %3, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %5 = stablehlo.transpose %0, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    return %4, %5 : tensor<1x100xf32>, tensor<1x100xf32>
  }
  func.func private @"batched_-_broadcast_scalar"(%arg0: tensor<10x100xf32>, %arg1: tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>, tensor<10x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %1 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<10x100xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %5 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    return %3, %4, %5 : tensor<10x100xf32>, tensor<10x100xf32>, tensor<10x100xf32>
  }
  func.func private @"batched_*_broadcast_scalar"(%arg0: tensor<10x100xf32>, %arg1: tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>, tensor<10x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %1 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<10x100xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %5 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    return %3, %4, %5 : tensor<10x100xf32>, tensor<10x100xf32>, tensor<10x100xf32>
  }
  func.func private @batched_literal_pow_broadcast_scalar(%arg0: tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<10x100xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %3 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    return %2, %3 : tensor<10x100xf32>, tensor<10x100xf32>
  }
  func.func private @"batched_-_broadcast_scalar1"(%arg0: tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %1 = stablehlo.negate %0 : tensor<10x100xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %3 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    return %2, %3 : tensor<10x100xf32>, tensor<10x100xf32>
  }
  func.func private @batched_exp_broadcast_scalar(%arg0: tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %1 = stablehlo.exponential %0 : tensor<10x100xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %3 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    return %2, %3 : tensor<10x100xf32>, tensor<10x100xf32>
  }
  func.func private @batched_softsign_broadcast_scalar1(%arg0: tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<40x100xf32>
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %1 = stablehlo.abs %0 : tensor<40x100xf32>
    %2 = stablehlo.add %cst, %1 : tensor<40x100xf32>
    %3 = stablehlo.divide %0, %2 : tensor<40x100xf32>
    %4 = stablehlo.transpose %3, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %5 = stablehlo.transpose %0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    return %4, %5 : tensor<40x100xf32>, tensor<40x100xf32>
  }
  func.func private @"batched_-_broadcast_scalar2"(%arg0: tensor<10x4000xf32>, %arg1: tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %1 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<10x4000xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %5 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    return %3, %4, %5 : tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>
  }
  func.func private @"batched_*_broadcast_scalar1"(%arg0: tensor<10x4000xf32>, %arg1: tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %1 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<10x4000xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %5 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    return %3, %4, %5 : tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>
  }
  func.func private @batched_literal_pow_broadcast_scalar1(%arg0: tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<10x4000xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %3 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    return %2, %3 : tensor<10x4000xf32>, tensor<10x4000xf32>
  }
  func.func private @"batched_-_broadcast_scalar3"(%arg0: tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %1 = stablehlo.negate %0 : tensor<10x4000xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %3 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    return %2, %3 : tensor<10x4000xf32>, tensor<10x4000xf32>
  }
  func.func private @batched_exp_broadcast_scalar1(%arg0: tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %1 = stablehlo.exponential %0 : tensor<10x4000xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %3 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    return %2, %3 : tensor<10x4000xf32>, tensor<10x4000xf32>
  }
  func.func private @batched_softsign_broadcast_scalar2(%arg0: tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<40x100xf32>
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %1 = stablehlo.abs %0 : tensor<40x100xf32>
    %2 = stablehlo.add %cst, %1 : tensor<40x100xf32>
    %3 = stablehlo.divide %0, %2 : tensor<40x100xf32>
    %4 = stablehlo.transpose %3, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %5 = stablehlo.transpose %0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    return %4, %5 : tensor<40x100xf32>, tensor<40x100xf32>
  }
  func.func private @"batched_-_broadcast_scalar4"(%arg0: tensor<10x4000xf32>, %arg1: tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %1 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<10x4000xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %5 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    return %3, %4, %5 : tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>
  }
  func.func private @"batched_*_broadcast_scalar2"(%arg0: tensor<10x4000xf32>, %arg1: tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %1 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<10x4000xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %5 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    return %3, %4, %5 : tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>
  }
  func.func private @batched_literal_pow_broadcast_scalar2(%arg0: tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<10x4000xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %3 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    return %2, %3 : tensor<10x4000xf32>, tensor<10x4000xf32>
  }
  func.func private @"batched_-_broadcast_scalar5"(%arg0: tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %1 = stablehlo.negate %0 : tensor<10x4000xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %3 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    return %2, %3 : tensor<10x4000xf32>, tensor<10x4000xf32>
  }
  func.func private @batched_exp_broadcast_scalar2(%arg0: tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %1 = stablehlo.exponential %0 : tensor<10x4000xf32>
    %2 = stablehlo.transpose %1, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %3 = stablehlo.transpose %0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    return %2, %3 : tensor<10x4000xf32>, tensor<10x4000xf32>
  }
  func.func @main(%arg0: tensor<100x1xf32>, %arg1: tensor<16800xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>, %arg4: tensor<10xf32>) -> (tensor<100x1xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<100x1xf32>, tensor<16800xf32>) {
    %c = stablehlo.constant dense<16400> : tensor<i64>
    %c_0 = stablehlo.constant dense<400> : tensor<i64>
    %cst = stablehlo.constant dense<4.500000e+00> : tensor<10x4000xf32>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %cst_2 = stablehlo.constant dense<4.500000e+00> : tensor<10x100xf32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %1 = stablehlo.transpose %arg1, dims = [0] : (tensor<16800xf32>) -> tensor<16800xf32>
    %2 = stablehlo.transpose %arg2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %3 = stablehlo.transpose %arg3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %4 = stablehlo.transpose %arg4, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %5 = stablehlo.transpose %0, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %6 = stablehlo.reshape %5 : (tensor<100x1xf32>) -> tensor<100x1xf32>
    %7 = stablehlo.transpose %6, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %9:2 = call @batched_softsign_broadcast_scalar(%8) : (tensor<1x100xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>)
    %10 = stablehlo.transpose %9#0, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %11 = stablehlo.reshape %10 : (tensor<100x1xf32>) -> tensor<100x1xf32>
    %12 = stablehlo.transpose %11, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<10x100xf32>
    %15 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0] : (tensor<10xf32>) -> tensor<10x100xf32>
    %17:3 = call @"batched_-_broadcast_scalar"(%14, %16) : (tensor<10x100xf32>, tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>, tensor<10x100xf32>)
    %18:3 = call @"batched_*_broadcast_scalar"(%17#0, %cst_2) : (tensor<10x100xf32>, tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>, tensor<10x100xf32>)
    %19 = stablehlo.broadcast_in_dim %18#0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %20:2 = call @batched_literal_pow_broadcast_scalar(%19) : (tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>)
    %21:2 = call @"batched_-_broadcast_scalar1"(%20#0) : (tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>)
    %22:2 = call @batched_exp_broadcast_scalar(%21#0) : (tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>)
    %23 = stablehlo.transpose %22#0, dims = [1, 0] : (tensor<10x100xf32>) -> tensor<100x10xf32>
    %24 = stablehlo.reshape %23 : (tensor<100x10xf32>) -> tensor<100x10xf32>
    %25 = stablehlo.transpose %24, dims = [1, 0] : (tensor<100x10xf32>) -> tensor<10x100xf32>
    %26 = stablehlo.dynamic_slice %1, %c_1, sizes = [400] : (tensor<16800xf32>, tensor<i64>) -> tensor<400xf32>
    %27 = stablehlo.transpose %26, dims = [0] : (tensor<400xf32>) -> tensor<400xf32>
    %28 = stablehlo.reshape %27 : (tensor<400xf32>) -> tensor<10x40xf32>
    %29 = stablehlo.transpose %28, dims = [1, 0] : (tensor<10x40xf32>) -> tensor<40x10xf32>
    %30 = stablehlo.dot_general %29, %25, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<40x10xf32>, tensor<10x100xf32>) -> tensor<40x100xf32>
    %31 = stablehlo.transpose %30, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %32 = stablehlo.reshape %31 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %33 = stablehlo.transpose %32, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %34 = stablehlo.transpose %33, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %35 = stablehlo.reshape %34 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %36 = stablehlo.transpose %35, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %38:2 = call @batched_softsign_broadcast_scalar1(%37) : (tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>)
    %39 = stablehlo.transpose %38#0, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %40 = stablehlo.reshape %39 : (tensor<100x40xf32>) -> tensor<4000x1xf32>
    %41 = stablehlo.transpose %40, dims = [1, 0] : (tensor<4000x1xf32>) -> tensor<1x4000xf32>
    %42 = stablehlo.broadcast_in_dim %41, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<1x4000xf32>
    %43 = stablehlo.broadcast_in_dim %42, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<10x4000xf32>
    %44 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0] : (tensor<10xf32>) -> tensor<10x4000xf32>
    %46:3 = call @"batched_-_broadcast_scalar2"(%43, %45) : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %47:3 = call @"batched_*_broadcast_scalar1"(%46#0, %cst) : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %48 = stablehlo.broadcast_in_dim %47#0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %49:2 = call @batched_literal_pow_broadcast_scalar1(%48) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %50:2 = call @"batched_-_broadcast_scalar3"(%49#0) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %51:2 = call @batched_exp_broadcast_scalar1(%50#0) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %52 = stablehlo.transpose %51#0, dims = [1, 0] : (tensor<10x4000xf32>) -> tensor<4000x10xf32>
    %53 = stablehlo.reshape %52 : (tensor<4000x10xf32>) -> tensor<100x400xf32>
    %54 = stablehlo.transpose %53, dims = [1, 0] : (tensor<100x400xf32>) -> tensor<400x100xf32>
    %55 = stablehlo.dynamic_slice %1, %c_0, sizes = [16000] : (tensor<16800xf32>, tensor<i64>) -> tensor<16000xf32>
    %56 = stablehlo.transpose %55, dims = [0] : (tensor<16000xf32>) -> tensor<16000xf32>
    %57 = stablehlo.reshape %56 : (tensor<16000xf32>) -> tensor<400x40xf32>
    %58 = stablehlo.transpose %57, dims = [1, 0] : (tensor<400x40xf32>) -> tensor<40x400xf32>
    %59 = stablehlo.dot_general %58, %54, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<40x400xf32>, tensor<400x100xf32>) -> tensor<40x100xf32>
    %60 = stablehlo.transpose %59, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %61 = stablehlo.reshape %60 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %62 = stablehlo.transpose %61, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %63 = stablehlo.transpose %62, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %64 = stablehlo.reshape %63 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %65 = stablehlo.transpose %64, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %66 = stablehlo.broadcast_in_dim %65, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %67:2 = call @batched_softsign_broadcast_scalar2(%66) : (tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>)
    %68 = stablehlo.transpose %67#0, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %69 = stablehlo.reshape %68 : (tensor<100x40xf32>) -> tensor<4000x1xf32>
    %70 = stablehlo.transpose %69, dims = [1, 0] : (tensor<4000x1xf32>) -> tensor<1x4000xf32>
    %71 = stablehlo.broadcast_in_dim %70, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<1x4000xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<10x4000xf32>
    %73 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %74 = stablehlo.broadcast_in_dim %73, dims = [0] : (tensor<10xf32>) -> tensor<10x4000xf32>
    %75:3 = call @"batched_-_broadcast_scalar4"(%72, %74) : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %76:3 = call @"batched_*_broadcast_scalar2"(%75#0, %cst) : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %77 = stablehlo.broadcast_in_dim %76#0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %78:2 = call @batched_literal_pow_broadcast_scalar2(%77) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %79:2 = call @"batched_-_broadcast_scalar5"(%78#0) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %80:2 = call @batched_exp_broadcast_scalar2(%79#0) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %81 = stablehlo.transpose %80#0, dims = [1, 0] : (tensor<10x4000xf32>) -> tensor<4000x10xf32>
    %82 = stablehlo.reshape %81 : (tensor<4000x10xf32>) -> tensor<100x400xf32>
    %83 = stablehlo.transpose %82, dims = [1, 0] : (tensor<100x400xf32>) -> tensor<400x100xf32>
    %84 = stablehlo.dynamic_slice %1, %c, sizes = [400] : (tensor<16800xf32>, tensor<i64>) -> tensor<400xf32>
    %85 = stablehlo.transpose %84, dims = [0] : (tensor<400xf32>) -> tensor<400xf32>
    %86 = stablehlo.reshape %85 : (tensor<400xf32>) -> tensor<400x1xf32>
    %87 = stablehlo.transpose %86, dims = [1, 0] : (tensor<400x1xf32>) -> tensor<1x400xf32>
    %88 = stablehlo.dot_general %87, %83, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x400xf32>, tensor<400x100xf32>) -> tensor<1x100xf32>
    %89 = stablehlo.transpose %88, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %90 = stablehlo.reshape %89 : (tensor<100x1xf32>) -> tensor<100x1xf32>
    %91 = stablehlo.transpose %90, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %92 = stablehlo.transpose %91, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %93 = stablehlo.transpose %2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %94 = stablehlo.transpose %3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %95 = stablehlo.transpose %4, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %96 = stablehlo.transpose %0, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %97 = stablehlo.transpose %1, dims = [0] : (tensor<16800xf32>) -> tensor<16800xf32>
    return %92, %93, %94, %95, %96, %97 : tensor<100x1xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<100x1xf32>, tensor<16800xf32>
  }
}