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
  func.func private @swish_broadcast_scalar(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.logistic %0 : tensor<f32>
    %2 = stablehlo.multiply %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4 : tensor<f32>, tensor<f32>
  }
  func.func private @"+_broadcast_scalar"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %arg1, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.add %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
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
  func.func private @swish_broadcast_scalar1(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.logistic %0 : tensor<f32>
    %2 = stablehlo.multiply %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4 : tensor<f32>, tensor<f32>
  }
  func.func private @"+_broadcast_scalar1"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %arg1, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.add %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
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
  func.func private @swish_broadcast_scalar2(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.logistic %0 : tensor<f32>
    %2 = stablehlo.multiply %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4 : tensor<f32>, tensor<f32>
  }
  func.func private @"+_broadcast_scalar2"(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %arg1, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.add %0, %1 : tensor<f32>
    %3 = stablehlo.transpose %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %5 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    return %3, %4, %5 : tensor<f32>, tensor<f32>, tensor<f32>
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
  func.func private @batched_swish_broadcast_scalar(%arg0: tensor<1x100xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %1 = stablehlo.logistic %0 : tensor<1x100xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x100xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    return %3, %4 : tensor<1x100xf32>, tensor<1x100xf32>
  }
  func.func private @"batched_+_broadcast_scalar"(%arg0: tensor<40x100xf32>, %arg1: tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>, tensor<40x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %1 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %2 = stablehlo.add %0, %1 : tensor<40x100xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %5 = stablehlo.transpose %1, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    return %3, %4, %5 : tensor<40x100xf32>, tensor<40x100xf32>, tensor<40x100xf32>
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
  func.func private @batched_swish_broadcast_scalar1(%arg0: tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %1 = stablehlo.logistic %0 : tensor<40x100xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<40x100xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    return %3, %4 : tensor<40x100xf32>, tensor<40x100xf32>
  }
  func.func private @"batched_+_broadcast_scalar1"(%arg0: tensor<40x100xf32>, %arg1: tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>, tensor<40x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %1 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %2 = stablehlo.add %0, %1 : tensor<40x100xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %5 = stablehlo.transpose %1, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    return %3, %4, %5 : tensor<40x100xf32>, tensor<40x100xf32>, tensor<40x100xf32>
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
  func.func private @batched_swish_broadcast_scalar2(%arg0: tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %1 = stablehlo.logistic %0 : tensor<40x100xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<40x100xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    return %3, %4 : tensor<40x100xf32>, tensor<40x100xf32>
  }
  func.func private @"batched_+_broadcast_scalar2"(%arg0: tensor<1x100xf32>, %arg1: tensor<1x100xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %1 = stablehlo.transpose %arg1, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %2 = stablehlo.add %0, %1 : tensor<1x100xf32>
    %3 = stablehlo.transpose %2, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %4 = stablehlo.transpose %0, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %5 = stablehlo.transpose %1, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    return %3, %4, %5 : tensor<1x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>
  }
  func.func @main(%arg0: tensor<100x1xf32>, %arg1: tensor<18480xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>, %arg4: tensor<10xf32>) -> (tensor<100x1xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<100x1xf32>, tensor<18480xf32>) {
    %c = stablehlo.constant dense<18440> : tensor<i64>
    %c_0 = stablehlo.constant dense<18040> : tensor<i64>
    %c_1 = stablehlo.constant dense<16440> : tensor<i64>
    %c_2 = stablehlo.constant dense<440> : tensor<i64>
    %cst = stablehlo.constant dense<4.500000e+00> : tensor<10x4000xf32>
    %c_3 = stablehlo.constant dense<400> : tensor<i64>
    %c_4 = stablehlo.constant dense<0> : tensor<i64>
    %cst_5 = stablehlo.constant dense<4.500000e+00> : tensor<10x100xf32>
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %1 = stablehlo.transpose %arg1, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
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
    %18:3 = call @"batched_*_broadcast_scalar"(%17#0, %cst_5) : (tensor<10x100xf32>, tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>, tensor<10x100xf32>)
    %19 = stablehlo.broadcast_in_dim %18#0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %20:2 = call @batched_literal_pow_broadcast_scalar(%19) : (tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>)
    %21:2 = call @"batched_-_broadcast_scalar1"(%20#0) : (tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>)
    %22:2 = call @batched_exp_broadcast_scalar(%21#0) : (tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>)
    %23 = stablehlo.transpose %22#0, dims = [1, 0] : (tensor<10x100xf32>) -> tensor<100x10xf32>
    %24 = stablehlo.reshape %23 : (tensor<100x10xf32>) -> tensor<100x10xf32>
    %25 = stablehlo.transpose %24, dims = [1, 0] : (tensor<100x10xf32>) -> tensor<10x100xf32>
    %26 = stablehlo.dynamic_slice %1, %c_4, sizes = [400] : (tensor<18480xf32>, tensor<i64>) -> tensor<400xf32>
    %27 = stablehlo.transpose %26, dims = [0] : (tensor<400xf32>) -> tensor<400xf32>
    %28 = stablehlo.reshape %27 : (tensor<400xf32>) -> tensor<10x40xf32>
    %29 = stablehlo.transpose %28, dims = [1, 0] : (tensor<10x40xf32>) -> tensor<40x10xf32>
    %30 = stablehlo.dot_general %29, %25, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<40x10xf32>, tensor<10x100xf32>) -> tensor<40x100xf32>
    %31 = stablehlo.dynamic_slice %1, %c_3, sizes = [40] : (tensor<18480xf32>, tensor<i64>) -> tensor<40xf32>
    %32 = stablehlo.transpose %31, dims = [0] : (tensor<40xf32>) -> tensor<40xf32>
    %33 = stablehlo.reshape %32 : (tensor<40xf32>) -> tensor<1x40xf32>
    %34 = stablehlo.transpose %33, dims = [1, 0] : (tensor<1x40xf32>) -> tensor<40x1xf32>
    %35 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %36:2 = call @batched_swish_broadcast_scalar(%35) : (tensor<1x100xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>)
    %37 = stablehlo.dot_general %34, %36#0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<40x1xf32>, tensor<1x100xf32>) -> tensor<40x100xf32>
    %38 = stablehlo.broadcast_in_dim %30, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %39 = stablehlo.broadcast_in_dim %37, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %40:3 = call @"batched_+_broadcast_scalar"(%38, %39) : (tensor<40x100xf32>, tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>, tensor<40x100xf32>)
    %41 = stablehlo.transpose %40#0, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %42 = stablehlo.reshape %41 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %43 = stablehlo.transpose %42, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %44 = stablehlo.transpose %43, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %45 = stablehlo.reshape %44 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %46 = stablehlo.transpose %45, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %47 = stablehlo.broadcast_in_dim %46, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %48:2 = call @batched_softsign_broadcast_scalar1(%47) : (tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>)
    %49 = stablehlo.transpose %48#0, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %50 = stablehlo.reshape %49 : (tensor<100x40xf32>) -> tensor<4000x1xf32>
    %51 = stablehlo.transpose %50, dims = [1, 0] : (tensor<4000x1xf32>) -> tensor<1x4000xf32>
    %52 = stablehlo.broadcast_in_dim %51, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<1x4000xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<10x4000xf32>
    %54 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0] : (tensor<10xf32>) -> tensor<10x4000xf32>
    %56:3 = call @"batched_-_broadcast_scalar2"(%53, %55) : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %57:3 = call @"batched_*_broadcast_scalar1"(%56#0, %cst) : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %58 = stablehlo.broadcast_in_dim %57#0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %59:2 = call @batched_literal_pow_broadcast_scalar1(%58) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %60:2 = call @"batched_-_broadcast_scalar3"(%59#0) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %61:2 = call @batched_exp_broadcast_scalar1(%60#0) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %62 = stablehlo.transpose %61#0, dims = [1, 0] : (tensor<10x4000xf32>) -> tensor<4000x10xf32>
    %63 = stablehlo.reshape %62 : (tensor<4000x10xf32>) -> tensor<100x400xf32>
    %64 = stablehlo.transpose %63, dims = [1, 0] : (tensor<100x400xf32>) -> tensor<400x100xf32>
    %65 = stablehlo.dynamic_slice %1, %c_2, sizes = [16000] : (tensor<18480xf32>, tensor<i64>) -> tensor<16000xf32>
    %66 = stablehlo.transpose %65, dims = [0] : (tensor<16000xf32>) -> tensor<16000xf32>
    %67 = stablehlo.reshape %66 : (tensor<16000xf32>) -> tensor<400x40xf32>
    %68 = stablehlo.transpose %67, dims = [1, 0] : (tensor<400x40xf32>) -> tensor<40x400xf32>
    %69 = stablehlo.dot_general %68, %64, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<40x400xf32>, tensor<400x100xf32>) -> tensor<40x100xf32>
    %70 = stablehlo.dynamic_slice %1, %c_1, sizes = [1600] : (tensor<18480xf32>, tensor<i64>) -> tensor<1600xf32>
    %71 = stablehlo.transpose %70, dims = [0] : (tensor<1600xf32>) -> tensor<1600xf32>
    %72 = stablehlo.reshape %71 : (tensor<1600xf32>) -> tensor<40x40xf32>
    %73 = stablehlo.transpose %72, dims = [1, 0] : (tensor<40x40xf32>) -> tensor<40x40xf32>
    %74 = stablehlo.broadcast_in_dim %46, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %75:2 = call @batched_swish_broadcast_scalar1(%74) : (tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>)
    %76 = stablehlo.dot_general %73, %75#0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<40x40xf32>, tensor<40x100xf32>) -> tensor<40x100xf32>
    %77 = stablehlo.broadcast_in_dim %69, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %78 = stablehlo.broadcast_in_dim %76, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %79:3 = call @"batched_+_broadcast_scalar1"(%77, %78) : (tensor<40x100xf32>, tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>, tensor<40x100xf32>)
    %80 = stablehlo.transpose %79#0, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %81 = stablehlo.reshape %80 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %82 = stablehlo.transpose %81, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %83 = stablehlo.transpose %82, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %84 = stablehlo.reshape %83 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %85 = stablehlo.transpose %84, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %86 = stablehlo.broadcast_in_dim %85, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %87:2 = call @batched_softsign_broadcast_scalar2(%86) : (tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>)
    %88 = stablehlo.transpose %87#0, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %89 = stablehlo.reshape %88 : (tensor<100x40xf32>) -> tensor<4000x1xf32>
    %90 = stablehlo.transpose %89, dims = [1, 0] : (tensor<4000x1xf32>) -> tensor<1x4000xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<1x4000xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<10x4000xf32>
    %93 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %94 = stablehlo.broadcast_in_dim %93, dims = [0] : (tensor<10xf32>) -> tensor<10x4000xf32>
    %95:3 = call @"batched_-_broadcast_scalar4"(%92, %94) : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %96:3 = call @"batched_*_broadcast_scalar2"(%95#0, %cst) : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %97 = stablehlo.broadcast_in_dim %96#0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %98:2 = call @batched_literal_pow_broadcast_scalar2(%97) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %99:2 = call @"batched_-_broadcast_scalar5"(%98#0) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %100:2 = call @batched_exp_broadcast_scalar2(%99#0) : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %101 = stablehlo.transpose %100#0, dims = [1, 0] : (tensor<10x4000xf32>) -> tensor<4000x10xf32>
    %102 = stablehlo.reshape %101 : (tensor<4000x10xf32>) -> tensor<100x400xf32>
    %103 = stablehlo.transpose %102, dims = [1, 0] : (tensor<100x400xf32>) -> tensor<400x100xf32>
    %104 = stablehlo.dynamic_slice %1, %c_0, sizes = [400] : (tensor<18480xf32>, tensor<i64>) -> tensor<400xf32>
    %105 = stablehlo.transpose %104, dims = [0] : (tensor<400xf32>) -> tensor<400xf32>
    %106 = stablehlo.reshape %105 : (tensor<400xf32>) -> tensor<400x1xf32>
    %107 = stablehlo.transpose %106, dims = [1, 0] : (tensor<400x1xf32>) -> tensor<1x400xf32>
    %108 = stablehlo.dot_general %107, %103, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x400xf32>, tensor<400x100xf32>) -> tensor<1x100xf32>
    %109 = stablehlo.dynamic_slice %1, %c, sizes = [40] : (tensor<18480xf32>, tensor<i64>) -> tensor<40xf32>
    %110 = stablehlo.transpose %109, dims = [0] : (tensor<40xf32>) -> tensor<40xf32>
    %111 = stablehlo.reshape %110 : (tensor<40xf32>) -> tensor<40x1xf32>
    %112 = stablehlo.transpose %111, dims = [1, 0] : (tensor<40x1xf32>) -> tensor<1x40xf32>
    %113 = stablehlo.broadcast_in_dim %85, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %114:2 = call @batched_swish_broadcast_scalar2(%113) : (tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>)
    %115 = stablehlo.dot_general %112, %114#0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x40xf32>, tensor<40x100xf32>) -> tensor<1x100xf32>
    %116 = stablehlo.broadcast_in_dim %108, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %117 = stablehlo.broadcast_in_dim %115, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %118:3 = call @"batched_+_broadcast_scalar2"(%116, %117) : (tensor<1x100xf32>, tensor<1x100xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>)
    %119 = stablehlo.transpose %118#0, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %120 = stablehlo.reshape %119 : (tensor<100x1xf32>) -> tensor<100x1xf32>
    %121 = stablehlo.transpose %120, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %122 = stablehlo.transpose %121, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %123 = stablehlo.transpose %2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %124 = stablehlo.transpose %3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %125 = stablehlo.transpose %4, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %126 = stablehlo.transpose %0, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %127 = stablehlo.transpose %1, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
    return %122, %123, %124, %125, %126, %127 : tensor<100x1xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<100x1xf32>, tensor<18480xf32>
  }
}