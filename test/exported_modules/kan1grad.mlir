module {
  func.func private @softsign_broadcast_scalar(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.abs %0 : tensor<f32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
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
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.abs %0 : tensor<f32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
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
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.abs %0 : tensor<f32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
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
  func.func private @l2_distance_loss_broadcast_scalar(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>) {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %arg1, dims = [] : (tensor<f32>) -> tensor<f32>
    %2 = stablehlo.subtract %0, %1 : tensor<f32>
    %3 = stablehlo.abs %2 : tensor<f32>
    %4 = stablehlo.multiply %3, %3 : tensor<f32>
    %5 = stablehlo.transpose %4, dims = [] : (tensor<f32>) -> tensor<f32>
    %6 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    %7 = stablehlo.transpose %1, dims = [] : (tensor<f32>) -> tensor<f32>
    return %5, %6, %7 : tensor<f32>, tensor<f32>, tensor<f32>
  }
  func.func private @identity_broadcast_scalar(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.transpose %arg0, dims = [] : (tensor<f32>) -> tensor<f32>
    %1 = stablehlo.transpose %0, dims = [] : (tensor<f32>) -> tensor<f32>
    return %1 : tensor<f32>
  }
  func.func private @"Const{typeof(loss)}(loss)_autodiff"(%arg0: tensor<18480xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>, %arg4: tensor<100x1xf32>, %arg5: tensor<100x1xf32>) -> (tensor<f32>, tensor<18480xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<100x1xf32>, tensor<100x1xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
    %1 = stablehlo.transpose %arg1, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %2 = stablehlo.transpose %arg2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %3 = stablehlo.transpose %arg3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %4 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %5 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %6 = stablehlo.transpose %4, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<100x1xf32>) -> tensor<100x1xf32>
    %8 = stablehlo.transpose %7, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %10:2 = enzyme.batch @softsign_broadcast_scalar(%9) {batch_shape = array<i64: 1, 100>} : (tensor<1x100xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>)
    %11 = stablehlo.transpose %10#0, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %12 = stablehlo.reshape %11 : (tensor<100x1xf32>) -> tensor<100x1xf32>
    %13 = stablehlo.transpose %12, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<10x100xf32>
    %16 = stablehlo.broadcast_in_dim %1, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<10xf32>) -> tensor<10x100xf32>
    %18:3 = enzyme.batch @"-_broadcast_scalar"(%15, %17) {batch_shape = array<i64: 10, 100>} : (tensor<10x100xf32>, tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>, tensor<10x100xf32>)
    %cst = stablehlo.constant dense<4.500000e+00> : tensor<10x100xf32>
    %19:3 = enzyme.batch @"*_broadcast_scalar"(%18#0, %cst) {batch_shape = array<i64: 10, 100>} : (tensor<10x100xf32>, tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>, tensor<10x100xf32>)
    %20 = stablehlo.broadcast_in_dim %19#0, dims = [0, 1] : (tensor<10x100xf32>) -> tensor<10x100xf32>
    %21:2 = enzyme.batch @literal_pow_broadcast_scalar(%20) {batch_shape = array<i64: 10, 100>} : (tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>)
    %22:2 = enzyme.batch @"-_broadcast_scalar1"(%21#0) {batch_shape = array<i64: 10, 100>} : (tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>)
    %23:2 = enzyme.batch @exp_broadcast_scalar(%22#0) {batch_shape = array<i64: 10, 100>} : (tensor<10x100xf32>) -> (tensor<10x100xf32>, tensor<10x100xf32>)
    %24 = stablehlo.transpose %23#0, dims = [1, 0] : (tensor<10x100xf32>) -> tensor<100x10xf32>
    %25 = stablehlo.reshape %24 : (tensor<100x10xf32>) -> tensor<100x10xf32>
    %26 = stablehlo.transpose %25, dims = [1, 0] : (tensor<100x10xf32>) -> tensor<10x100xf32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %27 = stablehlo.dynamic_slice %0, %c, sizes = [400] : (tensor<18480xf32>, tensor<i64>) -> tensor<400xf32>
    %28 = stablehlo.transpose %27, dims = [0] : (tensor<400xf32>) -> tensor<400xf32>
    %29 = stablehlo.reshape %28 : (tensor<400xf32>) -> tensor<10x40xf32>
    %30 = stablehlo.transpose %29, dims = [1, 0] : (tensor<10x40xf32>) -> tensor<40x10xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<40x100xf32>
    %31 = stablehlo.dot_general %30, %26, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<40x10xf32>, tensor<10x100xf32>) -> tensor<40x100xf32>
    %c_1 = stablehlo.constant dense<400> : tensor<i64>
    %32 = stablehlo.dynamic_slice %0, %c_1, sizes = [40] : (tensor<18480xf32>, tensor<i64>) -> tensor<40xf32>
    %33 = stablehlo.transpose %32, dims = [0] : (tensor<40xf32>) -> tensor<40xf32>
    %34 = stablehlo.reshape %33 : (tensor<40xf32>) -> tensor<1x40xf32>
    %35 = stablehlo.transpose %34, dims = [1, 0] : (tensor<1x40xf32>) -> tensor<40x1xf32>
    %36 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %37:2 = enzyme.batch @swish_broadcast_scalar(%36) {batch_shape = array<i64: 1, 100>} : (tensor<1x100xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>)
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<40x100xf32>
    %38 = stablehlo.dot_general %35, %37#0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<40x1xf32>, tensor<1x100xf32>) -> tensor<40x100xf32>
    %39 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %40 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %41:3 = enzyme.batch @"+_broadcast_scalar"(%39, %40) {batch_shape = array<i64: 40, 100>} : (tensor<40x100xf32>, tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>, tensor<40x100xf32>)
    %42 = stablehlo.transpose %41#0, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %43 = stablehlo.reshape %42 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %44 = stablehlo.transpose %43, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %45 = stablehlo.transpose %44, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %46 = stablehlo.reshape %45 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %47 = stablehlo.transpose %46, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %48 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %49:2 = enzyme.batch @softsign_broadcast_scalar1(%48) {batch_shape = array<i64: 40, 100>} : (tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>)
    %50 = stablehlo.transpose %49#0, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %51 = stablehlo.reshape %50 : (tensor<100x40xf32>) -> tensor<4000x1xf32>
    %52 = stablehlo.transpose %51, dims = [1, 0] : (tensor<4000x1xf32>) -> tensor<1x4000xf32>
    %53 = stablehlo.broadcast_in_dim %52, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<1x4000xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<10x4000xf32>
    %55 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %56 = stablehlo.broadcast_in_dim %55, dims = [0] : (tensor<10xf32>) -> tensor<10x4000xf32>
    %57:3 = enzyme.batch @"-_broadcast_scalar2"(%54, %56) {batch_shape = array<i64: 10, 4000>} : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %cst_3 = stablehlo.constant dense<4.500000e+00> : tensor<10x4000xf32>
    %58:3 = enzyme.batch @"*_broadcast_scalar1"(%57#0, %cst_3) {batch_shape = array<i64: 10, 4000>} : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %59 = stablehlo.broadcast_in_dim %58#0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %60:2 = enzyme.batch @literal_pow_broadcast_scalar1(%59) {batch_shape = array<i64: 10, 4000>} : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %61:2 = enzyme.batch @"-_broadcast_scalar3"(%60#0) {batch_shape = array<i64: 10, 4000>} : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %62:2 = enzyme.batch @exp_broadcast_scalar1(%61#0) {batch_shape = array<i64: 10, 4000>} : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %63 = stablehlo.transpose %62#0, dims = [1, 0] : (tensor<10x4000xf32>) -> tensor<4000x10xf32>
    %64 = stablehlo.reshape %63 : (tensor<4000x10xf32>) -> tensor<100x400xf32>
    %65 = stablehlo.transpose %64, dims = [1, 0] : (tensor<100x400xf32>) -> tensor<400x100xf32>
    %c_4 = stablehlo.constant dense<440> : tensor<i64>
    %66 = stablehlo.dynamic_slice %0, %c_4, sizes = [16000] : (tensor<18480xf32>, tensor<i64>) -> tensor<16000xf32>
    %67 = stablehlo.transpose %66, dims = [0] : (tensor<16000xf32>) -> tensor<16000xf32>
    %68 = stablehlo.reshape %67 : (tensor<16000xf32>) -> tensor<400x40xf32>
    %69 = stablehlo.transpose %68, dims = [1, 0] : (tensor<400x40xf32>) -> tensor<40x400xf32>
    %cst_5 = stablehlo.constant dense<0.000000e+00> : tensor<40x100xf32>
    %70 = stablehlo.dot_general %69, %65, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<40x400xf32>, tensor<400x100xf32>) -> tensor<40x100xf32>
    %c_6 = stablehlo.constant dense<16440> : tensor<i64>
    %71 = stablehlo.dynamic_slice %0, %c_6, sizes = [1600] : (tensor<18480xf32>, tensor<i64>) -> tensor<1600xf32>
    %72 = stablehlo.transpose %71, dims = [0] : (tensor<1600xf32>) -> tensor<1600xf32>
    %73 = stablehlo.reshape %72 : (tensor<1600xf32>) -> tensor<40x40xf32>
    %74 = stablehlo.transpose %73, dims = [1, 0] : (tensor<40x40xf32>) -> tensor<40x40xf32>
    %75 = stablehlo.broadcast_in_dim %47, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %76:2 = enzyme.batch @swish_broadcast_scalar1(%75) {batch_shape = array<i64: 40, 100>} : (tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>)
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<40x100xf32>
    %77 = stablehlo.dot_general %74, %76#0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<40x40xf32>, tensor<40x100xf32>) -> tensor<40x100xf32>
    %78 = stablehlo.broadcast_in_dim %70, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %79 = stablehlo.broadcast_in_dim %77, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %80:3 = enzyme.batch @"+_broadcast_scalar1"(%78, %79) {batch_shape = array<i64: 40, 100>} : (tensor<40x100xf32>, tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>, tensor<40x100xf32>)
    %81 = stablehlo.transpose %80#0, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %82 = stablehlo.reshape %81 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %83 = stablehlo.transpose %82, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %84 = stablehlo.transpose %83, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %85 = stablehlo.reshape %84 : (tensor<100x40xf32>) -> tensor<100x40xf32>
    %86 = stablehlo.transpose %85, dims = [1, 0] : (tensor<100x40xf32>) -> tensor<40x100xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %88:2 = enzyme.batch @softsign_broadcast_scalar2(%87) {batch_shape = array<i64: 40, 100>} : (tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>)
    %89 = stablehlo.transpose %88#0, dims = [1, 0] : (tensor<40x100xf32>) -> tensor<100x40xf32>
    %90 = stablehlo.reshape %89 : (tensor<100x40xf32>) -> tensor<4000x1xf32>
    %91 = stablehlo.transpose %90, dims = [1, 0] : (tensor<4000x1xf32>) -> tensor<1x4000xf32>
    %92 = stablehlo.broadcast_in_dim %91, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<1x4000xf32>
    %93 = stablehlo.broadcast_in_dim %92, dims = [0, 1] : (tensor<1x4000xf32>) -> tensor<10x4000xf32>
    %94 = stablehlo.broadcast_in_dim %3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0] : (tensor<10xf32>) -> tensor<10x4000xf32>
    %96:3 = enzyme.batch @"-_broadcast_scalar4"(%93, %95) {batch_shape = array<i64: 10, 4000>} : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %cst_8 = stablehlo.constant dense<4.500000e+00> : tensor<10x4000xf32>
    %97:3 = enzyme.batch @"*_broadcast_scalar2"(%96#0, %cst_8) {batch_shape = array<i64: 10, 4000>} : (tensor<10x4000xf32>, tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>, tensor<10x4000xf32>)
    %98 = stablehlo.broadcast_in_dim %97#0, dims = [0, 1] : (tensor<10x4000xf32>) -> tensor<10x4000xf32>
    %99:2 = enzyme.batch @literal_pow_broadcast_scalar2(%98) {batch_shape = array<i64: 10, 4000>} : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %100:2 = enzyme.batch @"-_broadcast_scalar5"(%99#0) {batch_shape = array<i64: 10, 4000>} : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %101:2 = enzyme.batch @exp_broadcast_scalar2(%100#0) {batch_shape = array<i64: 10, 4000>} : (tensor<10x4000xf32>) -> (tensor<10x4000xf32>, tensor<10x4000xf32>)
    %102 = stablehlo.transpose %101#0, dims = [1, 0] : (tensor<10x4000xf32>) -> tensor<4000x10xf32>
    %103 = stablehlo.reshape %102 : (tensor<4000x10xf32>) -> tensor<100x400xf32>
    %104 = stablehlo.transpose %103, dims = [1, 0] : (tensor<100x400xf32>) -> tensor<400x100xf32>
    %c_9 = stablehlo.constant dense<18040> : tensor<i64>
    %105 = stablehlo.dynamic_slice %0, %c_9, sizes = [400] : (tensor<18480xf32>, tensor<i64>) -> tensor<400xf32>
    %106 = stablehlo.transpose %105, dims = [0] : (tensor<400xf32>) -> tensor<400xf32>
    %107 = stablehlo.reshape %106 : (tensor<400xf32>) -> tensor<400x1xf32>
    %108 = stablehlo.transpose %107, dims = [1, 0] : (tensor<400x1xf32>) -> tensor<1x400xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<1x100xf32>
    %109 = stablehlo.dot_general %108, %104, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x400xf32>, tensor<400x100xf32>) -> tensor<1x100xf32>
    %c_11 = stablehlo.constant dense<18440> : tensor<i64>
    %110 = stablehlo.dynamic_slice %0, %c_11, sizes = [40] : (tensor<18480xf32>, tensor<i64>) -> tensor<40xf32>
    %111 = stablehlo.transpose %110, dims = [0] : (tensor<40xf32>) -> tensor<40xf32>
    %112 = stablehlo.reshape %111 : (tensor<40xf32>) -> tensor<40x1xf32>
    %113 = stablehlo.transpose %112, dims = [1, 0] : (tensor<40x1xf32>) -> tensor<1x40xf32>
    %114 = stablehlo.broadcast_in_dim %86, dims = [0, 1] : (tensor<40x100xf32>) -> tensor<40x100xf32>
    %115:2 = enzyme.batch @swish_broadcast_scalar2(%114) {batch_shape = array<i64: 40, 100>} : (tensor<40x100xf32>) -> (tensor<40x100xf32>, tensor<40x100xf32>)
    %cst_12 = stablehlo.constant dense<0.000000e+00> : tensor<1x100xf32>
    %116 = stablehlo.dot_general %113, %115#0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x40xf32>, tensor<40x100xf32>) -> tensor<1x100xf32>
    %117 = stablehlo.broadcast_in_dim %109, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %118 = stablehlo.broadcast_in_dim %116, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %119:3 = enzyme.batch @"+_broadcast_scalar2"(%117, %118) {batch_shape = array<i64: 1, 100>} : (tensor<1x100xf32>, tensor<1x100xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>)
    %120 = stablehlo.transpose %119#0, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %121 = stablehlo.reshape %120 : (tensor<100x1xf32>) -> tensor<100x1xf32>
    %122 = stablehlo.transpose %121, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %123 = stablehlo.broadcast_in_dim %122, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %124 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %125:3 = enzyme.batch @l2_distance_loss_broadcast_scalar(%123, %124) {batch_shape = array<i64: 1, 100>} : (tensor<1x100xf32>, tensor<1x100xf32>) -> (tensor<1x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>)
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %126 = stablehlo.broadcast_in_dim %125#0, dims = [0, 1] : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %127 = enzyme.batch @identity_broadcast_scalar(%126) {batch_shape = array<i64: 1, 100>} : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %128 = stablehlo.reduce(%127 init: %cst_13) applies stablehlo.add across dimensions = [0, 1] : (tensor<1x100xf32>, tensor<f32>) -> tensor<f32>
    %cst_14 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
    %129 = stablehlo.divide %128, %cst_14 : tensor<f32>
    %130 = stablehlo.transpose %129, dims = [] : (tensor<f32>) -> tensor<f32>
    %131 = stablehlo.transpose %0, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
    %132 = stablehlo.transpose %1, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %133 = stablehlo.transpose %2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %134 = stablehlo.transpose %3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %135 = stablehlo.transpose %4, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %136 = stablehlo.transpose %5, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    return %130, %131, %132, %133, %134, %135, %136 : tensor<f32>, tensor<18480xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<100x1xf32>, tensor<100x1xf32>
  }
  func.func @main(%arg0: tensor<18480xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>, %arg4: tensor<100x1xf32>, %arg5: tensor<100x1xf32>) -> (tensor<18480xf32>, tensor<18480xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<100x1xf32>, tensor<100x1xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
    %1 = stablehlo.transpose %arg1, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %2 = stablehlo.transpose %arg2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %3 = stablehlo.transpose %arg3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %4 = stablehlo.transpose %arg4, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %5 = stablehlo.transpose %arg5, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<18480xf32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %6 = stablehlo.transpose %0, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
    %7 = stablehlo.transpose %1, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %8 = stablehlo.transpose %2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %9 = stablehlo.transpose %3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %10 = stablehlo.transpose %4, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %11 = stablehlo.transpose %5, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %12 = stablehlo.transpose %cst_0, dims = [] : (tensor<f32>) -> tensor<f32>
    %13 = stablehlo.transpose %cst, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
    %14:7 = enzyme.autodiff @"Const{typeof(loss)}(loss)_autodiff"(%6, %7, %8, %9, %10, %11, %12, %13) {activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>], ret_activity = [#enzyme<activity enzyme_activenoneed>, #enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>, #enzyme<activity enzyme_const>]} : (tensor<18480xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<100x1xf32>, tensor<100x1xf32>, tensor<f32>, tensor<18480xf32>) -> (tensor<18480xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<100x1xf32>, tensor<100x1xf32>, tensor<18480xf32>)
    %15 = stablehlo.transpose %14#0, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
    %16 = stablehlo.transpose %14#1, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %17 = stablehlo.transpose %14#2, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %18 = stablehlo.transpose %14#3, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %19 = stablehlo.transpose %14#4, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %20 = stablehlo.transpose %14#5, dims = [1, 0] : (tensor<100x1xf32>) -> tensor<1x100xf32>
    %21 = stablehlo.transpose %14#6, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
    %22 = stablehlo.transpose %21, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
    %23 = stablehlo.transpose %15, dims = [0] : (tensor<18480xf32>) -> tensor<18480xf32>
    %24 = stablehlo.transpose %16, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %25 = stablehlo.transpose %17, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %26 = stablehlo.transpose %18, dims = [0] : (tensor<10xf32>) -> tensor<10xf32>
    %27 = stablehlo.transpose %19, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    %28 = stablehlo.transpose %20, dims = [1, 0] : (tensor<1x100xf32>) -> tensor<100x1xf32>
    return %22, %23, %24, %25, %26, %27, %28 : tensor<18480xf32>, tensor<18480xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<100x1xf32>, tensor<100x1xf32>
  }
}