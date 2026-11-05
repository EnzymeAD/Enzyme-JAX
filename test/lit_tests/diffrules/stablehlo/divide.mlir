// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=divide outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=divide outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --cse | FileCheck %s --check-prefix=STRONGZERO
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

func.func @divide(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.divide %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

func.func @divide_sz(%a : tensor<2xf32>, %b : tensor<2xf32>) -> tensor<2xf32> {
  %c = stablehlo.divide %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %c : tensor<2xf32>
}

func.func @atan2_deriv(%y : tensor<2xf32>) -> tensor<2xf32> {
  %one = stablehlo.constant dense<1.0> : tensor<2xf32>
  %yy = stablehlo.multiply %y, %y : tensor<2xf32>
  %den = stablehlo.add %one, %yy : tensor<2xf32>
  %res = stablehlo.divide %one, %den : tensor<2xf32>
  func.return %res : tensor<2xf32>
}

func.func @divide_complex(%a : tensor<2xcomplex<f32>>, %b : tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  %c = stablehlo.divide %a, %b : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %c : tensor<2xcomplex<f32>>
}

// FORWARD:  func.func @divide(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:    %0 = stablehlo.divide %arg0, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    %1 = stablehlo.multiply %arg3, %0 : tensor<2xf32>
// FORWARD-NEXT:    %2 = stablehlo.subtract %arg1, %1 : tensor<2xf32>
// FORWARD-NEXT:    %3 = stablehlo.divide %2, %arg2 : tensor<2xf32>
// FORWARD-NEXT:    return %0, %3 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:  }

// REVERSE:  func.func @divide(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// REVERSE-NEXT:    %0 = stablehlo.divide %arg2, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %1 = stablehlo.divide %arg0, %arg1 : tensor<2xf32>
// REVERSE-NEXT:    %2 = stablehlo.multiply %0, %1 : tensor<2xf32>
// REVERSE-NEXT:    %3 = stablehlo.negate %2 : tensor<2xf32>
// REVERSE-NEXT:    return %0, %3 : tensor<2xf32>, tensor<2xf32>
// REVERSE-NEXT:  }

// STRONGZERO-LABEL: func.func private @fwddiffedivide_sz(
// STRONGZERO-SAME: %[[A:.+]]: tensor<2xf32>, %[[DA:.+]]: tensor<2xf32>, %[[B:.+]]: tensor<2xf32>, %[[DB:.+]]: tensor<2xf32>) -> tensor<2xf32> {
// STRONGZERO-NEXT:    %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<2xf32>
// STRONGZERO-NEXT:    %[[Q:.+]] = stablehlo.divide %[[A]], %[[B]] : tensor<2xf32>
// STRONGZERO-NEXT:    %[[CMP0:.+]] = stablehlo.compare EQ, %[[DB]], %[[ZERO]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// STRONGZERO-NEXT:    %[[MUL:.+]] = stablehlo.multiply %[[DB]], %[[Q]] : tensor<2xf32>
// STRONGZERO-NEXT:    %[[DBQ:.+]] = stablehlo.select %[[CMP0]], %[[ZERO]], %[[MUL]] : tensor<2xi1>, tensor<2xf32>
// STRONGZERO-NEXT:    %[[NUM:.+]] = stablehlo.subtract %[[DA]], %[[DBQ]] : tensor<2xf32>
// STRONGZERO-NEXT:    %[[CMP1:.+]] = stablehlo.compare EQ, %[[NUM]], %[[ZERO]] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// STRONGZERO-NEXT:    %[[DIV:.+]] = stablehlo.divide %[[NUM]], %[[B]] : tensor<2xf32>
// STRONGZERO-NEXT:    %[[RES:.+]] = stablehlo.select %[[CMP1]], %[[ZERO]], %[[DIV]] : tensor<2xi1>, tensor<2xf32>
// STRONGZERO-NEXT:    return %[[RES]] : tensor<2xf32>
// STRONGZERO-NEXT:  }

func.func @main() {
  %a = stablehlo.constant dense<[4.0, 2.0]> : tensor<2xf32>
  %b = stablehlo.constant dense<[2.0, 5.0]> : tensor<2xf32>
  %output = stablehlo.constant dense<[2.0, 0.4]> : tensor<2xf32>

  %expected_da = stablehlo.constant dense<[0.5, 0.2]> : tensor<2xf32>
  %expected_db = stablehlo.constant dense<[-1.0, -0.08]> : tensor<2xf32>

  %dzero = stablehlo.constant dense<0.0> : tensor<2xf32>
  %done = stablehlo.constant dense<1.0> : tensor<2xf32>

  // fwd diff wrt a
  %fwd_a:2 = enzyme.fwddiff @divide(%a, %done, %b, %dzero) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_a#0, %output : tensor<2xf32>
  check.expect_almost_eq %fwd_a#1, %expected_da : tensor<2xf32>

  // fwd diff wrt b
  %fwd_b:2 = enzyme.fwddiff @divide(%a, %dzero, %b, %done) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd_b#0, %output : tensor<2xf32>
  check.expect_almost_eq %fwd_b#1, %expected_db : tensor<2xf32>

  // partial activity: b const, a const
  %fwd_const_b = enzyme.fwddiff @divide(%a, %done, %b) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_const>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  check.expect_almost_eq %fwd_const_b, %expected_da : tensor<2xf32>

  %fwd_const_a = enzyme.fwddiff @divide(%a, %b, %done) <{
    activity=[#enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  check.expect_almost_eq %fwd_const_a, %expected_db : tensor<2xf32>

  // divisor whose square overflows ([2e20]^2 = +inf) or underflows ([2e-23]^2 = 0.0) f32
  %ext_a = stablehlo.constant dense<[2.0e+20, 2.0e-23]> : tensor<2xf32>
  %ext_da = stablehlo.constant dense<[2.0e+20, 2.0e-23]> : tensor<2xf32>
  %ext_b = stablehlo.constant dense<[2.0e+20, 2.0e-23]> : tensor<2xf32>
  %ext_db = stablehlo.constant dense<[1.0e+20, 1.0e-23]> : tensor<2xf32>
  %ext_expected = stablehlo.constant dense<[0.5, 0.5]> : tensor<2xf32>
  %fwd_ext = enzyme.fwddiff @divide(%ext_a, %ext_da, %ext_b, %ext_db) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  check.expect_almost_eq %fwd_ext, %ext_expected : tensor<2xf32>

  // d/dy (1 / (1 + y^2)) at y = 2^34 (1 + y^2 = 2^68 finite in f32, (1 + y^2)^2 overflows f32)
  %atan_y = stablehlo.constant dense<[17179869184.0, -17179869184.0]> : tensor<2xf32>
  %atan_expected = stablehlo.constant dense<[-3.94430453e-31, 3.94430453e-31]> : tensor<2xf32>
  %fwd_atan = enzyme.fwddiff @atan2_deriv(%atan_y, %done) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>]
  }> : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  check.expect_almost_eq %fwd_atan, %atan_expected : tensor<2xf32>

  // strong_zero=true when a / b overflows f32 to +inf and db == 0
  %sz_a = stablehlo.constant dense<[1.0e+30, 4.0]> : tensor<2xf32>
  %sz_da = stablehlo.constant dense<[1.0e-30, 0.0]> : tensor<2xf32>
  %sz_b = stablehlo.constant dense<[1.0e-30, 2.0]> : tensor<2xf32>
  %sz_db = stablehlo.constant dense<[0.0, 0.0]> : tensor<2xf32>
  %sz_expected = stablehlo.constant dense<[1.0, 0.0]> : tensor<2xf32>
  %fwd_sz = enzyme.fwddiff @divide_sz(%sz_a, %sz_da, %sz_b, %sz_db) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>],
    strong_zero=true
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  check.expect_almost_eq %fwd_sz, %sz_expected : tensor<2xf32>

  // complex forward-mode division where b^2 would overflow or underflow f32
  %ca = stablehlo.constant dense<[(2.0e+20,0.0), (2.0e-23,0.0)]> : tensor<2xcomplex<f32>>
  %cda = stablehlo.constant dense<[(2.0e+20,0.0), (2.0e-23,0.0)]> : tensor<2xcomplex<f32>>
  %cb = stablehlo.constant dense<[(2.0e+20,0.0), (2.0e-23,0.0)]> : tensor<2xcomplex<f32>>
  %cdb = stablehlo.constant dense<[(1.0e+20,0.0), (1.0e-23,0.0)]> : tensor<2xcomplex<f32>>
  %cexpected = stablehlo.constant dense<[(0.5,0.0), (0.5,0.0)]> : tensor<2xcomplex<f32>>
  %fwd_c = enzyme.fwddiff @divide_complex(%ca, %cda, %cb, %cdb) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>]
  }> : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  check.expect_almost_eq %fwd_c, %cexpected : tensor<2xcomplex<f32>>

  // batched width=2 forward-mode division with overflow/underflow divisors
  %batch_da = stablehlo.constant dense<[[2.0e+20, 2.0e-23], [0.0, 0.0]]> : tensor<2x2xf32>
  %batch_db = stablehlo.constant dense<[[0.0, 0.0], [1.0e+20, 1.0e-23]]> : tensor<2x2xf32>
  %batch_expected = stablehlo.constant dense<[[1.0, 1.0], [-0.5, -0.5]]> : tensor<2x2xf32>
  %fwd_batch = enzyme.fwddiff @divide(%ext_a, %batch_da, %ext_b, %batch_db) <{
    activity=[#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>],
    width=2
  }> : (tensor<2xf32>, tensor<2x2xf32>, tensor<2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  check.expect_almost_eq %fwd_batch, %batch_expected : tensor<2x2xf32>

  // rev diff
  %rev:3 = enzyme.autodiff @divide(%a, %b, %done) <{
    activity=[#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev#0, %output : tensor<2xf32>
  check.expect_almost_eq %rev#1, %expected_da : tensor<2xf32>
  check.expect_almost_eq %rev#2, %expected_db : tensor<2xf32>

  func.return
}
