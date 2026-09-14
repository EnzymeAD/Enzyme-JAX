// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_const mode=ForwardMode" --canonicalize | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_const mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
	func.func @main(%arg0: tensor<10xf32>, %index: tensor<i32>) -> tensor<10xf32> {
		%cst = stablehlo.constant dense<1.0> : tensor<10xf32>

		%0 = "stablehlo.case"(%index) ({
			%1 = "stablehlo.add" (%arg0, %cst) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
			"stablehlo.return"(%1) : (tensor<10xf32>) -> ()
		}, {
			%2 = "stablehlo.multiply" (%arg0, %arg0): (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
			"stablehlo.return"(%2): (tensor<10xf32>) -> ()
		}, {
			"stablehlo.return"(%cst): (tensor<10xf32>) -> ()
		}): (tensor<i32>) -> tensor<10xf32>

		func.return %0 : tensor<10xf32>
	}
}

// FORWARD:    func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<i32>) -> (tensor<10xf32>, tensor<10xf32>) {
// FORWARD-NEXT:      %cst = arith.constant dense<0.000000e+00> : tensor<10xf32>
// FORWARD-NEXT:      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<10xf32>
// FORWARD-NEXT:      %0:2 = "stablehlo.case"(%arg2) ({
// FORWARD-NEXT:        %1 = stablehlo.add %arg1, %cst : tensor<10xf32>
// FORWARD-NEXT:        %2 = stablehlo.add %arg0, %cst_0 : tensor<10xf32>
// FORWARD-NEXT:        stablehlo.return %2, %1 : tensor<10xf32>, tensor<10xf32>
// FORWARD-NEXT:      }, {
// FORWARD-NEXT:        %1 = stablehlo.multiply %arg1, %arg0 : tensor<10xf32>
// FORWARD-NEXT:        %2 = stablehlo.multiply %arg1, %arg0 : tensor<10xf32>
// FORWARD-NEXT:        %3 = arith.addf %1, %2 fastmath<fast> : tensor<10xf32>
// FORWARD-NEXT:        %4 = stablehlo.multiply %arg0, %arg0 : tensor<10xf32>
// FORWARD-NEXT:        stablehlo.return %4, %3 : tensor<10xf32>, tensor<10xf32>
// FORWARD-NEXT:      }, {
// FORWARD-NEXT:        stablehlo.return %cst_0, %cst : tensor<10xf32>, tensor<10xf32>
// FORWARD-NEXT:      }) : (tensor<i32>) -> (tensor<10xf32>, tensor<10xf32>)
// FORWARD-NEXT:      return %0#0, %0#1 : tensor<10xf32>, tensor<10xf32>
// FORWARD-NEXT:    }
// FORWARD-NEXT:  }

// REVERSE: func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<i32>, %arg2: tensor<10xf32>) -> tensor<10xf32> {
// REVERSE-NEXT:    %cst = arith.constant dense<0.000000e+00> : tensor<10xf32>
// REVERSE-NEXT:    %0 = arith.addf %arg2, %cst fastmath<fast> : tensor<10xf32>
// REVERSE-NEXT:    %1:3 = "stablehlo.case"(%arg1) ({
// REVERSE-NEXT:      %2 = arith.addf %0, %cst fastmath<fast> : tensor<10xf32>
// REVERSE-NEXT:      %3 = arith.addf %2, %cst fastmath<fast> : tensor<10xf32>
// REVERSE-NEXT:      stablehlo.return %cst, %3, %cst : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      %2 = arith.addf %0, %cst fastmath<fast> : tensor<10xf32>
// REVERSE-NEXT:      %3 = stablehlo.multiply %2, %arg0 : tensor<10xf32>
// REVERSE-NEXT:      %4 = arith.addf %3, %cst fastmath<fast> : tensor<10xf32>
// REVERSE-NEXT:      %5 = stablehlo.multiply %2, %arg0 : tensor<10xf32>
// REVERSE-NEXT:      %6 = arith.addf %4, %5 fastmath<fast> : tensor<10xf32>
// REVERSE-NEXT:      stablehlo.return %cst, %6, %cst : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }, {
// REVERSE-NEXT:      stablehlo.return %cst, %cst, %cst : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>
// REVERSE-NEXT:    }) : (tensor<i32>) -> (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>)
// REVERSE-NEXT:    return %1#1 : tensor<10xf32>
// REVERSE-NEXT:  }
