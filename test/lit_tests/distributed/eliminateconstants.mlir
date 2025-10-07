// RUN: enzymexlamlir-opt --eliminate-constant-communication %s | FileCheck %s
distributed.leaf_device @myGpu
distributed.device_mesh @gpuMesh @myGpu [2, 2]
distributed.leaf_device @myCpu
distributed.channel @chan1 [@myCpu, @gpuMesh] [@gpuMesh, @myCpu]
distributed.device_group @gpusWithHost [@myGpu, @myCpu] [@chan1]

func.func @foo() {
   distributed.device_parallel @gpusWithHost {
       branch @myGpu {
            ^entry(%1: !distributed.token):
               distributed.device_parallel @gpuMesh {
                  branch @myGpu {
                    ^entry():
                  }
               }
               }
       branch @myCpu {
            ^entry(%1: !distributed.token):
               %output = stablehlo.constant() {
                    value = dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
                    } : () -> tensor<2x2xf32>
               distributed.send %1 tensor<2x2xf32> %output 

       }
       branch @chan1 {
            ^entry(%1: !distributed.token):
               %input = distributed.recv %1 tensor<2x2xf32>
               %sum = stablehlo.add %input, %input : tensor<2x2xf32>
       }
   }

    func.return
}

//CHECK: module {
//CHECK-NEXT:  distributed.leaf_device @myGpu
//CHECK-NEXT:  distributed.device_mesh @gpuMesh @myGpu [2, 2]
//CHECK-NEXT:  distributed.leaf_device @myCpu
//CHECK-NEXT:  distributed.channel @chan1 [@myCpu, @gpuMesh] [@gpuMesh, @myCpu]
//CHECK-NEXT:  distributed.device_group @gpusWithHost [@myGpu, @myCpu] [@chan1]
//CHECK-NEXT:  func.func @foo() {
//CHECK-NEXT:    distributed.device_parallel @gpusWithHost{ branch @myGpu{
//CHECK-NEXT:    ^bb0(%arg0: !distributed.token):
//CHECK-NEXT:      distributed.device_parallel @gpuMesh{ branch @myGpu{
//CHECK-NEXT:     }}
//CHECK-NEXT:    } branch @myCpu{
//CHECK-NEXT:    ^bb0(%arg0: !distributed.token):
//CHECK-NEXT{LITERAL}:      %cst = stablehlo.constant dense<[[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>
//CHECK-NEXT:    } branch @chan1{
//CHECK-NEXT:    ^bb0(%arg0: !distributed.token):
//CHECK-NEXT{LITERAL}:      %cst = stablehlo.constant dense<[[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>
//CHECK-NEXT:      %0 = stablehlo.add %cst, %cst : tensor<2x2xf32>
//CHECK-NEXT:    }}
//CHECK-NEXT:    return
//CHECK-NEXT:  }
//CHECK-NEXT:}