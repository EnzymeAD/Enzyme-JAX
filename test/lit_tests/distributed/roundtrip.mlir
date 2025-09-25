// RUN: enzymexlamlir-opt %s | FileCheck %s
distributed.leaf_device @myGpu
distributed.device_mesh @gpuMesh @myGpu [2, 2]
distributed.leaf_device @myCpu
distributed.channel @chan1 [@myCpu, @gpuMesh] [@gpuMesh, @myCpu]
distributed.device_group @gpusWithHost [@myGpu, @myCpu] [@chan1]

func.func @foo() {
   distributed.device_parallel @gpusWithHost {
       branch @myGpu {
            ^entry():
               distributed.device_parallel @gpuMesh {
                  branch @myGpu {
                    ^entry():
                        distributed.noop
                  }
               }
               }
       branch @myCpu {
            ^entry():
            distributed.noop
       }
       branch @chan1 {
            ^entry():
            distributed.noop
       }
   }

    func.return
}

//CHECK: module {
//CHECK-NEXT:   distributed.LeafDevice @myGpu
//CHECK-NEXT:   distributed.DeviceMesh @gpuMesh @myGpu [2, 2]
//CHECK-NEXT:   distributed.LeafDevice @myCpu
//CHECK-NEXT:   distributed.Channel @chan1 [@myCpu, @gpuMesh] [@gpuMesh, @myCpu]
//CHECK-NEXT:   distributed.DeviceGroup @gpusWithHost [@myGpu, @myCpu] [@chan1]
//CHECK-NEXT:   func.func @foo() {
//CHECK-NEXT:     distributed.GroupSplit @gpusWithHost  branch @myGpu{
//CHECK-NEXT:       distributed.MeshFor @gpuMesh {
//CHECK-NEXT:       }
//CHECK-NEXT:     } branch @myCpu{
//CHECK-NEXT:       %0 = distributed.DefineToken @chan1
//CHECK-NEXT:     }
//CHECK-NEXT:     return
//CHECK-NEXT:   }
//CHECK-NEXT: }