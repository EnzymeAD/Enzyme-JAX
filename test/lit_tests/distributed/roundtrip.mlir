// RUN: enzymexlamlir-opt %s | FileCheck %s
distributed.LeafDevice @myGpu
distributed.DeviceMesh @gpuMesh @myGpu [2, 2]
distributed.LeafDevice @myCpu
distributed.Channel @chan1 [@myCpu, @gpuMesh] [@gpuMesh, @myCpu]
distributed.DeviceGroup @gpusWithHost [@myGpu, @myCpu] [@chan1]

func.func @foo() {
    distributed.GroupSplit @gpusWithHost 
       branch @myGpu {
               distributed.MeshFor @gpuMesh {
               }
       }
       branch @myCpu {
          distributed.DefineToken @chan1
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