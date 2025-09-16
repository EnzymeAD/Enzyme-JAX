// RUN: enzymexlamlir-opt %s
distributed.LeafDevice @myGpu
distributed.DeviceMesh @gpuMesh @myGpu [2, 2]
distributed.LeafDevice @myCpu
distributed.Channel @chan1 [@myCpu, @gpuMesh] [@gpuMesh, @myCpu]
distributed.DeviceGroup @gpusWithHost [@myGpu, @myCpu] [@chan1]

func.func @foo() {
    distributed.GroupSplit @gpusWithHost {
        %tok = distributed.DefineToken @chan1
        distributed.SplitBranch @chan1 { }
        distributed.SplitBranch @myCpu {}
        distributed.SplitBranch @gpuMesh {
            distributed.MeshFor @gpuMesh {

            }
        }
    }
    func.return
}