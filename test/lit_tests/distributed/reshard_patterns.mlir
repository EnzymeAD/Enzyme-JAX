// RUN: enzymexlamlir-opt --split-input-file %s --sdy-propagation-pipeline --sdy-export-pipeline="enable-insert-explicit-collectives=true" --sdy-convert-global-to-local="enable-rgv3=true"
module {
    sdy.mesh @mesh = <["a" = 4, "b" = 2]>
    distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<2>]

    func.func @all_gather(
        %arg0: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}
    ) -> (tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
        func.return %arg0 : tensor<128x256xf32>
    }
}

// -----

module {
    sdy.mesh @mesh = <["a" = 4, "b" = 4]>
    distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<2>]
    func.func @gather_scatter(
        %arg0: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}
    ) -> (tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
        func.return %arg0 : tensor<128x256xf32>
    }
}

// -----

module {
    sdy.mesh @mesh = <["a" = 4, "b" = 4]>
    distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<2>]

    func.func @all_slice_all_gather(
        %arg0: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}
    ) -> (tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
        func.return %arg0 : tensor<128x256xf32>
    }
}

// -----

module {
    sdy.mesh @mesh = <["a" = 4, "b" = 4]>
    distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<2>]

    func.func @collective_permute(
        %arg0: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}
    ) -> (tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {"a"}]>}) {
        func.return %arg0 : tensor<128x256xf32>
    }
}

// -----

module {
    sdy.mesh @mesh = <["a" = 4, "b" = 3]>
    distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<2>]

    func.func @collective_permute_uneven(
        %arg0: tensor<120x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {}]>}
    ) -> (tensor<120x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a"}, {}]>}) {
        func.return %arg0 : tensor<120x256xf32>
    }
}

// -----

module {
    sdy.mesh @mesh = <["a" = 4, "b" = 2]>
    distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2>, !distributed.physical_comm_axis<2>]

    func.func @all_permute_all_to_all(
        %arg0: tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}
    ) -> (tensor<128x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {"a"}]>}) {
        func.return %arg0 : tensor<128x256xf32>
    }
}