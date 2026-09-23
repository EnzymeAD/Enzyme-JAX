// RUN: enzymexlamlir-opt --distributed-print-schedule-graph %s 2>&1 >/dev/null | FileCheck %s

// ACT's own patched kernel bodies (ActWorkerClient.cpp's spliceKernelBodyModule),
// once dissolved by distributed-inline-kernels into plain block-scope ops, look
// like the memref.alloc/subview/dealloc scaffolding below wrapped around calls
// to bodyless @act_stage_N/@act_launch/@act_fetch declarations: buildBlockNodes
// has to build real nodes for those memref ops (not silently drop them) and
// wire real ordering edges between calls that only ever share a memref
// operand, never an SSA result.
//
// - node 0 (alloc) has no memref operand of its own, so it starts the chain.
// - Every subview and the two stage calls are themselves the next op to
//   reference %alloc's root, so each chains onto whichever came right before
//   it: subview -> subview_0 -> subview_1 -> stage_0 -> stage_1, in addition
//   to each op's own direct SSA operand-producer edges. This is coarser than
//   the true, disjoint-region memory access it models (see
//   addMemrefSideEffectEdges in ScheduleGraph.cpp for why that's fine here).
// - act_launch has no SSA result feeding act_fetch: the only reason
//   act_fetch (which reads %subview_1) is ordered after act_launch (which
//   writes it) is that both reference the same memref root.
// - memref.dealloc %alloc is ordered after every other toucher of that root,
//   including act_fetch, the last of them in block order: without this edge,
//   distributed-reorder-schedule could sink the dealloc ahead of a real use,
//   producing an SSA dominance violation.
//
// CHECK: node 0: compute memref.alloc: flops=0 bytes=0 time=0 preds=[]
// CHECK: node 1: compute memref.subview: flops=0 bytes=0 time=0 preds=[0]
// CHECK: node 2: compute memref.subview: flops=0 bytes=0 time=0 preds=[0, 1]
// CHECK: node 3: compute memref.subview: flops=0 bytes=0 time=0 preds=[0, 2]
// CHECK: node 4: compute func.call: flops=0 bytes=0 time=0 preds=[1, 3]
// CHECK: node 5: compute func.call: flops=0 bytes=0 time=0 preds=[2, 4]
// CHECK: node 6: compute func.call: flops=8320 bytes=0 time=4.16 preds=[1, 2, 3, 5]
// CHECK: node 7: compute func.call: flops=0 bytes=0 time=0 preds=[3, 6]
// CHECK: node 8: compute memref.dealloc: flops=0 bytes=0 time=0 preds=[0, 7]
// CHECK: schedule graph: 9 nodes (9 compute, 0 communication, 0 call)
module {
  distributed.PhysicalMesh @mesh0 device_target "cpu" axes [!distributed.physical_comm_axis<2, 1>] {device_clock_frequency = 1.0e2}

  func.func @main(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %alloc = memref.alloc() : memref<320xi8>
    %subview = memref.subview %alloc[0] [128] [1] : memref<320xi8> to memref<128xi8, strided<[1]>>
    %subview_0 = memref.subview %alloc[128] [128] [1] : memref<320xi8> to memref<128xi8, strided<[1], offset: 128>>
    %subview_1 = memref.subview %alloc[256] [64] [1] : memref<320xi8> to memref<64xi8, strided<[1], offset: 256>>
    call @act_stage_0(%a, %subview) : (tensor<4x8xf32>, memref<128xi8, strided<[1]>>) -> ()
    call @act_stage_1(%b, %subview_0) : (tensor<8x4xf32>, memref<128xi8, strided<[1], offset: 128>>) -> ()
    call @act_launch(%subview, %subview_0, %subview_1) : (memref<128xi8, strided<[1]>>, memref<128xi8, strided<[1], offset: 128>>, memref<64xi8, strided<[1], offset: 256>>) -> ()
    %r = call @act_fetch(%subview_1) : (memref<64xi8, strided<[1], offset: 256>>) -> tensor<4x4xf32>
    memref.dealloc %alloc : memref<320xi8>
    return %r : tensor<4x4xf32>
  }

  func.func private @act_stage_0(tensor<4x8xf32>, memref<128xi8, strided<[1]>>) attributes {act.role = "stage"}
  func.func private @act_stage_1(tensor<8x4xf32>, memref<128xi8, strided<[1], offset: 128>>) attributes {act.role = "stage"}
  func.func private @act_launch(memref<128xi8, strided<[1]>>, memref<128xi8, strided<[1], offset: 128>>, memref<64xi8, strided<[1], offset: 256>>) attributes {act.cost = 4.16e2 : f64}
  func.func private @act_fetch(memref<64xi8, strided<[1], offset: 256>>) -> tensor<4x4xf32> attributes {act.role = "fetch"}
}
