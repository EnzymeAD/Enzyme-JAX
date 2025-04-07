#include "mhlo/IR/hlo_ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>

#define DEBUG_TYPE "shard-p2p"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SHARDPOINTTOPOINT
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

template <typename T> Attribute makeAttr(mlir::Type elemType, T val) {
  if (auto TT = dyn_cast<RankedTensorType>(elemType))
    return SplatElementsAttr::get(
        TT, ArrayRef(makeAttr<T>(TT.getElementType(), val)));
  if (isa<FloatType>(elemType))
    return FloatAttr::get(elemType, val);
  else
    return IntegerAttr::get(elemType, val);
}

/*

%26 = stablehlo.slice %arg11 [8:520, 1:1023, 2034:2040] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x6xf64> loc(#loc1191)
%27 = stablehlo.slice %arg11 [8:520, 1:1023, 8:2040] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64> loc(#loc1191)
%23 = stablehlo.slice %arg11 [8:520, 1:1023, 8:16] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64> loc(#loc1190)
%36 = stablehlo.concatenate %26, %27, %23, dim = 2 {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<512x1022x6xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x8xf64>) -> tensor<512x1022x2046xf64> loc(#loc1262


left pad = 8, right pad = 8

// https://github.com/openxla/shardy/blob/b242e869dbeb5a5f4bb8c36fd7d16d99a0413367/docs/compiler_api.md?plain=1#L143
%superset = stablehlo.slice %arg11 [8:530, 1:1023, 8:2040] [<@mesh, [{}, {ny}, {nx}]>] // this will lead to a single collective_permute
// https://openxla.org/shardy/sdy_dialect#sdymanual_computation_sdymanualcomputationop
// Total Size (T2) = (N2 - N1) + (N2 - N3) + (N4 - N1) // slices for the final concat
// T1 = N2 - N1
%36 = sdy.manual_computation(%superset) in_shardings=[<@mesh, [{}, {ny}, {nx}]>] out_shardings=[<@mesh, [{}, {ny}, {nx}]>] (%inner_arg1: // local size of inner_arg) {
    // N3:N2 N1:N2 N1:N4 -> to be concatenated
    // Original Split --> (N2 - N1) / nx
    // New Splits --> ((N2 - N1) + (N2 - N3) + (N4 - N1)) / nx

    // Part 1 of Comm slices from the middle part
    // if stablehlo.partition_id() == 0 // only for Iota Tiles
    %local_data_for_comm = slice %inner_arg1[:, :, ((T2 - T1) / ny):axes(inner_arg1, 3)]
    // elseif
    //     .... for all the partition_ids
      
    %result_1 = "stablehlo.collective_permute"(%local_data_for_comm) {
      // [0, 1], ... [ny - 2, ny - 1], [ny, ny + 1], .... // no need for ny - 1 -> ny and similarly 2ny - 1 -> 2ny
      source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], ...]> : tensor<...>,
      channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    } : (tensor<...) -> (....)

    %already_available_slice = slice %inner_arg1[:, :, 1:((T2 - T1) / ny)]

    if partition_id() >= 8 and <= 2040{
        // we already have the full data
        concatenate(%result_1, %already_available_slice)
    } else {
        // we might as well slice to prevent the branches????
        // we need to do extra comm to fetch the 2 ends
        if partition_id() == 0 or ny or 2ny or ...
            %end_slice = %inner_arg [:, :, 1:(N4 - N1)

        else
            %end_slice = zeros of the same size
        end

        %result_1 = "stablehlo.collective_permute"(%end_slice) {
             source_target_pairs = dense<[[0, ny - 1], [ny, 2ny - 1], ...]
          channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
        } : (tensor<...) -> (....)        
    }

        // we might as well slice to prevent the branches????
        if partition_id() == ny -1 or 2ny - 1 or ...
            %start_slice = %inner_arg:, :, [(N2 - N1):en:
        else
            %start_slice = zeros of same size

        %result_2 = "stablehlo.collective_permute"(%start_slice) {
             source_target_pairs = dense<[[0, ny - 1], [ny, 2ny - 1], ...]
             source_target_pairs = dense<[[ny - 1, 0], [2ny - 1, ny], ...]
          channel_handle = #stablehlo.channny -1, 0>, [2ny - 1, ny], ...]
        } : (tensor<...) -> (....)




        if partition_id() == 0 or ny or 2ny or ...
            concatenate(%result_2)        
    

        ....} : tensor<512x1022x(N2 - N1)xf32>) -> (512x1022xTotalSize) // global sizes


*/
struct SliceConcatSimplify : public OpRewritePattern<stablehlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConcatenateOp concat,
                                PatternRewriter &rewriter) const override { 
    if (concat.getNumOperands() != 3) {
      return failure();
    }

    SmallVector<stablehlo::SliceOp> ops;
    for (Value operand : concat.getOperands()) {
      auto sliceOp = operand.getDefiningOp<stablehlo::SliceOp>();
      if (!sliceOp)
        return failure();

      if (!sliceOp->hasOneUse())
        return failure();

      ops.push_back(sliceOp);
    }

    for (auto sl : ops) {
      for (int i=0; i<op.getShape().size(); i++) {
        if (sl.getStrides()[i] != 1) {
          return failure();
        }
        if (sl.getOperand()[0] != ops[0].getOperand()) {
          return failure();
        }
        if (i == concat.getDimension()) {
          continue;
        }
        if (sl.getStartIndices()[i] != ops[0].getStartIndices()[i]) {
          return failure();
        }
        if (sl.getLimitIndices()[i] != ops[0].getLimitIndices()[i]) {
          return failure();
        }
      }
    }



    /*    
    %26 = stablehlo.slice %arg11 [8:520, 1:1023, 2034:2040] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x6xf64> loc(#loc1191)
    %27 = stablehlo.slice %arg11 [8:520, 1:1023, 8:2040] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64> loc(#loc1191)
    %23 = stablehlo.slice %arg11 [8:520, 1:1023, 8:16] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64> loc(#loc1190)
    %36 = stablehlo.concatenate %26, %27, %23, dim = 2 {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<512x1022x6xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x8xf64>) -> tensor<512x1022x2046xf64> loc(#loc1262
    */
    
    // LHS = 6
    // RHS = 8 in above
    int lhsSize = ops[0].getType().getShape()[concat.getDimension()];
    int rhsSize = ops[1].getType().getShape()[concat.getDimension()]
    if (ops[2].getStartIndices()[concat.getDimension()] != lops[1].getStartIndices()[concat.getDimension()])
      return failure();

    if (ops[0].getLimitIndices()[concat.getDimension()] !+ lowerLim = uops[1].getLimitIndices()[concat.getDimension()])
      return failure();

    
    // https://github.com/openxla/shardy/blob/b242e869dbeb5a5f4bb8c36fd7d16d99a0413367/shardy/dialect/sdy/ir/utils.h 
      
    // query attr from input: sdy.sharding    // r_attr = MLIR.API.mlirDictionaryAttrGetElementByName(attr, "sdy.sharding")
    // mesh = sdyTensorShardingAttrGetMeshOrRef @mesh, ....    //
    // mlir_c.sdyTensorShardingAttrGetDimShardingsSize : tensorshardingattr -> int
    // sdyTensorShardingAttrGetDimShardingsElem : tenosrshardingattr, int -> dimensionshardingattribute 
    //
    // sdyDimensionShardingAttrGetAxesSize
    // sdyDimensionShardingAttrGetAxesElem
    //
    //
    // https://github.com/EnzymeAD/Reactant.jl/blob/0a460dc9915d2a11dc86691df244a4cb220c17ea/src/mlir/libMLIR_h.jl#L10396    // https://github.com/EnzymeAD/Reactant.jl/blob/0a460dc9915d2a11dc86691df244a4cb220c17ea/src/Sharding.jl#L1065-L1102 

    TensorShardingAttr op_shardings = {
      mlir::sdy::getSharding(ops[1])
    };
    ensorShardingPerValueAttr in_shardings; = TensorShardingPerValueAttr::get(concat.getContext(), op_shardings);
   TensorShardingPerValueAttr out_shardings; = in_shardings
    SmallVector<StringAttr> manual_axes;
    auto manual = rewriter.create<sdy::ManualComputation>(concat.getLoc(), TypeRange(concat.getType()), ValueRange(ops[1]), in_shardings, out_shardings, manual_axes);
     
    auto meshAttr = mlir::sdy::getCommonMesh(op_shardings, op_shardings, concat);

    auto dim_shardings = op_shardings[0].getDimShardings();
    SmallVector<int32_t> ndevices;
    for (auto dim_sharding : dim_shardings) {
      int32_t total_size = 1;
      for (auto axis : dim_sharding.getAxes()) {ops[1]
        total_size *= meshAttr.getAxisSize(axis.getName());
      }
      ndevices.push_back(total_size);
    }

    SmallVector<int64_t> localShape = llvm::to_vector(ops[1].getType().getShape());
    assert(localShape.size() == ndevices.size());
    for (int i=0; i<localShape.size(); i++) {
      localShape[i] /= ndevices[i];
    }

    mlir::Type in_tys[1] {
      RankedTensorType::get(localShape, ops[1].getType().getElementType())
    };
    mlir::Type in_locs[] = { ops[1].getLoc() };

    uto blk = rewriter.createBlock(manual.getBody(), manual.getBody().begin(), in_tys, in_locs);
    auto arg = blk->getArgument(0);
   auto id = rewriter.craete<stablehlo::PartitionIdOp>(concat.getLoc());
     

    /*    
    %26 = stablehlo.slice %arg11 [8:520, 1:1023, 2034:2040] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x6xf64> loc(#loc1191)
    %27 = stablehlo.slice %arg11 [8:520, 1:1023, 8:2040] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x2032xf64> loc(#loc1191)
    %23 = stablehlo.slice %arg11 [8:520, 1:1023, 8:16] {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<528x1024x2048xf64>) -> tensor<512x1022x8xf64> loc(#loc1190)
    %36 = stablehlo.concatenate %26, %27, %23, dim = 2 {mhlo.sharding = "{devices=[1,1,2]<=[2]}"} : (tensor<512x1022x6xf64>, tensor<512x1022x2032xf64>, tensor<512x1022x8xf64>) -> tensor<512x1022x2046xf64> loc(#loc1262
    */

    auto T1 = ops[1].getType().getShape()[concat.getDimension()];
    auto T2 = lhsSize;

    // %local_data_for_comm = slice %inner_arg1[:, :, ((T2 - T1) / ny):axes(inner_arg1, 3)]
    // T2 = size(%26, concat_dim) + 

    
    // // uto cond = rewriter.create<stablehlo::CompareOp>(

    if (ndevices.size() != 3 || ndevices[0] != 1)
      return failure();
    }

    int64_t nx, ny;
    if (concat.getDimension() == 1) {
      ny = ndevices[1];
      nx = ndevices[2];
    } else {
      ny = ndevices[2];
      nx = ndevices[1];
    }

    auto devices_along_dim = ndevices[concat.getDimension()];

    SmallVector<int64_t> inner_starts(concat.getType().getShape().size(), 0);
    SmallVector<int64_t> inner_limits = llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
    SmallVector<int64_t> inner_strides(concat.getType().getShape().size(), 1);
    inner_starts[concat.getDimension()] = (T2 - T1) / ny;
    auto local_data_for_comm = rewriter.create<stablehlo::SliceOp>(concat.getLoc(), inner_arg1, inner_starts, inner_limits, inner_strides);
     
    // SmallVector<int64_t> source_ids((ny - 1) * nx);
    // SmallVector<int64_t> target_ids((ny - 1) * nx);
    // send_idx = 0;
    // for (int i = 0; i < (ny - 1) * nx; i++) {
    //   source_ids[i] = send_idx;
    //   target_ids[i] = send_idx + 1;
    // }

    // for simplicity send for all pairs
    SmallVector<int64_t> source_target_ids(2 * ny * nx);
    for (int i = 0; i < ny * nx; i++) {
      source_target_ids[2 * i] = i;
      source_target_ids[2 * i + 1] = i;
    }

    // conver this to source_target_pairs 

    auto source_target_pairs_ty = RankedTensorType::get({(int64_t)(nx * ny), (int64_t)2}, rewriter.getI64Type());
    
    auto cperm = rewriter.create<stablehlo::CollectivePermuteOp>(
      concat.getLoc(), 
      local_data_for_comm->getResult(0),
      DenseIntElementsAttr::get(source_target_pairs_ty, source_target_ids),
      stablehlo::ChannelHandleAttr::get(concat.getContext(),/*handle*/0, /*type*/0)
    );

      / %already_available_slice = slice %inner_arg1[:, :, 1:((T2 - T1) / ny)]


    // partition_id in 0, ny, 2ny ... (nx - 1)ny
    Value leftSide = rewriter.create<stablehlo::RemOp>(concat.getLoc(), partition_id, 
      rewriter.create<stablehlo::ConstantOp>(concat.getLoc(), partition_id.getType(), makeAttr(partition_id.getType(), ny).cast<ElementsAttr>())
    );
    leftSize = rewriter.create<stablehlo::CompareOp>(concat.getLoc(),
      leftSide,
      rewriter.getZeroAttr(leftSide.getType()), stablehlo::ComparisonDirection::EQ);

    // partition_id == (ny -1) (2ny - 1) ... 
    Value rightSide = rewriter.create<stablehlo::AddOp>(concat.getLoc(), partition_id, makeAttr(partition_id.getType(), 1).cast<ElementsAttr>());
    rightSide = rewriter.create<stablehlo::RemOp>(concat.getLoc(), rightSide, 
      rewriter.create<stablehlo::ConstantOp>(concat.getLoc(), partition_id.getType(), makeAttr(partition_id.getType(), ny).cast<ElementsAttr>())
    );
    rightSide = rewriter.create<stablehlo::CompareOp>(concat.getLoc(),
    rightSide,
      rewriter.getZeroAttr(leftSide.getType()), stablehlo::ComparisonDirection::EQ);

    // if ..... !leftSide  && !rightSide
      SmallVector<int64_t> inner_starts2(concat.getType().getShape().size(), 0);
      SmallVector<int64_t> inner_limits2 = llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
      inner_limits2[concat.getDimension()] = (T2 - T1) / ny;
      auto already_available_slice = rewriter.create<stablehlo::SliceOp>(
        concat.getLoc(), inner_arg1, inner_starts2, inner_limits2, inner_strides);

        Value concat_args[] = {cperm, already_available_slice};        // fix api
        
      rewriter.createOp<stablehlo::ConcatenateOp>(
        concat.getLoc(), {cperm, already_available_slice}, concat.getDimension()
      );
  
    // else       
      SmallVector<int64_t> inner_starts3(concat.getType().getShape().size(), 0);
      SmallVector<int64_t> inner_limits3 = llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
      inner_limits3[concat.getDimension()] = rhsSize;
      auto end_slice = rewriter.create<stablehlo::SliceOp>(
        concat.getLoc(), inner_arg1, inner_starts3, inner_limits3, inner_strides);
      
      SmallVector<int64_t> source_target_ids2(2 * nx);
      for (int i = 0; i < nx; i++) {
        source_target_ids2[2 * i] = i;
        source_target_ids2[2 * i + 1] = (i + 1) * ny - 1;
      }

      auto result_1 = rewriter.create<stablehlo::CollectivePermuteOp>(
        concat.getLoc(), 
        end_slice,
        DenseIntElementsAttr::get(source_target_ids2, source_target_ids),
        stablehlo::ChannelHandleAttr::get(concat.getContext(),/*handle*/0, /*type*/0)
      );
    //       channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
    //     } : (tensor<...) -> (....)        
    // }
    
    // uto cond = rewriter.create<stablehlo::CompareOp>(concat.getLoc(), id, rewriter.getZeroAttr(unrankedTensorType),
     //      stablehlo::ComparisonDirection::EQ);
    m// lir::Type DataForCondType;
    a// uto ifOp = rewriter.create<stablehlo::IfOp>(concat.getLoc(), 
    //  //  cond,
    //  //  DataForCondType
    // )// ;
    // TODO fill in if branches

      SmallVector<int64_t> inner_starts4(concat.getType().getShape().size(), 0);
      SmallVector<int64_t> inner_limits4 = llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
      inner_starts4[concat.getDimension()] = // size(%26, concatDim)
      auto start_slice = rewriter.create<stablehlo::SliceOp>(
        concat.getLoc(), inner_arg1, inner_starts4, inner_limits4, inner_strides);
    //     %result_2 = "stablehlo.collective_permute"(%start_slice) {
    //          source_target_pairs = dense<[[0, ny - 1], [ny, 2ny - 1], ...]
    //       channel_handle = #stablehlo.channny -1, 0>, [2ny - 1, ny], ...]
    //     } : (tensor<...) -> (....)

      SmallVector<int64_t> source_target_ids3(2 * nx);
      for (int i = 0; i < nx; i++) {
        source_target_ids3[2 * i] = (i + 1) * ny - 1;
        source_target_ids3[2 * i + 1] = i;
      }

  //     %result_2 = "stablehlo.collective_permute"(%start_slice) {
  //       source_target_pairs = dense<[[ny - 1, 0], [2ny - 1, ny], ...]
  //    channel_handle = #stablehlo.channny -1, 0>, [2ny - 1, ny], ...]
  //  } : (tensor<...) -> (....)


    //     if lhsSide
               auto l

              SmallVector<int64_t> inner_starts5(concat.getType().getShape().size(), 0);
              SmallVector<int64_t> inner_limits5 = llvm::to_vector(cast<RankedTensorType>(arg.getType()).getShape());
              inner_starts4[concat.getDimension()] = // size(%26, concatDim)hsRightSlice = rewriter.createOp<stablehlo::SliceOp>(
                 

              auto l               )

;    //         concatenate(%result_2)      
    //     else
    //         

   


    auto ResultType = ;
    Value full_data_condition = ;
    auto result = rewriter.create<stablehlo::IfOp>(concat.getLoc(), 
      full_data_condition,
      ResultType
    );
    rewriter.create<stablehlo::ReturnOp>(concat.getLoc(), ValueRange(result));

    {
    rewriter.createBlock(result.getTrueBranch(), result.getTrueBranch().begin());
    
    Value tomerge[] = { already_available_slice, cperm };
    rewriter.create<stablehlo::ReturnOp>(concat.getLoc(),
      ValueRange(
        rewriter.create<stable::ConcatenateOp>(concat.getLoc(), tomerge, concat.getDimension())
      )
    );
    }
    
    {
    rewriter.createBlock(result.getFalseBranch(), result.getFalseBranch().begin());
    
    rewriter.create<stablehlo::ReturnOp>(concat.getLoc(),
      ValueRange(
        extra_comm_to_fetch_the_two_ends
      )
    );
  }


  // {
  //   rewriter.createBlock(ifOp.getTrueBranch(), ifOp.getTrueBranch().begin());
    
  //   }
    
  //   {
  //   rewriter.createBlock(ifOp.getFalseBranch(), ifOp.getFalseBranch().begin());
    
  //   rewriter.create<stablehlo::ReturnOp>(concat.getLoc(),
  //     ValueRange(
  //       extra_comm_to_fetch_the_two_ends
  //     )
  //   );
  }

    
    rewriter.replaceOp(concat, result);

    return success();
  }
};
struct ShardPointToPointPass
    : public enzyme::impl::ShardPointToPointBase<ShardPointToPointPass> {
  using Base::Base;
  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<SliceConcatSimplify>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
