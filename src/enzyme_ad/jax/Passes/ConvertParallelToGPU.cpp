//===- ConvertParallelToGPU.cpp - Remove unused private functions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
// #include "enzymexla/BarrierUtils.h"
// #include "enzymexla/Ops.h"
// #include "enzymexla/Passes/Passes.h"
// #include "enzymexla/Passes/Utils.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

#include "mlir/Dialect/Async/IR/Async.h"

#include <llvm/ADT/StringRef.h>
#include <optional>

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CONVERTPARALLELTOGPU1
#define GEN_PASS_DEF_CONVERTPARALLELTOGPU2
#define GEN_PASS_DEF_MERGEGPUMODULESPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

#include "ParallelLoopUnroll.h"
#include "RuntimeWrapperUtils.h"

using namespace mlir::enzyme;

static llvm::cl::opt<bool> GPUKernelEmitCoarsenedAlternatives(
    "gpu-kernel-emit-coarsened-alternatives", llvm::cl::init(false),
    llvm::cl::desc("Emit alternative kernels with coarsened threads"));

static llvm::cl::opt<bool> GPUKernelEnableBlockCoarsening(
    "gpu-kernel-enable-block-coarsening", llvm::cl::init(true),
    llvm::cl::desc("When emitting coarsened kernels, enable block coarsening"));

static llvm::cl::opt<bool> GPUKernelEnableCoalescingFriendlyUnroll(
    "gpu-kernel-enable-coalescing-friendly-unroll", llvm::cl::init(false),
    llvm::cl::desc("When thread coarsening, do coalescing-friendly unrolling"));

// TODO when we add other backends, we would need to to add an argument to the
// pass which one we are compiling to to provide the appropriate error id
#if POLYGEIST_ENABLE_CUDA
#include <cuda.h>
#else
#define CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES 701
#endif

using namespace mlir;
using namespace enzymexla;

#define DEBUG_TYPE "convert-parallel-to-gpu"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

#define POLYGEIST_REMARK_TYPE "CONVERT_PARALLEL_TO_GPU"
#define POLYGEIST_REMARK(X)                                                    \
  do {                                                                         \
    if (getenv("POLYGEIST_EMIT_REMARKS_" POLYGEIST_REMARK_TYPE)) {             \
      X;                                                                       \
    }                                                                          \
  } while (0)

// From ParallelLICM.cpp
void moveParallelLoopInvariantCode(scf::ParallelOp looplike);

namespace {
static void shrinkAlternativesOp(enzymexla::AlternativesOp alternativesOp,
                                 unsigned size, PatternRewriter &rewriter) {
  // New AOP with the exact number of regions needed
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(alternativesOp);
  auto newAop = enzymexla::AlternativesOp::create(
      rewriter, alternativesOp->getLoc(), size);
  newAop->setAttr("alternatives.type",
                  alternativesOp->getAttr("alternatives.type"));
  assert(newAop->getNumRegions() > 0);

  auto oldDescs =
      alternativesOp->getAttrOfType<ArrayAttr>("alternatives.descs");

  std::vector<Attribute> descs;
  for (unsigned i = 0; i < newAop->getNumRegions(); i++) {
    auto &region = alternativesOp->getRegion(i);
    auto &newRegion = newAop->getRegion(i);
    rewriter.eraseBlock(&newRegion.front());
    rewriter.inlineRegionBefore(region, newRegion, newRegion.begin());
    descs.push_back(oldDescs[i]);
  }
  newAop->setAttr("alternatives.descs", rewriter.getArrayAttr(descs));
  rewriter.eraseOp(alternativesOp);
}
static void shrinkAlternativesOp(enzymexla::AlternativesOp alternativesOp,
                                 unsigned size, OpBuilder &builder) {
  // New AOP with the exact number of regions needed
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(alternativesOp);
  auto newAop = enzymexla::AlternativesOp::create(
      builder, alternativesOp->getLoc(), size);
  newAop->setAttr("alternatives.type",
                  alternativesOp->getAttr("alternatives.type"));
  assert(newAop->getNumRegions() > 0);

  auto oldDescs =
      alternativesOp->getAttrOfType<ArrayAttr>("alternatives.descs");

  std::vector<Attribute> descs;
  for (unsigned i = 0; i < newAop->getNumRegions(); i++) {
    auto &region = alternativesOp->getRegion(i);
    newAop->getRegion(i).takeBody(region);
    descs.push_back(oldDescs[i]);
  }
  newAop->setAttr("alternatives.descs", builder.getArrayAttr(descs));
  alternativesOp->erase();
}
std::optional<int> getConstantInteger(Value v) {
  if (auto cstint = dyn_cast_or_null<arith::ConstantIntOp>(v.getDefiningOp())) {
    return cstint.value();
  } else if (auto cstindex =
                 dyn_cast_or_null<arith::ConstantIndexOp>(v.getDefiningOp())) {
    return cstindex.value();
  } else {
    return {};
  }
}
template <typename T>
bool hasEffect(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  for (auto it : effects)
    if (isa<T>(it.getEffect()))
      return true;
  return false;
}

template <int S = 3> SmallVector<Value, S> getUpperBounds(scf::ParallelOp pop) {
  SmallVector<Value, S> bounds;
  for (auto bound : pop.getUpperBound()) {
    bounds.push_back(bound);
  }
  return bounds;
}

void insertReturn(PatternRewriter &rewriter, func::FuncOp f) {
  func::ReturnOp::create(rewriter, rewriter.getUnknownLoc());
}
void insertReturn(PatternRewriter &rewriter, LLVM::LLVMFuncOp f) {
  LLVM::ReturnOp::create(rewriter, rewriter.getUnknownLoc(),
                         std::vector<Value>{});
}

scf::ParallelOp
getDirectlyNestedSingleParallel_(const char *PATTERN, Block *block,
                                 bool allowAllocas = false,
                                 bool allowIndexComputation = false) {
  auto it = block->begin();
  while ((allowAllocas && isa<memref::AllocaOp>(&*it)) ||
         (allowIndexComputation && isa<arith::ArithDialect>(it->getDialect())))
    it++;
  auto pop = dyn_cast<scf::ParallelOp>(&*it);
  it++;
  if (!pop) {
    LLVM_DEBUG(DBGS() << "[pop-to-launch] need directly nested parallelop\n");
    return nullptr;
  }
  if (block->getTerminator() != &*it) {
    LLVM_DEBUG(DBGS() << "[pop-to-launch] stray ops in block\n");
    return nullptr;
  }
  it++;
  assert(it == block->end());
  return pop;
}

#define getDirectlyNestedSingleParallel(...)                                   \
  getDirectlyNestedSingleParallel_(PATTERN, __VA_ARGS__)

// Set launch bound attributes
//
// TODO Add a NVVM::NVVMDialect::getLaunchBoundAttrName() (or a gpu dialect one?
// refer to how the KernelAttrName is done for gpu, nvvm, rocdl - needs upstream
// mlir patch) and
struct AddLaunchBounds : public OpRewritePattern<gpu::LaunchFuncOp> {
  using OpRewritePattern<gpu::LaunchFuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gpu::LaunchFuncOp launchOp,
                                PatternRewriter &rewriter) const override {
    // TODO Currently this can be done safely because the enzymexla pipeline
    // generates a different kernel for each _callsite_ and not for each source
    // kernel, we must actually look at whether the symbol is private and
    // whether _all_ call sites use the same const params and only then do this
    // (so we should actually match gpu::GPUFuncOp's and not
    // gpu::LaunchFuncOp's)
    auto gpuFuncOp = launchOp->getParentOfType<ModuleOp>().lookupSymbol(
        launchOp.getKernel());
    auto blockDims = launchOp.getBlockSizeOperandValues();
    auto bx = getConstantInteger(blockDims.x);
    auto by = getConstantInteger(blockDims.y);
    auto bz = getConstantInteger(blockDims.z);
    if (!bx || !by || !bz)
      return failure();
    // TODO should we only set idx or separately set idx, idy, idz? clang seems
    // to only set idx to the total num
    // TODO grab the attr name from the NVVM dialect after bumping llvm
    bool succeeded = false;
    int blockSize = *bx * *by * *bz;
    llvm::StringRef attrName = "nvvm.maxntidx";
    if (!gpuFuncOp->hasAttr(attrName)) {
      gpuFuncOp->setAttr(attrName, rewriter.getIntegerAttr(
                                       rewriter.getIndexType(), blockSize));
      succeeded = true;
    } else {
      assert(blockSize ==
             dyn_cast<IntegerAttr>(gpuFuncOp->getAttr(attrName)).getInt());
      succeeded = false;
    }
    attrName = "rocdl.max_flat_work_group_size";
    if (!gpuFuncOp->hasAttr(attrName)) {
      gpuFuncOp->setAttr(attrName, rewriter.getIntegerAttr(
                                       rewriter.getIndexType(), blockSize));
      assert(succeeded);
      (void)succeeded;
      return success();
    } else {
      assert(blockSize ==
             dyn_cast<IntegerAttr>(gpuFuncOp->getAttr(attrName)).getInt());
      assert(!succeeded);
      (void)succeeded;
      return failure();
    }
  }
};

template <typename FuncType>
struct RemoveFunction : public OpRewritePattern<FuncType> {
  using OpRewritePattern<FuncType>::OpRewritePattern;
  LogicalResult matchAndRewrite(FuncType f,
                                PatternRewriter &rewriter) const override {
    if (!isa<ModuleOp>(f->getParentOp())) {
      return failure();
    }
    auto V = f->getAttr("enzymexla.device_only_func");
    if (!V) {
      return failure();
    }
    Region *region = &f.getBody();
    if (region->empty())
      return failure();
    rewriter.eraseOp(f);
    // TODO leave an empty function to pass to cudaSetCacheConfig
    // Region *region = &f.getBody();
    // if (region->empty())
    //  return failure();
    // rewriter.eraseBlock(&region->front());
    // region->push_back(new Block());
    // rewriter.setInsertionPointToEnd(&region->front());
    // insertReturn(rewriter, f);
    return success();
  }
};

struct SharedLLVMAllocaToGlobal : public OpRewritePattern<LLVM::AllocaOp> {
  using OpRewritePattern<LLVM::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AllocaOp ao,
                                PatternRewriter &rewriter) const override {
    auto PT = cast<LLVM::LLVMPointerType>(ao.getType());
    if (PT.getAddressSpace() != 5) {
      return failure();
    }

    auto loc = ao->getLoc();
    auto name = "shared_mem_" + std::to_string((long long int)(Operation *)ao);

    auto module = ao->getParentOfType<gpu::GPUModuleOp>();
    if (!module) {
      return failure();
    }

    rewriter.setInsertionPointToStart(module.getBody());

    auto globalOp = LLVM::GlobalOp::create(
        rewriter, loc, ao.getElemType(), /* isConstant */ false,
        LLVM::Linkage::Internal, name, mlir::Attribute(),
        /* alignment */ 0, /* addrSpace */ 3);
    rewriter.setInsertionPoint(ao);
    auto aoo = LLVM::AddressOfOp::create(rewriter, loc, globalOp);

    rewriter.replaceOp(ao, aoo->getResults());

    return success();
  }
};

struct SharedMemrefAllocaToGlobal : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp ao,
                                PatternRewriter &rewriter) const override {
    auto mt = ao.getType();
    if (mt.getMemorySpaceAsInt() != 5) {
      return failure();
    }

    auto type = MemRefType::get(mt.getShape(), mt.getElementType(), {},
                                /* memspace */ 3);
    auto loc = ao->getLoc();
    auto name = "shared_mem_" + std::to_string((long long int)(Operation *)ao);

    auto module = ao->getParentOfType<gpu::GPUModuleOp>();
    if (!module) {
      return failure();
    }

    rewriter.setInsertionPointToStart(module.getBody());

    auto initial_value = rewriter.getUnitAttr();
    memref::GlobalOp::create(rewriter, loc, rewriter.getStringAttr(name),
                             /* sym_visibility */ mlir::StringAttr(),
                             mlir::TypeAttr::get(type), initial_value,
                             mlir::UnitAttr(), /* alignment */ nullptr);
    rewriter.setInsertionPoint(ao);
    auto getGlobalOp = memref::GetGlobalOp::create(rewriter, loc, type, name);

    rewriter.replaceOp(ao, getGlobalOp->getResults());

    return success();
  }
};

/// TODO implement code motion across the gpu_wrapper, then we would have
/// two options for gpu_wrappers without any parallel ops in them - we
/// could either hoist the computation to the cpu with added cpu-gpu copies or
/// we could run a single iteration gpu kernel - whichever we think might be
/// better for each case
///
/// gpu_wrapper {
///   A()
/// }
/// ->
/// gpu_wrapper {
///   parallel _ = 0 to 1 {
///     parallel _ = 0 to 1 {
///       A()
///     }
///   }
/// }
///
struct CreateParallelOps : public OpRewritePattern<enzymexla::GPUWrapperOp> {
  using OpRewritePattern<enzymexla::GPUWrapperOp>::OpRewritePattern;
  const char *PATTERN = "create-parallel-ops";
  LogicalResult matchAndRewrite(enzymexla::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    scf::ParallelOp pop = nullptr;
    for (Operation &op : *wrapper.getBody()) {
      if (auto p = dyn_cast<scf::ParallelOp>(&op)) {
        pop = p;
      }
    }
    if (pop) {
      LLVM_DEBUG(DBGS() << "parallel already exists\n");
      return failure();
    }
    auto loc = wrapper->getLoc();
    auto terminator = wrapper.getBody()->getTerminator();
    rewriter.setInsertionPoint(wrapper);
    auto zeroindex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto oneindex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value, 1> one(1, oneindex);
    SmallVector<Value, 1> zero(1, zeroindex);
    rewriter.setInsertionPointToEnd(wrapper.getBody());
    auto gridPop = scf::ParallelOp::create(rewriter, loc, zero, one, one);
    rewriter.clone(*terminator);
    rewriter.setInsertionPointToStart(gridPop.getBody());
    auto blockPop = scf::ParallelOp::create(rewriter, loc, zero, one, one);
    rewriter.setInsertionPointToStart(blockPop.getBody());

    SmallVector<Operation *> toErase;
    IRMapping mapping;
    for (Operation &op : *wrapper.getBody()) {
      toErase.push_back(&op);
      if (terminator == &op)
        break;
      rewriter.clone(op, mapping);
    }
    for (Operation *op : llvm::reverse(toErase))
      rewriter.eraseOp(op);

    return success();
  }
};

/// TODO Look or any barriers and if they are we must preserve the threads it
/// syncs at to be block threads
///
/// parallel {
///   A()
/// }
///
/// ->
///
/// parallel {
///   parallel {
///     A()
///   }
/// }
///
///
/// Splitting an iteration variable to grid/block one:
/// parallel i = 0 to i_bound {
///   A(i)
/// }
/// ->
/// parallel i = 0 to i_bound / BLOCK_SIZE {
///   parallel j = 0 to BLOCK_SIZE {
///     A(i * BLOCK_SIZE + j)
///   }
/// }
///
/// Making iteration variables with constant bounds block iteration variables:
/// parallel i = 0 to var_i_bound, j = 0 to const_j_bound {
///   A(i, j)
/// }
/// ->
/// parallel i = 0 to var_i_bound {
///   parallel j = 0 to const_j_bound {
///     A(i, j)
///   }
/// }
///
#if 1
struct SplitParallelOp : public OpRewritePattern<enzymexla::GPUWrapperOp> {
  using OpRewritePattern<enzymexla::GPUWrapperOp>::OpRewritePattern;
  const char *PATTERN = "split-parallel-op";

  // TODO this should differ from arch to arch
  const unsigned MAX_GPU_THREADS = 1024;

  const std::vector<unsigned> ALTERNATIVE_KERNEL_BLOCK_SIZES = {
      32 * 1, 32 * 2, 32 * 4, 32 * 8, 32 * 16, 32 * 32};

  LogicalResult matchAndRewrite(enzymexla::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    scf::ParallelOp pop = getDirectlyNestedSingleParallel(wrapper.getBody());
    if (!pop)
      return failure();
    bool child = false;
    pop->walk([&](scf::ParallelOp p) {
      if (pop != p)
        child = true;
    });
    if (child) {
      LLVM_DEBUG(DBGS() << "only single parallel ops\n");
      return failure();
    }

    auto loc = pop->getLoc();

    int curRegion = 0;
    llvm::SmallSet<int, 6> emittedBlockSizes;
    std::vector<Attribute> descs;
    auto emitAlternative = [&](int defaultThreads,
                               enzymexla::AlternativesOp alternativesOp) {
      auto block = &*alternativesOp->getRegion(curRegion).begin();
      rewriter.setInsertionPointToStart(block);
      // TODO not very efficient...
      auto newWrapper = rewriter.clone(*wrapper.getOperation());
      auto blockSize = createSplitOp(cast<enzymexla::GPUWrapperOp>(newWrapper),
                                     defaultThreads, rewriter);
      if (emittedBlockSizes.contains(blockSize) ||
          /* failed */ blockSize == -1) {
      } else {
        emittedBlockSizes.insert(blockSize);
        descs.push_back(rewriter.getStringAttr(
            std::string("block_size=" + std::to_string(blockSize) + ",")));
        curRegion++;
      }
    };
    auto exactMatch = [&](enzymexla::AlternativesOp alternativesOp) {
      auto block = &*alternativesOp->getRegion(curRegion).begin();
      rewriter.setInsertionPointToStart(block);
      // TODO not very efficient...
      auto newWrapper = cast<enzymexla::GPUWrapperOp>(
          rewriter.clone(*wrapper.getOperation()));
      scf::ParallelOp pop =
          getDirectlyNestedSingleParallel(newWrapper.getBody());
      auto loc = pop->getLoc();

      auto upperBounds = getUpperBounds<6>(pop);

      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(pop);
      auto gridPop = scf::ParallelOp::create(
          rewriter, loc, pop.getLowerBound().slice(0, 3),
          pop.getUpperBound().slice(0, 3), pop.getStep().slice(0, 3));
      rewriter.setInsertionPointToStart(gridPop.getBody());
      auto blockPop = scf::ParallelOp::create(
          rewriter, loc, pop.getLowerBound().slice(3, 3),
          pop.getUpperBound().slice(3, 3), pop.getStep().slice(3, 3));
      rewriter.setInsertionPointToStart(blockPop.getBody());

      IRMapping mapping;
      for (unsigned i = 0; i < 3; i++)
        mapping.map(pop.getBody()->getArgument(i),
                    gridPop.getBody()->getArgument(i));
      for (unsigned i = 0; i < 3; i++)
        mapping.map(pop.getBody()->getArgument(i + 3),
                    blockPop.getBody()->getArgument(i));

      rewriter.eraseOp(pop.getBody()->getTerminator());
      for (auto &op : *pop.getBody())
        rewriter.clone(op, mapping);
      rewriter.eraseOp(pop);
      emittedBlockSizes.insert(-1);
      descs.push_back(rewriter.getStringAttr(
          std::string("block_size=" + std::to_string(-1) + ",")));
      curRegion++;
    };

    if (char *blockSizeStr = getenv("POLYGEIST_GPU_KERNEL_BLOCK_SIZE")) {
      auto alternativesOp = enzymexla::AlternativesOp::create(rewriter, loc, 1);
      alternativesOp->setAttr("alternatives.type",
                              rewriter.getStringAttr("gpu_kernel"));
      llvm::errs() << "Emitting kernel with " << atoi(blockSizeStr)
                   << " threads\n";
      emitAlternative(atoi(blockSizeStr), alternativesOp);
      if (curRegion == 0) {
        llvm::errs() << " Failed to make kernel with exact dimension\n";
        assert(wrapper.getOperands().size() == 6);
        assert(pop.getUpperBound().size() == 6 &&
               pop.getUpperBound() == wrapper.getOperands());
        exactMatch(alternativesOp);
      }
      alternativesOp->setAttr("alternatives.descs",
                              rewriter.getArrayAttr(descs));
    } else if (shouldEmitAlternatives(pop)) {
      auto alternativesOp = enzymexla::AlternativesOp::create(
          rewriter, loc,
          ALTERNATIVE_KERNEL_BLOCK_SIZES.size() +
              (pop.getUpperBound().size() == 6 &&
               pop.getUpperBound() == wrapper.getOperands()));
      alternativesOp->setAttr("alternatives.type",
                              rewriter.getStringAttr("gpu_kernel"));
      for (unsigned blockSize : ALTERNATIVE_KERNEL_BLOCK_SIZES) {
        emitAlternative(blockSize, alternativesOp);
      }
      assert(wrapper.getOperands().size() == 6);
      if (pop.getUpperBound().size() == 6 &&
          pop.getUpperBound() == wrapper.getOperands()) {
        exactMatch(alternativesOp);
      }
      alternativesOp->setAttr("alternatives.descs",
                              rewriter.getArrayAttr(descs));
      shrinkAlternativesOp(alternativesOp, curRegion, rewriter);
    } else {
      auto alternativesOp = enzymexla::AlternativesOp::create(rewriter, loc, 1);
      alternativesOp->setAttr("alternatives.type",
                              rewriter.getStringAttr("gpu_kernel"));
      emitAlternative(-1, alternativesOp);
      alternativesOp->setAttr("alternatives.descs",
                              rewriter.getArrayAttr(descs));
    }

    rewriter.eraseOp(wrapper);

    return success();
  }

  // Get the IVs that are synced over
  llvm::SmallVector<BlockArgument, 3> getSyncIVs(scf::ParallelOp pop) const {
    llvm::SmallVector<BlockArgument, 3> syncIVs;
    pop->walk([&](enzymexla::BarrierOp barrier) {
      for (auto o : barrier.getOperands())
        if (auto ba = dyn_cast<BlockArgument>(o))
          if (std::find(syncIVs.begin(), syncIVs.end(), ba) == syncIVs.end())
            syncIVs.push_back(ba);
    });
    return syncIVs;
  }

  // If the parallel op is suitable for emitting alternatives
  bool shouldEmitAlternatives(scf::ParallelOp pop) const {
    auto dea = pop->getAttrOfType<DenseElementsAttr>(
        "enzymexla.kernel_thread_indices");

    auto syncIVs = getSyncIVs(pop);

    if (!dea && syncIVs.empty())
      return true;

    auto upperBounds = getUpperBounds<6>(pop);

    // If any of the original dimensions were not consts we cannot effectively
    // vary the block size as we cannot make proper assumptions about the
    // resulting block size and transforming what was originally a block dim
    // into a grid dim usually has bad effects on performance
    return (!dea ||
            llvm::all_of(dea.getValues<IntegerAttr>(),
                         [&](auto index_) {
                           auto index = index_.getValue().getLimitedValue();
                           auto cst = getConstantInteger(upperBounds[index]);
                           return cst;
                         })) &&
           llvm::all_of(syncIVs, [&](BlockArgument bo) {
             auto cst = getConstantInteger(upperBounds[bo.getArgNumber()]);
             return cst;
           });
  }

  // If maxThreads == -1, then pick the original block dims, otherwise try to
  // maximize the block size up to maxThreads
  //
  // Returns the resulting block size if it is static; if it is dynamic, returns
  // 0, if we failed to split the parallel op, then returns -1
  int createSplitOp(enzymexla::GPUWrapperOp wrapper, int maxThreads,
                    PatternRewriter &rewriter) const {
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    scf::ParallelOp pop = getDirectlyNestedSingleParallel(wrapper.getBody());

    auto loc = pop->getLoc();

    auto upperBounds = getUpperBounds<6>(pop);
    int totalDims = upperBounds.size();

    SmallVector<Value, 3> blockDims;
    SmallVector<Value, 3> gridDims;
    // Arg ids in the original parallel block
    SmallVector<int, 3> blockArgId;
    SmallVector<int, 3> gridArgId;

    SmallVector<int, 3> originalBlockDimIndices = [&]() {
      SmallVector<int, 3> tmp;
      if (maxThreads == -1) {
        auto dea = pop->getAttrOfType<DenseElementsAttr>(
            "enzymexla.kernel_thread_indices");
        assert(dea);
        for (auto index_ : dea.getValues<IntegerAttr>()) {
          auto index = index_.getValue().getLimitedValue();
          tmp.push_back(index);
        }
      }
      return tmp;
    }();
    auto isOriginalBlockDim = [&](int index) {
      return std::find(originalBlockDimIndices.begin(),
                       originalBlockDimIndices.end(),
                       index) != originalBlockDimIndices.end();
    };

    auto mustBeBlockIVs = getSyncIVs(pop);
    auto isMustBeBlockIV = [&](int index) {
      auto ba = pop.getBody()->getArgument(index);
      return std::find(mustBeBlockIVs.begin(), mustBeBlockIVs.end(), ba) !=
             mustBeBlockIVs.end();
    };

    int threadNum = 1;

    for (int i = totalDims - 1; i >= 0; i--) {
      auto &bound = upperBounds[i];
      int val = 0;
      APInt aval;
      if (matchPattern(bound, m_ConstantInt(&aval)))
        val = (int)aval.getSExtValue();
      if (isMustBeBlockIV(i)) {
        blockDims.insert(blockDims.begin(), bound);
        blockArgId.insert(blockArgId.begin(), i);
        threadNum *= val;
      }
    }
    assert(threadNum >= 0 && threadNum <= (int)MAX_GPU_THREADS);

    if (mustBeBlockIVs.empty()) {
      // TODO We can actually add more block dims even if there were IVs that
      // need to be synced over if we can prove that there will be no barriers
      // in divergent branches (i.e. same loop trip count or if conditions of
      // regions containing barriers, check the parallel loop unroll logic)
      for (int i = totalDims - 1; i >= 0; i--) {
        if (isMustBeBlockIV(i))
          // Already added
          continue;
        auto &bound = upperBounds[i];
        int val = 1;
        APInt aval;
        bool cst = false;
        if (matchPattern(bound, m_ConstantInt(&aval))) {
          val = (int)aval.getSExtValue();
          cst = true;
        }
        bool isBlockDim = [&]() {
          if (maxThreads != -1) {
            return cst && blockDims.size() < 3 && threadNum * val <= maxThreads;
          } else {
            return isOriginalBlockDim(i);
          }
        }();

        if (isBlockDim) {
          blockDims.insert(blockDims.begin(), bound);
          blockArgId.insert(blockArgId.begin(), i);
          threadNum *= val;
        } else {
          gridDims.insert(gridDims.begin(), bound);
          gridArgId.insert(gridArgId.begin(), i);
        }
      }
    } else {
      for (int i = totalDims - 1; i >= 0; i--) {
        if (isMustBeBlockIV(i))
          // Already added
          continue;
        auto &bound = upperBounds[i];
        gridDims.insert(gridDims.begin(), bound);
        gridArgId.insert(gridArgId.begin(), i);
      }
    }

    // TODO if we have too many dims, we have to merge some of them - currently
    // unsupported - we need to have some kernel structure preservation to
    // support all cases currently
    if (gridDims.size() > 3) {
      rewriter.setInsertionPoint(wrapper);
      auto err = arith::ConstantIndexOp::create(
          rewriter, loc, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
      rewriter.replaceOp(wrapper, err->getResults());
      return -1;
    }

    rewriter.setInsertionPoint(pop);
    auto zeroindex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto oneindex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    unsigned splitDims = 0;
    SmallVector<int, 3> gi;
    SmallVector<int, 3> bi;
    if (gridDims.size() == 0) {
      gridDims.push_back(oneindex);
      // Put a random index, we will override it
      gridArgId.push_back(0);
    } else if (maxThreads != -1 && threadNum <= maxThreads / 2 &&
               mustBeBlockIVs.empty()) {
      // If we are not getting enough parallelism in the block, use part of the
      // grid dims

      // TODO we have to be careful to not exceed max z dimension in block, it
      // is lower than the 1024 max for the x and y

      // TODO we can actually generate multiple kernels here and dynamically
      // split from the grid dimension that has enough parallelism in it

      unsigned threadsLeft =
          (llvm::bit_floor(static_cast<unsigned>(maxThreads) /
                           static_cast<unsigned>(threadNum)));
      threadNum *= threadsLeft;
      assert(threadNum <= maxThreads);

      // TODO what should the number here be
      // How many dims to take from the grid
      splitDims = 1;
      assert(splitDims <= gridDims.size());
      assert(splitDims + blockDims.size() <= 3);

      // Which grid dims to take
      for (unsigned i = 0; i < splitDims; i++)
        gi.push_back(gridDims.size() - 1 - i);
      // Which block dims they correspond to
      for (unsigned i = 0; i < splitDims; i++) {
        bi.push_back(i);
        blockArgId.insert(blockArgId.begin(), gridArgId[gi[i]]);
      }

      SmallVector<Value, 3> newBlockDims;
      // TODO try our best to make them divisors of the gridDims
      rewriter.setInsertionPoint(pop);
      if (splitDims == 1)
        newBlockDims = {
            arith::ConstantIndexOp::create(rewriter, loc, threadsLeft),
        };
      else if (splitDims == 2)
        // TODO
        assert(0);
      else if (splitDims == 3)
        // TODO
        assert(0);
      else
        assert(0);
      newBlockDims.insert(newBlockDims.end(), blockDims.begin(),
                          blockDims.end());

      for (unsigned i = 0; i < splitDims; i++) {
        // newGridDims[j] = ((gridDims[j] - 1) / newBlockDims[i]) + 1;
        auto sub =
            arith::SubIOp::create(rewriter, loc, gridDims[gi[i]], oneindex);
        auto div =
            arith::DivUIOp::create(rewriter, loc, sub, newBlockDims[bi[i]]);
        gridDims[gi[i]] = arith::AddIOp::create(rewriter, loc, div, oneindex);
      }
      blockDims = newBlockDims;
    }

    LLVM_DEBUG(DBGS() << "converting to block with threadNum: " << threadNum
                      << ", dims: " << blockDims.size() << "\n";);

    SmallVector<Value, 3> lowerBoundsGrid(gridDims.size(), zeroindex);
    SmallVector<Value, 3> stepsGrid(gridDims.size(), oneindex);
    SmallVector<Value, 3> lowerBoundsBlock(blockDims.size(), zeroindex);
    SmallVector<Value, 3> stepsBlock(blockDims.size(), oneindex);

    rewriter.setInsertionPoint(pop);
    auto gridPop = scf::ParallelOp::create(rewriter, loc, lowerBoundsGrid,
                                           gridDims, stepsGrid);
    rewriter.setInsertionPointToStart(gridPop.getBody());
    auto blockPop = scf::ParallelOp::create(rewriter, loc, lowerBoundsBlock,
                                            blockDims, stepsBlock);
    rewriter.setInsertionPointToStart(blockPop.getBody());

    IRMapping mapping;
    for (unsigned i = 0; i < gridDims.size(); i++)
      mapping.map(pop.getBody()->getArgument(gridArgId[i]),
                  gridPop.getBody()->getArgument(i));
    for (unsigned i = 0; i < blockDims.size(); i++)
      mapping.map(pop.getBody()->getArgument(blockArgId[i]),
                  blockPop.getBody()->getArgument(i));

    // For the split dims, calculate the equivalent threadId and map that
    // instead
    if (splitDims > 0) {
      Value cond;
      // SmallVector<Value, 3> threadId(splitDims);
      for (unsigned i = 0; i < splitDims; i++) {
        // threadIndex = blockIdx * blockDim + threadIdx
        // threadIndex < original upperBound
        //
        // Currently we do not care if the split dim correspond to the same
        // block/thread index, so we might do something like blockIdx.x *
        // blockDim.x + threadIdx.y, should we try to rearrange dims to match
        // them?
        auto mul = arith::MulIOp::create(rewriter, loc,
                                         gridPop.getBody()->getArgument(gi[i]),
                                         blockDims[bi[i]]);
        auto threadId = arith::AddIOp::create(
            rewriter, loc, mul, blockPop.getBody()->getArgument(bi[i]));
        assert(blockArgId[bi[i]] == gridArgId[gi[i]]);
        mapping.map(pop.getBody()->getArgument(gridArgId[gi[i]]), threadId);
        auto threadCond =
            arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                  threadId, upperBounds[gridArgId[gi[i]]]);
        if (i == 0)
          cond = threadCond.getResult();
        else
          cond = arith::AndIOp::create(rewriter, loc, threadCond, cond)
                     .getResult();
      }
      auto ifOp = scf::IfOp::create(rewriter, loc, cond);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
    }

    rewriter.eraseOp(pop.getBody()->getTerminator());
    for (auto &op : *pop.getBody())
      rewriter.clone(op, mapping);

    rewriter.eraseOp(pop);

    return threadNum;
  }
};
#endif

// TODO handle something like this if it happens
//
// scf.parallel {
//   scf.parallel {
//     A()
//   }
//   scf.parallel {
//     B()
//   }
// }
//

/// scf.parallel {
///   A()
///   scf.parallel {
///     B()
///   }
///   C()
/// }
///
/// ->
///
/// scf.parallel {
///   scf.parallel {
///     A'()
///     barrier
///     B()
///     barrier
///     C'()
///   }
/// }
struct ParallelizeBlockOps : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  const char *PATTERN = "parallelize-block-ops";
  LogicalResult matchAndRewrite(scf::ParallelOp pop,
                                PatternRewriter &rewriter) const override {
    if (!pop->getParentOfType<scf::ParallelOp>()) {
      LLVM_DEBUG(DBGS() << "ignoring non nested parallel op\n");
      return failure();
    }
    auto loc = pop->getLoc();
    Block *outerBlock = pop->getBlock();
    Block *innerBlock = pop.getBody();

    if (getDirectlyNestedSingleParallel(outerBlock, /* allowAllocas */ true)) {
      LLVM_DEBUG(DBGS() << "no ops to parallelize\n");
      return failure();
    }

    // Handle ops before the parallel
    scf::IfOp ifOp = nullptr;
    auto getIf = [&]() {
      if (!ifOp) {
        auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value cond;
        for (unsigned i = 0; i < innerBlock->getNumArguments(); i++) {
          auto threadId = innerBlock->getArgument(i);
          auto threadCond = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, threadId, zero);
          if (i == 0)
            cond = threadCond.getResult();
          else
            cond = arith::AndIOp::create(rewriter, loc, threadCond, cond)
                       .getResult();
        }
        ifOp = scf::IfOp::create(rewriter, loc, cond);
        mlir::enzymexla::BarrierOp::create(rewriter, loc,
                                           innerBlock->getArguments());
        rewriter.setInsertionPoint(ifOp);
      }
    };

    rewriter.setInsertionPointToStart(innerBlock);
    auto it = outerBlock->begin();
    auto end = outerBlock->getTerminator()->getIterator();
    SmallVector<Operation *> toErase;
    IRMapping mapping;
    for (; &*it != pop.getOperation(); ++it) {
      Operation &op = *it;
      Operation *newOp;
      if (isa<scf::ParallelOp>(&op)) {
        llvm_unreachable("Unhandled case");
        break;
      } else if (it == end) {
        llvm_unreachable("Impossible");
        continue;
      } else if (auto alloca = dyn_cast<memref::AllocaOp>(&op)) {
        continue;
      } else if (auto alloca = dyn_cast<LLVM::AllocaOp>(&op)) {
        continue;
      } else {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        SmallVector<MemoryEffects::EffectInstance> effects;
        collectEffects(&op, effects, /*ignoreBarriers*/ false);
        if (effects.empty()) {
        } else if (hasEffect<MemoryEffects::Allocate>(effects)) {
          llvm_unreachable("??");
        } else if (hasEffect<MemoryEffects::Free>(effects)) {
          llvm_unreachable("??");
        } else if (hasEffect<MemoryEffects::Write>(effects)) {
          getIf();
          assert(ifOp);
          rewriter.setInsertionPoint(ifOp.thenBlock()->getTerminator());
          // TODO currently we assume that ops with write effects will have no
          // uses - we have to introduce shared mem otherwise
          if (!op.use_empty()) {
            llvm_unreachable("could not fix parallel fusion");
          }
        } else if (hasEffect<MemoryEffects::Read>(effects)) {
          // Reads-only ops are legal to parallelize
        }
        newOp = rewriter.clone(op, mapping);
      }
      rewriter.replaceOpUsesWithinBlock(&op, newOp->getResults(), innerBlock);
      toErase.push_back(&op);
    }
    it++;

    // Handle ops after the parallel
    {
      auto zeroindex = arith::ConstantIndexOp::create(rewriter, loc, 0);
      rewriter.setInsertionPoint(innerBlock->getTerminator());
      auto cmpOp =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                zeroindex, innerBlock->getArgument(0));
      Value condition = cmpOp.getResult();
      for (unsigned i = 1; i < innerBlock->getNumArguments(); i++) {
        auto cmpOp2 =
            arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                  zeroindex, innerBlock->getArgument(i));
        auto andOp = arith::AndIOp::create(rewriter, loc, condition, cmpOp2);
        condition = andOp.getResult();
      }
      auto ifOp = scf::IfOp::create(rewriter, loc, condition);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      for (; it != end; ++it) {
        Operation &op = *it;
        if (isa<scf::ParallelOp>(&op)) {
          llvm_unreachable("Unhandled case");
          break;
        } else if (auto alloca = dyn_cast<memref::AllocaOp>(&op)) {
          llvm_unreachable("Unhandled case");
          break;
        } else if (auto alloca = dyn_cast<LLVM::AllocaOp>(&op)) {
          llvm_unreachable("Unhandled case");
          break;
        } else {
          rewriter.clone(op, mapping);
        }
        toErase.push_back(&op);
      }
    }

    for (Operation *op : llvm::reverse(toErase))
      rewriter.eraseOp(op);

    return success();
  }
};

bool hasNestedParallel(Operation *topLevelOp) {
  auto walkRes = topLevelOp->walk(
      [&](scf::ParallelOp) { return WalkResult::interrupt(); });
  return walkRes.wasInterrupted();
}

/// If we find an alloca at top level in the wrapper it means (currently at
/// least, as we are only lowering cuda kernels to wrapped parallels and nothing
/// else) that that alloca is shared mem allocation and the single trip grid
/// parallel was removed - this pass restores it
struct HandleWrapperRootAlloca
    : public OpRewritePattern<enzymexla::GPUWrapperOp> {
  using OpRewritePattern<enzymexla::GPUWrapperOp>::OpRewritePattern;

  const char *PATTERN = "handle wrapper root alloca";
  LogicalResult matchAndRewrite(enzymexla::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    auto loc = wrapper->getLoc();
    auto wrapperBody = wrapper.getBody();
    if (!hasNestedParallel(wrapper)) {
      LLVM_DEBUG(DBGS() << "wrapper has no parallel\n");
      return failure();
    }
    bool allocFound = false;
    for (Operation &op : *wrapperBody) {
      if (isa<memref::AllocaOp>(&op)) {
        allocFound = true;
        break;
      }
    }
    if (!allocFound) {
      LLVM_DEBUG(DBGS() << "no alloc in \n");
      return failure();
    }

    auto terminator = wrapper.getBody()->getTerminator();
    rewriter.setInsertionPoint(wrapper);
    auto zeroindex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto oneindex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value, 1> one(1, oneindex);
    SmallVector<Value, 1> zero(1, zeroindex);
    rewriter.setInsertionPointToEnd(wrapper.getBody());
    auto gridPop = scf::ParallelOp::create(rewriter, loc, zero, one, one);
    rewriter.clone(*terminator);
    rewriter.setInsertionPointToStart(gridPop.getBody());

    SmallVector<Operation *> toErase;
    IRMapping mapping;
    for (Operation &op : *wrapper.getBody()) {
      toErase.push_back(&op);
      if (terminator == &op)
        break;
      rewriter.clone(op, mapping);
    }
    for (Operation *op : llvm::reverse(toErase))
      rewriter.eraseOp(op);

    return success();
  }
};

// TODO
// this doesnt work if we actually have two parallels like this:
//
// gpu_wrapper {
//   A()
//   parallel {
//   }
//   parallel {
//   }
// }
//

/// gpu_wrapper {
///   A()
///   parallel {
///     ...
///   }
///   ...
/// }
/// ->
/// A1()
/// gpu_wrapper {
///   A2()
/// }
/// gpu_wrapper {
///   parallel {
///     A3()
///     ...
///   }
///   ...
/// }
struct HandleWrapperRootOps : public OpRewritePattern<enzymexla::GPUWrapperOp> {
  using OpRewritePattern<enzymexla::GPUWrapperOp>::OpRewritePattern;

  const char *PATTERN = "handle-wrapper-root-ops";
  LogicalResult matchAndRewrite(enzymexla::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    auto loc = wrapper->getLoc();
    auto wrapperBody = wrapper.getBody();
    auto it = wrapperBody->begin();
    if (isa<scf::ParallelOp>(&*it)) {
      LLVM_DEBUG(DBGS() << "first op is a parellel\n");
      return failure();
    }
    SmallVector<Operation *> toHandle;
    Operation *pop;
    Operation *firstGridOp;
    for (;; ++it) {
      if (&*it == wrapperBody->getTerminator())
        return failure();
      if (hasNestedParallel(&*it) && isa<scf::ParallelOp, scf::IfOp>(&*it)) {
        pop = &*it;
        // TODO handle ifs with elses
        if (auto ifOp = dyn_cast<scf::IfOp>(&*it))
          assert(ifOp.getElseRegion().empty());
        firstGridOp = &*pop->getRegion(0).begin()->begin();
        break;
      }
      toHandle.push_back(&*it);
    }
    if (toHandle.size() == 0) {
      LLVM_DEBUG(DBGS() << "empty wrapper\n");
      return failure();
    }
    rewriter.setInsertionPoint(wrapper);
    auto newWrapper =
        enzymexla::GPUWrapperOp::create(rewriter, loc, wrapper.getOperands());
    IRMapping hoistMapping;
    IRMapping splitMapping;
    IRMapping parallelizedMapping;
    for (Operation *op : toHandle) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      collectEffects(op, effects, /*ignoreBarriers*/ false);
      bool read = hasEffect<MemoryEffects::Read>(effects);
      bool write = hasEffect<MemoryEffects::Write>(effects);
      SmallVector<Value, 1> cloned;
      // Special case for get_global because what if actually refers to is the
      // device-side global, so this must remain in the gpu wrapper
      if (isa<memref::GetGlobalOp>(op)) {
        // This is the same as the case for a parallelizable read op
        rewriter.setInsertionPoint(newWrapper.getBody()->getTerminator());
        rewriter.clone(*op, splitMapping);
        rewriter.setInsertionPoint(firstGridOp);
        cloned = rewriter.clone(*op, parallelizedMapping)->getResults();
      } else if (effects.empty()) {
        rewriter.setInsertionPoint(firstGridOp);
        rewriter.clone(*op, parallelizedMapping);
        rewriter.setInsertionPoint(newWrapper.getBody()->getTerminator());
        rewriter.clone(*op, splitMapping);
        rewriter.setInsertionPoint(newWrapper);
        cloned = rewriter.clone(*op, hoistMapping)->getResults();
      } else if (hasEffect<MemoryEffects::Allocate>(effects)) {
        // I think this can actually happen if we lower a kernel with a barrier
        // and shared memory with gridDim = 1 TODO handle
        llvm_unreachable("what?");
      } else if (hasEffect<MemoryEffects::Free>(effects)) {
        llvm_unreachable("what?");
      } else if (write) {
        rewriter.setInsertionPoint(newWrapper.getBody()->getTerminator());
        cloned = rewriter.clone(*op, splitMapping)->getResults();
        // TODO if we find that this has results that get used later in the
        // final parallel we need to introduce temporary gpu cache memory to
        // pass it on
      } else if (read) {
        // Check if we can safely put the read in the grid parallel op, i.e. the
        // ops up to and including the next parallel op may not write to where
        // we read from

        // TODO for recursive mem effects ops, try to collect all memrefs we
        // load from and do the checks on them
        bool canParallelize = true;
        if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
          auto loadMemRef = loadOp.getMemref();
          Operation *op = loadOp;
          while (op != pop) {
            op = op->getNextNode();
            if (mayWriteTo(op, loadMemRef, /*ignoreBarrier*/ false)) {
              canParallelize = false;
            }
          }
        } else {
          canParallelize = false;
        }
        if (canParallelize) {
          rewriter.setInsertionPoint(newWrapper.getBody()->getTerminator());
          rewriter.clone(*op, splitMapping);
          rewriter.setInsertionPoint(firstGridOp);
          cloned = rewriter.clone(*op, parallelizedMapping)->getResults();
        } else {
          // If it is not used beyond the parallel, we can just put it out in
          // the newWrapper
          bool usedOnlyBeforePop = true;
          for (auto v : op->getResults()) {
            for (auto &u : llvm::make_early_inc_range(v.getUses())) {
              auto *user = u.getOwner();
              while (user->getBlock() != pop->getBlock())
                user = user->getBlock()->getParentOp();
              if (!user->isBeforeInBlock(pop)) {
                usedOnlyBeforePop = false;
                break;
              }
            }
          }
          if (usedOnlyBeforePop) {
            rewriter.setInsertionPoint(newWrapper.getBody()->getTerminator());
            // We will be trying to replace uses of this in the pop but it does
            // not matter as we confirmed there are none
            cloned = rewriter.clone(*op, splitMapping)->getResults();
          } else {
            rewriter.setInsertionPoint(newWrapper.getBody()->getTerminator());
            auto clonedOp = rewriter.clone(*op, splitMapping);

            // TODO it might be better to load this from the host and pass it as
            // a parameter
            SmallVector<Value, 1> cacheLoads;
            cacheLoads.reserve(op->getNumResults());
            for (auto v : clonedOp->getResults()) {
              rewriter.setInsertionPoint(newWrapper);
              auto mt = MemRefType::get({}, v.getType());
              // TODO we never actually free this...
              auto alloc = gpu::AllocOp::create(
                  rewriter, loc, mt, /* asyncToken type */ nullptr,
                  /* TODO asyncDependencies */ ValueRange(),
                  /* dynamicSizes */ ValueRange(),
                  /* symbolOperands */ ValueRange());

              rewriter.setInsertionPoint(newWrapper.getBody()->getTerminator());
              memref::StoreOp::create(rewriter, loc, v, alloc.getMemref());

              rewriter.setInsertionPoint(firstGridOp);
              cacheLoads.push_back(
                  memref::LoadOp::create(rewriter, loc, alloc.getMemref()));
            }

            cloned = cacheLoads;
          }
        }
      } else {
        llvm_unreachable("are there other effects?");
      }
      rewriter.replaceUsesWithIf(op->getResults(), cloned, [&](OpOperand &use) {
        Operation *owner = use.getOwner();
        while (owner->getBlock() != pop->getBlock())
          owner = owner->getParentOp();
        return pop->getPrevNode()->isBeforeInBlock(owner);
      });
    }
    for (Operation *op : llvm::reverse(toHandle)) {
      assert(op->use_empty());
      rewriter.eraseOp(op);
    }

    return success();
  }
};

/// Removes the enzymexla.noop ops that is used to prevent optimizations from
/// changing the GPU kernel structure
///
/// Currently, this is the structure we expect when we encounter a noop:
///
/// gpu_wrapper {
///   ...
///   parallel bix, biy, biz {
///     noop(bix, biy, biz)
///     ...
///     parallel tix, tiy, tiz {
///       noop(tix, tiy, tiz)
///       ...
///     }
///   }
/// }
///
/// or:
///
/// gpu_wrapper {
///   ...
///   parallel bix, biy, biz {
///     ...
///     parallel tix, tiy, tiz {
///       noop(tix, tiy, tiz)
///       ...
///     }
///   }
/// }
///
/// or in case the block and thread parallels were merged:
///
/// gpu_wrapper {
///   ...
///   parallel bix, [biy, biz,] tix, [tiy, tiz] {
///       noop(tix, tiy, tiz)
///       ...
///     }
///   }
/// }
struct RemovePolygeistNoopOp : public OpRewritePattern<enzymexla::NoopOp> {
  using OpRewritePattern<enzymexla::NoopOp>::OpRewritePattern;
  const char *PATTERN = "remove-enzymexla-noop";
  LogicalResult matchAndRewrite(enzymexla::NoopOp noop,
                                PatternRewriter &rewriter) const override {
    if (!noop->getParentOfType<enzymexla::GPUWrapperOp>()) {
      LLVM_DEBUG(DBGS() << "not in a gpu wrapper\n");
      return failure();
    }
    auto noopType =
        noop->getAttrOfType<StringAttr>("enzymexla.noop_type").getValue();
    if (!noopType.starts_with("gpu_kernel.")) {
      LLVM_DEBUG(DBGS() << "noop does not have the appropriate attribute\n");
      return failure();
    }
    // TODO rotate the parallel loop dims in the order they appear in the
    // noop op to completely restore the original structure
    auto loc = noop->getLoc();
    // If all of the operands to the block/thread noop are not block args
    // that means the parallel loop whose indices were used for the operands got
    // optimized away (trip count = 1), reinsert it
    if (llvm::all_of(noop.getOperands(),
                     [](Value v) { return !isa<BlockArgument>(v); })) {
      // TODO check that the args _are actually_ constants = 1
      Block *block = noop->getBlock();
      auto term = block->getTerminator();
      rewriter.setInsertionPointToStart(block);
      auto zeroindex = arith::ConstantIndexOp::create(rewriter, loc, 0);
      auto oneindex = arith::ConstantIndexOp::create(rewriter, loc, 1);
      SmallVector<Value, 1> one(1, oneindex);
      SmallVector<Value, 1> zero(1, zeroindex);
      auto pop = scf::ParallelOp::create(rewriter, loc, zero, one, one);

      Operation *toClone = pop->getNextNode();
      SmallVector<Operation *> toErase;
      IRMapping mapping;
      rewriter.setInsertionPointToStart(pop.getBody());
      while (toClone != term) {
        Operation *cloned = rewriter.clone(*toClone, mapping);
        toErase.push_back(toClone);
        if (toClone == noop.getOperation())
          noop = cast<enzymexla::NoopOp>(cloned);
        toClone = toClone->getNextNode();
      }
      for (Operation *op : llvm::reverse(toErase))
        rewriter.eraseOp(op);
    }

    // If the type is gpu_kernel.thread_only we need to indicate which of the
    // parallel dims were originally the block dims in case the block and thread
    // parallel ops got merged
    //
    // If the type is gpu_kernel.thread, then that means we have inserted
    // gpu_kernel.block noop as well to prevent the merge so we already have the
    // information
    if (noopType == "gpu_kernel.thread_only") {
      // Find which parallel op args originally correspond to thread indices
      assert(noop.getNumOperands() == 3 &&
             "noop must have 3 args corresponding to the thread indices");
      scf::ParallelOp pop = nullptr;
      SmallVector<int, 3> threadIndices;
      for (auto operand : noop.getOperands()) {
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (auto _pop = dyn_cast<scf::ParallelOp>(
                  blockArg.getOwner()->getParentOp())) {
            if (!pop)
              pop = _pop;
            assert(_pop == pop && "noop operands take thread indices so they "
                                  "must belong to the same parallel op");
            threadIndices.push_back(blockArg.getArgNumber());
          } else {
            llvm_unreachable(
                "noop block arg operands must be scf parallel op args");
          }
        } else {
          auto cst = getConstantInteger(operand);
          (void)cst;
          assert(cst && *cst == 0 && "non block arg operands must be const 0");
        }
      }
      pop->setAttr("enzymexla.kernel_thread_indices",
                   rewriter.getI32VectorAttr(threadIndices));
    }

    rewriter.eraseOp(noop);
    return success();
  }
};

/// Removes the enzymexla.gpu_{block,thread} ops that are used to prevent
/// optimizations from changing the GPU kernel structure
template <typename OpType>
struct RemovePolygeistGPUWrapperOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  const char *PATTERN = "remove-enzymexla-gpu-wrapper-op";
  LogicalResult matchAndRewrite(OpType wrapper,
                                PatternRewriter &rewriter) const override {
    // TODO rotate the parallel loop dims in the order they appear in the
    // wrapper op to completely restore the original structure
    auto loc = wrapper->getLoc();
    // If all of the operands to the block/thread wrapper are not block args
    // that means the parallel loop whose indices were used for the operands got
    // optimized away (trip count = 1), reinsert it
    if (llvm::all_of(wrapper.getOperands(),
                     [](Value v) { return !isa<BlockArgument>(v); }) &&
        !isa<scf::ParallelOp>(wrapper->getParentOp())) {
      // TODO check that the args _are actually_ constants = 1
      Block *block = wrapper->getBlock();
      auto term = block->getTerminator();
      rewriter.setInsertionPointToStart(block);
      auto zeroindex = arith::ConstantIndexOp::create(rewriter, loc, 0);
      auto oneindex = arith::ConstantIndexOp::create(rewriter, loc, 1);
      SmallVector<Value, 1> one(1, oneindex);
      SmallVector<Value, 1> zero(1, zeroindex);
      auto pop = scf::ParallelOp::create(rewriter, loc, zero, one, one);

      Operation *toClone = pop->getNextNode();
      SmallVector<Operation *> toErase;
      IRMapping mapping;
      rewriter.setInsertionPointToStart(pop.getBody());
      while (toClone != term) {
        Operation *cloned = rewriter.clone(*toClone, mapping);
        toErase.push_back(toClone);
        if (toClone == wrapper.getOperation())
          wrapper = cast<OpType>(cloned);
        toClone = toClone->getNextNode();
      }
      for (Operation *op : llvm::reverse(toErase))
        rewriter.eraseOp(op);
    }
    rewriter.eraseOp(wrapper.getBody()->getTerminator());
    rewriter.setInsertionPoint(wrapper);
    rewriter.inlineBlockBefore(wrapper.getBody(), wrapper);
    rewriter.eraseOp(wrapper);
    return success();
  }
};

struct InterchangeIfOp : public OpRewritePattern<enzymexla::GPUWrapperOp> {
  using OpRewritePattern<enzymexla::GPUWrapperOp>::OpRewritePattern;
  const char *PATTERN = "interchange-if-op";
  LogicalResult matchAndRewrite(enzymexla::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    auto loc = wrapper->getLoc();
    auto wrapperBody = wrapper.getBody();
    auto ifOp = dyn_cast<scf::IfOp>(&*wrapperBody->begin());
    if (!ifOp) {
      LLVM_DEBUG(DBGS() << "first op is not an if\n");
      return failure();
    }
    if (&*std::prev(wrapperBody->end(), 2) != ifOp.getOperation()) {
      LLVM_DEBUG(DBGS() << "if is not the only op\n");
      return failure();
    }

    // TODO Currently it has to be the only remaining op in the wrapper
    // and we assume it only has a then
    assert(ifOp.getElseRegion().empty());
    rewriter.setInsertionPoint(wrapper);
    auto newIf = rewriter.cloneWithoutRegions(ifOp);
    newIf.getThenRegion().push_back(new Block());
    rewriter.setInsertionPointToStart(&*newIf.getThenRegion().begin());
    auto newWrapper = rewriter.cloneWithoutRegions(wrapper);
    scf::YieldOp::create(rewriter, loc);
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), newWrapper.getRegion(),
                                newWrapper.getRegion().end());
    rewriter.eraseOp(newWrapper.getBody()->getTerminator());
    rewriter.setInsertionPointToEnd(newWrapper.getBody());
    enzymexla::PolygeistYieldOp::create(rewriter, loc);

    rewriter.eraseOp(wrapper);

    return success();
  }
};

/// gpu_wrapper {
///   parallel {
///     ...
///   }
///   A()
/// }
/// ->
/// gpu_wrapper {
///   parallel {
///     ...
///   }
/// }
/// gpu_wrapper {
///   A()
/// }
struct SplitOffParallel : public OpRewritePattern<enzymexla::GPUWrapperOp> {
  using OpRewritePattern<enzymexla::GPUWrapperOp>::OpRewritePattern;
  const char *PATTERN = "split-off-parallel";
  LogicalResult matchAndRewrite(enzymexla::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    auto loc = wrapper->getLoc();
    auto pop = dyn_cast<scf::ParallelOp>(&(*wrapper.getBody()->begin()));
    if (!pop) {
      LLVM_DEBUG(DBGS() << "first op is not a parellel\n");
      return failure();
    }
    if (pop->getNextNode() == wrapper.getBody()->getTerminator()) {
      LLVM_DEBUG(DBGS() << "pop is the only op in the block\n");
      return failure();
    }
    assert(pop->getNumResults() == 0);

    rewriter.setInsertionPoint(wrapper);
    auto newWrapper =
        enzymexla::GPUWrapperOp::create(rewriter, loc, wrapper.getOperands());
    rewriter.setInsertionPointToStart(newWrapper.getBody());
    rewriter.clone(*pop.getOperation());
    rewriter.eraseOp(pop);
    return success();
  }
};

/// gpu_wrapper {
///   parallel grid_bounds {
///     parallel block_bounds {
///       A()
///     }
///   }
/// }
/// ->
/// gpu.launch grid_bounds, block_bounds {
///   A()
/// }
struct ParallelToGPULaunch : public OpRewritePattern<enzymexla::GPUWrapperOp> {
  using OpRewritePattern<enzymexla::GPUWrapperOp>::OpRewritePattern;
  const char *PATTERN = "parallel-to-gpu-launch";
  LogicalResult matchAndRewrite(enzymexla::GPUWrapperOp wrapper,
                                PatternRewriter &rewriter) const override {
    auto loc = wrapper->getLoc();
    if (wrapper->getParentOfType<enzymexla::GPUWrapperOp>()) {
      LLVM_DEBUG(DBGS() << "[pop-to-launch] ignoring nested parallel op\n");
      return failure();
    }
    rewriter.setInsertionPoint(wrapper);
    auto oneindex = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // TODO we currently assume that all parallel ops we encouter are already
    // prepared for conversion to gpu.launch, i.e. two nested parallel loops
    // with lower bounds zero and constant upper bounds for the inner parallel,
    // the memory they use is on the gpu, are there more conditions?
    scf::ParallelOp gridPop =
        getDirectlyNestedSingleParallel(wrapper.getBody());
    if (!gridPop)
      return failure();
    scf::ParallelOp blockPop = getDirectlyNestedSingleParallel(
        gridPop.getBody(), /* allowAllocas */ true);
    if (!blockPop)
      return failure();

    rewriter.setInsertionPoint(wrapper);
    auto errOp = enzymexla::GPUErrorOp::create(rewriter, loc);

    for (auto atname : {"passthrough", "target_features"})
      if (auto attr = wrapper->getAttr(atname)) {
        errOp->setAttr(atname, attr);
      }
    rewriter.setInsertionPointToStart(errOp.getBody());
    rewriter.eraseOp(wrapper.getBody()->getTerminator());
    rewriter.inlineBlockBefore(wrapper.getBody(),
                               errOp.getBody()->getTerminator());
    rewriter.replaceOp(wrapper, errOp->getResults());

    // TODO make sure we start at zero or else convert the parallel ops to start
    // at 0
    Value gridBounds[3];
    auto popGridBounds = getUpperBounds(gridPop);
    for (unsigned int i = 0; i < 3; i++) {
      if (i < popGridBounds.size())
        gridBounds[i] = popGridBounds[i];
      else
        gridBounds[i] = oneindex;
    }
    Value blockBounds[3];
    auto popBlockBounds = getUpperBounds(blockPop);
    for (unsigned int i = 0; i < 3; i++) {
      if (i < popBlockBounds.size())
        blockBounds[i] = popBlockBounds[i];
      else
        blockBounds[i] = oneindex;
    }

    // TODO handle stream and dependencies - we would have to convert an
    // async{parallel {parallel {}}} to a gpu.launch
    // TODO handle dyn shmem
    rewriter.setInsertionPoint(gridPop);

    // Launch only if the grid size > 0 because launching with 0 blocks is an
    // error in cuda (we do not need the same for blocks because this only
    // happens when we coarsen the blocks with an epilogue loop)
    auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto cond0 = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                       gridBounds[0], one);
    auto cond1 = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                       gridBounds[1], one);
    auto cond2 = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                       gridBounds[2], one);
    auto cond = arith::AndIOp::create(rewriter, loc, cond0, cond1);
    cond = arith::AndIOp::create(rewriter, loc, cond, cond2);
    auto ifOp = scf::IfOp::create(rewriter, loc, cond, /*hasElse*/ false);

    rewriter.setInsertionPointToStart(&*ifOp.getThenRegion().begin());
    auto launchOp = gpu::LaunchOp::create(
        rewriter, loc, gridBounds[0], gridBounds[1], gridBounds[2],
        blockBounds[0], blockBounds[1], blockBounds[2],
        /*dynamic shmem size*/ nullptr,
        /*token type*/ nullptr,
        /*dependencies*/ SmallVector<Value, 1>());

    auto getDim = [](unsigned index) {
      // TODO what should the order be
      if (index == 0)
        return gpu::Dimension::x;
      if (index == 1)
        return gpu::Dimension::y;
      if (index == 2)
        return gpu::Dimension::z;
      llvm_unreachable("Invalid index");
      return gpu::Dimension::z;
    };

    auto launchBlock = &launchOp.getRegion().front();
    rewriter.setInsertionPointToStart(launchBlock);
    SmallVector<Value, 3> argReplacements;
    for (auto en : llvm::enumerate(blockPop.getBody()->getArguments())) {
      gpu::Dimension dim = getDim(en.index());
      auto blockIdx = gpu::ThreadIdOp::create(
          rewriter, loc, mlir::IndexType::get(rewriter.getContext()), dim);
      argReplacements.push_back(blockIdx);
    }
    rewriter.mergeBlocks(blockPop.getBody(), launchBlock, argReplacements);
    rewriter.setInsertionPointToStart(launchBlock);
    for (auto it = gridPop.begin(); !isa<scf::ParallelOp>(&*it); it++) {
      if (auto alloca = dyn_cast<memref::AllocaOp>(&*it)) {
        auto mt = alloca.getType();
        auto type = MemRefType::get(mt.getShape(), mt.getElementType(), {},
                                    /* memspace */ 5);
        auto newAlloca =
            memref::AllocaOp::create(rewriter, alloca.getLoc(), type);
        auto cast = memref::CastOp::create(rewriter, alloca.getLoc(),
                                           alloca.getType(), newAlloca);
        it->replaceAllUsesWith(cast);
      } else {
        assert(0);
      }
    }
    rewriter.setInsertionPointToStart(launchBlock);

    for (auto en : llvm::enumerate(gridPop.getBody()->getArguments())) {
      gpu::Dimension dim = getDim(en.index());
      auto gridIdx = gpu::BlockIdOp::create(
          rewriter, loc, mlir::IndexType::get(rewriter.getContext()), dim);
      en.value().replaceAllUsesWith(gridIdx);
      argReplacements.push_back(gridIdx);
    }

    rewriter.setInsertionPointToEnd(launchBlock);
    gpu::TerminatorOp::create(rewriter, loc);

    rewriter.eraseOp(gridPop);

    Operation *yieldOp = nullptr;
    for (auto &op : *launchBlock) {
      if (auto y = dyn_cast<scf::ReduceOp>(&op)) {
        assert(!yieldOp && "Multiple yields in the final block? why?");
        yieldOp = y;
      }
    }
    rewriter.eraseOp(yieldOp);

    launchBlock->walk([&](mlir::enzymexla::BarrierOp op) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<gpu::BarrierOp>(op);
    });

    return success();

    // enzymexla::BarrierOp barrier = nullptr;
    // std::vector<BlockArgument> barrierArgs;
    // gridPop->walk([&](enzymexla::BarrierOp b) {
    //   // TODO maybe do some barrier checks here, but for now we just assume
    //   // verything is fine and is generated from gpu code
    //   auto args = b->getOpOperands();
    //   if (barrier) {
    //     // assert(args == barrierArgs);
    //   }
    //   barrier = b;
    //   // barrierArgs = args;
    // });
    // return success();
  }
};

struct AsyncGPULaunch : public OpRewritePattern<async::ExecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(async::ExecuteOp async,
                                PatternRewriter &rewriter) const override {
    for (auto res : async.getResults()) {
      if (!res.use_empty()) {
        return failure();
      }
    }
    for (auto dep : async.getDependencies()) {
      if (!dep.getDefiningOp<enzymexla::StreamToTokenOp>()) {
        return failure();
      }
    }
    SmallVector<gpu::LaunchFuncOp> launches;
    SmallVector<gpu::LaunchOp> launches2;
    if (async
            ->walk<WalkOrder::PreOrder>([&](Operation *op) {
              if (auto launch = dyn_cast<gpu::LaunchFuncOp>(op)) {
                launches.push_back(launch);
                return WalkResult::skip();
              }
              if (auto launch = dyn_cast<gpu::LaunchOp>(op)) {
                launches2.push_back(launch);
                return WalkResult::skip();
              }
              if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
                return WalkResult::advance();
              }
              if (isa<enzymexla::AlternativesOp>(op)) {
                return WalkResult::advance();
              }
              if (isPure(op))
                return WalkResult::advance();
              return WalkResult::interrupt();
            })
            .wasInterrupted())
      return failure();

    SmallVector<Value> gpudeps;
    for (auto dep : async.getDependencies()) {
      gpudeps.push_back(enzymexla::StreamToTokenOp::create(
          rewriter, dep.getLoc(), rewriter.getType<gpu::AsyncTokenType>(),
          dep.getDefiningOp<enzymexla::StreamToTokenOp>().getOperand()));
    }

    for (auto launch : launches) {
      rewriter.modifyOpInPlace(launch, [&]() {
        launch.getAsyncDependenciesMutable().append(gpudeps);
      });
    }

    for (auto launch : launches2) {
      rewriter.modifyOpInPlace(launch, [&]() {
        launch.getAsyncDependenciesMutable().append(gpudeps);
      });
    }

    rewriter.eraseOp(async.getBody()->getTerminator());
    rewriter.inlineBlockBefore(async.getBody(), async);
    rewriter.eraseOp(async);

    return success();
  }
};

uint64_t getSharedMemUsage(scf::ParallelOp pop) {
  uint64_t shmem = 0;
  ModuleOp moduleOp = pop->getParentOfType<ModuleOp>();
  DataLayout DLI(moduleOp);
  for (auto &op : *pop.getBody()) {
    if (auto alloca = dyn_cast<memref::AllocaOp>(&op)) {
      auto mt = alloca.getType();
      auto elTy = mt.getElementType();
      auto elSize = DLI.getTypeSize(elTy);
      auto size = elSize * mt.getNumElements();
      shmem += size;
    }
  }
  return shmem;
}

struct InnerParallelSerialization : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const {
    Location loc = parallelOp.getLoc();
    auto reductionOp =
        dyn_cast<scf::ReduceOp>(parallelOp.getBody()->getTerminator());
    if (!reductionOp) {
      return failure();
    }
    if (!parallelOp->getParentOfType<enzymexla::GPUWrapperOp>()) {
      return failure();
    }
    size_t parallelCount = 0;
    auto par = parallelOp;
    while ((par = par->getParentOfType<scf::ParallelOp>())) {
      parallelCount++;
    }
    // is presently one of the three outer parallel loops;
    if (parallelCount < 2)
      return failure();

    // For a parallel loop, we essentially need to create an n-dimensional loop
    // nest. We do this by translating to scf.for ops and have those lowered in
    // a further rewrite. If a parallel loop contains reductions (and thus
    // returns values), forward the initial values for the reductions down the
    // loop hierarchy and bubble up the results by modifying the "yield"
    // terminator.
    SmallVector<Value, 4> iterArgs =
        llvm::to_vector<4>(parallelOp.getInitVals());
    SmallVector<Value, 4> ivs;
    ivs.reserve(parallelOp.getNumLoops());
    bool first = true;
    SmallVector<Value, 4> loopResults(iterArgs);
    for (auto [iv, lower, upper, step] :
         llvm::zip(parallelOp.getInductionVars(), parallelOp.getLowerBound(),
                   parallelOp.getUpperBound(), parallelOp.getStep())) {
      scf::ForOp forOp =
          scf::ForOp::create(rewriter, loc, lower, upper, step, iterArgs);
      ivs.push_back(forOp.getInductionVar());
      auto iterRange = forOp.getRegionIterArgs();
      iterArgs.assign(iterRange.begin(), iterRange.end());

      if (first) {
        // Store the results of the outermost loop that will be used to replace
        // the results of the parallel loop when it is fully rewritten.
        loopResults.assign(forOp.result_begin(), forOp.result_end());
        first = false;
      } else if (!forOp.getResults().empty()) {
        // A loop is constructed with an empty "yield" terminator if there are
        // no results.
        rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
        scf::YieldOp::create(rewriter, loc, forOp.getResults());
      }

      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    // First, merge reduction blocks into the main region.
    SmallVector<Value> yieldOperands;
    yieldOperands.reserve(parallelOp.getNumResults());
    for (int64_t i = 0, e = parallelOp.getNumResults(); i < e; ++i) {
      Block &reductionBody = reductionOp.getReductions()[i].front();
      Value arg = iterArgs[yieldOperands.size()];
      yieldOperands.push_back(
          cast<scf::ReduceReturnOp>(reductionBody.getTerminator()).getResult());
      rewriter.eraseOp(reductionBody.getTerminator());
      rewriter.inlineBlockBefore(&reductionBody, reductionOp,
                                 {arg, reductionOp.getOperands()[i]});
    }
    rewriter.eraseOp(reductionOp);

    // Then merge the loop body without the terminator.
    Block *newBody = rewriter.getInsertionBlock();
    if (newBody->empty())
      rewriter.mergeBlocks(parallelOp.getBody(), newBody, ivs);
    else
      rewriter.inlineBlockBefore(parallelOp.getBody(), newBody->getTerminator(),
                                 ivs);

    // Finally, create the terminator if required (for loops with no results, it
    // has been already created in loop construction).
    if (!yieldOperands.empty()) {
      rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
      scf::YieldOp::create(rewriter, loc, yieldOperands);
    }

    rewriter.replaceOp(parallelOp, loopResults);

    return success();
  }
};

// TODO parallel wrapper LICM
struct ConvertParallelToGPU1Pass
    : public enzyme::impl::ConvertParallelToGPU1Base<
          ConvertParallelToGPU1Pass> {
  using ConvertParallelToGPU1Base::ConvertParallelToGPU1Base;
  void runOnOperation() override {
    auto m = getOperation();
    {
      RewritePatternSet patterns(&getContext());
      patterns.insert<InnerParallelSerialization>(&getContext());
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(m, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    }
    // TODO we need to transform all parallels in gpu_wrappers to have lower
    // bounds of 0 and steps of 1 as we kind of assume that in many patterns (or
    // have the patterns check)
    auto removeNoops = [&]() {
      RewritePatternSet patterns(&getContext());
      // clang-format off
      patterns.insert<
        RemovePolygeistGPUWrapperOp<enzymexla::GPUThreadOp>,
        RemovePolygeistGPUWrapperOp<enzymexla::GPUBlockOp>,
        RemovePolygeistNoopOp
        >(&getContext());
      // clang-format on
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(m, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    };
    auto populateNormalizationPatterns = [&](RewritePatternSet &patterns) {
      // clang-format off
      patterns.insert<
        //BarrierElim</*TopLevelOnly*/ false>,
        InterchangeIfOp,
        SplitOffParallel,
        HandleWrapperRootAlloca,
        HandleWrapperRootOps,
        CreateParallelOps,
        ParallelizeBlockOps
        >(&getContext());
      patterns.insert<SplitParallelOp>(&getContext());
      // clang-format on
    };
    auto runNormalization = [&]() {
      removeNoops();
      RewritePatternSet patterns(&getContext());
      populateNormalizationPatterns(patterns);
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(m, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    };
    runNormalization();

    // Thread/Block coarsening
    {
      auto runLICM = [&]() {
        m->walk([&](LoopLikeOpInterface loopLike) {
          auto op = (Operation *)loopLike;
          if (auto par = dyn_cast<scf::ParallelOp>(op)) {
            // Only LICM outermost parallel loops in GPU regions which would be
            // the ones affected by thread/block coarsening
            if (op->getParentOfType<enzymexla::GPUWrapperOp>() &&
                !op->getParentOfType<scf::ParallelOp>()) {
              moveParallelLoopInvariantCode(par);
            }
          }
        });
      };

      auto getBlockUnrollFactors = [&](uint64_t unrollFactor,
                                       unsigned gridDims) {
        std::vector<uint64_t> divisors;
        for (unsigned i = 2; unrollFactor != 1; ++i) {
          while (unrollFactor % i == 0) {
            divisors.push_back(i);
            unrollFactor /= i;
          }
        }
        SmallVector<uint64_t, 3> unrollFactors;
        for (unsigned i = 0; i < gridDims; i++)
          unrollFactors.push_back(1);
        for (unsigned i = 0; i < divisors.size(); i++)
          unrollFactors[i % gridDims] *= divisors[i];
        std::sort(unrollFactors.begin(), unrollFactors.end(),
                  [](auto a, auto b) { return a > b; });
        for (unsigned i = 0; i < gridDims; i++)
          llvm::errs() << unrollFactors[i] << " ";
        llvm::errs() << "\n";
        return unrollFactors;
      };
      auto getThreadUnrollFactors = [&](unsigned unrollFactor,
                                        unsigned blockDims) {
        unsigned powsOf2 = std::log2(unrollFactor);
        unsigned initial = std::pow(2, powsOf2 / blockDims);
        unsigned currentFactor = 1;
        SmallVector<uint64_t, 3> unrollFactors;
        for (unsigned i = 0; i < blockDims; i++) {
          unrollFactors.push_back(initial);
          currentFactor *= initial;
        }
        for (unsigned i = blockDims - 1; currentFactor < unrollFactor; i--) {
          currentFactor *= 2;
          unrollFactors[i] *= 2;
        }
        return unrollFactors;
      };
      SmallVector<uint64_t, 3> noCoarsening = {1, 1, 1};
      auto convertToFactors = [&](char *str_, unsigned dims, auto fun) {
        if (!str_)
          return noCoarsening;
        StringRef str(str_);
        uint64_t x, y, z;
        str.consumeInteger(10, x);
        if (str.size() == 0)
          return fun(x, dims);
        str.consume_front(",");
        str.consumeInteger(10, y);
        str.consume_front(",");
        str.consumeInteger(10, z);
        return SmallVector<uint64_t, 3>({x, y, z});
      };
      auto isValid = [&](SmallVectorImpl<uint64_t> &c) {
        return llvm::all_of(c, [&](auto x) { return x >= 1; });
      };

      // These can either be one number `total_factor` or three factors for the
      // three dimensions `x_factor,y_factor,z_factor`
      char *coarsenThreads = getenv("POLYGEIST_GPU_KERNEL_COARSEN_THREADS");
      char *coarsenBlocks = getenv("POLYGEIST_GPU_KERNEL_COARSEN_BLOCKS");

      if (coarsenThreads || coarsenBlocks) {
        std::vector<enzymexla::GPUWrapperOp> toHandle;
        m->walk([&](enzymexla::GPUWrapperOp wrapper) {
          toHandle.push_back(wrapper);
        });
        for (enzymexla::GPUWrapperOp wrapper : toHandle) {
          const char *PATTERN = "coarsen-threads";
          scf::ParallelOp gridPop =
              getDirectlyNestedSingleParallel(wrapper.getBody());
          assert(gridPop);
          scf::ParallelOp blockPop = getDirectlyNestedSingleParallel(
              gridPop.getBody(), /* allowAllocas */ true);
          assert(blockPop);

          auto ubs = gridPop.getUpperBound();
          int gridDims = ubs.size();
          assert(gridDims >= 1 && gridDims <= 3);

          SmallVector<uint64_t, 3> blockUnrollFactors =
              convertToFactors(coarsenBlocks, gridDims, getBlockUnrollFactors);

          if (blockUnrollFactors != noCoarsening &&
              isValid(blockUnrollFactors)) {
            if (enzymexla::scfParallelUnrollByFactors(
                    gridPop, ArrayRef<uint64_t>(blockUnrollFactors),
                    /* generateEpilogueLoop */ true,
                    /* coalescingFriendlyIndexing */ false, nullptr)
                    .failed())
              wrapper->emitRemark("Failed to coarsen blocks");
          }
          blockPop = getDirectlyNestedSingleParallel(
              gridPop.getBody(), /*allowAllocas*/ true,
              /*allowIndexComputation*/ true);
          ubs = blockPop.getUpperBound();
          int blockDims = ubs.size();
          assert(blockDims >= 1 && blockDims <= 3);

          SmallVector<uint64_t, 3> threadUnrollFactors = convertToFactors(
              coarsenThreads, blockDims, getThreadUnrollFactors);

          if (threadUnrollFactors != noCoarsening &&
              isValid(threadUnrollFactors)) {
            // TODO We kind of assume that the upper bounds will be divisible by
            // the factors and in that case this will succeed if the upper
            // bounds are dynamic - we need to insert runtime checks and
            // fallback to a non-coarsened kernel, or have an 'if' statement in
            // the unrolled parallel that will do the "epilogue" part
            if (enzymexla::scfParallelUnrollByFactors(
                    blockPop, ArrayRef<uint64_t>(threadUnrollFactors),
                    /* generateEpilogueLoop */ false,
                    GPUKernelEnableCoalescingFriendlyUnroll, nullptr)
                    .failed())
              wrapper->emitRemark("Failed to coarsen threads");
          }
        }
        runLICM();
      } else if (GPUKernelEmitCoarsenedAlternatives) {
        // If the user did not specify coarsening factors, generate
        // pre-determined set of alternative coarsened kernels

        std::vector<enzymexla::GPUWrapperOp> toHandle;
        m->walk([&](enzymexla::GPUWrapperOp wrapper) {
          toHandle.push_back(wrapper);
        });
        for (enzymexla::GPUWrapperOp wrapper : toHandle) {
          // clang-format off
          const std::vector<std::vector<std::vector<uint64_t>>> UNROLL_FACTORS =
              {{},
               {
                 {32},
                 {16},
                 {8},
                 {4},
                 {2},
                 {1}
               },
               {
                 {4, 8},
                 {4, 4},
                 {2, 4},
                 {2, 2},
                 {1, 2},
                 {1, 1}
               },
               {
                 {2, 4, 4},
                 {2, 2, 4},
                 {2, 2, 2},
                 {1, 2, 2},
                 {1, 1, 2},
                 {1, 1, 1}
               },
              };
          // clang-format on

          const char *PATTERN = "coarsen-threads";
          scf::ParallelOp gridPop =
              getDirectlyNestedSingleParallel(wrapper.getBody());
          assert(gridPop);
          scf::ParallelOp blockPop = getDirectlyNestedSingleParallel(
              gridPop.getBody(), /* allowAllocas */ true);
          assert(blockPop);

          auto ubs = blockPop.getUpperBound();
          int blockDims = ubs.size();
          assert(blockDims >= 1 && blockDims <= 3);
          auto gubs = gridPop.getUpperBound();
          int gridDims = gubs.size();
          assert(gridDims >= 1 && gridDims <= 3);

          unsigned originalThreadNum = 1;
          for (auto ub : ubs) {
            if (auto cio = ub.getDefiningOp<arith::ConstantIndexOp>()) {
              originalThreadNum *= cio.value();
            } else {
              originalThreadNum = 0;
              break;
            }
          }
          unsigned firstUnrollFactorId = 0;
          if (originalThreadNum > 0)
            while (firstUnrollFactorId < UNROLL_FACTORS[1].size() - 1 &&
                   originalThreadNum /
                           UNROLL_FACTORS[1][firstUnrollFactorId][0] <
                       32)
              firstUnrollFactorId++;

          // If we have already varied the block size in SplitParallelOp, avoid
          // doing that here too.
          bool altBlockSize = false;
          if (auto aop = wrapper->getParentOfType<enzymexla::AlternativesOp>())
            if (aop->getAttrOfType<StringAttr>("alternatives.type")
                    .getValue() == "gpu_kernel")
              altBlockSize = true;

          auto loc = wrapper->getLoc();

          auto getMaxSharedMem = [](StringRef arch) -> unsigned {
            if (arch.consume_front("sm_")) {
              return 48 * 1024;
              // TODO we can use more than 48KB (below) but it is only
              // available as dynamic shared mem
              int sm;
              arch.getAsInteger(10, sm);
              if (50 <= sm && sm <= 62)
                return 48 * 1024;
              else if (70 <= sm && sm <= 72)
                return 96 * 1024;
              else if (75 == sm)
                return 64 * 1024;
              else if (80 == sm)
                return 163 * 1024;
              else if (86 == sm || 89 == sm)
                return 99 * 1024;
              else if (90 == sm)
                return 227 * 1024;
            } else if (arch.consume_front("gfx")) {
              // TODO find the proper value for this
              return 48 * 1024;
            }
            return 48 * 1024;
          };

          OpBuilder builder(wrapper);

          auto emitBlockRemCheck = [&](unsigned iThread) {
            auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
            auto unrollFactors = UNROLL_FACTORS[blockDims][iThread];
            Value cond = nullptr;
            for (unsigned i = 0; i < unrollFactors.size(); i++) {
              auto unrollFactor = unrollFactors[i];
              auto unrollFactorCst =
                  arith::ConstantIndexOp::create(builder, loc, unrollFactor);
              auto rem =
                  arith::RemUIOp::create(builder, loc, ubs[i], unrollFactorCst);
              auto cmp = arith::CmpIOp::create(
                  builder, loc, arith::CmpIPredicate::eq, rem, zero);
              if (cond)
                cond = arith::AndIOp::create(builder, loc, cond, cmp);
              else
                cond = cmp;
            }

            auto ifOp = scf::IfOp::create(builder, loc, cond, /*hasElse*/ true);
            auto elseBuilder = ifOp.getElseBodyBuilder();
            GpuRuntimeCallBuilders callBuilder(
                ifOp.getContext(), /*pointerBitwidth, doesnt matter*/ 64);
            callBuilder.abortCallBuilder(loc, elseBuilder, {});
          };

          // Coarsen blocks with all factors, and coarsen threads only by
          // factors which do not bring the number of threads under 32
          unsigned numAlternatives =
              UNROLL_FACTORS[gridDims].size() *
              (UNROLL_FACTORS[blockDims].size() - firstUnrollFactorId);
          auto alternativesOp =
              enzymexla::AlternativesOp::create(builder, loc, numAlternatives);
          alternativesOp->setAttr("alternatives.type",
                                  builder.getStringAttr("gpu_kernel"));
          std::vector<Attribute> descs;
          unsigned curRegion = 0;

          auto emitAlternative = [&](unsigned iBlock, unsigned iThread) {
            // Do not coarsen with factor of over 32
            if (UNROLL_FACTORS[1][iThread][0] * UNROLL_FACTORS[1][iBlock][0] >
                32)
              return failure();
            auto block = &*alternativesOp->getRegion(curRegion).begin();
            builder.setInsertionPointToStart(block);
            emitBlockRemCheck(iThread);
            auto newWrapper = cast<enzymexla::GPUWrapperOp>(
                builder.clone(*wrapper.getOperation()));
            scf::ParallelOp gridPop =
                getDirectlyNestedSingleParallel(newWrapper.getBody());
            assert(gridPop);
            bool succeeded = true;
            auto unrollFactors = UNROLL_FACTORS[gridDims][iBlock];
            if (enzymexla::scfParallelUnrollByFactors(
                    gridPop, ArrayRef<uint64_t>(unrollFactors),
                    /* generateEpilogueLoop */ true,
                    /* coalescingFriendlyIndexing */ false, nullptr)
                    .failed()) {
              wrapper->emitRemark("Failed to coarsen blocks");
              succeeded = false;
            }
            scf::ParallelOp blockPop = getDirectlyNestedSingleParallel(
                gridPop.getBody(), /*allowAllocas*/ true,
                /*allowIndexComputation*/ true);
            assert(blockPop);
            unrollFactors = UNROLL_FACTORS[blockDims][iThread];
            if (enzymexla::scfParallelUnrollByFactors(
                    blockPop, ArrayRef<uint64_t>(unrollFactors),
                    /* generateEpilogueLoop */ false,
                    GPUKernelEnableCoalescingFriendlyUnroll, nullptr)
                    .failed()) {
              wrapper->emitRemark("Failed to coarsen threads");
              llvm::errs() << "Failed to coarsen threads\n";
              succeeded = false;
            }

            if (getSharedMemUsage(gridPop) > getMaxSharedMem(arch))
              succeeded = false;

            if (succeeded) {
              curRegion++;
              descs.push_back(builder.getStringAttr(
                  std::string("block_factor=") +
                  std::to_string(UNROLL_FACTORS[1][iBlock][0]) + "," +
                  std::string("thread_factor=") +
                  std::to_string(UNROLL_FACTORS[1][iThread][0]) + ","));
              return success();
            } else {
              // Clear block
              auto newBlock = new Block();
              block->getParent()->push_front(newBlock);
              OpBuilder::atBlockBegin(newBlock).clone(*block->getTerminator());
              block->erase();
              return failure();
            }
          };

          if (altBlockSize) {
            bool failed = false;
            unsigned unrollFactorOne = UNROLL_FACTORS[blockDims].size() - 1;
            if (GPUKernelEnableBlockCoarsening) {
              for (unsigned iBlock = 0;
                   iBlock < UNROLL_FACTORS[gridDims].size(); iBlock++) {
                if ((failed =
                         emitAlternative(iBlock, unrollFactorOne).failed()))
                  break;
              }
            } else {
              failed = true;
            }
            if (failed) {
              curRegion = 0;
              for (unsigned iThread = firstUnrollFactorId;
                   iThread < UNROLL_FACTORS[blockDims].size(); iThread++) {
                auto succeeded =
                    emitAlternative(unrollFactorOne, iThread).succeeded();
                (void)succeeded;
                assert(succeeded);
              }
            }
          } else {
            for (unsigned iThread = firstUnrollFactorId;
                 iThread < UNROLL_FACTORS[blockDims].size(); iThread++) {
              for (unsigned iBlock = GPUKernelEnableBlockCoarsening
                                         ? 0
                                         : UNROLL_FACTORS[gridDims].size() - 1;
                   iBlock < UNROLL_FACTORS[gridDims].size(); iBlock++) {
                (void)emitAlternative(iBlock, iThread);
              }
            }
          }

          wrapper->erase();

          alternativesOp->setAttr("alternatives.descs",
                                  builder.getArrayAttr(descs));

          shrinkAlternativesOp(alternativesOp, curRegion, builder);
        }
        // Make sure LICM doesnt change the structure
        m->walk([&](scf::ParallelOp pop) {
          if (!pop->getParentOfType<enzymexla::GPUWrapperOp>())
            return;
          auto loc = pop->getLoc();
          OpBuilder builder = OpBuilder::atBlockBegin(pop.getBody());
          auto noop = enzymexla::NoopOp::create(builder, loc,
                                                pop.getBody()->getArguments());
          if (pop->getParentOfType<scf::ParallelOp>()) {
            noop->setAttr(
                "enzymexla.noop_type",
                StringAttr::get(noop->getContext(), "gpu_kernel.thread"));
          } else {
            noop->setAttr(
                "enzymexla.noop_type",
                StringAttr::get(noop->getContext(), "gpu_kernel.block"));
          }
        });
        runLICM();
      }
    }

    runNormalization();

    {
      RewritePatternSet patterns(&getContext());
      // clang-format off
      patterns.insert<
        ParallelToGPULaunch
        >(&getContext());
      // clang-format on
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(m, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      // clang-format off
      patterns.insert<
	AsyncGPULaunch
        >(&getContext());
      // clang-format on
      GreedyRewriteConfig config;
      if (failed(
              applyPatternsAndFoldGreedily(m, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    }

    // Sink constants in the body
    m->walk([](gpu::LaunchOp launchOp) {
      Region &launchOpBody = launchOp.getBody();
      SetVector<Value> sinkCandidates;
      getUsedValuesDefinedAbove(launchOpBody, sinkCandidates);
      SetVector<Operation *> toBeSunk;
      for (Value operand : sinkCandidates) {
        Operation *operandOp = operand.getDefiningOp();
        if (operandOp && operandOp->hasTrait<OpTrait::ConstantLike>() &&
            operandOp->getNumOperands() == 0)
          toBeSunk.insert(operandOp);
      }

      if (toBeSunk.empty())
        return;

      OpBuilder builder(launchOpBody);
      for (Operation *op : toBeSunk) {
        Operation *clonedOp = builder.clone(*op);
        // Only replace uses within the launch op.
        for (auto pair : llvm::zip(op->getResults(), clonedOp->getResults()))
          replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair),
                                     launchOp.getBody());
      }
    });

    m->walk([](scf::ParallelOp pop) {
      if (pop->getParentOfType<gpu::GPUModuleOp>()) {
        POLYGEIST_REMARK(
            pop->emitRemark("Could not use parallel loop for parallelism in "
                            "GPU kernel - serializing instead"));
        // TODO the gpu.funcs we created and serialize any stray
        // parallels that may remain (optimally we would want to use them for
        // the gpu.launch op but there may be cases where we cannot?)
      }
    });
  }
};

struct ConvertParallelToGPU2Pass
    : public enzyme::impl::ConvertParallelToGPU2Base<
          ConvertParallelToGPU2Pass> {
  using ConvertParallelToGPU2Base::ConvertParallelToGPU2Base;
  void runOnOperation() override {

    /*
std::vector<enzymexla::GetDeviceGlobalOp> gdgops;
getOperation()->walk(
 [&](enzymexla::GetDeviceGlobalOp gdgo) { gdgops.push_back(gdgo); });
for (auto gdgo : gdgops) {
auto builder = OpBuilder(gdgo);
auto ggo = memref::GetGlobalOp::create(builder,
   gdgo->getLoc(), gdgo.getType(), gdgo.getNameAttr());
gdgo->replaceAllUsesWith(ggo);
gdgo->erase();
}
*/

    RewritePatternSet patterns(&getContext());
    if (emitGPUKernelLaunchBounds)
      patterns.insert<AddLaunchBounds>(&getContext());
    patterns
        .insert<SharedLLVMAllocaToGlobal, SharedMemrefAllocaToGlobal,
                RemoveFunction<func::FuncOp>, RemoveFunction<LLVM::LLVMFuncOp>>(
            &getContext());
    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
      return;
    }
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(getOperation());
    getOperation()->walk([&](GPUErrorOp err) {
      std::string sm;
      if (auto attr =
              dyn_cast_or_null<ArrayAttr>(err->getAttr("passthrough"))) {
        for (auto a : attr) {
          if (auto ar = dyn_cast<ArrayAttr>(a)) {
            if (ar.size() != 2)
              continue;
            auto s0 = dyn_cast<StringAttr>(ar[0]);
            auto s1 = dyn_cast<StringAttr>(ar[1]);
            if (!s0 || !s1)
              continue;
            if (s0.getValue() == "target-cpu")
              sm = s1.getValue();
            if (backend == "rocm") {
              if (sm.find("sm_") != std::string::npos) {
                llvm::errs() << "Error: Found NVIDIA architecture while "
                                "targeting ROCm.\n";
                std::abort();
              }
            }
          }
        }
      }
      std::string feat;
      if (auto attr = dyn_cast_or_null<LLVM::TargetFeaturesAttr>(
              err->getAttr("target_features"))) {
        feat = attr.getFeaturesString();
      }

      err->walk([&](gpu::LaunchFuncOp launch) {
        auto gfunc = dyn_cast_or_null<gpu::GPUFuncOp>(
            symbolTable.lookupNearestSymbolFrom(launch, launch.getKernel()));
        if (!gfunc)
          return;
        auto gmod = cast<gpu::GPUModuleOp>(gfunc->getParentOp());
        if (!gmod.getTargetsAttr()) {
          Attribute target;
          if (backend == "rocm") {
            auto chip = "gfx900";
            auto features = "+wavefront64";
            target = ROCDL::ROCDLTargetAttr::get(
                gmod.getContext(),
                /*optLevel=*/3, /*triple=*/"amdgcn-amd-amdhsa", chip, features,
                /*abiVersion=*/"600");
          } else {
            auto chip = sm;
            if (chip.size() == 0)
              chip = "sm_80";
            auto features = feat;
            if (features.size() == 0)
              features = "+ptx73";
            target = NVVM::NVVMTargetAttr::get(
                gmod.getContext(), /*optLevel*/ 3,
                /*triple*/ "nvptx64-nvidia-cuda", chip, features);
          }
          gmod.setTargetsAttr(ArrayAttr::get(gmod.getContext(), target));

          DataLayoutSpecInterface dataLayout = {};
          // Set index type size to 32 bits
          {
            auto ctx = gmod.getContext();
            llvm::DenseMap<mlir::TypeAttr, mlir::DataLayoutEntryInterface>
                typeEntries;
            auto type = IndexType::get(ctx);
            auto key = mlir::TypeAttr::get(type);
            uint64_t size = 32;
            auto params =
                IntegerAttr::get(mlir::IntegerType::get(ctx, 64), size);
            typeEntries.try_emplace(key,
                                    DataLayoutEntryAttr::get(type, params));
            SmallVector<DataLayoutEntryInterface> entries;
            entries.reserve(typeEntries.size());
            for (const auto &it : typeEntries)
              entries.push_back(it.second);
            dataLayout = DataLayoutSpecAttr::get(ctx, entries);
          }
          // gpuModule->setAttr(
          //     LLVM::LLVMDialect::getDataLayoutAttrName(),
          //     deviceModule->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName()));
          gmod->setAttr(DLTIDialect::kDataLayoutAttrName, dataLayout);
        }
      });
    });
  }
};

struct MergeGPUModulesPass
    : public enzyme::impl::MergeGPUModulesPassBase<MergeGPUModulesPass> {
  using MergeGPUModulesPassBase::MergeGPUModulesPassBase;
  void runOnOperation() override {
    auto m = getOperation();
    Region &moduleRegion = m->getRegion(0);
    OpBuilder mBuilder(moduleRegion);
    std::string newModuleName = "__enzymexla_gpu_module";
    auto newGpuModule =
        gpu::GPUModuleOp::create(mBuilder, m->getLoc(), newModuleName);
    OpBuilder gpumBuilder(newGpuModule->getRegion(0));
    std::vector<gpu::GPUModuleOp> toErase;
    m->walk([&](gpu::GPUModuleOp gpum) {
      if (gpum == newGpuModule)
        return;
      toErase.push_back(gpum);
      for (auto &op : *gpum.getBody()) {
        auto cloneIf = [&](auto op) {
          if (op) {
            if (!SymbolTable::lookupSymbolIn(newGpuModule, op.getName())) {
              gpumBuilder.clone(*op.getOperation());
            }
            return true;
          }
          return false;
        };

        if (auto f = dyn_cast<gpu::GPUFuncOp>(&op)) {
          auto newF = cast<gpu::GPUFuncOp>(gpumBuilder.clone(op));
          if (SymbolTable::lookupSymbolIn(newGpuModule, f.getName())) {
            auto newKernelName =
                std::string(f.getName()) +
                std::to_string(reinterpret_cast<intptr_t>(f.getOperation()));
            newF.setName(newKernelName);
          }
          auto symbolUses = SymbolTable::getSymbolUses(f.getOperation(), m);
          assert(symbolUses);
          for (auto symbolUse : *symbolUses) {
            if (auto launchOp =
                    dyn_cast<gpu::LaunchFuncOp>(symbolUse.getUser())) {
              auto kernelSymbol =
                  SymbolRefAttr::get(newGpuModule.getNameAttr(),
                                     {SymbolRefAttr::get(newF.getNameAttr())});
              launchOp->setAttr(
                  gpu::LaunchFuncOp::getKernelAttrName(launchOp->getName()),
                  kernelSymbol);
            } else {
              f.emitError("Unexpected user of gpu func op");
              assert(0);
            }
          }
        } else if (!(cloneIf(dyn_cast<memref::GlobalOp>(&op)) ||
                     cloneIf(dyn_cast<LLVM::GlobalOp>(&op)) ||
                     cloneIf(dyn_cast<func::FuncOp>(&op)) ||
                     cloneIf(dyn_cast<LLVM::LLVMFuncOp>(&op)))) {
          op.emitError("Unexpected global type in gpu module");
          op.dump();
          assert(0);
        }
      }
    });

    if (toErase.size() == 0)
      newGpuModule->erase();

    for (auto gpum : toErase)
      gpum->erase();
  }
};

} // namespace
