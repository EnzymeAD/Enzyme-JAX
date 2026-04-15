#include "src/enzyme_ad/jax/Passes/Distributed/TimingAnalysis.h"

#include <algorithm>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/Dialect.h"

namespace mlir {
namespace enzyme {
namespace distributed {

namespace {

constexpr double kKflopsPerFlop = 1.0 / 1000.0;
constexpr double kKbytesPerByte = 1.0 / 1000.0;

int64_t getTypeBitWidth(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return intTy.getWidth();
  if (auto floatTy = dyn_cast<FloatType>(ty))
    return floatTy.getWidth();
  if (auto complexTy = dyn_cast<ComplexType>(ty)) {
    Type elemTy = complexTy.getElementType();
    if (auto elemFloat = dyn_cast<FloatType>(elemTy))
      return elemFloat.getWidth() * 2;
  }
  llvm::report_fatal_error("unsupported element type for cost model");
}

int64_t getTensorSizeBytesFromType(Type ty) {
  auto shapedTy = dyn_cast<ShapedType>(ty);
  if (!shapedTy || !shapedTy.hasStaticShape() ||
      !shapedTy.getElementType().isIntOrFloat()) {
    llvm::report_fatal_error(
        "timing cost model requires static int/float tensor types");
  }
  int64_t bitWidth = getTypeBitWidth(shapedTy.getElementType());
  return shapedTy.getNumElements() * ((bitWidth + 7) / 8);
}

int64_t getTensorNumElementsFromType(Type ty) {
  auto shapedTy = dyn_cast<ShapedType>(ty);
  if (!shapedTy || !shapedTy.hasStaticShape() ||
      !shapedTy.getElementType().isIntOrFloat()) {
    llvm::report_fatal_error(
        "timing cost model requires static int/float tensor types");
  }
  return shapedTy.getNumElements();
}

double estimateDotLikeOpKflops(ShapedType lhsTy, ShapedType rhsTy,
                 ShapedType outTy,
                 int64_t reductionSize) {
  if (!lhsTy || !rhsTy || !outTy || !lhsTy.hasStaticShape() ||
      !rhsTy.hasStaticShape() || !outTy.hasStaticShape() ||
      !lhsTy.getElementType().isIntOrFloat() ||
      !rhsTy.getElementType().isIntOrFloat()) {
    llvm::report_fatal_error(
        "unsupported stablehlo dot shape/type in timing model");
  }

  int64_t lhsElems = lhsTy.getNumElements();
  int64_t rhsElems = rhsTy.getNumElements();
  (void)lhsElems;
  (void)rhsElems;
  int64_t outElems = outTy.getNumElements();
  if (outElems == 0)
    return 0.0;

  int64_t dotOps = outElems * reductionSize;
  return static_cast<double>(dotOps) * kKflopsPerFlop;
}

int64_t getDotGeneralReductionSize(stablehlo::DotGeneralOp dot,
                                   ShapedType lhsTy, ShapedType rhsTy) {
  auto lhsShape = lhsTy.getShape();
  auto rhsShape = rhsTy.getShape();

  auto dimNumbers = dot.getDotDimensionNumbers();
  auto lhsContracting = dimNumbers.getLhsContractingDimensions();
  auto rhsContracting = dimNumbers.getRhsContractingDimensions();
  if (lhsContracting.size() != rhsContracting.size()) {
    llvm::report_fatal_error(
        "stablehlo.dot_general has mismatched contracting dimensions");
  }

  int64_t reductionSize = 1;
  for (size_t i = 0, e = lhsContracting.size(); i < e; ++i) {
    int64_t lhsDim = lhsContracting[i];
    int64_t rhsDim = rhsContracting[i];
    if (lhsDim < 0 || rhsDim < 0 || lhsDim >= static_cast<int64_t>(lhsShape.size()) ||
        rhsDim >= static_cast<int64_t>(rhsShape.size())) {
      llvm::report_fatal_error(
          "stablehlo.dot_general has out-of-bounds contracting dimension");
    }
    if (lhsShape[lhsDim] != rhsShape[rhsDim]) {
      llvm::report_fatal_error(
          "stablehlo.dot_general has incompatible contracting dimensions");
    }
    reductionSize *= lhsShape[lhsDim];
  }
  return reductionSize;
}

int64_t getDotReductionSize(ShapedType lhsTy, ShapedType rhsTy) {
  auto lhsShape = lhsTy.getShape();
  auto rhsShape = rhsTy.getShape();
  if (lhsShape.empty() || rhsShape.empty())
    return 1;

  int64_t lhsContractDim = lhsShape.back();
  int64_t rhsContractDim = rhsShape.front();
  if (lhsContractDim != rhsContractDim) {
    llvm::report_fatal_error("stablehlo.dot has incompatible contracted dimensions");
  }
  return lhsContractDim;
}

double getOpToKflops(Operation *op) {
  if (isa<SendOp, RecvOp, distributed::DistributedYieldOp, stablehlo::ReshapeOp,
          stablehlo::SliceOp, stablehlo::ConcatenateOp, TransferOp>(op)) {
    return 0.0;
  }

  if (auto dot = dyn_cast<stablehlo::DotGeneralOp>(op)) {
    auto lhsTy = dyn_cast<ShapedType>(dot.getLhs().getType());
    auto rhsTy = dyn_cast<ShapedType>(dot.getRhs().getType());
    auto outTy = dyn_cast<ShapedType>(dot.getResult().getType());
    int64_t reductionSize = getDotGeneralReductionSize(dot, lhsTy, rhsTy);
    return estimateDotLikeOpKflops(lhsTy, rhsTy, outTy, reductionSize);
  }

  if (auto dot = dyn_cast<stablehlo::DotOp>(op)) {
    auto lhsTy = dyn_cast<ShapedType>(dot.getLhs().getType());
    auto rhsTy = dyn_cast<ShapedType>(dot.getRhs().getType());
    auto outTy = dyn_cast<ShapedType>(dot.getResult().getType());
    int64_t reductionSize = getDotReductionSize(lhsTy, rhsTy);
    return estimateDotLikeOpKflops(lhsTy, rhsTy, outTy, reductionSize);
  }

  llvm::report_fatal_error(
      llvm::Twine("unsupported op in timing op-to-flop model: ") +
      op->getName().getStringRef());
}

double getTransferSizeBytes(Operation *op) {
  if (auto transfer = dyn_cast<TransferOp>(op)) {
    Value token = transfer.getToken();
    if (auto collective = token.getDefiningOp<CollectiveOp>()) {
      return static_cast<double>(getTensorSizeBytesFromType(
          collective.getLocalOutputTensorType()));
    }
    if (auto parts = token.getDefiningOp<SubmeshCollectivePartsOp>()) {
      return static_cast<double>(
          getTensorSizeBytesFromType(parts.getOutputTensorType()));
    }
    llvm::report_fatal_error(
        "unsupported transfer token producer in timing model");
  }

  return 0.0;
}

} // namespace

double AffineTimingCostModel::getOperationDuration(Operation *op) const {
  // TODO: Incomplete prototype model for all-reduce. This approximates one
  // local tile add and one network transfer of that same tile.
  if (op->getName().getStringRef() == "sdy.all_reduce") {
    if (op->getNumOperands() == 0) {
      llvm::report_fatal_error(
          "sdy.all_reduce must have a tensor operand for timing model");
    }
    Type tileType = op->getOperand(0).getType();
    double tileKflops =
        static_cast<double>(getTensorNumElementsFromType(tileType)) *
        kKflopsPerFlop;
    double tileKbytes =
        static_cast<double>(getTensorSizeBytesFromType(tileType)) *
        kKbytesPerByte;
    return params.opIntercept + params.kflopCoeff * tileKflops +
           params.transferIntercept + params.transferKbyteCoeff * tileKbytes;
  }

  if (isa<TransferOp>(op)) {
    double kbytes = getTransferSizeBytes(op) * kKbytesPerByte;
    return params.transferIntercept + params.transferKbyteCoeff * kbytes;
  }

  double kflops = getOpToKflops(op);
  return params.opIntercept + params.kflopCoeff * kflops;
}

TimingAnalysis::TimingAnalysis(Operation *op, AnalysisManager &am) {
  (void)op;
  const auto &hbAnalysis = am.getAnalysis<HappensBeforeAnalysis>();
  hb = &hbAnalysis;
  AffineTimingCostModel affineCostModel;
  buildTimingMap(hbAnalysis, affineCostModel);
}

TimingAnalysis::TimingAnalysis(const HappensBeforeAnalysis &hb,
                               const TimingCostModel &costModel) {
  this->hb = &hb;
  buildTimingMap(hb, costModel);
}

void TimingAnalysis::buildTimingMap(const HappensBeforeAnalysis &hb,
                                    const TimingCostModel &costModel) {
  rootToTimeRange.clear();

  // classesInTopologicalOrder() guarantees predecessors come before successors,
  // so a single forward pass is sufficient to compute start/end times.
  for (Operation *root : hb.classesInTopologicalOrder()) {
    double duration = 0.0;
    for (Operation *member : hb.classList(root)) {
      duration = std::max(duration, costModel.getOperationDuration(member));
    }

    double startTime = 0.0;
    for (Operation *pred : hb.predecessorClasses(root)) {
      auto predIt = rootToTimeRange.find(pred);
      if (predIt != rootToTimeRange.end()) {
        // TODO: can do a critical path analysis with argmax
        startTime = std::max(startTime, predIt->second.second);
      }
    }

    rootToTimeRange[root] = {startTime, startTime + duration};
  }
}

TimingAnalysis::TimeRange TimingAnalysis::getTimeRange(Operation *op) const {
  assert(hb && "TimingAnalysis has no backing HappensBeforeAnalysis");
  Operation *root = hb->classRoot(op);
  assert(root && "Operation is not tracked by HappensBeforeAnalysis");
  auto it = rootToTimeRange.find(root);
  assert(it != rootToTimeRange.end() && "Operation class root has no timing entry");
  return it->second;
}

bool TimingAnalysis::isInvalidated(
    const AnalysisManager::PreservedAnalyses &pa) {
  return !pa.isPreserved<TimingAnalysis>() ||
         !pa.isPreserved<HappensBeforeAnalysis>();
}

} // namespace distributed
} // namespace enzyme
} // namespace mlir
