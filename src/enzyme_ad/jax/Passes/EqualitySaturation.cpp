#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/deps/include/ReactantExtra.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/service/platform_util.h"

#include "cxxbridge/deps/tensat/src/input.rs.h"
#include "rust/cxx.h"

#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <memory>
#include <sstream>

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace rust::cxxbridge1;

class OperationMapInfo : public llvm::DenseMapInfo<Operation*> {
public:
  static unsigned getHashValue(const Operation* val) {
    return OperationEquivalence::computeHash(
      const_cast<Operation*>(val),
      // Operands, values and locations don't matter for runtime - we just
      // need the operation, attributes and types to be the same.
      OperationEquivalence::ignoreHashValue,
      OperationEquivalence::ignoreHashValue,
      OperationEquivalence::IgnoreLocations);
  }

  // Adapted from llvm-project/mlir/lib/Transforms/CSE.cpp
  static bool isEqual(const Operation* lhsC, const Operation* rhsC) {
    auto* lhs = const_cast<Operation*>(lhsC);
    auto* rhs = const_cast<Operation*>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        lhs, rhs,
        OperationEquivalence::ignoreValueEquivalence,
        nullptr,
        OperationEquivalence::IgnoreLocations);
  }
};

class OperationTimer {
public:
  /**
   * Measure cost of operation (execution time in microseconds) by running it many times and measuring the time taken.
   * TODO: Make cloning optional
   * TODO: Preserve context across runs so that we're not creating unnecessary contexts
  */
  static uint64_t getCost(Operation* op, unsigned warmup, unsigned repetitions) {
    if (!logsInitialized) {
      InitializeLogs();
      logsInitialized = true;
    }

    std::string opName = op->getName().getStringRef().str();

    // TODO: Have a whitelist instead?
    if (op->getDialect()->getNamespace() != "stablehlo" || opName == "stablehlo.constant" 
        || opName == "stablehlo.return" || opName == "stablehlo.compare")
      return 0;
    
    if (runtimeCache.contains(op)) {
      return runtimeCache[op];
    }

    auto context = OperationTimer::getContext();

    ModuleOp wrapperModule = createModuleFromOperation(context, op);
    
    // TODO: GPU
    xla::PjRtClient *client = MakeCPUClient(0, 1, 1);
    auto executable = prepareExecutable(client, wrapperModule);

    unsigned numArgs = op->getNumOperands();
    unsigned numResults = op->getNumResults();
    uint8_t futures = 0;

    xla::PjRtBuffer* args[numArgs];
    uint8_t isArgDonatable[numArgs];
    xla::PjRtBuffer* res[numResults];

    for (int i = 0; i < numArgs; i++) {
      args[i] = getRandomInput(client, op->getOperand(i).getType());
      isArgDonatable[i] = false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    for (unsigned i = 0; i < warmup + repetitions; i++) {
      if (i == warmup) t1 = std::chrono::high_resolution_clock::now();
      XLAExecute(executable, numArgs, args, isArgDonatable, numResults, res, &futures, nullptr);

      // Cleanup
      for (int i = 0; i < numResults; i++) {
        PjRtBufferFree(res[i]);
      }
    }
    
    assert(!futures);

    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    FreeClient(executable->client());
    ExecutableFree(executable);

    wrapperModule.erase();

    // std::cout << op->getName().getStringRef().str() << "\n";
    auto indexOp = op->clone();
    runtimeCache.try_emplace(indexOp, duration);
    return duration;
  }

  static Operation *getDummyOp(OpBuilder &builder, Type type) {
    // Zero-initialise inputs with same operand shape
    Attribute zeroAttr = builder.getZeroAttr(type);
    OperationState zeroState(builder.getUnknownLoc(), "stablehlo.constant");
    zeroState.addTypes(type);
    zeroState.addAttribute("value", zeroAttr);

    return builder.create(zeroState);
  }

  static Operation *cloneOpInContext(OpBuilder &builder, Operation *op) {
    IRMapping mapping;
    return cloneOpInContext(builder, op, mapping);
  }

  static MLIRContext *getContext() {
    if (!context) {
      DialectRegistry registry;
      InitializeRegistryAndPasses(wrap(&registry));
      context = new MLIRContext(registry);
      RegisterDialects(wrap(context));
    }
    return context;
  }
private:
  static llvm::DenseMap<Operation*, uint64_t, OperationMapInfo> runtimeCache;
  static MLIRContext* context;
  inline static bool logsInitialized;
  
  /**
   * Create a clone of the operation in the new context recursively (i.e. going down to the regions).
   * Just using op->clone() will preserve context of the original operation, which poses a problem later
   * since stablehlo -> mhlo legalization pass will not match the new operation. 
   * 
   * Like the normal op->clone(), any operands that use values outside of the operations are remapped using 
   * the map that is provided (leaving them alone if no entry is present).
   * 
   * TODO: Surely there's a simpler way to do this?
  */
  static Operation *cloneOpInContext(OpBuilder &builder,
                                     Operation *op,
                                     IRMapping& mapping) {
    Location location = builder.getUnknownLoc();

    // Recursively clone regions
    llvm::SmallVector<std::unique_ptr<Region>> regions;

    for (auto& region : op->getRegions()) {
      auto newRegion = std::make_unique<Region>();

      for (auto& block : region.getBlocks()) {
        auto newBlock = new Block();

        // Map from old block arguments to new ones
        for (auto& arg : block.getArguments()) {
          mapping.map(arg, newBlock->addArgument(arg.getType(), location));
        }

        for (auto& nestedOp : block.getOperations()) {
          auto *newNestedOp = cloneOpInContext(builder, &nestedOp, mapping);  
          newBlock->push_back(newNestedOp);
          
          // Map result of old operation to that of new operation, so that operations after can use it
          for (int i = 0; i < nestedOp.getNumResults(); i++) {
            mapping.map(nestedOp.getResult(i), newNestedOp->getResult(i));
          }
        }
        newRegion->push_back(newBlock);
      }
      regions.push_back(std::move(newRegion));
    }

    OperationState opState(location, 
                            op->getName().getStringRef().str(),   // Use string to make a new name, rather than reusing the OperationName
                            op->getOperands(),
                            op->getResultTypes(),
                            op->getAttrs(),
                            {},
                            regions);

    auto *newOp = builder.create(opState);
    
    for (int i = 0; i < newOp->getNumOperands(); i++) {
      newOp->setOperand(i, mapping.lookupOrDefault(newOp->getOperand(i)));
    }

    return newOp;
  }

  /**
   * Wrap operation into a module, where its operands are mapped to inputs of main. 
   * Doesn't mutate op (instead it creates a copy).
   */
  static ModuleOp createModuleFromOperation(MLIRContext *context, Operation *op) {
    // Wrap operation into a module with dummy inputs
    OpBuilder builder(context);
    Location location = builder.getUnknownLoc();
    ModuleOp wrapperModule = ModuleOp::create(location);

    auto block = wrapperModule.getBodyRegion().begin();

    auto *newOp = cloneOpInContext(builder, op);

    // Create a func.func to wrap newOp around
    FunctionType funcType = FunctionType::get(context, op->getOperandTypes(), op->getResultTypes());
    func::FuncOp funcOp = builder.create<func::FuncOp>(location, "main", funcType);
    block->push_back(funcOp);

    Block *entryBlock = funcOp.addEntryBlock();

    for (int i = 0; i < op->getNumOperands(); i++) {
      newOp->setOperand(i, funcOp.getArgument(i));
    }

    entryBlock->push_back(newOp);
    
    auto returnOp = builder.create<func::ReturnOp>(location, newOp->getResults());
    entryBlock->push_back(returnOp);

    return std::move(wrapperModule);
  }

  /**
   * Wrap and compile operation into a PjRtLoadedExecutable, to be passed into XLAExecute.
  */
  static xla::PjRtLoadedExecutable* prepareExecutable(xla::PjRtClient *client, ModuleOp &wrapperModule) {
    if (failed(verify(wrapperModule))) {
      llvm::errs() << "Module verification error\n";
    }

    xla::PjRtLoadedExecutable *executable = ClientCompile(client, wrap(wrapperModule));

    return executable;
  }

  /**
   * Create a PjRtBuffer from a given MLIR type with random elements.
   */

  static xla::PjRtBuffer* getRandomInput(xla::PjRtClient* client, mlir::Type type) {
    assert(isa<RankedTensorType>(type)); // TODO: not true in general
    auto ranked = type.cast<RankedTensorType>();
    auto elementType = ranked.getElementType();
    auto shape = ranked.getShape();

    auto width = (elementType.getIntOrFloatBitWidth() + 7) / 8; // round up to nearest byte
    int numElements = 1;
    for (auto i : shape) numElements *= i;

    void* data = malloc(width * numElements);

    auto device = ClientGetAddressableDevice(client, 0);
    auto buffer = ArrayFromHostBuffer(client, data, wrap(elementType), shape.size(), shape.data(), device);
    
    free(data);
    return buffer;
  }
};

llvm::DenseMap<Operation*, uint64_t, OperationMapInfo> OperationTimer::runtimeCache;
MLIRContext* OperationTimer::context = nullptr;

/**
* Create a new mlir::RankedTensorType based on the type of an existing mlir::Value and the provided shape.
*/
mlir::RankedTensorType deriveOutputType(mlir::Value &input, llvm::ArrayRef<int64_t> shape) {
  auto inputType = input.getType();
  assert(isa<RankedTensorType>(inputType));
  auto ranked = inputType.cast<RankedTensorType>();
  RankedTensorType::Builder builder(ranked);
  return builder.setShape(shape);
}

llvm::ArrayRef<int64_t> getShape(mlir::Value &input) {
  auto inputType = input.getType();
  assert(isa<RankedTensorType>(inputType));
  auto ranked = inputType.cast<RankedTensorType>();
  return ranked.getShape();
}

std::vector<int64_t> rust_vec_to_cpp_vector(tensat::Shape input_slice) {
  std::vector<int64_t> result;
  for (const auto& value : input_slice.shape) {
    result.push_back(value);
  }
  return result;
}

/**
 * https://github.com/google/jax/blob/c08656c61d0e2460f5d902b0af808b74c76a48ca/jax/_src/lax/lax.py#L2729
 */
std::vector<int64_t> vectorExcept(llvm::ArrayRef<int64_t> &vec,
                                  std::vector<int64_t> &indices) {
  std::vector<int64_t> result;
  for (int i = 0; i < vec.size(); i++) {
    if (std::find(indices.begin(), indices.end(), i) == indices.end()) {
      result.push_back(vec[i]);
    }
  }
  return result;
}

/**
 * https://github.com/google/jax/blob/c08656c61d0e2460f5d902b0af808b74c76a48ca/jax/_src/lax/lax.py#L2720C5-L2720C35
 */
std::vector<int64_t> dotGeneralShapeComputation(llvm::ArrayRef<int64_t> &lhs_shape, 
                                                llvm::ArrayRef<int64_t> &rhs_shape, 
                                                std::vector<int64_t> &lhs_batch,
                                                std::vector<int64_t> &rhs_batch,
                                                std::vector<int64_t> &lhs_contracting,
                                                std::vector<int64_t> &rhs_contracting) {
  std::vector<int64_t> batch_shape;
  for (auto i : lhs_batch) batch_shape.push_back(lhs_shape[i]);

  std::vector<int64_t> lhs_contract_or_batch = lhs_contracting;
  lhs_contract_or_batch.insert(lhs_contract_or_batch.end(), lhs_batch.begin(), lhs_batch.end());
  std::sort(lhs_contract_or_batch.begin(), lhs_contract_or_batch.end());

  auto lhs_tensored_shape = vectorExcept(lhs_shape, lhs_contract_or_batch);

  std::vector<int64_t> rhs_contract_or_batch = rhs_contracting;
  rhs_contract_or_batch.insert(rhs_contract_or_batch.end(), rhs_batch.begin(), rhs_batch.end());
  std::sort(rhs_contract_or_batch.begin(), rhs_contract_or_batch.end());

  auto rhs_tensored_shape = vectorExcept(rhs_shape, rhs_contract_or_batch);

  auto shape = batch_shape;
  shape.insert(shape.end(), lhs_tensored_shape.begin(), lhs_tensored_shape.end());
  shape.insert(shape.end(), rhs_tensored_shape.begin(), rhs_tensored_shape.end());

  return shape;
}

Operation* createStableHloOp(
    OpBuilder &builder,
    tensat::Ops op,
    SmallVector<Value> &operands,
    std::vector<std::vector<int64_t>> &other_vecs,
    std::vector<int64_t> &int_args,
    MLIRContext *context
  ) {
  Operation* mlirOp = nullptr;

  switch (op) {
    case tensat::Ops::AddOp:
      mlirOp = builder.create<stablehlo::AddOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::MulOp:
      mlirOp = builder.create<stablehlo::MulOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::DivOp:
      mlirOp = builder.create<stablehlo::DivOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::SubtractOp:
      mlirOp = builder.create<stablehlo::SubtractOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::NegOp:
      mlirOp = builder.create<stablehlo::NegOp>(builder.getUnknownLoc(), operands[0]);
      break;
    case tensat::Ops::TanhOp:
      mlirOp = builder.create<stablehlo::TanhOp>(builder.getUnknownLoc(), operands[0]);
      break;
    case tensat::Ops::ExpOp:
      mlirOp = builder.create<stablehlo::ExpOp>(builder.getUnknownLoc(), operands[0]);
      break;
    case tensat::Ops::MinOp:
      mlirOp = builder.create<stablehlo::MinOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::MaxOp:
      mlirOp = builder.create<stablehlo::MaxOp>(builder.getUnknownLoc(), operands[0], operands[1]);
      break;
    case tensat::Ops::TransposeOp:
      mlirOp = builder.create<stablehlo::TransposeOp>(builder.getUnknownLoc(), operands[0], other_vecs[0]);
      break;
    case tensat::Ops::ReshapeOp:
      mlirOp = builder.create<stablehlo::ReshapeOp>(builder.getUnknownLoc(), deriveOutputType(operands[0], other_vecs[0]), operands[0]);
      break;
    case tensat::Ops::DotGeneralOp: {
      std::vector<int64_t> lhs_batch_dim = other_vecs[0];
      std::vector<int64_t> rhs_batch_dim = other_vecs[1];
      std::vector<int64_t> lhs_contract_dim = other_vecs[2];
      std::vector<int64_t> rhs_contract_dim = other_vecs[3];
      std::vector<int64_t> precision_config = other_vecs[4];
      auto lhs_shape = getShape(operands[0]);
      auto rhs_shape = getShape(operands[1]);
      std::vector<int64_t> shape = dotGeneralShapeComputation(
        lhs_shape, rhs_shape,
        lhs_batch_dim, rhs_batch_dim,
        lhs_contract_dim, rhs_contract_dim
      );

      std::vector<Attribute> precisionVec;

      for (auto& precision : precision_config) {
        switch (precision) {
          case 0:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(context, stablehlo::Precision::DEFAULT)); break;
          case 1:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(context, stablehlo::Precision::HIGH)); break;
          case 2:
            precisionVec.push_back(stablehlo::PrecisionAttr::get(context, stablehlo::Precision::HIGHEST)); break;
        }
      }

      mlirOp = builder.create<stablehlo::DotGeneralOp>(
        builder.getUnknownLoc(),
        deriveOutputType(operands[1], shape),
        operands[0],
        operands[1],
        stablehlo::DotDimensionNumbersAttr::get(context, lhs_batch_dim, rhs_batch_dim, lhs_contract_dim, rhs_contract_dim),
        mlir::ArrayAttr::get(context, llvm::ArrayRef(precisionVec)),
        nullptr
      );
      break;
    }
    case tensat::Ops::SliceOp: {
      mlirOp = builder.create<stablehlo::SliceOp>(
        builder.getUnknownLoc(), operands[0], other_vecs[0], other_vecs[1], other_vecs[2]
      );
      break;
    }
    case tensat::Ops::ConcatenateOp: {
      mlirOp = builder.create<stablehlo::ConcatenateOp>(
        builder.getUnknownLoc(), operands, int_args[0] 
      );
      break;
    }
    case tensat::Ops::PadOp: {
      mlirOp = builder.create<stablehlo::PadOp>(builder.getUnknownLoc(), operands[0], operands[1], other_vecs[0], other_vecs[1], other_vecs[2]);
      break;
    }
    default:
      std::cout << "EGRAPH INVALID, UNSUPPORTED OP SHAPE REQUESTED" << "\n";
      assert(false);
      return nullptr;
  }

  return mlirOp;
}

// TODO: Avoid creating dummy inputs (we need them again for cost measurement, so duplicated)
uint64_t tensat::get_cost(
    tensat::Ops op,
    rust::Vec<tensat::Shape> operand_dims,
    rust::Vec<tensat::Type> operand_types,
    rust::Vec<tensat::Shape> other_vector_args,
    rust::Vec<int64_t> int_args) {
  auto context = OperationTimer::getContext();
  OpBuilder builder(context);

  // Create operands and other args
  SmallVector<Value> operands;
  for (const auto& [dim_slice, type] : llvm::zip(operand_dims, operand_types)) {
    auto tensor_type = tensat::newTensorType(builder, dim_slice, type);
    operands.push_back(OperationTimer::getDummyOp(builder, tensor_type)->getResult(0));
  }

  std::vector<std::vector<int64_t>> other_vecs;
  for (const auto& vec : other_vector_args)
    other_vecs.push_back(rust_vec_to_cpp_vector(vec));

  std::vector<int64_t> int_args_as_vec;
  for (const auto& num : int_args)
    int_args_as_vec.push_back(num);
  
  // Create the MLIR operation
  Operation* mlirOp = createStableHloOp(builder, op, operands, other_vecs, int_args_as_vec, context);

  if (mlirOp) {
    auto cost = OperationTimer::getCost(mlirOp, 100, 100);
    mlirOp->erase();
    return cost;
  }
  return 100000;
}

mlir::Type tensat::newTensorType(OpBuilder& builder, tensat::Shape &shape, tensat::Type type) {
  auto dims = shape.shape;
  auto dimsRef = llvm::ArrayRef(dims.data(), dims.size());
  auto mlirType = tensatTypeToMlirType(builder, type);
  return RankedTensorType::get(dimsRef, mlirType);
}

mlir::Type tensat::tensatTypeToMlirType(OpBuilder& builder, tensat::Type type) {
  switch (type) {
    case tensat::Type::i32:
      return builder.getI32Type();
    case tensat::Type::f32:
      return builder.getF32Type();
    default:
      assert(false);
  }
}

// SHAPE INFERENCE 
std::vector<int32_t> castArrayRefToInt32(llvm::ArrayRef<int64_t> shape) {
  std::vector<int32_t> dims;
  dims.reserve(shape.size());
  for (int64_t dim : shape) {
    dims.push_back(static_cast<int32_t>(dim));
  }
  return dims;
}

rust::Vec<int64_t> castArrayRefToRustVec(llvm::ArrayRef<int64_t> shape) {
  rust::Vec<int64_t> dims;
  for (int64_t dim : shape) {
    dims.push_back(static_cast<int64_t>(dim));
  }
  return dims;
}

rust::Vec<tensat::Shape> tensat::get_shape(
    Ops op,
    rust::Vec<tensat::Shape> operand_dims,
    rust::Vec<Type> operand_types,
    rust::Vec<tensat::Shape> other_vector_args,
    rust::Vec<int64_t> int_args) {
  auto context = OperationTimer::getContext();
  OpBuilder builder(context);

  // Create operands and other args
  SmallVector<Value> operands;
  for (const auto& [dim_slice, type] : llvm::zip(operand_dims, operand_types)) {
    auto tensor_type = newTensorType(builder, dim_slice, type);
    operands.push_back(OperationTimer::getDummyOp(builder, tensor_type)->getResult(0));
  }

  std::vector<std::vector<int64_t>> other_vecs;
  for (const auto& vec : other_vector_args)
    other_vecs.push_back(rust_vec_to_cpp_vector(vec));

  std::vector<int64_t> int_args_as_vec;
  for (const auto& num : int_args)
    int_args_as_vec.push_back(num);

  // Create the MLIR operation
  Operation* mlirOp = createStableHloOp(builder, op, operands, other_vecs, int_args_as_vec, context);
  if (mlirOp) {
    rust::Vec<tensat::Shape> shapes;
    for (auto res : mlirOp->getResults()) {
      auto output_tensor = res.getType().cast<TensorType>();
      auto shape = castArrayRefToRustVec(output_tensor.getShape());
      shapes.push_back({shape});
    }
    mlirOp->erase();
    return shapes;
  }
  return {};
}

// std::unique_ptr<tensat::ShapeInference> tensat::newShapeInference() {
//   return std::make_unique<tensat::ShapeInference>();
// }

namespace {
  class EqualitySaturationPass
    : public PassWrapper<EqualitySaturationPass, OperationPass<ModuleOp>> {
    public:
      StringRef getArgument() const override { return "equality-saturation-pass"; }
      StringRef getDescription() const override {
      return "Optimizes HLO graph using a Rust-based optimizer";
    }

    int getValueIndex(Operation* definingOp, Value &value) {
      auto results = definingOp->getResults();
      for (int i = 0; i < results.size(); i++) {
        if (results[i] == value) return i;
      }
      return -1;
    }

    tensat::TensorInfo* handleOperand(
        Value &operand,
        std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
        std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
        std::vector<Operation*> *blackboxIDToTensorInfo,
        OpBuilder &builder,
        Box<tensat::CppGraphConverter> &graph) {
      if (auto defOp = operand.getDefiningOp()) {
        // Use existing TensorInfo if already processed
        int index = getValueIndex(defOp, operand);
        assert(index >= 0);
        auto convertedOperand = dfs(defOp, opToTensorInfo, blockArgToTensorInfo, blackboxIDToTensorInfo, builder, graph);
        if (index == 0) {
          return convertedOperand;
        } else {
          auto indexOperand = graph->new_index(index, *convertedOperand).into_raw();
          opToTensorInfo->insert({defOp, indexOperand});
          return indexOperand;
        }
      } else if (auto arg = operand.dyn_cast<BlockArgument>()) {
        // Handle BlockArguments which represent function parameters
        if (isa<TensorType>(operand.getType())) {
          int32_t block_arg_number = arg.getArgNumber();
          auto &tensorInfo = (*blockArgToTensorInfo)[block_arg_number];
          if (!tensorInfo) {
            auto shape = operand.getType().cast<TensorType>().getShape();
            auto dims = castArrayRefToInt32(shape);
            auto input_slice = rust::Slice<const int32_t>{
              dims.data(), static_cast<size_t>(dims.size())};
            tensorInfo = graph->new_input(block_arg_number, input_slice).into_raw();
            (*blockArgToTensorInfo)[block_arg_number] = tensorInfo;
          }
          return tensorInfo;
        } else {
          std::cout
            << "EqualitySaturationPass does not support this argument type!"
            << "\n";
          operand.getType().dump();
          assert(false);
        }
      }
      std::cout
        << "EqualitySaturationPass: encountered operand that is neither the result of an Op nor a BlockArgument."
        << "\n";
      assert(false);
    }

// // Handle integral types, simply return the operand as-is
// template<typename T>
// typename std::enable_if<std::is_integral<T>::value, T>::type
// handleOperand(T operand, std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
//               std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
//               std::vector<Operation*> *blackboxIDToTensorInfo,
//               OpBuilder &builder, 
//               Box<tensat::CppGraphConverter> &graph) {
//     return operand;
// }

    // Handle rust::Slice<const int> directly, assuming we can pass it through without modification
    std::unique_ptr<rust::Slice<int>> handleOperand(
        std::unique_ptr<rust::Slice<int>> operand,
        std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
        std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
        std::vector<Operation*> *blackboxIDToTensorInfo,
        OpBuilder &builder, 
        Box<tensat::CppGraphConverter> &graph) {
        return operand;
    }

    template <typename CreateOpFunc, typename... Args>
    tensat::TensorInfo* handleOperation(
        Operation* op,
        CreateOpFunc createOpFunc,
        std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
        std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
        std::vector<Operation*> *blackboxIDToTensorInfo,
        OpBuilder &builder,
        Box<tensat::CppGraphConverter> &graph,
        Args&&... args) {
      auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
      auto handleArgs = [&](auto&&... operands) {
        return std::make_tuple(handleOperand(operands, opToTensorInfo, blockArgToTensorInfo, blackboxIDToTensorInfo, builder, graph)...);
      };

      auto operandInfos = std::apply(handleArgs, args_tuple);

      // Use std::apply to unpack operandInfos into the function call
      return std::apply([&](auto&&... unpacked) {
        return std::invoke(createOpFunc, *graph, *unpacked...).into_raw();
      }, operandInfos);
    }

    tensat::TensorInfo *dfs(Operation* op,
      std::unordered_map<Operation*, tensat::TensorInfo*> *opToTensorInfo,
      std::unordered_map<int, tensat::TensorInfo*> *blockArgToTensorInfo,
      std::vector<Operation*> *blackboxIDToTensorInfo,
      OpBuilder &builder,
      Box<tensat::CppGraphConverter> &graph) {
      // std::cout << "DFS AT " << op->getName().getStringRef().str() << "\n";
      if (opToTensorInfo->find(op) != opToTensorInfo->end()) {
        return opToTensorInfo->at(op);
      }
      tensat::TensorInfo *tensorInfo = nullptr;
      auto handleOperandPartial = [&](auto operand) {
        return handleOperand(operand, opToTensorInfo, blockArgToTensorInfo, blackboxIDToTensorInfo, builder, graph); 
      };
      // auto handleOperationPartial = [&](auto&& createOpFunc, auto&&... operands) {
      //   return handleOperation(op, createOpFunc, opToTensorInfo, blockArgToTensorInfo, blackboxIDToTensorInfo, builder, graph, std::forward<decltype(operands)>(operands)...);
      // }; 

      /*
      if (isa<stablehlo::ConstantOp>(op)) {
        auto constant = cast<stablehlo::ConstantOp>(op);
        auto output_tensor = constant->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };

        tensorInfo = graph->new_constant_op(shape).into_raw();
      } */
      if (isa<stablehlo::MulOp>(op)) {
        auto mul = cast<stablehlo::MulOp>(op);
        auto output_tensor = mul->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	      auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_mul_op(*handleOperandPartial(mul.getLhs()), *handleOperandPartial(mul.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::SubtractOp>(op)) {
        auto subtract = cast<stablehlo::SubtractOp>(op);
        auto output_tensor = subtract->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	      auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_subtract_op(*handleOperandPartial(subtract.getLhs()), *handleOperandPartial(subtract.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::DivOp>(op)) {
        auto div = cast<stablehlo::DivOp>(op);
        auto output_tensor = div->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	      auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_div_op(*handleOperandPartial(div.getLhs()), *handleOperandPartial(div.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::AddOp>(op)) {
        auto add = cast<stablehlo::AddOp>(op);
        auto output_tensor = add->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	      auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_add_op(*handleOperandPartial(add.getLhs()), *handleOperandPartial(add.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::MinOp>(op)) {
        auto min = cast<stablehlo::MinOp>(op);
        auto output_tensor = min->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	      auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_min_op(*handleOperandPartial(min.getLhs()), *handleOperandPartial(min.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::MaxOp>(op)) {
        auto max = cast<stablehlo::MaxOp>(op);
        auto output_tensor = max->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	      auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_max_op(*handleOperandPartial(max.getLhs()), *handleOperandPartial(max.getRhs()), shape).into_raw();
      } else if (isa<stablehlo::TanhOp>(op)) {
        auto tanh = cast<stablehlo::TanhOp>(op);
        auto output_tensor = tanh->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	      auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_tanh_op(*handleOperandPartial(tanh.getOperand()), shape).into_raw();
      } else if (isa<stablehlo::NegOp>(op)) {
        auto neg = cast<stablehlo::NegOp>(op);
        auto output_tensor = neg->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	      auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_neg_op(*handleOperandPartial(neg.getOperand()), shape).into_raw();
      } else if (isa<stablehlo::ExpOp>(op)) {
        auto exp = cast<stablehlo::ExpOp>(op);
        auto output_tensor = exp->getResult(0).getType().cast<TensorType>();
        auto shape_array = castArrayRefToInt32(output_tensor.getShape());
	      auto shape = rust::Slice<const int>{ shape_array.data(), shape_array.size() };
        tensorInfo = graph->new_exp_op(*handleOperandPartial(exp.getOperand()), shape).into_raw();
      } else if (isa<stablehlo::TransposeOp>(op)) {
        auto transpose = cast<stablehlo::TransposeOp>(op);
        std::vector<int32_t> permutation = castArrayRefToInt32(transpose.getPermutation());
        auto permutation_slice = rust::Slice<const int> {
          permutation.data(), static_cast<size_t>(permutation.size())};
        auto output_shape = castArrayRefToInt32(transpose->getResult(0).getType().cast<TensorType>().getShape());
	      auto output_shape_slice = rust::Slice<const int> {
		      output_shape.data(), output_shape.size() };
        tensorInfo = graph->new_transpose_op(
          *handleOperandPartial(transpose.getOperand()),
          permutation_slice,
          output_shape_slice
        ).into_raw();
      } else if (isa<stablehlo::ReshapeOp>(op)) {
        auto reshape = cast<stablehlo::ReshapeOp>(op);
        if (auto output_tensor = reshape.getResult().getType().cast<TensorType>()) {
          auto shape = castArrayRefToInt32(output_tensor.getShape());
          auto output_shape_slice = rust::Slice<const int> {
            shape.data(), shape.size()};
          tensorInfo = graph->new_reshape_op(
            *handleOperandPartial(reshape.getOperand()),
            output_shape_slice
          ).into_raw();
        } else {
          std::cout << "EqualitySaturationPass: result of stablehlo::ReshapeOp has non-tensor type" << std::endl;
        }
      } else if (isa<stablehlo::IotaOp>(op)) {
        auto iota = cast<stablehlo::IotaOp>(op);
        int32_t iota_dimension = iota.getIotaDimension();
        if (auto output_tensor = iota.getResult().getType().cast<TensorType>()) {
          auto shape = castArrayRefToInt32(output_tensor.getShape());
          auto output_shape_slice = rust::Slice<const int>{
            shape.data(), static_cast<size_t>(shape.size())};
          tensorInfo = graph->new_iota_op(
            iota_dimension,
            output_shape_slice
          ).into_raw();
        } else {
          std::cout << "EqualitySaturationPass: result of stablehlo::IotaOp has non-tensor type" << std::endl;
        }
      } else if (isa<stablehlo::DotGeneralOp>(op)) {
        // we might need more guards here
        auto dot_general = cast<stablehlo::DotGeneralOp>(op);
        auto dot_dim_attrs = dot_general.getDotDimensionNumbersAttr();
        auto lhs_batch_dim = castArrayRefToInt32(dot_dim_attrs.getLhsBatchingDimensions());
        auto rhs_batch_dim = castArrayRefToInt32(dot_dim_attrs.getRhsBatchingDimensions());
        auto lhs_contract_dim = castArrayRefToInt32(dot_dim_attrs.getLhsContractingDimensions());
        auto rhs_contract_dim = castArrayRefToInt32(dot_dim_attrs.getRhsContractingDimensions());
       
        mlir::ArrayAttr precision = dot_general.getPrecisionConfig().value_or(mlir::ArrayAttr());
        std::vector<int> precision_configs;
        for (int i = 0; i < precision.size(); i++) {
          auto precisionAttr = precision[i].dyn_cast<mlir::stablehlo::PrecisionAttr>();
          if (!precisionAttr) continue;  // Skip if it's not a PrecisionAttr, although such attributes should not exist here
          mlir::stablehlo::Precision val = precisionAttr.getValue();
          switch (val) {
            case mlir::stablehlo::Precision::DEFAULT:
              precision_configs.push_back(0);
              break;
            case mlir::stablehlo::Precision::HIGH:
              precision_configs.push_back(1);
              break;
            case mlir::stablehlo::Precision::HIGHEST:
              precision_configs.push_back(2);
              break;
          }
        }
        auto precision_config_slice = rust::Slice<const int>{
          precision_configs.data(), precision_configs.size()};

        if (auto output_tensor = dot_general.getResult().getType().cast<TensorType>()) {
          auto shape = castArrayRefToInt32(output_tensor.getShape());
          auto output_shape_slice = rust::Slice<const int> {
            shape.data(), shape.size()};

          tensorInfo = graph->new_dot_general_op(
            *handleOperandPartial(dot_general.getLhs()),
            *handleOperandPartial(dot_general.getRhs()),
            {lhs_batch_dim.data(), lhs_batch_dim.size()},
            {rhs_batch_dim.data(), rhs_batch_dim.size()},
            {lhs_contract_dim.data(), lhs_contract_dim.size()},
            {rhs_contract_dim.data(), rhs_contract_dim.size()},
            precision_config_slice,
            output_shape_slice
          ).into_raw();
        } else {
          std::cout << "EqualitySaturationPass: result of stablehlo::DotGeneralOp has non-tensor type" << std::endl;
        }
      } else if (isa<stablehlo::ConcatenateOp>(op)) {
        auto concat = cast<stablehlo::ConcatenateOp>(op);
        auto output_tensor = concat->getResult(0).getType().cast<TensorType>();
        auto output_shape_array = castArrayRefToInt32(output_tensor.getShape());
        std::vector<tensat::TensorInfo*> inputs;
        for (auto input : concat.getInputs()) {
          inputs.push_back(handleOperandPartial(input));
        }
        int32_t dimension = concat.getDimension();
        tensorInfo = graph->new_concatenate_op(
          { inputs.data(), inputs.size() },
          dimension,
          { output_shape_array.data(), output_shape_array.size() }
        ).into_raw();
      } else if (isa<stablehlo::SliceOp>(op)) {
        auto slice = cast<stablehlo::SliceOp>(op);
        auto output_tensor = slice->getResult(0).getType().cast<TensorType>();
        auto output_shape_array = castArrayRefToInt32(output_tensor.getShape());
        std::vector<tensat::TensorInfo*> inputs;
        auto operand = handleOperandPartial(slice.getOperand());
        auto start_indices = castArrayRefToInt32(slice.getStartIndices());
        auto limit_indices = castArrayRefToInt32(slice.getLimitIndices());
        auto strides = castArrayRefToInt32(slice.getStrides());
        tensorInfo = graph->new_slice_op(
          *operand,
          { start_indices.data(), start_indices.size() },
          { limit_indices.data(), limit_indices.size() },
          { strides.data(), strides.size() },
          { output_shape_array.data(), output_shape_array.size() }
        ).into_raw();
      } else if (isa<stablehlo::PadOp>(op)) {
        auto pad = cast<stablehlo::PadOp>(op);
        auto output_tensor = pad->getResult(0).getType().cast<TensorType>();
        auto output_shape_array = castArrayRefToInt32(output_tensor.getShape());
        auto operand = handleOperandPartial(pad.getOperand());
        auto padding_value = handleOperandPartial(pad.getPaddingValue());
        auto edge_padding_low = castArrayRefToInt32(pad.getEdgePaddingLow());
        auto edge_padding_high = castArrayRefToInt32(pad.getEdgePaddingHigh());
        auto interior_padding = castArrayRefToInt32(pad.getInteriorPadding());
        tensorInfo = graph->new_pad_op(
          *operand,
          *padding_value,
          { edge_padding_low.data(), edge_padding_low.size() },
          { edge_padding_high.data(), edge_padding_high.size() },
          { interior_padding.data(), interior_padding.size() },
          { output_shape_array.data(), output_shape_array.size() }
        ).into_raw();
      } else if (isa<func::ReturnOp>(op)) {
        rust::vec<tensat::Shape> shapes;
        int numOperands = op->getNumOperands();
        std::vector<tensat::TensorInfo *> processedOperands;
        for (size_t i = 0; i < numOperands; i++) {
          auto operand = handleOperandPartial(op->getOperand(i));
          processedOperands.push_back(operand);
        }
        auto operandPtrsSlice = rust::Slice<tensat::TensorInfo *const>{
            processedOperands.data(),
            static_cast<size_t>(processedOperands.size())};
        tensorInfo = graph->new_return_op(operandPtrsSlice).into_raw();
      } else {
        int numOperands = op->getNumOperands();
        rust::vec<tensat::Shape> shapes;
        for (auto result : op->getResults()) {
          auto output_tensor = result.getType().cast<TensorType>();
          auto shape_array = castArrayRefToRustVec(output_tensor.getShape());
          shapes.push_back(tensat::Shape {shape_array});
        }
        auto output_tensor = op->getResult(0).getType().cast<TensorType>();
        std::vector<tensat::TensorInfo*> processedOperands;
        auto copy = op->clone(Operation::CloneOptions(/* cloneRegions = */ true, /* cloneOperands = */ false));
        blackboxIDToTensorInfo->push_back(copy);
        int blackboxOpID = blackboxIDToTensorInfo->size()-1;
        for (size_t i = 0; i < numOperands; i++) {
          auto operand = handleOperandPartial(op->getOperand(i));
          processedOperands.push_back(operand);
        }
        auto operandPtrsSlice = rust::Slice<tensat::TensorInfo* const>{processedOperands.data(), static_cast<size_t>(processedOperands.size())};
        tensorInfo = graph->new_blackbox_op(
          operandPtrsSlice,
          blackboxOpID,
          shapes
        ).into_raw();
      }
      if (tensorInfo != nullptr) {
        opToTensorInfo->insert({op, tensorInfo});
        return tensorInfo;
      }
      return tensorInfo;
    }

    Box<tensat::CppGraphConverter> createEgraph(
        std::vector<Operation*> *blackboxIDToTensorInfo,
        OpBuilder &builder,
        ModuleOp module) {

      auto graph = tensat::new_converter();
      // members of the class
      std::unordered_map<Operation*, tensat::TensorInfo*> opToTensorInfo;
      std::unordered_map<int, tensat::TensorInfo*> blockArgToTensorInfo;

      module.walk([&](func::ReturnOp op) {
        dfs(op, &opToTensorInfo, &blockArgToTensorInfo, blackboxIDToTensorInfo, builder, graph);
      });

      // graph->print_rec_expr();
      return graph;
    }

    template <typename T>
    Operation* createUnaryOp(OpBuilder &builder, std::vector<Value>& opVals, tensat::Node &node) {
      auto location = builder.getUnknownLoc();
      return builder.create<T>(location, opVals[node.operands[0]]);
    }

    template <typename T>
    Operation* createBinaryOp(OpBuilder &builder, std::vector<Value>& opVals, tensat::Node &node) {
      auto location = builder.getUnknownLoc();
      return builder.create<T>(location, opVals[node.operands[0]], opVals[node.operands[1]]);
    }

    /**
     * Parse the Vec nodes with Nums (e.g Vec(Num(128), Num(128))) emitted by tensat node construction.
     */
    std::vector<int64_t> parseNumVec(rust::vec<tensat::Node> &nodes, tensat::Node &seq) {
      assert(seq.name == "Vec");
      std::vector<int64_t> result;
      
      for (auto i : seq.operands) {
        assert(nodes[i].name == "Num");
        result.push_back(nodes[i].operands[0]);
      }

      return result;
    }

    /**
     * Parse the Vec nodes with arbitrary operations (e.g Vec(Input(...), AddOp(...))) emitted by tensat node construction.
     */
    std::vector<Value> parseOpVec(std::vector<Value> &opVals, tensat::Node &seq) {
      assert(seq.name == "Vec");
      std::vector<Value> result;

      for (auto i : seq.operands) {
        assert(opVals[i] != nullptr);
        result.push_back(opVals[i]);
      }
      
      return result;
   }


    void reconstructStablehlo(ModuleOp *root, std::vector<Operation*> *blackboxIDToTensorInfo, rust::vec<tensat::Node> &nodes, OpBuilder &builder) {
      auto context = root->getContext();
      std::vector<Value> opVals;

      // Find funcOp to get the block.
      func::FuncOp funcOp;

      for (auto &op : root->getBody()->getOperations()) {
        if (isa<func::FuncOp>(op)) {
          funcOp = cast<func::FuncOp>(op);
          break;
        }
      }

      auto& region = funcOp.getRegion();
      auto& block = funcOp.getRegion().front();

      block.clear();

      auto location = builder.getUnknownLoc();

      for (auto& node : nodes) {
        Operation* newOp = nullptr;
        // Create the new operation based on the operands
        if (node.name == "Var" || node.name == "Num" || node.name == "Vec") {
          /* do nothing */
        } else if (node.name == "Input") {
          int blockArgNumber = nodes[node.operands[1]].operands[0];
          opVals.push_back(block.getArgument(blockArgNumber));
          continue;
        } else if (node.name == "Index") {
          int index = nodes[node.operands[0]].operands[0];
          int input = node.operands[1];
          opVals.push_back(opVals[input].getDefiningOp()->getResult(index));
          continue;
        } else if (node.name == "NegOp") {
          newOp = createUnaryOp<stablehlo::NegOp>(builder, opVals, node);
        } else if (node.name == "TanhOp") {
          newOp = createUnaryOp<stablehlo::TanhOp>(builder, opVals, node);
        } else if (node.name == "ExpOp") {
          newOp = createUnaryOp<stablehlo::ExpOp>(builder, opVals, node);
        } else if (node.name == "AddOp") {
          newOp = createBinaryOp<stablehlo::AddOp>(builder, opVals, node);
        } else if (node.name == "SubtractOp") {
          newOp = createBinaryOp<stablehlo::SubtractOp>(builder, opVals, node);
        } else if (node.name == "MulOp") {
          newOp = createBinaryOp<stablehlo::MulOp>(builder, opVals, node);
        } else if (node.name == "DivOp") {
          newOp = createBinaryOp<stablehlo::DivOp>(builder, opVals, node);
        } else if (node.name == "MinOp") {
          newOp = createBinaryOp<stablehlo::MinOp>(builder, opVals, node);
        } else if (node.name == "MaxOp") {
          newOp = createBinaryOp<stablehlo::MaxOp>(builder, opVals, node);
        } else if (node.name == "TransposeOp") {
          auto input = opVals[node.operands[0]];
          auto permutation = parseNumVec(nodes, nodes[node.operands[1]]);
          newOp = builder.create<stablehlo::TransposeOp>(location, input, permutation);
        } else if (node.name == "ReshapeOp") {
          auto input = opVals[node.operands[0]];
          auto shape = parseNumVec(nodes, nodes[node.operands[1]]);
          auto newType = deriveOutputType(input, shape);
          newOp = builder.create<stablehlo::ReshapeOp>(location, newType, input);
        } else if (node.name == "DotGeneralOp") {
          auto lhs = opVals[node.operands[0]];
          auto rhs = opVals[node.operands[1]];

          auto lhsBatchDim = parseNumVec(nodes, nodes[node.operands[2]]);
          auto rhsBatchDim = parseNumVec(nodes, nodes[node.operands[3]]);
          auto lhsContractDim = parseNumVec(nodes, nodes[node.operands[4]]);
          auto rhsContractDim = parseNumVec(nodes, nodes[node.operands[5]]);
          auto precisionConfig = parseNumVec(nodes, nodes[node.operands[6]]);
          auto lhsShape = getShape(lhs);
          auto rhsShape = getShape(rhs);
          auto shape = dotGeneralShapeComputation(lhsShape, rhsShape, lhsBatchDim, rhsBatchDim, lhsContractDim, rhsContractDim);

          auto dotDimensionNumbersAttr = stablehlo::DotDimensionNumbersAttr::get(context, lhsBatchDim, rhsBatchDim, lhsContractDim, rhsContractDim);
          
          std::vector<Attribute> precisionVec;

          for (auto& precision : precisionConfig) {
            switch (precision) {
              case 0:
                precisionVec.push_back(stablehlo::PrecisionAttr::get(context, stablehlo::Precision::DEFAULT)); break;
              case 1:
                precisionVec.push_back(stablehlo::PrecisionAttr::get(context, stablehlo::Precision::HIGH)); break;
              case 2:
                precisionVec.push_back(stablehlo::PrecisionAttr::get(context, stablehlo::Precision::HIGHEST)); break;
            }
          }

          // TODO: Is lhs correct here?
          auto newType = deriveOutputType(lhs, shape);
          newOp = builder.create<stablehlo::DotGeneralOp>(location, newType, lhs, rhs, dotDimensionNumbersAttr, mlir::ArrayAttr::get(context, llvm::ArrayRef(precisionVec)), nullptr);
        } else if (node.name == "ConcatenateOp") {
          auto inputs = parseOpVec(opVals, nodes[node.operands[0]]);
          auto dimension = nodes[node.operands[1]].operands[0];
          newOp = builder.create<stablehlo::ConcatenateOp>(location, inputs, dimension);
        } else if (node.name == "SliceOp") {
          auto operand = opVals[node.operands[0]];
          auto startIndices = parseNumVec(nodes, nodes[node.operands[1]]);
          auto limitIndices = parseNumVec(nodes, nodes[node.operands[2]]);
          auto strides = parseNumVec(nodes, nodes[node.operands[3]]);
          newOp = builder.create<stablehlo::SliceOp>(location, operand, startIndices, limitIndices, strides);
        } else if (node.name == "PadOp") {
          auto operand = opVals[node.operands[0]];
          auto paddingValue = opVals[node.operands[1]];
          auto edgePaddingLow = parseNumVec(nodes, nodes[node.operands[2]]);
          auto edgePaddingHigh = parseNumVec(nodes, nodes[node.operands[3]]);
          auto interiorPadding = parseNumVec(nodes, nodes[node.operands[4]]);
          newOp = builder.create<stablehlo::PadOp>(location, operand, paddingValue, edgePaddingLow, edgePaddingHigh, interiorPadding);
        } else if (node.name == "ReturnOp") {
          auto inputs = parseOpVec(opVals, nodes[node.operands[0]]);
          newOp = builder.create<func::ReturnOp>(location, inputs);       
        } else if (node.name == "blackbox") {
          assert(node.operands.size() > 0);
          size_t numOperands = node.operands.size() - 1;
          assert(nodes[node.operands[numOperands]].name == "Num");
          auto blackboxID = nodes[node.operands[numOperands]].operands[0];
          newOp = blackboxIDToTensorInfo->at(blackboxID);
          assert(numOperands == newOp->getNumOperands());
    
          // Really subtle error arose here from not handling Num properly.
          // We might want to have a Num hashmap 
          std::vector<Value> operands;
          for (size_t i = 0; i < numOperands; ++i) {
            auto operandIndex = node.operands[i];
            auto operand = opVals[operandIndex];
            operands.push_back(operand);
          }
          newOp->insertOperands(0, operands);
        } else {
          // TODO: implement other operations
          std::cout << "UNIMPLEMENTED " << node.name << "\n";
        }
        if (newOp) {
          block.push_back(newOp);
          opVals.push_back(newOp->getResult(0));
        } else {
          // This is bad practice, as we're pushing nullptr
          // to ops in case of Input, Num, or Var nodes. This
          // is unsafe, but maintains indexing. We could use
          // some llvm no-op, but that would not be much better.
          opVals.push_back(nullptr);
        }
      }
    }

    void runOnOperation() override {
      ModuleOp module = getOperation();
      // std::cout << "ORIGINAL MODULE" << "\n";
      // module.dump();
      std::vector<Operation*> blackboxIDToTensorInfo;
      auto context = module->getContext();
      OpBuilder builder(context);
      auto graph = createEgraph(&blackboxIDToTensorInfo, builder, module);
      auto optimized = graph->optimize();

      // std::cout << "reconstructing\n";
      reconstructStablehlo(&module, &blackboxIDToTensorInfo, optimized, builder);
      // std::cout << "Optimised module" << "\n";
      // module.dump();
    }
  };
}  // end anonymous namespace

namespace mlir {
  namespace enzyme {
    std::unique_ptr<Pass> createEqualitySaturationPass() {
      return std::make_unique<EqualitySaturationPass>();
    }
  }  // end namespace enzyme
}  // end namespace mlir
