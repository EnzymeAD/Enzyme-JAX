// #include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/LinalgUtils.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
// #include "stablehlo/dialect/StablehloOps.h"
// #include "llvm/ADT/DynamicAPInt.h"
// #include "llvm/ADT/SetVector.h"
// #include "llvm/ADT/SmallVector.h"
// #include "llvm/Support/ErrorHandling.h"
// #include "llvm/Support/LogicalResult.h"
// #include "llvm/Support/MathExtras.h"
// #include <algorithm>
// #include <cstdint>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEXLAMPIPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

static FailureOr<int32_t> mapMPIToNCCLDatatype(enzymexla::MPIDatatype dt) {
  switch (dt) {
  case enzymexla::MPIDatatype::MPI_INT32_T:
  case enzymexla::MPIDatatype::MPI_INT:
    return 2; // ncclInt32
  case enzymexla::MPIDatatype::MPI_UINT32_T:
  case enzymexla::MPIDatatype::MPI_UNSIGNED:
    return 3; // ncclUint32
  case enzymexla::MPIDatatype::MPI_INT64_T:
  case enzymexla::MPIDatatype::MPI_LONG_LONG_INT:
    return 4; // ncclInt64
  case enzymexla::MPIDatatype::MPI_UINT64_T:
  case enzymexla::MPIDatatype::MPI_UNSIGNED_LONG_LONG:
    return 5; // ncclUint64
  case enzymexla::MPIDatatype::MPI_FLOAT:
    return 7; // ncclFloat32
  case enzymexla::MPIDatatype::MPI_DOUBLE:
    return 8; // ncclFloat64
  default:
    return failure();
  }
}

static FailureOr<int32_t> mapMPIToNCCLRedOp(enzymexla::MPIOp op) {
  switch (op) {
  case enzymexla::MPIOp::MPI_SUM:
    return 0; // ncclSum
  case enzymexla::MPIOp::MPI_PROD:
    return 1; // ncclProd
  case enzymexla::MPIOp::MPI_MAX:
    return 2; // ncclMax
  case enzymexla::MPIOp::MPI_MIN:
    return 3; // ncclMin
  default:
    return failure();
  }
}

struct MPICommRankOpLowering
    : public OpRewritePattern<enzymexla::MPICommRankOp> {

  std::string backend;
  MPICommRankOpLowering(std::string backend, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPICommRankOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);
      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Comm_rank";

      // For now we just hard code MPI_COMM_WORLD as the communicator.
      // TODO make this more flexible
      std::string communicatorName = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper_MPI_Comm_rank LLVM function body
      std::string wrapperFunctionName = "enzymexla_wrapper_" + mpiFunctionName;

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the function type
        auto funcType =
            LLVM::LLVMFunctionType::get(llvmVoidType,  // void return type
                                        {llvmPtrType}, // pointer parameter
                                        false);        // is variadic: false

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute
        wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
                               memoryEffectsAttr);

        // Get the rank pointer from the argument
        Value rankPtr = entryBlock->getArgument(0);

        // Get the address of the communicator
        // NOTE these symbols are not ABI-stable until MPI 5.0, but in practice,
        // they are represented as word-size values (i.e. `int` or ptr)
        Value addressOfComm = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, communicatorName);

        // TODO error checking
        // MPI_Comm_rank returns i32 error code which we're ignoring here
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(context, mpiFunctionName),
                             ValueRange{addressOfComm, rankPtr});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Comm_rank function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type, {llvmPtrType, llvmPtrType}, false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               communicatorName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Create a constant tensor to hold the result
      auto tensorType = llvm::cast<RankedTensorType>(op->getResultTypes()[0]);
      auto constantAttr =
          DenseIntElementsAttr::get(tensorType, ArrayRef<int32_t>{-1});
      Value constantTensor = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), tensorType, constantAttr);

      // Call the LLVM function with enzymexla.jit_call
      auto aliasAttr = stablehlo::OutputOperandAliasAttr::get(
          context,
          /*outputTupleIndices=*/ArrayRef<int64_t>{},
          /*operandIndex=*/0,
          /*operandTupleIndices=*/ArrayRef<int64_t>{});

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{constantTensor}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr({aliasAttr}),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall.getResult(0));

      return success();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct MPICommSizeOpLowering
    : public OpRewritePattern<enzymexla::MPICommSizeOp> {

  std::string backend;
  MPICommSizeOpLowering(std::string backend, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPICommSizeOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);

      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Comm_size";

      // For now we just hard code MPI_COMM_WORLD as the communicator.
      // TODO make this more flexible
      std::string communicatorName = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper_MPI_Comm_size LLVM function body
      std::string wrapperFunctionName = "enzymexla_wrapper_" + mpiFunctionName;

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the function type
        auto funcType =
            LLVM::LLVMFunctionType::get(llvmVoidType,  // void return type
                                        {llvmPtrType}, // parameter types
                                        false);        // is variadic: false

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute
        wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
                               memoryEffectsAttr);

        // Get the first (and only) argument of the function
        Value sizePtr = entryBlock->getArgument(0);

        // Get the address of the communicator
        // NOTE these symbols are not ABI-stable until MPI 5.0, but in practice,
        // they are represented as w ord-size values (i.e. `int` or ptr)
        Value addressOfComm = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, communicatorName);

        // TODO error checking
        // MPI_Comm_size returns i32 error code which we're ignoring here
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(context, mpiFunctionName),
                             ValueRange{addressOfComm, sizePtr});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Comm_size function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type, {llvmPtrType, llvmPtrType}, false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               communicatorName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Create a constant tensor to hold the result
      auto tensorType = llvm::cast<RankedTensorType>(op->getResultTypes()[0]);
      auto constantAttr =
          DenseIntElementsAttr::get(tensorType, ArrayRef<int32_t>{-1});
      Value constantTensor = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), tensorType, constantAttr);

      // Call the LLVM function with enzymexla.jit_call
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{},
          /*operand_index=*/0,
          /*operand_tuple_indices=*/std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{constantTensor}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct MPIBarrierOpLowering : public OpRewritePattern<enzymexla::MPIBarrierOp> {

  std::string backend;
  MPIBarrierOpLowering(std::string backend, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPIBarrierOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);

      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Barrier";

      // TODO For now we just hard code MPI_COMM_WORLD as the communicator.
      std::string communicatorName = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper_MPI_Barrier LLVM function body
      std::string wrapperFunctionName = "enzymexla_wrapper_" + mpiFunctionName;

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the function type
        auto funcType = LLVM::LLVMFunctionType::get(llvmVoidType, {}, false);

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Get the address of the communicator
        // NOTE these symbols are not ABI-stable until MPI 5.0, but in practice,
        // they are represented as w ord-size values (i.e. `int` or ptr)
        Value addressOfComm = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, communicatorName);

        // Call MPI_Barrier
        // int MPI_Barrier(MPI_Comm comm)
        // TODO returns i32 error code which we're ignoring here
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(context, mpiFunctionName),
                             ValueRange{addressOfComm});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Barrier function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType =
            LLVM::LLVMFunctionType::get(i32Type, {llvmPtrType}, false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               communicatorName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Call the LLVM function with enzymexla.jit_call
      enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{},
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/nullptr,
          /*xla_side_effect_free=*/nullptr);

      rewriter.eraseOp(op);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct MPISendOpLowering : public OpRewritePattern<enzymexla::MPISendOp> {

  std::string backend;
  MPISendOpLowering(std::string backend, MLIRContext *context,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPISendOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);

      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Send";

      // get the MPI datatype
      auto datatype = op.getDatatype();
      StringRef datatypeName = stringifyMPIDatatype(datatype);

      // For now we just hard code MPI_COMM_WORLD as the communicator.
      // TODO make this more flexible
      std::string communicatorName = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName =
          "enzymexla_wrapper_" + mpiFunctionName + "_" + datatypeName.str();

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType},
            false);

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 4; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects",
                                 memoryEffectsAttr);
        }

        // Get the function arguments
        Value bufPtr = entryBlock->getArgument(0);
        Value countPtr = entryBlock->getArgument(1);
        Value destPtr = entryBlock->getArgument(2);
        Value tagPtr = entryBlock->getArgument(3);

        // Load the count, dest, tag values
        Value count =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, countPtr);

        Value dest =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, destPtr);

        Value tag =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, tagPtr);

        // Get the address of the datatype
        // NOTE these symbols are not ABI-stable until MPI 5.0, but in practice,
        // they are represented as w ord-size values (i.e. `int` or ptr)
        Value addressOfDtype = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, datatypeName);

        // Get the address of the communicator
        Value addressOfComm = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, communicatorName);

        // Call MPI_Send
        // int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int
        //     dest, int tag, MPI_Comm comm)
        // TODO returns i32 error code which we're ignoring here
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(context, mpiFunctionName),
                             ValueRange{bufPtr, count, addressOfDtype, dest,
                                        tag, addressOfComm});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Send function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, i32Type, llvmPtrType, i32Type, i32Type, llvmPtrType},
            false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               communicatorName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Insert datatype declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(datatypeName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               datatypeName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Get all orinigal op operands
      auto operands = op.getOperands();

      // Call the LLVM function with enzymexla.jit_call
      enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{},
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{operands}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/nullptr,
          /*xla_side_effect_free=*/nullptr);

      rewriter.eraseOp(op);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct MPIRecvOpLowering : public OpRewritePattern<enzymexla::MPIRecvOp> {

  std::string backend;
  MPIRecvOpLowering(std::string backend, MLIRContext *context,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPIRecvOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);

      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Recv";

      // get the MPI datatype
      auto datatype = op.getDatatype();
      StringRef datatypeName = stringifyMPIDatatype(datatype);

      // For now we just hard code MPI_COMM_WORLD as the communicator.
      // TODO make this more flexible
      std::string communicatorName = "MPI_COMM_WORLD";

      std::string statusName = "MPI_STATUS_IGNORE";

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName =
          "enzymexla_wrapper_" + mpiFunctionName + "_" + datatypeName.str();

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType},
            false);

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 4; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects",
                                 memoryEffectsAttr);
        }

        // Get the function arguments
        Value bufPtr = entryBlock->getArgument(0);
        Value countPtr = entryBlock->getArgument(1);
        Value srcPtr = entryBlock->getArgument(2);
        Value tagPtr = entryBlock->getArgument(3);

        // Load the count, src, tag values
        Value count =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, countPtr);

        Value src =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, srcPtr);

        Value tag =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, tagPtr);

        // Get the address of the datatype
        // NOTE these symbols are not ABI-stable until MPI 5.0, but in practice,
        // they are represented as w ord-size values (i.e. `int` or ptr)
        Value addressOfDtype = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, datatypeName);

        // Get the address of the communicator
        Value addressOfComm = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, communicatorName);

        // Get the address of the status
        Value addressOfStatus = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, statusName);

        // Call MPI_Recv
        // int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int
        //     source, int tag, MPI_Comm comm, MPI_Status* status)
        // TODO returns i32 error code which we're ignoring here
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(context, mpiFunctionName),
                             ValueRange{bufPtr, count, addressOfDtype, src, tag,
                                        addressOfComm, addressOfStatus});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Recv function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, i32Type, llvmPtrType, i32Type, i32Type, llvmPtrType,
             llvmPtrType},
            false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Insert MPI_STATUS_IGNORE declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               statusName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               communicatorName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Insert datatype declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(datatypeName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               datatypeName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Get all orinigal op operands
      auto operands = op.getOperands();

      // Call the LLVM function with enzymexla.jit_call
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{},
          /*operand_index=*/0,
          /*operand_tuple_indices=*/std::vector<int64_t>{}));

      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{operands}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct MPIIsendOpLowering : public OpRewritePattern<enzymexla::MPIIsendOp> {

  std::string backend;
  MPIIsendOpLowering(std::string backend, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPIIsendOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);

      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Isend";

      // get the MPI datatype
      auto datatype = op.getDatatype();
      StringRef datatypeName = stringifyMPIDatatype(datatype);

      // For now we just hard code MPI_COMM_WORLD as the communicator.
      // TODO make this more flexible
      std::string communicatorName = "MPI_COMM_WORLD";

      std::string wrapperFunctionName =
          "enzymexla_wrapper_" + mpiFunctionName + "_" + datatypeName.str();

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType},
            false);

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 5; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects",
                                 memoryEffectsAttr);
        }

        // Get the function arguments
        Value bufPtr = entryBlock->getArgument(0);
        Value countPtr = entryBlock->getArgument(1);
        Value destPtr = entryBlock->getArgument(2);
        Value tagPtr = entryBlock->getArgument(3);
        Value requestPtr = entryBlock->getArgument(4);

        // Load the count, dest, tag values
        Value count =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, countPtr);

        Value dest =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, destPtr);

        Value tag =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, tagPtr);

        // Get the address of the datatype
        // NOTE these symbols are not ABI-stable until MPI 5.0, but in practice,
        // they are represented as w ord-size values (i.e. `int` or ptr)
        Value addressOfDtype = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, datatypeName);

        // Get the address of the communicator
        Value addressOfComm = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, communicatorName);

        // Call MPI_Isend
        // int MPI_Isend(void* buf, int count, MPI_Datatype datatype, int
        //               dest, int tag, MPI_Comm comm, MPI_Request* request)
        // TODO returns i32 error code which we're ignoring here
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(context, mpiFunctionName),
                             ValueRange{bufPtr, count, addressOfDtype, dest,
                                        tag, addressOfComm, requestPtr});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Isend function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, i32Type, llvmPtrType, i32Type, i32Type, llvmPtrType,
             llvmPtrType},
            false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               communicatorName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Insert datatype declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(datatypeName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               datatypeName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Get all orinigal op operands
      auto opOperands = op.getOperands();

      // Create a constant tensor to hold request
      auto tensorType = RankedTensorType::get({}, i32Type);
      auto constantAttr =
          DenseIntElementsAttr::get(tensorType, ArrayRef<int32_t>{-1});
      Value constantTensor = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), tensorType, constantAttr);

      // Combine all operands
      SmallVector<Value> jitCallOperands(opOperands.begin(), opOperands.end());
      jitCallOperands.push_back(constantTensor);

      // Add request to output operand aliases
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{},
          /*operand_index=*/4,
          /*operand_tuple_indices=*/std::vector<int64_t>{}));

      // Call the LLVM function with enzymexla.jit_call
      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          jitCallOperands, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct MPIIrecvOpLowering : public OpRewritePattern<enzymexla::MPIIrecvOp> {

  std::string backend;
  MPIIrecvOpLowering(std::string backend, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPIIrecvOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);

      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Irecv";

      // get the MPI datatype
      auto datatype = op.getDatatype();
      StringRef datatypeName = stringifyMPIDatatype(datatype);

      // For now we just hard code MPI_COMM_WORLD as the communicator.
      // TODO make this more flexible
      std::string communicatorName = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName =
          "enzymexla_wrapper_" + mpiFunctionName + "_" + datatypeName.str();

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType, llvmPtrType},
            false);

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 5; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects",
                                 memoryEffectsAttr);
        }

        // Get the function arguments
        Value bufPtr = entryBlock->getArgument(0);
        Value countPtr = entryBlock->getArgument(1);
        Value srcPtr = entryBlock->getArgument(2);
        Value tagPtr = entryBlock->getArgument(3);
        Value requestPtr = entryBlock->getArgument(4);

        // Load the count, src, tag values
        Value count =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, countPtr);

        Value src =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, srcPtr);

        Value tag =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, tagPtr);

        // Get the address of the datatype
        // NOTE these symbols are not ABI-stable until MPI 5.0, but in practice,
        // they are represented as w ord-size values (i.e. `int` or ptr)
        Value addressOfDtype = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, datatypeName);

        // Get the address of the communicator
        Value addressOfComm = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, communicatorName);

        // Call MPI_Irecv
        // int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int
        //               source, int tag, MPI_Comm comm, MPI_Request* request)
        // TODO returns i32 error code which we're ignoring here
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(context, mpiFunctionName),
                             ValueRange{bufPtr, count, addressOfDtype, src, tag,
                                        addressOfComm, requestPtr});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Irecv function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, i32Type, llvmPtrType, i32Type, i32Type, llvmPtrType,
             llvmPtrType},
            false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               communicatorName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Insert datatype declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(datatypeName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               datatypeName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Get all orinigal op operands
      auto opOperands = op.getOperands();

      // Create a constant tensor to hold request
      auto tensorType = RankedTensorType::get({}, i32Type);
      auto constantAttr =
          DenseIntElementsAttr::get(tensorType, ArrayRef<int32_t>{-1});
      Value constantTensor = stablehlo::ConstantOp::create(
          rewriter, op.getLoc(), tensorType, constantAttr);

      // Combine all operands
      SmallVector<Value> jitCallOperands(opOperands.begin(), opOperands.end());
      jitCallOperands.push_back(constantTensor);

      // Add buffer to output operand aliases
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{0},
          /*operand_index=*/0,
          /*operand_tuple_indices=*/std::vector<int64_t>{}));

      // Add request to output operand aliases
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{1},
          /*operand_index=*/4,
          /*operand_tuple_indices=*/std::vector<int64_t>{}));

      // Call the LLVM function with enzymexla.jit_call
      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{jitCallOperands}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct MPIWaitOpLowering : public OpRewritePattern<enzymexla::MPIWaitOp> {

  std::string backend;
  MPIWaitOpLowering(std::string backend, MLIRContext *context,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPIWaitOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);

      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Wait";

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName = "enzymexla_wrapper_" + mpiFunctionName;

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType =
            LLVM::LLVMFunctionType::get(llvmVoidType, {llvmPtrType}, false);

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        wrapperFunc.setArgAttr(0, "enzymexla.memory_effects",
                               memoryEffectsAttr);

        // Get the function argument
        Value requestPtr = entryBlock->getArgument(0);

        // Allocate a 1x!llvm.array<6 x i32> that we use in place of MPI_Status
        // Size of status is implem dependendent, this should cover the max
        Value numElements = arith::ConstantOp::create(
            rewriter, op.getLoc(), i32Type, rewriter.getI32IntegerAttr(1));

        auto arrayType = LLVM::LLVMArrayType::get(i32Type, 6);

        Value statusPtr = LLVM::AllocaOp::create(
            rewriter, op.getLoc(), llvmPtrType, arrayType, numElements);

        // Call MPI_Wait
        // int MPI_Wait(MPI_Request* request, MPI_Status* status)
        // TODO returns i32 error code which we're ignoring here
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(context, mpiFunctionName),
                             ValueRange{requestPtr, statusPtr});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Wait function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type, {llvmPtrType, llvmPtrType}, false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Get the request operand
      auto request = op.getRequest();

      // Call the LLVM function with enzymexla.jit_call
      enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{},
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{request}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/nullptr,
          /*xla_side_effect_free=*/nullptr);

      rewriter.eraseOp(op);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct MPIWaitallOpLowering : public OpRewritePattern<enzymexla::MPIWaitallOp> {

  std::string backend;
  MPIWaitallOpLowering(std::string backend, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPIWaitallOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);

      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Waitall";
      auto requests = op.getRequests();
      unsigned numRequests = requests.size();

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName = "enzymexla_wrapper_" + mpiFunctionName +
                                        "_" + std::to_string(numRequests);

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        SmallVector<Type> wrapperArgumentTypes(numRequests, llvmPtrType);
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, wrapperArgumentTypes, false);

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < numRequests; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects",
                                 memoryEffectsAttr);
        }

        Value count = arith::ConstantOp::create(
            rewriter, op.getLoc(), i32Type,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(numRequests)));

        // Pack the scalar requests into the native contiguous request array.
        Value requestsPtr = LLVM::AllocaOp::create(rewriter, op.getLoc(),
                                                   llvmPtrType, i32Type, count);
        SmallVector<Value> requestElementPtrs;
        requestElementPtrs.reserve(numRequests);
        for (unsigned index = 0; index < numRequests; ++index) {
          Value requestPtr = entryBlock->getArgument(index);
          Value request =
              LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, requestPtr);
          Value requestElementPtr = LLVM::GEPOp::create(
              rewriter, op.getLoc(), llvmPtrType, i32Type, requestsPtr,
              ArrayRef<LLVM::GEPArg>{static_cast<int32_t>(index)});
          LLVM::StoreOp::create(rewriter, op.getLoc(), request,
                                requestElementPtr);
          requestElementPtrs.push_back(requestElementPtr);
        }

        // Allocate a count x !llvm.array<6 x i32> for the array of statuses
        // Size of status is implem dependendent, 6 should cover the max
        auto arrayType = LLVM::LLVMArrayType::get(i32Type, 6);

        Value statusPtr = LLVM::AllocaOp::create(rewriter, op.getLoc(),
                                                 llvmPtrType, arrayType, count);

        // Call MPI_Waitall
        // int MPI_Waitall(int count, MPI_Request array_of_requests[],
        // MPI_Status *array_of_statuses)
        // TODO returns i32 error code which we're ignoring here
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(context, mpiFunctionName),
                             ValueRange{count, requestsPtr, statusPtr});

        // MPI_Waitall sets completed requests to MPI_REQUEST_NULL.
        for (unsigned index = 0; index < numRequests; ++index) {
          Value requestPtr = entryBlock->getArgument(index);
          Value requestElementPtr = requestElementPtrs[index];
          Value request = LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type,
                                               requestElementPtr);
          LLVM::StoreOp::create(rewriter, op.getLoc(), request, requestPtr);
        }

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Waitall function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type, {i32Type, llvmPtrType, llvmPtrType}, false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Call the LLVM function with enzymexla.jit_call
      enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), TypeRange{},
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName), requests,
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/nullptr,
          /*xla_side_effect_free=*/nullptr);

      rewriter.eraseOp(op);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct MPIAllreduceOpLowering
    : public OpRewritePattern<enzymexla::MPIAllreduceOp> {

  std::string backend;
  size_t ncclCommPtr;
  MPIAllreduceOpLowering(std::string backend, size_t ncclCommPtr,
                         MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend),
        ncclCommPtr(ncclCommPtr) {}

  LogicalResult matchAndRewrite(enzymexla::MPIAllreduceOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);

      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Allreduce";

      // get the MPI datatype
      auto datatype = op.getDatatype();
      StringRef datatypeName = stringifyMPIDatatype(datatype);

      // get the MPI Op type
      StringRef mpiOpName = stringifyMPIOp(op.getOp());

      // TODO For now we just hard code MPI_COMM_WORLD as the communicator.
      std::string communicatorName = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName = "enzymexla_wrapper_" + mpiFunctionName +
                                        "_" + mpiOpName.str() + "_" +
                                        datatypeName.str();

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType}, false);

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 3; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects",
                                 memoryEffectsAttr);
        }

        // Get the function arguments
        Value sendbufPtr = entryBlock->getArgument(0);
        Value inbufPtr = entryBlock->getArgument(1);
        Value countPtr = entryBlock->getArgument(2);

        // Load the count value
        Value count =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, countPtr);

        // Get the address of the datatype
        // NOTE these symbols are not ABI-stable until MPI 5.0, but in practice,
        // they are represented as w ord-size values (i.e. `int` or ptr)
        Value addressOfDtype = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, datatypeName);

        // Get the address of the communicator
        Value addressOfComm = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, communicatorName);

        // Get the address of the MPI Op
        Value addressOfMPIOp = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, mpiOpName);

        // Call MPI_Allreduce
        // int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count,
        //     MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
        // TODO returns i32 error code which we're ignoring here
        LLVM::CallOp::create(rewriter, op.getLoc(), TypeRange{i32Type},
                             SymbolRefAttr::get(context, mpiFunctionName),
                             ValueRange{sendbufPtr, inbufPtr, count,
                                        addressOfDtype, addressOfMPIOp,
                                        addressOfComm});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Allreduce function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType =
            LLVM::LLVMFunctionType::get(i32Type,
                                        {llvmPtrType, llvmPtrType, i32Type,
                                         llvmPtrType, llvmPtrType, llvmPtrType},
                                        false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               communicatorName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Insert datatype declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(datatypeName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               datatypeName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Insert MPI_Op declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(mpiOpName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               mpiOpName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Get all orinigal op operands
      auto operands = op.getOperands();

      // Add inbuf to output operand aliases
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{},
          /*operand_index=*/1,
          /*operand_tuple_indices=*/std::vector<int64_t>{}));

      // Call the LLVM function with enzymexla.jit_call
      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{operands}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();

    } else if (backend == "cuda") {

      auto moduleOp = op->getParentOfType<ModuleOp>();
      auto sendbufType = cast<RankedTensorType>(op.getOperand(0).getType());
      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);
      auto i32Type = IntegerType::get(context, 32);
      auto i64Type = IntegerType::get(context, 64);

      std::string ncclFunctionName = "ncclAllReduce";

      auto datatype = op.getDatatype();
      StringRef datatypeName = stringifyMPIDatatype(datatype);
      auto ncclDatatype = mapMPIToNCCLDatatype(datatype);

      auto mpiOp = op.getOp();
      StringRef mpiOpName = stringifyMPIOp(mpiOp);
      auto ncclRedOp = mapMPIToNCCLRedOp(mpiOp);

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName =
          "enzymexla_wrapper_" + ncclFunctionName + "_" + mpiOpName.str() +
          "_" + datatypeName.str();

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType}, false);

        auto wrapperFunc = rewriter.create<LLVM::LLVMFuncOp>(
            op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);
        wrapperFunc->setAttr("enzymexla.device_abi",
                             rewriter.getStringAttr("cuda"));

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 2; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects",
                                 memoryEffectsAttr);
        }

        // Get the function arguments
        Value sendbufPtr = entryBlock->getArgument(0);
        Value recvbufPtr = entryBlock->getArgument(1);

        Value count = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), i64Type,
            rewriter.getI64IntegerAttr(sendbufType.getNumElements()));

        Value dtype = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), i32Type, rewriter.getI32IntegerAttr(*ncclDatatype));

        // Get the address of the communicator
        Value ncclCommInt = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), i64Type, rewriter.getI64IntegerAttr(ncclCommPtr));
        Value ncclComm = rewriter.create<LLVM::IntToPtrOp>(
            op.getLoc(), llvmPtrType, ncclCommInt);

        Value redOp = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), i32Type, rewriter.getI32IntegerAttr(*ncclRedOp));

        Value stream =
            enzymexla::GetStreamOp::create(rewriter, op.getLoc(), llvmPtrType);

        Value sendDataPtr =
            rewriter.create<LLVM::LoadOp>(op.getLoc(), llvmPtrType, sendbufPtr);
        Value recvDataPtr =
            rewriter.create<LLVM::LoadOp>(op.getLoc(), llvmPtrType, recvbufPtr);

        // Call ncclAllReduce
        // TODO error handling
        rewriter.create<LLVM::CallOp>(
            op.getLoc(), TypeRange{i32Type},
            SymbolRefAttr::get(context, ncclFunctionName),
            ValueRange{sendDataPtr, recvDataPtr, count, dtype, redOp, ncclComm,
                       stream});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert ncclAllReduce function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(ncclFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(i32Type,
                                                    {llvmPtrType, llvmPtrType,
                                                     i64Type, i32Type, i32Type,
                                                     llvmPtrType, llvmPtrType},
                                                    false);

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), ncclFunctionName,
                                          funcType, LLVM::Linkage::External);
      }

      Value operands[] = {op.getOperand(0), op.getOperand(1)};

      // Add inbuf to output operand aliases
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{},
          /*operand_index=*/1,
          /*operand_tuple_indices=*/std::vector<int64_t>{}));

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(), op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{operands}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();

    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct MPIBcastOpLowering : public OpRewritePattern<enzymexla::MPIBcastOp> {

  std::string backend;
  MPIBcastOpLowering(std::string backend, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), backend(backend) {}

  LogicalResult matchAndRewrite(enzymexla::MPIBcastOp op,
                                PatternRewriter &rewriter) const override {
    auto context = op->getContext();

    if (backend == "cpu") {

      auto moduleOp = op->getParentOfType<ModuleOp>();

      auto llvmPtrType = LLVM::LLVMPointerType::get(context);
      auto llvmVoidType = LLVM::LLVMVoidType::get(context);

      auto i32Type = IntegerType::get(context, 32);

      std::string mpiFunctionName = "MPI_Bcast";

      // get the MPI datatype
      auto datatype = op.getDatatype();
      StringRef datatypeName = stringifyMPIDatatype(datatype);

      // TODO For now we just hard code MPI_COMM_WORLD as the communicator.
      std::string communicatorName = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName =
          "enzymexla_wrapper_" + mpiFunctionName + "_" + datatypeName.str();

      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(wrapperFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, {llvmPtrType, llvmPtrType, llvmPtrType}, false);

        auto wrapperFunc = LLVM::LLVMFuncOp::create(
            rewriter, op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr(
            {rewriter.getStringAttr("read"), rewriter.getStringAttr("write"),
             rewriter.getStringAttr("allocate"),
             rewriter.getStringAttr("free")});
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 3; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects",
                                 memoryEffectsAttr);
        }

        // Get the function arguments
        Value bufPtr = entryBlock->getArgument(0);
        Value countPtr = entryBlock->getArgument(1);
        Value rootPtr = entryBlock->getArgument(2);

        // Load the count and root values
        Value count =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, countPtr);

        Value root =
            LLVM::LoadOp::create(rewriter, op.getLoc(), i32Type, rootPtr);

        // Get the address of the datatype
        // NOTE these symbols are not ABI-stable until MPI 5.0, but in practice,
        // they are represented as word-size values (i.e. `int` or ptr)
        Value addressOfDtype = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, datatypeName);

        // Get the address of the communicator
        Value addressOfComm = LLVM::AddressOfOp::create(
            rewriter, op.getLoc(), llvmPtrType, communicatorName);

        // Call MPI_Bcast
        // int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype,
        //     int root, MPI_Comm comm)
        // TODO returns i32 error code which we're ignoring here
        LLVM::CallOp::create(
            rewriter, op.getLoc(), TypeRange{i32Type},
            SymbolRefAttr::get(context, mpiFunctionName),
            ValueRange{bufPtr, count, addressOfDtype, root, addressOfComm});

        LLVM::ReturnOp::create(rewriter, op.getLoc(), ValueRange{});
      }

      // Insert MPI_Bcast function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type, {llvmPtrType, i32Type, llvmPtrType, i32Type, llvmPtrType},
            false);

        LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), mpiFunctionName,
                                 funcType, LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               communicatorName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Insert datatype declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(datatypeName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        LLVM::GlobalOp::create(rewriter, op.getLoc(), llvmPtrType,
                               /*isConstant=*/true, LLVM::Linkage::External,
                               datatypeName,
                               /*value=*/Attribute(),
                               /*alignment=*/0,
                               /*addrSpace=*/0);
      }

      // Get all original op operands
      auto operands = op.getOperands();

      // Add buf to output operand aliases
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{},
          /*operand_index=*/0,
          /*operand_tuple_indices=*/std::vector<int64_t>{}));

      // Call the LLVM function with enzymexla.jit_call
      auto jitCall = enzymexla::JITCallOp::create(
          rewriter, op.getLoc(), op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{operands}, rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Backend not supported: " + backend);
    }
  }
};

struct LowerEnzymeXLAMPIPass
    : public enzyme::impl::LowerEnzymeXLAMPIPassBase<LowerEnzymeXLAMPIPass> {
  using Base::Base;

  void runOnOperation() override {
    // TODO: declare LowerEnzymeXLAMPIPass as a ModuleOp pass in Passes.td
    // so getOperation() has the right static type here.
    ModuleOp module = dyn_cast<ModuleOp>(getOperation());
    if (!module) {
      getOperation()->emitError()
          << "lower-enzymexla-mpi must run on a builtin.module";
      signalPassFailure();
      return;
    }

    auto context = module->getContext();

    if (backend == "cuda" && !ncclCommPtr) {
      if (module
              .walk([&](Operation *op) -> WalkResult {
                if (isa<enzymexla::MPIBarrierOp, enzymexla::MPISendOp,
                        enzymexla::MPIRecvOp, enzymexla::MPIIsendOp,
                        enzymexla::MPIIrecvOp, enzymexla::MPIWaitOp,
                        enzymexla::MPIWaitallOp, enzymexla::MPIAllreduceOp,
                        enzymexla::MPIBcastOp>(op)) {
                  return WalkResult::interrupt();
                }
                return WalkResult::advance();
              })
              .wasInterrupted()) {
        module.emitError() << "lower-enzymexla-mpi with backend=cuda requires "
                              "a valid NCCL communicator pointer";
        signalPassFailure();
        return;
      }
    }

    if (backend == "cuda") {
      bool hasUnsupportedMPIOp = false;
      module.walk([&](Operation *op) {
        if (isa<enzymexla::MPICommRankOp, enzymexla::MPICommSizeOp,
                enzymexla::MPIBarrierOp, enzymexla::MPISendOp,
                enzymexla::MPIRecvOp, enzymexla::MPIIsendOp,
                enzymexla::MPIIrecvOp, enzymexla::MPIWaitOp,
                enzymexla::MPIWaitallOp, enzymexla::MPIBcastOp>(op)) {
          op->emitError() << "MPI operation not supported by backend cuda";
          hasUnsupportedMPIOp = true;
        }
      });
      if (hasUnsupportedMPIOp) {
        signalPassFailure();
        return;
      }

      bool hasUnsupportedAllreduce = false;
      module.walk([&](enzymexla::MPIAllreduceOp op) {
        auto sendbufType =
            dyn_cast<RankedTensorType>(op.getOperand(0).getType());
        auto recvbufType =
            dyn_cast<RankedTensorType>(op.getOperand(1).getType());
        if (!sendbufType || !sendbufType.hasStaticShape()) {
          op.emitError() << "CUDA NCCL allreduce lowering requires statically "
                            "shaped sendbuf to derive the element count";
          hasUnsupportedAllreduce = true;
          return;
        }
        if (!recvbufType || !recvbufType.hasStaticShape()) {
          op.emitError() << "CUDA NCCL allreduce lowering requires statically "
                            "shaped recvbuf to validate the element count";
          hasUnsupportedAllreduce = true;
          return;
        }
        if (sendbufType.getShape() != recvbufType.getShape()) {
          op.emitError() << "CUDA NCCL allreduce lowering requires sendbuf and "
                            "recvbuf to have the same shape";
          hasUnsupportedAllreduce = true;
          return;
        }

        auto datatype = op.getDatatype();
        if (failed(mapMPIToNCCLDatatype(datatype))) {
          op.emitError() << "MPI datatype not supported by NCCL lowering: "
                         << stringifyMPIDatatype(datatype);
          hasUnsupportedAllreduce = true;
          return;
        }

        auto mpiOp = op.getOp();
        if (failed(mapMPIToNCCLRedOp(mpiOp))) {
          op.emitError() << "MPI reduction op not supported by NCCL lowering: "
                         << stringifyMPIOp(mpiOp);
          hasUnsupportedAllreduce = true;
        }
      });
      if (hasUnsupportedAllreduce) {
        signalPassFailure();
        return;
      }
    }

    RewritePatternSet patterns(context);

    patterns.add<MPICommRankOpLowering>(backend, context);
    patterns.add<MPICommSizeOpLowering>(backend, context);
    patterns.add<MPIBarrierOpLowering>(backend, context);
    patterns.add<MPISendOpLowering>(backend, context);
    patterns.add<MPIRecvOpLowering>(backend, context);
    patterns.add<MPIIsendOpLowering>(backend, context);
    patterns.add<MPIIrecvOpLowering>(backend, context);
    patterns.add<MPIWaitOpLowering>(backend, context);
    patterns.add<MPIWaitallOpLowering>(backend, context);
    patterns.add<MPIAllreduceOpLowering>(backend, ncclCommPtr, context);
    patterns.add<MPIBcastOpLowering>(backend, context);

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
