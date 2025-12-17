// #include "mhlo/IR/hlo_ops.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

struct MPICommRankOpLowering
    : public OpRewritePattern<enzymexla::MPICommRankOp> {

  std::string backend;
  MPICommRankOpLowering(std::string backend, MLIRContext *context, PatternBenefit benefit = 1)
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
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the function type
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, // void return type
            {llvmPtrType},   // parameter types
            false);          // is variadic: false

        auto wrapperFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr({
            rewriter.getStringAttr("read"),
            rewriter.getStringAttr("write"),
            rewriter.getStringAttr("allocate"),
            rewriter.getStringAttr("free")
        });
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute
        wrapperFunc.setArgAttr(0, "enzymexla.memory_effects", memoryEffectsAttr);

        // Get the first (and only) argument of the function
        Value rankPtr = entryBlock->getArgument(0);

        // Get the address of the communicator
        Value addressOfComm = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          communicatorName
        );

        // TODO error checking
        // MPI_Comm_rank returns i32 error code which we're ignoring here
        rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            TypeRange{i32Type},
            SymbolRefAttr::get(context, mpiFunctionName),
            ValueRange{addressOfComm, rankPtr});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert MPI_Comm_rank function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, llvmPtrType},
            false
        );

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), mpiFunctionName, funcType,
                                 LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          communicatorName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Create a placeholder dense tensor constant with arbitrary value
      auto resultType = op.getResult().getType();
      auto rankedTensorType = cast<RankedTensorType>(resultType);
      auto elementType = rankedTensorType.getElementType();
      auto attr = DenseElementsAttr::get(
          rankedTensorType,
          rewriter.getIntegerAttr(elementType, 0));
      auto placeholderValue = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), attr);

      // Call the LLVM function with enzymexla.jit_call
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{},
          /*operand_index=*/0,
          /*operand_tuple_indices=*/std::vector<int64_t>{})
      );

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(),
          TypeRange{resultType},
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{placeholderValue},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Backend not supported: " + backend);
    }

  }

};


struct MPICommSizeOpLowering
    : public OpRewritePattern<enzymexla::MPICommSizeOp> {

  std::string backend;
  MPICommSizeOpLowering(std::string backend, MLIRContext *context, PatternBenefit benefit = 1)
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
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the function type
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType, // void return type
            {llvmPtrType},   // parameter types
            false);          // is variadic: false

        auto wrapperFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr({
            rewriter.getStringAttr("read"),
            rewriter.getStringAttr("write"),
            rewriter.getStringAttr("allocate"),
            rewriter.getStringAttr("free")
        });
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute
        wrapperFunc.setArgAttr(0, "enzymexla.memory_effects", memoryEffectsAttr);

        // Get the first (and only) argument of the function
        Value sizePtr = entryBlock->getArgument(0);

        // Get the address of the communicator
        Value addressOfComm = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          communicatorName
        );

        // TODO error checking
        // MPI_Comm_size returns i32 error code which we're ignoring here
        rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            TypeRange{i32Type},
            SymbolRefAttr::get(context, mpiFunctionName),
            ValueRange{addressOfComm, sizePtr});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert MPI_Comm_size function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, llvmPtrType},
            false
        );

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), mpiFunctionName, funcType,
                                 LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          communicatorName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Create a placeholder dense tensor constant with arbitrary value
      auto resultType = op.getResult().getType();
      auto rankedTensorType = cast<RankedTensorType>(resultType);
      auto elementType = rankedTensorType.getElementType();
      auto attr = DenseElementsAttr::get(
          rankedTensorType,
          rewriter.getIntegerAttr(elementType, 0));
      auto placeholderValue = rewriter.create<stablehlo::ConstantOp>(
          op.getLoc(), attr);

      // Call the LLVM function with enzymexla.jit_call
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{},
          /*operand_index=*/0,
          /*operand_tuple_indices=*/std::vector<int64_t>{})
      );

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(),
          TypeRange{resultType},
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{placeholderValue},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Backend not supported: " + backend);
    }

  }

};


struct MPIBarrierOpLowering
    : public OpRewritePattern<enzymexla::MPIBarrierOp> {

  std::string backend;
  MPIBarrierOpLowering(std::string backend, MLIRContext *context, PatternBenefit benefit = 1)
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
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the function type
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {},
            false);

        auto wrapperFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr({
            rewriter.getStringAttr("read"),
            rewriter.getStringAttr("write"),
            rewriter.getStringAttr("allocate"),
            rewriter.getStringAttr("free")
        });
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Get the address of the communicator
        Value addressOfComm = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          communicatorName
        );

        // Call MPI_Barrier
        // int MPI_Barrier(MPI_Comm comm)
        // TODO returns i32 error code which we're ignoring here
        rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            TypeRange{i32Type},
            SymbolRefAttr::get(context, mpiFunctionName),
            ValueRange{addressOfComm});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert MPI_Barrier function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType},
            false
        );

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), mpiFunctionName, funcType,
                                 LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          communicatorName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Call the LLVM function with enzymexla.jit_call
      rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(),
          TypeRange{},
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{},
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
      return rewriter.notifyMatchFailure(op, "Backend not supported: " + backend);
    }

  }

};


struct MPISendOpLowering
    : public OpRewritePattern<enzymexla::MPISendOp> {

  std::string backend;
  MPISendOpLowering(std::string backend, MLIRContext *context, PatternBenefit benefit = 1)
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

      // TODO we just assume it's MPI_DOUBLE for now
      std::string datatypeName = "MPI_DOUBLE";

      // For now we just hard code MPI_COMM_WORLD as the communicator.
      // TODO make this more flexible
      std::string communicatorName = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName = "enzymexla_wrapper_" + mpiFunctionName + datatypeName;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType,llvmPtrType,llvmPtrType,llvmPtrType},
            false);

        auto wrapperFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr({
            rewriter.getStringAttr("read"),
            rewriter.getStringAttr("write"),
            rewriter.getStringAttr("allocate"),
            rewriter.getStringAttr("free")
        });
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 4; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects", memoryEffectsAttr);
        }

        // Get the function arguments
        Value bufPtr = entryBlock->getArgument(0);
        Value countPtr = entryBlock->getArgument(1);
        Value destPtr = entryBlock->getArgument(2);
        Value tagPtr = entryBlock->getArgument(3);

        // Load the count, dest, tag values
        Value count = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            countPtr
        );

        Value dest = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            destPtr
        );

        Value tag = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            tagPtr
        );

        // Get the address of the datatype
        Value addressOfDtype = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          datatypeName
        );

        // Get the address of the communicator
        Value addressOfComm = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          communicatorName
        );

        // Call MPI_Send
        // int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int
        //     dest, int tag, MPI_Comm comm)
        // TODO returns i32 error code which we're ignoring here
        rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            TypeRange{i32Type},
            SymbolRefAttr::get(context, mpiFunctionName),
            ValueRange{bufPtr, count, addressOfDtype, dest, tag, addressOfComm});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert MPI_Send function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, i32Type, llvmPtrType, i32Type, i32Type, llvmPtrType},
            false
        );

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), mpiFunctionName, funcType,
                                 LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          communicatorName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Insert MPI_DOUBLE declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(datatypeName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          datatypeName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Get all orinigal op operands
      auto operands = op.getOperands();

      // Call the LLVM function with enzymexla.jit_call
      rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(),
          TypeRange{},
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{operands},
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
      return rewriter.notifyMatchFailure(op, "Backend not supported: " + backend);
    }

  }

};


struct MPIRecvOpLowering
    : public OpRewritePattern<enzymexla::MPIRecvOp> {

  std::string backend;
  MPIRecvOpLowering(std::string backend, MLIRContext *context, PatternBenefit benefit = 1)
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

      // TODO we just assume it's MPI_DOUBLE for now
      std::string datatypeName = "MPI_DOUBLE";

      // For now we just hard code MPI_COMM_WORLD as the communicator.
      // TODO make this more flexible
      std::string communicatorName = "MPI_COMM_WORLD";

      std::string statusName = "MPI_STATUS_IGNORE";

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName = "enzymexla_wrapper_" + mpiFunctionName + datatypeName;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType,llvmPtrType,llvmPtrType,llvmPtrType},
            false);

        auto wrapperFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr({
            rewriter.getStringAttr("read"),
            rewriter.getStringAttr("write"),
            rewriter.getStringAttr("allocate"),
            rewriter.getStringAttr("free")
        });
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 4; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects", memoryEffectsAttr);
        }

        // Get the function arguments
        Value bufPtr = entryBlock->getArgument(0);
        Value countPtr = entryBlock->getArgument(1);
        Value srcPtr = entryBlock->getArgument(2);
        Value tagPtr = entryBlock->getArgument(3);

        // Load the count, src, tag values
        Value count = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            countPtr
        );

        Value src = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            srcPtr
        );

        Value tag = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            tagPtr
        );

        // Get the address of the datatype
        // TODO make a comment on what exactly we're doing here
        Value addressOfDtype = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          datatypeName
        );

        // Get the address of the communicator
        Value addressOfComm = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          communicatorName
        );

        // Get the address of the status
        Value addressOfStatus = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          statusName
        );

        // Call MPI_Recv
        // int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int
        //     source, int tag, MPI_Comm comm, MPI_Status* status)
        // TODO returns i32 error code which we're ignoring here
        rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            TypeRange{i32Type},
            SymbolRefAttr::get(context, mpiFunctionName),
            ValueRange{
              bufPtr, 
              count, 
              addressOfDtype, 
              src, 
              tag, 
              addressOfComm, 
              addressOfStatus
            }
          );

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert MPI_Recv function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, i32Type, llvmPtrType, i32Type, i32Type, llvmPtrType, llvmPtrType},
            false
        );

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), mpiFunctionName, funcType,
                                 LLVM::Linkage::External);
      }

      // Insert MPI_STATUS_IGNORE declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          statusName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          communicatorName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Insert MPI_DOUBLE declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(datatypeName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          datatypeName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Get all orinigal op operands
      auto operands = op.getOperands();

      // Call the LLVM function with enzymexla.jit_call
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{},
          /*operand_index=*/0,
          /*operand_tuple_indices=*/std::vector<int64_t>{})
      );

      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(),
          op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{operands},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Backend not supported: " + backend);
    }

  }

};


struct MPIIsendOpLowering
    : public OpRewritePattern<enzymexla::MPIIsendOp> {

  std::string backend;
  MPIIsendOpLowering(std::string backend, MLIRContext *context, PatternBenefit benefit = 1)
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

      // TODO we just assume it's MPI_DOUBLE for now
      std::string datatypeName = "MPI_DOUBLE";

      // For now we just hard code MPI_COMM_WORLD as the communicator.
      // TODO make this more flexible
      std::string communicatorName = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName = "enzymexla_wrapper_" + mpiFunctionName + datatypeName;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType,llvmPtrType,llvmPtrType,llvmPtrType,llvmPtrType},
            false);

        auto wrapperFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr({
            rewriter.getStringAttr("read"),
            rewriter.getStringAttr("write"),
            rewriter.getStringAttr("allocate"),
            rewriter.getStringAttr("free")
        });
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 5; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects", memoryEffectsAttr);
        }

        // Get the function arguments
        Value bufPtr = entryBlock->getArgument(0);
        Value countPtr = entryBlock->getArgument(1);
        Value destPtr = entryBlock->getArgument(2);
        Value tagPtr = entryBlock->getArgument(3);
        Value requestPtr = entryBlock->getArgument(4);

        // Load the count, dest, tag values
        Value count = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            countPtr
        );

        Value dest = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            destPtr
        );

        Value tag = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            tagPtr
        );

        // Get the address of the datatype
        // TODO make a comment on what exactly we're doing here
        Value addressOfDtype = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          datatypeName
        );

        // Get the address of the communicator
        Value addressOfComm = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          communicatorName
        );

        // Call MPI_Isend
        // int MPI_Isend(void* buf, int count, MPI_Datatype datatype, int
        //               dest, int tag, MPI_Comm comm, MPI_Request* request)
        // TODO returns i32 error code which we're ignoring here
        rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            TypeRange{i32Type},
            SymbolRefAttr::get(context, mpiFunctionName),
            ValueRange{
              bufPtr, 
              count, 
              addressOfDtype, 
              dest, 
              tag, 
              addressOfComm, 
              requestPtr
            }
          );

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert MPI_Isend function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, i32Type, llvmPtrType, i32Type, i32Type, llvmPtrType, llvmPtrType},
            false
        );

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), mpiFunctionName, funcType,
                                 LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          communicatorName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Insert MPI_DOUBLE declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(datatypeName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          datatypeName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Get all orinigal op operands
      auto operands = op.getOperands();

      // Add request to output operand aliases
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{},
          /*operand_index=*/4,
          /*operand_tuple_indices=*/std::vector<int64_t>{})
      );

      // Call the LLVM function with enzymexla.jit_call
      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(),
          op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{operands},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Backend not supported: " + backend);
    }

  }

};


struct MPIIrecvOpLowering
    : public OpRewritePattern<enzymexla::MPIIrecvOp> {

  std::string backend;
  MPIIrecvOpLowering(std::string backend, MLIRContext *context, PatternBenefit benefit = 1)
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

      // TODO we just assume it's MPI_DOUBLE for now
      std::string datatypeName = "MPI_DOUBLE";

      // For now we just hard code MPI_COMM_WORLD as the communicator.
      // TODO make this more flexible
      std::string communicatorName = "MPI_COMM_WORLD";

      // Generate the enzymexla_wrapper LLVM function body
      std::string wrapperFunctionName = "enzymexla_wrapper_" + mpiFunctionName + datatypeName;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType,llvmPtrType,llvmPtrType,llvmPtrType,llvmPtrType},
            false);

        auto wrapperFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr({
            rewriter.getStringAttr("read"),
            rewriter.getStringAttr("write"),
            rewriter.getStringAttr("allocate"),
            rewriter.getStringAttr("free")
        });
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        for (unsigned i = 0; i < 5; ++i) {
          wrapperFunc.setArgAttr(i, "enzymexla.memory_effects", memoryEffectsAttr);
        }

        // Get the function arguments
        Value bufPtr = entryBlock->getArgument(0);
        Value countPtr = entryBlock->getArgument(1);
        Value srcPtr = entryBlock->getArgument(2);
        Value tagPtr = entryBlock->getArgument(3);
        Value requestPtr = entryBlock->getArgument(4);

        // Load the count, src, tag values
        Value count = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            countPtr
        );

        Value src = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            srcPtr
        );

        Value tag = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            i32Type,
            tagPtr
        );

        // Get the address of the datatype
        // TODO make a comment on what exactly we're doing here
        Value addressOfDtype = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          datatypeName
        );

        // Get the address of the communicator
        Value addressOfComm = rewriter.create<LLVM::AddressOfOp>(
          op.getLoc(),
          llvmPtrType,
          communicatorName
        );

        // Call MPI_Irecv
        // int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int
        //               source, int tag, MPI_Comm comm, MPI_Request* request)
        // TODO returns i32 error code which we're ignoring here
        rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            TypeRange{i32Type},
            SymbolRefAttr::get(context, mpiFunctionName),
            ValueRange{
              bufPtr, 
              count, 
              addressOfDtype, 
              src, 
              tag, 
              addressOfComm, 
              requestPtr
            }
          );

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert MPI_Irecv function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, i32Type, llvmPtrType, i32Type, i32Type, llvmPtrType, llvmPtrType},
            false
        );

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), mpiFunctionName, funcType,
                                 LLVM::Linkage::External);
      }

      // Insert MPI_COMM_WORLD declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(communicatorName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          communicatorName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Insert MPI_DOUBLE declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::GlobalOp>(datatypeName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        rewriter.create<LLVM::GlobalOp>(
          op.getLoc(),
          llvmPtrType,
          /*isConstant=*/true,
          LLVM::Linkage::External,
          datatypeName,
          /*value=*/Attribute(),
          /*alignment=*/0,
          /*addrSpace=*/0
        );
      }

      // Get all orinigal op operands
      auto operands = op.getOperands();

      // Add buffer to output operand aliases
      SmallVector<Attribute> aliases;
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{0},
          /*operand_index=*/0,
          /*operand_tuple_indices=*/std::vector<int64_t>{})
      );

      // Add request to output operand aliases
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          context,
          /*output_operand_aliases=*/std::vector<int64_t>{1},
          /*operand_index=*/4,
          /*operand_tuple_indices=*/std::vector<int64_t>{})
      );

      // Call the LLVM function with enzymexla.jit_call
      auto jitCall = rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(),
          op->getResultTypes(),
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{operands},
          rewriter.getStringAttr(""),
          /*operand_layouts=*/nullptr,
          /*result_layouts=*/nullptr,
          /*arg_attrs=*/nullptr,
          /*res_attrs=*/nullptr,
          /*output_operand_aliases=*/rewriter.getArrayAttr(aliases),
          /*xla_side_effect_free=*/nullptr);

      rewriter.replaceOp(op, jitCall);

      return success();
    } else {
      return rewriter.notifyMatchFailure(op, "Backend not supported: " + backend);
    }

  }

};


struct MPIWaitOpLowering
    : public OpRewritePattern<enzymexla::MPIWaitOp> {

  std::string backend;
  MPIWaitOpLowering(std::string backend, MLIRContext *context, PatternBenefit benefit = 1)
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
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        // Create the wrapper function decl
        auto funcType = LLVM::LLVMFunctionType::get(
            llvmVoidType,
            {llvmPtrType},
            false);

        auto wrapperFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), wrapperFunctionName, funcType);

        // Add function-level memory effects attribute
        auto memoryEffectsAttr = rewriter.getArrayAttr({
            rewriter.getStringAttr("read"),
            rewriter.getStringAttr("write"),
            rewriter.getStringAttr("allocate"),
            rewriter.getStringAttr("free")
        });
        wrapperFunc->setAttr("enzymexla.memory_effects", memoryEffectsAttr);

        Block *entryBlock = wrapperFunc.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        // Add argument-level memory effects attribute to all arguments
        wrapperFunc.setArgAttr(0, "enzymexla.memory_effects", memoryEffectsAttr);

        // Get the function argument
        Value requestPtr = entryBlock->getArgument(0);

        // Allocate a 1x!llvm.array<6 x i32> that we use in place of MPI_Status
        // Size of status is implem dependendent, this should cover the max
        Value numElements = rewriter.create<arith::ConstantOp>(
            op.getLoc(), i32Type, rewriter.getI32IntegerAttr(1));
        
        auto arrayType = LLVM::LLVMArrayType::get(i32Type, 6);
        
        Value statusPtr = rewriter.create<LLVM::AllocaOp>(
            op.getLoc(), 
            llvmPtrType,
            arrayType,
            numElements);

        // Call MPI_Wait
        // int MPI_Wait(MPI_Request* request, MPI_Status* status)
        // TODO returns i32 error code which we're ignoring here
        rewriter.create<LLVM::CallOp>(
            op.getLoc(),
            TypeRange{i32Type},
            SymbolRefAttr::get(context, mpiFunctionName),
            ValueRange{requestPtr, statusPtr});

        rewriter.create<LLVM::ReturnOp>(op.getLoc(), ValueRange{});
      }

      // Insert MPI_Wait function declaration if not already present
      if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(mpiFunctionName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());

        auto funcType = LLVM::LLVMFunctionType::get(
            i32Type,
            {llvmPtrType, llvmPtrType},
            false
        );

        rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), mpiFunctionName, funcType,
                                 LLVM::Linkage::External);
      }

      // Get the request operand
      auto request = op.getInrequest();

      // Call the LLVM function with enzymexla.jit_call
      rewriter.create<enzymexla::JITCallOp>(
          op.getLoc(),
          TypeRange{},
          mlir::FlatSymbolRefAttr::get(context, wrapperFunctionName),
          ValueRange{request},
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
      return rewriter.notifyMatchFailure(op, "Backend not supported: " + backend);
    }

  }

};


struct LowerEnzymeXLAMPIPass
    : public enzyme::impl::LowerEnzymeXLAMPIPassBase<
          LowerEnzymeXLAMPIPass> {
  using Base::Base;

  void runOnOperation() override {
    auto context = getOperation()->getContext();
    RewritePatternSet patterns(context);

    patterns.add<MPICommRankOpLowering>(backend, context);
    patterns.add<MPICommSizeOpLowering>(backend, context);
    patterns.add<MPIBarrierOpLowering>(backend, context);
    patterns.add<MPISendOpLowering>(backend, context);
    patterns.add<MPIRecvOpLowering>(backend, context);
    patterns.add<MPIIsendOpLowering>(backend, context);
    patterns.add<MPIIrecvOpLowering>(backend, context);
    patterns.add<MPIWaitOpLowering>(backend, context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      signalPassFailure();
    }
  }
};
