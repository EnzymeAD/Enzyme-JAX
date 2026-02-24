#ifndef ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H
#define ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"

#include "shardy/dialect/sdy/ir/dialect.h"

// Include the dialect
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedDialect.h.inc"
// Traits and interfaces
#include "Traits.h"
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedInterfaces.h.inc"
// Types
#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedTypes.h.inc"
// Operations
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.h.inc"

// Utilities
namespace mlir::enzyme::distributed {

::mlir::FailureOr<PhysicalCommAxisOpInterface>
resolvePhysicalAxisInterfaceFromAttr(::mlir::Operation *from,
                                     ::mlir::Attribute axisAttr);

/**
 * Decomposes a logical axis into the SSA values resulting from 
 * the `factor` calls on a physical axis.
 */
::mlir::LogicalResult resolveLogicalAxisToAtomicFactors(
    ::mlir::Value logicalAxis,
    ::llvm::SmallVectorImpl<::mlir::Value> &atomicFactors);

::mlir::LogicalResult resolveLogicalMeshToAtomicFactors(
    LogicalMeshOp logicalMesh,
    ::llvm::SmallVectorImpl<::mlir::Value> &atomicFactors);

/**
 * Returns true if a logical mesh is disjoint: all atomic factors are unique,
 * and all factors referencing the same physical axis come from the same
 * AxisFactorOp.
 */
bool isLogicalMeshDisjoint(LogicalMeshOp logicalMesh);

/**
 * Returns true if a mesh is a submesh. A submesh is defined as any logical mesh
 * whose physical axis factors are a subset of the parents physical axis
 * factors. This is distinct from another potential definition of submesh as a
 * contiguous slice of a parent mehs, or as a subset of the logical axis of the
 * parent mesh.
 */
bool isLogicalMeshSubmesh(LogicalMeshOp logicalMesh, LogicalMeshOp submesh);

/**
 * Returns the total number of devices in a logical mesh, computed as the
 * product of the sizes of all of its atomic factors.
 */
::mlir::FailureOr<int64_t> getLogicalMeshSize(LogicalMeshOp logicalMesh);

} // namespace mlir::enzyme::distributed

#endif // ENZYME_AD_JAX_DIALECT_DISTRIBUTED_DIALECT_H
