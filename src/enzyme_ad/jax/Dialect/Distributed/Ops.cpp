#include "CollectiveOps.h"

// Central emission point for generated distributed op class definitions.
// Keep this in a dedicated file so op definitions do not depend on any
// specific op implementation unit remaining present.
#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Distributed/DistributedOps.cpp.inc"
