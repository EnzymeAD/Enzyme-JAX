#ifndef ENZYME_DISTRIBUTED_DIALECT_TYPES_H
#define ENZYME_DISTRIBUTED_DIALECT_TYPES_H

include "Dialect.td"

def ReadTokenType : DistributedType<"ReadToken", "read_token">;
def WriteTokenType : DistributedType<"WriteToken", "write_token">;

#endif // ENZYME_DISTRIBUTED_DIALECT_TYPES_H