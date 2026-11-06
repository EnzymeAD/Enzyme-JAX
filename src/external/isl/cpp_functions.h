#ifndef CPP_FUNCTIONS_H_
#define CPP_FUNCTIONS_H_

#include <isl/schedule_type.h>
#include <isl/union_map_type.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
isl_union_map *isl_ast_generate_array_expansion_indexing(
	isl_schedule_node *_node, isl_union_map *extra_umap, isl_union_map *lrs);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CPP_FUNCTIONS_H_
