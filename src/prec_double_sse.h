#define DOUBLEPREC 1
#define MPI_REAL_T MPI_DOUBLE
typedef double real_t;

#define SSE 1

#define VECWIDTH 2
#define ALIGNREQUIREMENT 2

typedef __m128d vector_t;

#define VECTOR_ADD _mm_add_pd
#define VECTOR_SUB _mm_sub_pd
#define VECTOR_MUL _mm_mul_pd
#define VECTOR_DIV _mm_div_pd

#define VECTOR_STORE _mm_store_pd
#define VECTOR_SET1 _mm_set1_pd
#define VECTOR_LOAD _mm_load_pd
