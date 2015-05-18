#define DOUBLEPREC 1
#define MPI_REAL_T MPI_DOUBLE
typedef double real_t;

#define AVX 1

#define VECWIDTH 4
#define ALIGNREQUIREMENT 4

typedef __m256d vector_t;

#define VECTOR_ADD _mm256_add_pd
#define VECTOR_SUB _mm256_sub_pd
#define VECTOR_MUL _mm256_mul_pd
#define VECTOR_DIV _mm256_div_pd

#define VECTOR_STORE _mm256_store_pd
#define VECTOR_SET1 _mm256_set1_pd
#define VECTOR_LOAD _mm256_load_pd
