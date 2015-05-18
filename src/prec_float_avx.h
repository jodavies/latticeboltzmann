#define SINGLEPREC 1
#define MPI_REAL_T MPI_FLOAT
typedef float real_t;

#define AVX 1

#define VECWIDTH 8
#define ALIGNREQUIREMENT 8

typedef __m256 vector_t;

#define VECTOR_ADD _mm256_add_ps
#define VECTOR_SUB _mm256_sub_ps
#define VECTOR_MUL _mm256_mul_ps
#define VECTOR_DIV _mm256_div_ps

#define VECTOR_STORE _mm256_store_ps
#define VECTOR_SET1 _mm256_set1_ps
#define VECTOR_LOAD _mm256_load_ps
