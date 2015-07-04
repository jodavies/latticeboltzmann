#define SINGLEPREC 1
#define MPI_REAL_T MPI_FLOAT
typedef float real_t;
#define REAL_T(x) x##f

#define SSE 1

#define VECWIDTH 4
#define ALIGNREQUIREMENT 4

typedef __m128 vector_t;

#define VECTOR_ADD _mm_add_ps
#define VECTOR_SUB _mm_sub_ps
#define VECTOR_MUL _mm_mul_ps
#define VECTOR_DIV _mm_div_ps

#define VECTOR_STORE _mm_store_ps
#define VECTOR_SET1 _mm_set1_ps
#define VECTOR_LOAD _mm_load_ps
