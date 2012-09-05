#include <emmintrin.h>
const int mask = 0xFFFF;
void sgemm( int m, int n, float *A, float *C )
{
  float* transpose = (float*) malloc(4*n*m);
  for( int r = 0; r < m; r++ ) {
    for( int c = 0; c < n; c++) {
      transpose[r+c*m] = A[c+r*n]; // (row, column) = (column # + row * #columns)
    }
  }
  int asdf;
  int qwer;
  // Go one row at a time:
  for (int rnumber = 0; rnumber < m; rnumber++) {//for every row, do this
    for (int os = 0; os < n; os += 4) {
      float values[4] = {0,0,0,0};
      for (int offset = 0; offset < m; offset += 4) {
        __m128 horizontal = _mm_loadu_ps(transpose+offset+rnumber*n);
        __m128 vert[4];
        for (int p = 0; p < 4; p++) {
          vert[p] = _mm_loadu_ps(A+offset+p*m);
        }
        for (int current = 0; current < 4; current++) {
          __m128 product = _mm_mul_ps(horizontal, vert[current]);
          qwer = 0;
          float total[4] = {0,0,0,0};
          _mm_storeu_ps(total,product);
          for (asdf = 0; asdf < 4; asdf++) {
            qwer += total[asdf];
          }
          values[current] += qwer;
        }
      }
      for (int q = 0; q < 4; q++) {
        C[rnumber+(os+q)*m] = values[q];
      }
    }
  }

  free(transpose);
} 
