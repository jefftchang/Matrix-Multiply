#include <nmmintrin.h>
#include <omp.h>

void sgemm( int m, int n, float *A, float *C )
{
  omp_set_num_threads(8);
  
  
  int mpaddingsize = 4 - (m % 4);
  int npaddingsize = 4 - (n % 4);
  
  
  if (mpaddingsize != 4 | npaddingsize != 4) {
    if (mpaddingsize == 4) {
      mpaddingsize = 0;
    }
    if (npaddingsize == 4) {
      npaddingsize = 0;
    }
    int onecolumn = n+npaddingsize;
    int twocolumn = 2*onecolumn;
    int threecolumn = 3*onecolumn;
    int onerow = m+mpaddingsize;
    float paddedMatrix[onerow*onecolumn];

    
    //printf("need to pad m:%d n: %d\n", mpaddingsize, npaddingsize);
    int originalCounter = 0;
    int newCounter = 0;
    for (int col = 0; col < n; col++) {
      for (int row = 0; row < m; row++) {
        paddedMatrix[newCounter] = A[originalCounter];
        newCounter++;
        originalCounter++;
        //printf("padded col:%d row:%d as %f\n", col, row, paddedMatrix[newCounter-1]);
      }
      for (int row = 0; row < mpaddingsize; row++) {
        paddedMatrix[newCounter] = 0;
        newCounter++;
        //printf("padded col:%d row:%d as %f\n", col, row+m, paddedMatrix[newCounter-1]);
      }
    }
    for (int col = n; col < onecolumn; col++) {
      for (int row = 0; row < onerow; row++) {
        paddedMatrix[newCounter] = 0;
        newCounter++;
        //printf("padded col:%d row:%d as %f\n", col, row, paddedMatrix[newCounter-1]);
      }
    }
    

    
    float* transpose = (float*) malloc(sizeof(float)*(n+npaddingsize)*(m+mpaddingsize));
    
      for( int r = 0; r < onerow; r++ ) {
        for( int c = 0; c < onecolumn; c++) {
          transpose[c+r*onecolumn] = paddedMatrix[r+c*onerow]; // (row, column) = (column # + row * #columns)
        }
      }

      __m128 temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
      __m128 word1, word2, word3, word4;
      __m128 scalar1, scalar2;
      float* wordaddress;
      float* scalaraddress;
      float* answeraddress;



    #pragma omp parallel for private (temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, word1, word2, word3, word4, scalar1, scalar2, wordaddress, scalaraddress, answeraddress)
      for (int row = 0; row < m; row += 4) {
        //transpose column offset = row
        for (int col = 0; col < m; col+= 2) {
          //transpose row offset = col
          temp1 = _mm_setzero_ps();
          temp2 = _mm_setzero_ps();
          temp3 = _mm_setzero_ps();
          temp4 = _mm_setzero_ps();
          temp5 = _mm_setzero_ps();
          temp6 = _mm_setzero_ps();
          temp7 = _mm_setzero_ps();
          temp8 = _mm_setzero_ps();

          for (int offset = 0; offset < onecolumn; offset += 4) {
            wordaddress = transpose+row*onecolumn+offset;
            scalaraddress = transpose+col*onecolumn+offset;
            //offsetting makes both words and byteblocks go down
            word1 = _mm_loadu_ps(wordaddress);
            word2 = _mm_loadu_ps(wordaddress+onecolumn);
            word3 = _mm_loadu_ps(wordaddress+twocolumn);
            word4 = _mm_loadu_ps(wordaddress+threecolumn);
            scalar1 = _mm_loadu_ps(scalaraddress);
            scalar2 = _mm_loadu_ps(scalaraddress+onecolumn);

            temp1 = _mm_add_ps(temp1, _mm_mul_ps(word1, scalar1));
            temp2 = _mm_add_ps(temp2, _mm_mul_ps(word2, scalar1));
            temp3 = _mm_add_ps(temp3, _mm_mul_ps(word3, scalar1));
            temp4 = _mm_add_ps(temp4, _mm_mul_ps(word4, scalar1));
            temp5 = _mm_add_ps(temp5, _mm_mul_ps(word1, scalar2));
            temp6 = _mm_add_ps(temp6, _mm_mul_ps(word2, scalar2));
            temp7 = _mm_add_ps(temp7, _mm_mul_ps(word3, scalar2));
            temp8 = _mm_add_ps(temp8, _mm_mul_ps(word4, scalar2));

          }
          answeraddress = C+row+col*m;
          temp1 = _mm_hadd_ps(temp1, temp1);
          temp1 = _mm_hadd_ps(temp1, temp1);
          _mm_store_ss(answeraddress, temp1);
          if (row+1 < m) {
            temp2 = _mm_hadd_ps(temp2, temp2);
            temp2 = _mm_hadd_ps(temp2, temp2);
            _mm_store_ss(answeraddress+1, temp2);
            if (row + 2 < m) {
              temp3 = _mm_hadd_ps(temp3, temp3);
              temp3 = _mm_hadd_ps(temp3, temp3);
              _mm_store_ss(answeraddress+2, temp3);
              if (row + 3 < m) {
                temp4 = _mm_hadd_ps(temp4, temp4);
                temp4 = _mm_hadd_ps(temp4, temp4);
                _mm_store_ss(answeraddress+3, temp4);
              }
            }
          }
          if (col + 1 < m) {
            temp5 = _mm_hadd_ps(temp5, temp5);
            temp5 = _mm_hadd_ps(temp5, temp5);
            _mm_store_ss(answeraddress+m, temp5);
            if (row + 1 < m) {
              temp6 = _mm_hadd_ps(temp6, temp6);
              temp6 = _mm_hadd_ps(temp6, temp6);
              _mm_store_ss(answeraddress+m+1, temp6);
              if (row + 2 < m) {
                temp7 = _mm_hadd_ps(temp7, temp7);
                temp7 = _mm_hadd_ps(temp7, temp7);
                _mm_store_ss(answeraddress+m+2, temp7);
                if (row + 3 < m) {
                  temp8 = _mm_hadd_ps(temp8, temp8);
                  temp8 = _mm_hadd_ps(temp8, temp8);
                  _mm_store_ss(answeraddress+m+3, temp8);
                }
              }
            }
          }
        }
      }
    return;
  }
  
  

  
  
  
  float* transpose = (float*) malloc(sizeof(float)*n*m);
  for( int r = 0; r < m; r++ ) {
    for( int c = 0; c < n; c++) {
      transpose[c+r*n] = A[r+c*m]; // (row, column) = (column # + row * #columns)
    }
  }
  
  __m128 temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
  __m128 word1, word2, word3, word4;
  __m128 scalar1, scalar2;
  float* wordaddress;
  float* scalaraddress;
  float* answeraddress;
  
  
  
#pragma omp parallel for private (temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, word1, word2, word3, word4, scalar1, scalar2, wordaddress, scalaraddress, answeraddress)
  for (int row = 0; row < m; row += 4) {
    //transpose column offset = row
    for (int col = 0; col < m; col+= 2) {
      //transpose row offset = col
      temp1 = _mm_setzero_ps();
      temp2 = _mm_setzero_ps();
      temp3 = _mm_setzero_ps();
      temp4 = _mm_setzero_ps();
      temp5 = _mm_setzero_ps();
      temp6 = _mm_setzero_ps();
      temp7 = _mm_setzero_ps();
      temp8 = _mm_setzero_ps();

      for (int offset = 0; offset < n; offset += 4) {
        wordaddress = transpose+row*n+offset;
        scalaraddress = transpose+col*n+offset;
        //offsetting makes both words and byteblocks go down
        word1 = _mm_loadu_ps(wordaddress);
        word2 = _mm_loadu_ps(wordaddress+n);
        word3 = _mm_loadu_ps(wordaddress+2*n);
        word4 = _mm_loadu_ps(wordaddress+3*n);
        scalar1 = _mm_loadu_ps(scalaraddress);
        scalar2 = _mm_loadu_ps(scalaraddress+n);
        
        temp1 = _mm_add_ps(temp1, _mm_mul_ps(word1, scalar1));
        temp2 = _mm_add_ps(temp2, _mm_mul_ps(word2, scalar1));
        temp3 = _mm_add_ps(temp3, _mm_mul_ps(word3, scalar1));
        temp4 = _mm_add_ps(temp4, _mm_mul_ps(word4, scalar1));
        temp5 = _mm_add_ps(temp5, _mm_mul_ps(word1, scalar2));
        temp6 = _mm_add_ps(temp6, _mm_mul_ps(word2, scalar2));
        temp7 = _mm_add_ps(temp7, _mm_mul_ps(word3, scalar2));
        temp8 = _mm_add_ps(temp8, _mm_mul_ps(word4, scalar2));
        
      }
      answeraddress = C+row+col*m;
      temp1 = _mm_hadd_ps(temp1, temp1);
      temp1 = _mm_hadd_ps(temp1, temp1);
      _mm_store_ss(answeraddress, temp1);
      temp2 = _mm_hadd_ps(temp2, temp2);
      temp2 = _mm_hadd_ps(temp2, temp2);
      _mm_store_ss(answeraddress+1, temp2);
      temp3 = _mm_hadd_ps(temp3, temp3);
      temp3 = _mm_hadd_ps(temp3, temp3);
      _mm_store_ss(answeraddress+2, temp3);
      temp4 = _mm_hadd_ps(temp4, temp4);
      temp4 = _mm_hadd_ps(temp4, temp4);
      _mm_store_ss(answeraddress+3, temp4);
      temp5 = _mm_hadd_ps(temp5, temp5);
      temp5 = _mm_hadd_ps(temp5, temp5);
      _mm_store_ss(answeraddress+m, temp5);
      temp6 = _mm_hadd_ps(temp6, temp6);
      temp6 = _mm_hadd_ps(temp6, temp6);
      _mm_store_ss(answeraddress+m+1, temp6);
      temp7 = _mm_hadd_ps(temp7, temp7);
      temp7 = _mm_hadd_ps(temp7, temp7);
      _mm_store_ss(answeraddress+m+2, temp7);
      temp8 = _mm_hadd_ps(temp8, temp8);
      temp8 = _mm_hadd_ps(temp8, temp8);
      _mm_store_ss(answeraddress+m+3, temp8);
    }
  }
}

