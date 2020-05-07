#include <math.h>

#include "matutil.h"
#include "output.h"

int matutil_initialize(void) { return 0; }

int matutil_teardown(void) { return 0; }

void matutil_get_new_dimensions(int h1, int w1, int h2, int w2, int *hr,
                                int *wr) {
  *hr = h1;
  *wr = w2;
}

int matutil_multiply(float *m1, int h1, int w1, float *m2, int h2, int w2,
                     float *ret) {
  // check dimensions
  if (w1 != h2) {
    print_err(
            "Matrices have incompatible dimensions for multiplication %dx%d "
            "and %dx%d\n",
            h1, w1, h2, w2);
    return -1;
  }

  int hr = h1, wr = w2;
  for (int i = 0; i < hr; ++i) { // coordinates in ret
    for (int j = 0; j < wr; ++j) {
      ret[i * wr + j] = 0;

      for(int mul_i = 0, mul_j = 0; mul_i < h2; ++mul_i, ++mul_j)
        ret[i*wr + j] += m1[i*w1 + mul_j] * m2[mul_i*w2 + j];
    }
  }
  return 0;
}

int matutil_add(float *m1, int h1, int w1, float *m2, int h2, int w2,
                float *ret) {
  if (h1 != h2 || w1 != w2) {
    print_err(
	    "Matrices have incompatible dimensions for addition %dx%d and %dx%d\n",
	    h1, w1, h2, w2);
    return -1;
  }

  for (int i = 0; i < h1; ++i) {
    for (int j = 0; j < w1; ++j) {
      int coord = i * w1 + j;
      ret[coord] = m1[coord] + m2[coord];
    }
  }
  return 0;
}

int matutil_sep_conv1(float *input, int steps, int c, int f, float *depth_kernels, float *point_kernels, int ks, float *biases, float *ret){
  int len_ret = steps*f;
  for (int i = 0; i < len_ret; ++i) {
    ret[i] = 0;
  }

  int min_offset = ks/2;
  for (int i = 0; i < steps; ++i) {
    for (int di = 0; di < ks; ++di) {
      int input_i = i - min_offset + di;
      if(input_i < 0 || input_i >= steps)
	continue;
      
      for (int ci = 0; ci < c; ++ci) {
	for (int fi = 0; fi < f; ++fi) {
	  ret[i*f + fi] += input[input_i*c + ci] * depth_kernels[di*c + ci] * point_kernels[ci*f + fi];
	}      
      }
    }


    for (int fi = 0; fi < f; ++fi) {
      ret[i*f + fi] += biases[fi];
    }

  }

  return 0;
}

int matutil_conv2(float *input, int h, int w, int c, int f, float *kernels,
                 int kh, int kw, float *biases, float *ret) {
  // clear ret
  int len_ret = h * w * f;
  for (int i = 0; i < len_ret; ++i) {
    ret[i] = 0;
  }

  int min_row_offset = kh / 2;
  int min_col_offset = kw / 2;

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      for (int ki = 0; ki < kh; ++ki) {
        for (int kj = 0; kj < kw; ++kj) {
          for (int ci = 0; ci < c; ++ci) {
            for (int fi = 0; fi < f; ++fi) {
              int input_i = i - min_row_offset + ki;
              int input_j = j - min_col_offset + kj;
	      
              // zero padding // TODO: make this data independent
              if (input_i < 0 || input_i >= h || input_j < 0 || input_j >= w)
                continue;

              ret[i * w * f + j * f + fi] +=
		input[input_i * w * c + input_j * c + ci] *
		kernels[ki * kw * c * f + kj * c * f + ci * f + fi];
            }
          }
        }
      }
      
      for (int fi = 0; fi < f; ++fi) {
	ret[i*w*f + j*f + fi] += biases[fi];
      }
    }
  }

  return 0;
}

void matutil_relu(float *m, int r, int c) {
  for (int i = 0; i < r * c; i++)
    if (m[i] < 0)
      m[i] = 0;
}

void matutil_global_average_pooling_1d(float *m, int steps, int c, float *ret){
  for (int ci = 0; ci < c; ++ci) {
    ret[ci] = 0;
  }

  for (int i = 1; i < steps; ++i) {
    for (int ci = 0; ci < c; ++ci) {
      ret[ci] += m[i*c + ci];
    }
  }

  for (int ci=0; ci < c; ++ci) {
    ret[ci] /= steps;
  }
}

void matutil_global_average_pooling_2d(float *m, int h, int w, int c, float *ret){
  //calculate the average per channel (averaging over h and w)
  for (int i=0; i < c; ++i) {
    ret[i] = 0;
  }

  for (int i = 0; i<h; ++i) {
    for (int j = 0; j<w; ++j) {
      for(int ci=0; ci<c; ++ci) {
	ret[ci] += m[i*w*c + j*c + ci];
      }
    }
  }

  int div = h*w;
  for (int ci=0; ci < c; ++ci) {
    ret[ci] /= div;
  }
}

void matutil_max_pooling_1d(float *m, int steps, int c, int pool_size, float *ret){
  int ret_steps = steps/pool_size;

  for (int i = 0; i < ret_steps; ++i) {
    int input_start = i*pool_size;
    
    for (int ci = 0; ci < c; ++ci) {
      float current_max = m[input_start*c + ci];

      for (int di=0; di < pool_size; ++di) {
	int current_i = input_start+di;
	float to_compare = m[current_i*c + ci];
	current_max = to_compare > current_max ? to_compare : current_max;
      }
      
      ret[i*c + ci] = current_max;
    }
  }
}

void matutil_max_pooling_2d(float *m, int h, int w, int c, int pool_size, float *ret){
  int ret_h = h/pool_size;
  int ret_w = w/pool_size;
  
  for (int i = 0; i < ret_h; ++i) {
    int input_i = i*pool_size;
    for (int j = 0; j < ret_w; ++j) {
      int input_j = j*pool_size;

      for (int ci = 0; ci < c; ++ci) {
	float current_max = m[input_i*w*c + input_j*c + ci];
	
	for (int di=0; di < pool_size; ++di) {
	  for (int dj=0; dj < pool_size; ++dj) {
	    int current_i = input_i + di;
	    int current_j = input_j + dj;
	    float to_compare = m[current_i*w*c + current_j*c + ci];
	    current_max = to_compare > current_max ? to_compare : current_max;
	  }
	}

	ret[i*ret_w*c + j*c + ci] = current_max;
      }
    }
  }
}

int matutil_depthwise_conv2(float *input, int h, int w, int c, int padding, float *kernels, int kh, int kw, float *ret){
  int len_ret = h * w * c;
  for(int i = 0; i< len_ret; ++i)
    ret[i] = 0;

  int min_row_offset = kh / 2;
  int min_col_offset = kw / 2;

  int row_start, row_end, col_start, col_end;
  if(padding == PADDING_SAME){
    row_start = 0;
    row_end = h;
    col_start = 0;
    col_end = w;
  } else if (padding == PADDING_VALID){
    row_start = min_row_offset;
    row_end = h - min_row_offset;
    col_start = min_col_offset;
    col_end = h - min_col_offset;
  } else {
    print_out("Unknown padding\n");
    return 1;
  }

  for(int i = 0; i<h; ++i){
    for(int j = 0; j<w; ++j){
      for(int ki = 0; ki<kh; ++ki){
        for(int kj = 0; kj<kw; ++kj){
          for(int ci = 0; ci<c; ++ci){
              int input_i = i - min_row_offset + ki;
              int input_j = j - min_col_offset + kj;
	      
              // zero padding // TODO: make this data independent
              if (input_i < 0 || input_i >= h || input_j < 0 || input_j >= w)
                continue;

              ret[i*w*c + j*c + ci] +=
                input[input_i*w*c + input_j*c + ci] *
                kernels[ki*kw*c + kj*c + ci];
          }
        }
      }
    }
  }

  return 0;
}

void matutil_zero_pad2(float *m, int h, int w, int c, int top_pad, int bottom_pad, int left_pad, int right_pad, float *ret){
  int new_width = w + left_pad + right_pad;

  //top pad
  for(int i=0; i<top_pad; ++i)
    for(int j=0; j<new_width; ++j)
      for(int ci=0; ci<c; ++ci)
        ret[i*new_width*c + j*c + ci] = 0;

  //copy contents
  for(int i=0; i<h; ++i){
    //left pad
    for(int lj=0; lj<left_pad; ++lj)
      for(int ci=0; ci<c; ++ci)
        ret[i*new_width*c + lj*c + ci] = 0;

    for(int j=0; j<w; ++j)
      for(int ci=0; ci<c; ++ci)
        ret[i*new_width*c + (left_pad+j)*c + ci] = m[i*w*c + j*c + ci];

    //right pad
    for(int rj=0; rj<left_pad; ++rj)
      for(int ci=0; ci<c; ++ci)
        ret[i*new_width*c + (left_pad+w+rj)*c + ci] = 0;
  }

  //bottom pad
  for(int i=0; i<bottom_pad; ++i)
    for(int j=0; j<new_width; ++j)
      for(int ci=0; ci<c; ++ci)
        ret[(top_pad+h+i)*new_width*c + j*c + ci] = 0;
}


void matutil_dump_matrix(float *m, int r, int c) {
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      print_out("%.09f, ", m[i * c + j]);
    }
    print_out("\n");
  }
}

void matutil_dump_matrix3(float *m, int h, int w, int c){
  for (int ci = 0; ci < c; ++ci) {
    print_out("Ci=%d:\n", ci);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
	print_out("%.07f, ", m[i*w*c + j*c + ci]);
      }
      print_out("\n");
    }
  }
}