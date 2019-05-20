#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <ctime>
#include "icd.h"

int circ(int idx,int maxlen){
  while(idx < 0){
    idx += maxlen;
  }
  return idx % maxlen;
}

icd::icd(std::vector< std::vector<double> >y_py, std::vector< std::vector<double> >h_py, int _K, double _lambd, double _sigw)
{
  y = y_py;
  h = h_py;
  K = _K;
  lambd = _lambd;
  sigw = _sigw;
  rows_lr = y.size();
  cols_lr = y.at(0).size();
  h_rows = h.size();
  h_cols = h.at(0).size();
  rows_hr = rows_lr * K;
  cols_hr = cols_lr * K;
  std::vector<double> zero_row_lr(cols_lr,0);
  std::vector<double> zero_row_hr(cols_hr,0);
  for (int i = 0 ; i < rows_lr; ++i){
    e.push_back(zero_row_lr);
  }
}

icd::~icd(){
}


std::vector< std::vector<double> > icd::update(std::vector< std::vector<double> > x, std::vector< std::vector<double> > xtilde){
  double GtG, etG, alpha;
  //std::vector< std::vector<double> > map_img = x;
  // initialize e
  for (int i = 0; i < rows_lr; ++i){
    for (int j = 0; j < cols_lr; ++j){
      e[i][j] = apply_h(i*K,j*K,rows_hr, cols_hr, x) - y[i][j];
    }
  } // end init e
  // icd main loop
  for (int i = 0; i < rows_hr; ++i){
    for (int j = 0; j < cols_hr; ++j){
      // calculate inner products
      etG = 0.;
      GtG = 0.;
      for (int di = -h_rows/2; di <= h_rows/2; ++di){
        for (int dj = -h_cols/2; dj <= h_cols/2; ++dj){
          if ((circ(di+i,rows_hr)%K == 0) && (circ(dj+j,cols_hr)%K == 0)){ 
            etG += e[circ(di+i,rows_hr)/K][circ(dj+j,cols_hr)/K] * h[di+h_rows/2][dj+h_cols/2];
            GtG += h[di+h_rows/2][dj+h_cols/2]*h[di+h_rows/2][dj+h_cols/2];
          }
        }
      }
      alpha = (lambd*(xtilde[i][j]-x[i][j]) - etG/(sigw*sigw)) / (lambd + GtG/(sigw*sigw));
      //std::cout<<"GtG="<<GtG<<std::endl;
      x[i][j] = x[i][j] + alpha;
      // update error image
      for (int di = -h_rows/2; di <= h_rows/2; ++di){
        for (int dj = -h_cols/2; dj <= h_cols/2; ++dj){
          if ((circ(di+i,rows_hr)%K == 0) && (circ(dj+j,cols_hr)%K == 0)){
            e[circ(di+i,rows_hr)/K][circ(dj+j,cols_hr)/K] += alpha * h[di+h_rows/2][dj+h_cols/2];
          }
        }
      } // end update error image
    } // end for j
  } // end for i
  return x;
}


double icd::apply_h(int i, int j, int rows, int cols, std::vector< std::vector<double> >& x){
  double retval = 0;
  for (int di = -h_rows/2; di <= h_rows/2; di++){
    for (int dj = -h_cols/2; dj <= h_cols/2; dj++){
      retval += x[circ(i+di,rows)][circ(j+dj,cols)] * h[di+h_rows/2][dj+h_cols/2];
    }
  }
  return retval; 
}
