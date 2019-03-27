#include <iostream>
#include <algorithm>
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
    Gs.push_back(zero_row_lr);
  }
  for (int i = 0; i < rows_hr; ++i)
    Hs.push_back(zero_row_hr);
    // initialize Hx
  for (int i = -h_rows/2; i <= h_rows/2; ++i){
    for (int j = -h_cols/2; j <= h_cols/2; ++j){
      Hs[circ(i,rows_hr)][circ(j,cols_hr)] = h[i+h_rows/2][j+h_cols/2];
    }
  }
}

icd::~icd(){
}


void icd::down_sample(std::vector< std::vector<double> >& hr_img, std::vector< std::vector<double> >& lr_img){
  for (int i = 0; i < rows_lr; ++i){
    for (int j = 0; j < cols_lr; ++j){
      lr_img[i][j] = hr_img[i*K][j*K];
    }
  }
  return;
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
      // calculate Gs
      down_sample(Hs, Gs);
      // calculate inner products
      etG = 0.;
      GtG = 0.;
      for (int di = 0; di < rows_lr; ++di){
        for (int dj = 0; dj < cols_lr; ++dj){
          etG += e[di][dj] * Gs[di][dj];
          GtG += Gs[di][dj] * Gs[di][dj];
        }
      }
      alpha = std::max(-x[i][j],(lambd*(xtilde[i][j]-x[i][j]) - etG/(sigw*sigw)) / (lambd + GtG/(sigw*sigw)));
      //std::cout<<"GtG="<<GtG<<std::endl;
      x[i][j] = x[i][j] + alpha;
      // update error image
      for (int k = 0; k < rows_lr; ++k){
        for (int l = 0; l < cols_lr; ++l){
          e[k][l] += Gs[k][l]*alpha;
        }
      }
      // shift Hx
      for (int di = 0; di < rows_hr; ++di){
        std::rotate(Hs.at(di).begin(), Hs.at(di).begin() + Hs.at(di).size()-1, Hs.at(di).end());
      }
    }
    std::rotate(Hs.begin(), Hs.begin() + Hs.size()-1, Hs.end());
  } 
  return x;
}


double icd::apply_h(int i, int j, int rows, int cols, std::vector< std::vector<double> >& x){
  if ( (i >= rows) || (j >= cols))
    throw std::runtime_error("index of pixel out of range");
  double retval = 0;
  for (int di = -h_rows/2; di <= h_rows/2; di++){
    for (int dj = -h_cols/2; dj <= h_cols/2; dj++){
      retval += x[circ(i+di,rows)][circ(j+dj,cols)] * h[di+h_rows/2][dj+h_cols/2];
    }
  }
  return retval; 
}
