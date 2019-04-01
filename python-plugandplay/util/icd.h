/*
Passing variables / arrays between cython and cpp
Example from 
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

Adapted to include passing of multidimensional arrays

*/

#include <vector>
#include <iostream>

class icd{
    public:
        icd(std::vector< std::vector<double> >y_py, std::vector< std::vector<double> >h_py, int _K, double _lambd, double _sigw);
        ~icd();
        std::vector< std::vector<double> > update(std::vector< std::vector<double> > x, std::vector< std::vector<double> > xtilde);
    private:
        int rows_hr,cols_hr,rows_lr, cols_lr, h_rows, h_cols, K;
        double lambd, sigw;
        std::vector< std::vector<double> > y;
        std::vector< std::vector<double> > h;
        std::vector< std::vector<double> > e;
        double apply_h(int i, int j, int rows, int cols, std::vector< std::vector<double> >& x);
};
