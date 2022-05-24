use std::ops::{Add, Sub, Mul};
use std::cmp::PartialEq;

use crate::coords::{Vector, Point, Coord};
use crate::utils::{float_eq, float_approx};

#[derive(Debug, Clone)]
struct Matrix {
    width: usize,
    height: usize,
    data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn new(n: usize, m: usize) -> Self {
        let row = vec![0.0; m];
        Self {
            width: m,
            height: n,
            data: vec![row; n]
        }
    }

    pub fn construct(mat: Vec<Vec<f64>>) -> Self {
        Self {
            width: mat[0].len(),
            height: mat.len(),
            data: mat
        }
    }

    pub fn get(&self, n: usize, m: usize) -> f64 {
        self.data[n][m]
    }

    pub fn rows(&self) -> usize {
        self.height
    }

    pub fn columns(&self) -> usize {
        self.width
    }

    pub fn ident(n: usize, m: usize) -> Self {
        let row = vec![0.0; m];
        let mut mat = vec![row; n];
        for n in 0..mat.len() {
            for m in 0..mat[n].len() {
                if n == m {
                    mat[m][n] = 1.0;
                }
            }            
        }

        Self {
            width: m,
            height: n,
            data: mat,
        }
    }

    pub fn transpose(mut self) -> Self {
        assert_eq!(self.columns(), self.rows());
        for n in 0..self.rows() {
            for m in n..(self.columns()) {
                let temp = self.data[n][m];
                self.data[n][m] = self.data[m][n];
                self.data[m][n] = temp;
            }
        }
        self
    }

    pub fn inverse(&self) -> Option<Self> {
        if !self.is_invertible() {
            //not invertible
            None
        } else {
            let mut inv = vec![vec![0.0; self.columns()]; self.rows()];
            let det = self.det();
            for n in 0..self.rows() {
                for m in 0..self.columns() {
                    inv[m][n] = self.cofactor(n,m)/det;
                }
            }
            Some(Matrix::construct(inv))
        }
    }

    pub fn is_invertible(&self) -> bool {
        !float_eq(self.det(), 0.0) && self.columns() == self.rows()
    }
    
    /// Matrix determinant. Valid only for square matrices.
    fn det(&self) -> f64 {
        assert_eq!(self.rows(), self.columns());
        Matrix::det_helper(self)
    }

    fn det_helper(sub: &Matrix) -> f64 {
        if sub.rows() == 2 {
            sub.data[0][0] * sub.data[1][1] - sub.data[0][1]*sub.data[1][0]
        } else {
            let mut det = 0.;
            for n in 0..sub.columns() {
                det += sub.get(0, n)*sub.cofactor(0, n);
            }
            det
        }
    }

    fn submatrix(&self, i: usize, j: usize) -> Matrix {
        let mut sub = vec![vec![0.0; self.columns() - 1]; self.rows()-1];
        let mut skipped = 0;
        
        for n in 0..self.rows() {
            let mut skipped_inner = 0;
            if n == i {
                skipped = 1;
                continue;
            }
            for m in 0..self.columns() {
                if m == j {
                    skipped_inner = 1;
                    continue;
                }
                sub[n - skipped][m - skipped_inner] = self.data[n][m];
            }
        }
        Matrix::construct(sub)
    }
    
    /// compute the determinant of a submatrix
    fn minor(&self, row: usize, col: usize) -> f64 {
        let sub = self.submatrix(row, col);
        Matrix::det_helper(&sub)
    }

    fn cofactor(&self, row: usize, col: usize) -> f64 {
        if (row + col) % 2 == 0 {
            1.0*self.minor(row, col)
        } else {
            -1.0*self.minor(row,col)
        }
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows() != other.rows() || self.columns() != other.columns() {
            false
        } else {
            for n in 0..self.data.len() {
                for m in 0..self.data[n].len() {
                    //5 sig figs of tolerance is good enough
                    if !float_approx(self.data[n][m], other.data[n][m], 0.00001) {
                        return false;
                    }
                 }
            }
            true
        }
    }
}

impl Mul for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.columns(), rhs.rows());
        let mut output = Self::Output::new(self.rows(), rhs.columns());

        for n in 0..self.rows() {
            for m in 0..rhs.columns() {
                for i in 0..self.columns() {
                    output.data[n][m] += self.data[n][i]*rhs.data[i][m];
                }
            }
        }
        output
    }
}

impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        assert_eq!(self.columns(), rhs.rows());
        let mut output = Self::Output::new(self.rows(), rhs.columns());

        for n in 0..self.rows() {
            for m in 0..rhs.columns() {
                for i in 0..self.columns() {
                    output.data[n][m] += self.data[n][i]*rhs.data[i][m];
                }
            }
        }
        output
    }
}

impl Mul<Vector> for &Matrix {
    type Output = Vector;
    fn mul(self, rhs: Vector) -> Self::Output {
        assert_eq!(self.columns(), 4);
        let mut v = Vector::zero();
        v.x = self.data[0][0]*rhs.x + self.data[0][1]*rhs.y + self.data[0][2]*rhs.z + self.data[0][3]*0.0;
        v.y = self.data[1][0]*rhs.x + self.data[1][1]*rhs.y + self.data[1][2]*rhs.z + self.data[1][3]*0.0;
        v.z = self.data[2][0]*rhs.x + self.data[2][1]*rhs.y + self.data[2][2]*rhs.z + self.data[2][3]*0.0;
        //no modification of last row
        v
    }
}

impl Mul<Point> for &Matrix {
    type Output = Point;
    fn mul(self, rhs: Point) -> Self::Output {
        assert_eq!(self.columns(), 4);
        let mut v = Point::zero();
        v.x = self.data[0][0]*rhs.x + self.data[0][1]*rhs.y + self.data[0][2]*rhs.z + self.data[0][3]*1.0;
        v.y = self.data[1][0]*rhs.x + self.data[1][1]*rhs.y + self.data[1][2]*rhs.z + self.data[1][3]*1.0;
        v.z = self.data[2][0]*rhs.x + self.data[2][1]*rhs.y + self.data[2][2]*rhs.z + self.data[2][3]*1.0;
        assert_eq!(self.data[3].iter().sum::<f64>(), 1.0);
        //no modification of last row
        v
    }
}

#[test]
fn construct_matrix(){
    let data = vec![vec![1.,2.,3.,4.], vec![5.5, 6.5, 7.5, 8.5], vec![9., 10., 11., 12.], vec![13.5, 14.5, 15.5, 16.5]];
    let m = Matrix::construct(data);
    
    assert!(float_eq(m.get(0,0), 1.0));
    assert!(float_eq(m.get(0,3), 4.0));
    assert!(float_eq(m.get(1,2), 7.5));
    assert!(float_eq(m.get(3,2), 15.5));
    println!("{:?}", m);
}

#[test]
fn construct_differently_sized_matrix(){
    let d1 = vec![vec![-3., 5.], vec![1., -2.]];
    let m = Matrix::construct(d1);
    assert!(float_eq(m.get(0,0), -3.));
    assert!(float_eq(m.get(1,0), 1.));

    let d2 = vec![vec![-3., 5., 0.], vec![1., -2., -7.], vec![0., 1., 1.]];
    let m = Matrix::construct(d2);
    assert!(float_eq(m.get(0,0), -3.));
    assert!(float_eq(m.get(1,1), -2.));
    assert!(float_eq(m.get(2,2), 1.));
}

#[test]
fn matrix_equality(){
    let d1 = vec![
        vec![1., 2., 3., 4.],
        vec![5., 6., 7., 8.],
        vec![9., 8., 7., 6.],
        vec![5., 4., 3., 2.],
    ];

    let m1 = Matrix::construct(d1);


    let d2 = vec![
        vec![1., 2., 3., 4.],
        vec![5., 6., 7., 8.],
        vec![9., 8., 7., 6.],
        vec![5., 4., 3., 2.],
    ];

    let m2 = Matrix::construct(d2);

    assert_eq!(m1, m2);

    let d3 = vec![
        vec![2., 3., 4., 5.],
        vec![6., 7., 8., 9.],
        vec![8., 7., 6., 5.],
        vec![4., 3., 2., 1.],
    ];

    let m3 = Matrix::construct(d3);
    assert_ne!(m1, m3);
}

#[test]
fn matrix_multiplication(){
    let d1 = vec![
        vec![1., 2., 3., 4.],
        vec![5., 6., 7., 8.],
        vec![9., 8., 7., 6.],
        vec![5., 4., 3., 2.],
    ];

    let m1 = Matrix::construct(d1);

    let d2 = vec![
        vec![ -2., 1.,  2.,  3.],
        vec![ 3., 2.,  1.,  -1.],
        vec![ 4., 3.,  6.,  5.],
        vec![ 1., 2.,  7.,  8.],
    ];

    let m2 = Matrix::construct(d2);

    let d3 = vec![
        vec![20., 22., 50., 48.],
        vec![44., 54., 114., 108.],
        vec![40., 58., 110., 102.],
        vec![16., 26., 46., 42.],
    ];

    let m3 = Matrix::construct(d3);
    assert_eq!(m1*m2, m3);
}

#[test]
fn identity_matrix() {
    let d1 = vec![
        vec![1., 2., 3., 4.],
        vec![5., 6., 7., 8.],
        vec![9., 8., 7., 6.],
        vec![5., 4., 3., 2.],
    ];
    let m1 = Matrix::construct(d1);
    
    let ident = Matrix::ident(4, 4);
    assert_eq!(m1.clone()*ident.clone(), m1.clone());
    assert_eq!(ident*m1.clone(), m1);
}

#[test]
fn transpose_matrix() {
    let d1 = vec![
        vec![0., 9., 3., 0.],
        vec![9., 8., 0., 8.],
        vec![1., 8., 5., 3.],
        vec![0., 0., 5., 8.],
    ];

    let m = Matrix::construct(d1);
    let t = m.transpose();
    let d2 = vec![
        vec![0., 9., 1., 0.],
        vec![9., 8., 8., 0.],
        vec![3., 0., 5., 5.],
        vec![0., 8., 3., 8.],
    ];

    let m2 = Matrix::construct(d2);
    assert_eq!(t, m2);

}

#[test]
fn make_submatrix() {
    let d1 = vec![
        vec![1., 5., 0.],
        vec![-3.,2., 7.],
        vec![0., 6.,-3.],
    ];

    let m = Matrix::construct(d1);
    let sm =  m.submatrix(0, 2);
    let d2 = vec![
        vec![-3., 2.],
        vec![0., 6.],
    ];

    let m2 = Matrix::construct(d2);

    assert_eq!(sm, m2);
}

#[test]
fn make_submatrix2() {
    let d1 = vec![
        vec![-6., 1., 1., 6.],
        vec![-8., 5., 8., 6.],
        vec![-1., 0., 8., 2.],
        vec![-7., 1., -1., 1.]
    ];

    let m = Matrix::construct(d1);
    let sm =  m.submatrix(2, 1);
    let d2 = vec![
        vec![-6., 1., 6.],
        vec![-8., 8., 6.],
        vec![-7., -1., 1.],
    ];

    let m2 = Matrix::construct(d2);

    assert_eq!(sm, m2);
}

#[test]
fn cofactor_3x3() {
    let d1 = vec![
        vec![3., 5., 0.],
        vec![2.,-1., -7.],
        vec![6.,-1.,5.],
    ];

    let m = Matrix::construct(d1);
    assert!(float_eq(m.minor(0,0), -12.));
    assert!(float_eq(m.cofactor(0,0), -12.));
    assert!(float_eq(m.minor(1,0), 25.));
    assert!(float_eq(m.cofactor(1,0), -25.));
}

#[test]
fn determinant_3x3() {
    let d1 = vec![
        vec![1., 2., 6.],
        vec![-5.,8., -4.],
        vec![2.,6.,4.],
    ];

    let m = Matrix::construct(d1);
    assert!(float_eq(m.cofactor(0,0), 56.0));
    assert!(float_eq(m.cofactor(0,1), 12.0));
    assert!(float_eq(m.cofactor(0,2), -46.0));
    assert!(float_eq(m.det(), -196.0));
}

#[test]
fn determinant_4x4() {
    let d1 = vec![
        vec![-2., -8., 3., 5.],
        vec![-3., 1., 7., 3.],
        vec![ 1., 2., -9., 6.],
        vec![-6., 7., 7., -9.]
    ];

    let m = Matrix::construct(d1);
    assert!(float_eq(m.cofactor(0,0), 690.));
    assert!(float_eq(m.cofactor(0,1), 447.));
    assert!(float_eq(m.cofactor(0,2), 210.));
    assert!(float_eq(m.cofactor(0,3), 51.));
    assert!(float_eq(m.det(), -4071.));
}

#[test]
fn invertible_4x4(){
    let d1 = vec![
        vec![6., 4., 4., 4.],
        vec![5., 5., 7., 6.],
        vec![ 4., -9., 3., -7.],
        vec![9., 1., 7., -6.]
    ];
    let m1 = Matrix::construct(d1);

    let d2 = vec![
        vec![-4., 2., -2., -3.],
        vec![9., 6., 2., 6.],
        vec![ 0., -5., 1., -5.],
        vec![0., 0., 0., 0.]
    ];

    let m2 = Matrix::construct(d2);
    
    assert!(m1.is_invertible());
    assert!(!m2.is_invertible());
}

#[test]
fn invert_4x4() {
    let d1 = vec![
        vec![-5., 2., 6., -8.],
        vec![1., -5., 1., 8.],
        vec![ 7., 7., -6., -7.],
        vec![1.,-3., 7., 4.]
    ];

    let m1 = Matrix::construct(d1);

    assert!(float_eq(m1.det(), 532.));
    assert!(float_eq(m1.cofactor(2,3), -160.));
    let inv = m1.inverse().unwrap();
    assert!(float_eq(inv.get(3,2), -160./532.));

    let d2 = vec![
        vec![ 0.21805, 0.45113, 0.24060,-0.04511],
        vec![-0.80827,-1.45677,-0.44361, 0.52068],
        vec![-0.07895,-0.22368,-0.05263, 0.19737],
        vec![-0.52256,-0.81391,-0.30075, 0.30639],
    ];

    let m2 = Matrix::construct(d2);
    assert_eq!(m2, inv);
}

#[test]
fn invert_4x4_2() {
    let d1 = vec![
        vec![ 8. , -5., 9., 2. ],
        vec![ 7. , 5., 6., 1. ],
        vec![ -6. , 0., 9., 6. ],
        vec![ -3. , 0., -9., -4. ],
    ];

    let m1 = Matrix::construct(d1);
    let d2 = vec![
        vec![ -0.15385 , -0.15385 , -0.28205 , -0.53846 ],
        vec![ -0.07692 , 0.12308 , 0.02564 , 0.03077 ],
        vec![ 0.35897 , 0.35897 , 0.43590 , 0.92308 ],
        vec![ -0.69231 , -0.69231 , -0.76923 , -1.92308 ],
    ];

    let m2 = Matrix::construct(d2);

    let inv = m1.inverse().unwrap();
    assert_eq!(inv, m2);
}


#[test]
fn invert_4x4_3(){
    let d1 = vec![
        vec![ 9. , 3. , 0. , 9. ],
        vec![ -5. , -2. , -6. , -3. ],
        vec![ -4. , 9. , 6. , 4. ],
        vec![ -7. , 6. , 6. , 2. ],
    ];

    let m1 = Matrix::construct(d1);

    let d2 = vec![
        vec![ -0.04074 , -0.07778 , 0.14444 , -0.22222 ],
        vec![ -0.07778 , 0.03333 , 0.36667 , -0.33333 ],
        vec![ -0.02901 , -0.14630 , -0.10926 , 0.12963 ],
        vec![ 0.17778 , 0.06667 , -0.26667 , 0.33333 ],
    ];

    let m2 = Matrix::construct(d2);

    let inv = m1.inverse().unwrap();
    assert_eq!(inv, m2);
}

#[test]
fn inverse_multiplied_by_product(){
    let d1 = vec![
        vec![ 3., -9. , 7., 3.],
        vec![ 3., -8. , 2., -9.],
        vec![ -4., 4. , 4., 1.],
        vec![ -6., 5. , -1., 1.],
    ];

    let m1 = Matrix::construct(d1);

    let d2 = vec![
        vec![8.,  2., 2., 2.],
        vec![3.,  -1., 7., 0.],
        vec![7.,  0., 5., 4.],
        vec![6.,  -2., 0., 5.],
    ];

    let m2 = Matrix::construct(d2);

    let c = &m1*&m2;
    assert_eq!(c*m2.inverse().unwrap(), m1);
}