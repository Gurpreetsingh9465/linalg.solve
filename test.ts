import {solve, det, is_upper, is_lower, diagonal, eigvals, slogdet, covarianceMatrix, cholesky } from './linalg';
import * as tf from '@tensorflow/tfjs';

//http://www.cs.unc.edu/techreports/96-043.pdf
// https://github.com/sympy/sympy/blob/master/sympy/matrices/matrices.py
let nullspace = (a: tf.Tensor, eps = 1.0/(Math.pow(10,10))) => {
    //row reduce and backsubsitute 
    const [n,m] = a.shape;
    let get = (a:tf.Tensor,i:number,j:number) => {
        const [n,m] = a.shape;
        return a.dataSync()[m*i+j];
    }
    let swapRow = (a: tf.Tensor, row1: number, row2: number) => {
        const [n,m] = a.shape;
        for(let i = 0 ; i < m ; i++) {
            let temp: number;
            temp = get(a,row1,i);
            set(a,get(a,row2,i),row1,i);
            set(a,temp,row2,i);
        } 
    }
    let set = (a:tf.Tensor,val:number,i:number,j:number) => {
        const [n,m] = a.shape;
        a.dataSync()[m*i+j] = val;
    }
    let crossCancel = (mat: tf.Tensor,i: number,j: number,a: number,b: number) => {
        //Does the row op row[i] = a*row[i] - b*row[j]
        for(let c = 0; c < m ;c++) {
            let val = a* get(mat,i,c) - b* get(mat, j,c);
            set(mat,val,i,c);
        }
    }
    if(is_lower(a) && is_upper(a)) { // not sure about this fact confirm it with sympy nullspace function
        let result = new Array(n).fill(0);
        for(let i=0;i<n;i++) {
            if(get(a,i,i) === 0)
                result[i] = 1;
        }
        return result;
    } else {
        return tf.tidy(() => {
            let mat = tf.concat([a,tf.zeros([n,1])],1);
            mat.print();
            for(let y = 0; y < n ; y++) {
                let maxrow = y;
                for(let y2 = y+1 ; y2 < n ; y2++) {
                    if(Math.abs(get(mat,y2,y)) > Math.abs(get(mat,maxrow,y)))
                        maxrow = y2;
                }
                swapRow(mat,y,maxrow);
                console.log(Math.abs(get(mat,y,y)));
                if( Math.abs(get(mat,y,y)) <= eps ) // singular matrix
                    return [];
                for(let y2 = y+1; y2 < n ;y2++) {
                    let c = get(mat,y2,y) / get(mat,y,y);
                    for(let x = y ; x < m; x++) {
                        let num = get(mat,y2,x) - get(mat,y,x);
                        set(mat,num,y2,x);
                    }
                }
            }
            mat.print();
            return [];
        })
        
    }
    
}

let eig = (a: tf.Tensor) => {
    const [n,_] = a.shape;
    let vals = eigvals(a);
    let vec = [];
    for(let val of vals) {
        let mat_minus_lambda_i = tf.sub(a,tf.mul(val,tf.eye(n)));
        let nullSp = nullspace(mat_minus_lambda_i);
        vec.push(nullSp);
    }
    return [vals,vec];
}

let SVD = (X: tf.Tensor) => {
    const [n,d] = X.shape;
    let S = []; // n
    let xmulxtrans = tf.matMul(X,X.transpose());
    let xtransmulx = tf.matMul(X.transpose(),X);
    let [vals,U] = eig(xmulxtrans);
    for(let val of vals) {
        S.push(Math.sqrt(val*(n-1)));
    }
    let [_,Vtrans] = eig(xtransmulx);
    return [U,S,Vtrans];
}


let a = tf.tensor([[ 5.,  4.,  2.],
    [ 4.,  8., 10.],
    [ 2.,  1., 11.]])
// let test = tf.tensor([[ 5,  5],
//     [-8,  9]])
let ch  = cholesky(a);
ch.print();
// tf.matMul(ch,ch.transpose()).print();
console.log(eigvals(a));
// tf.div(tf.matMul(a.transpose(),a),3).print();
