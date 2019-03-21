import * as tf from '@tensorflow/tfjs';


let diagonal = (a:tf.Tensor) => {
    const [n,m] = a.shape;
    if(n !== m) {
        throw new Error('Matrix is not sqaure matrix')
    }
    if(a.rank !== 2) {
        throw new Error('Matrix was expected but got a Tensor')
    }
    let get = (a:tf.Tensor,i:number,j:number) => {
        return a.dataSync()[n*i+j];
    }
    let diagonal = [];
    for(let i = 0;i < n;i++) {
        diagonal.push(get(a,i,i));
    }
    return diagonal;
}


let det = (m: tf.Tensor) => {
    return tf.tidy(() => {
        const [r, _] = m.shape;
        if(r == 1) {
            return m.as1D().slice([0], [1]).dataSync()[0]
        }
        if (r === 2) {
            const t = m.as1D()
            const a = t.slice([0], [1]).dataSync()[0]
            const b = t.slice([1], [1]).dataSync()[0]
            const c = t.slice([2], [1]).dataSync()[0]
            const d = t.slice([3], [1]).dataSync()[0]
            let result:number = a * d - b * c
            return result

        } else {
            let s: number = 0;
            let rows = Array.from(Array(r).keys());
            for (let i = 0; i < r; i++) {
                let mul = m.slice([i], [1]).dataSync()[0]
                let sub_m = m.gather(tf.tensor1d(rows.filter(e => e !== i), 'int32'))
                let sli = sub_m.slice([0, 1], [r - 1, r - 1])
                s += mul * Math.pow(-1, i) * det(sli)
            }
            return s
        }
    })
}

let invertMatrix = (m: tf.Tensor) => {
    return tf.tidy(() => {
        const [r, c] = m.shape;
        if(r !== c) {
            let message: string = 'Matrix m is a Singular Matrix of shape '+r+' '+c;
            throw new Error(message);
        }
        const d = det(m);
        if (d === 0) {
            let message: string = 'Matrix m is Singular Matrix';
            throw new Error(message);
        }
        let rows = Array.from(Array(r).keys());
        let dets = [];
        for (let i = 0; i < r; i++) {
            for (let j = 0; j < r; j++) {
                const sub_m = m.gather(tf.tensor1d(rows.filter(e => e !== i), 'int32'))
                let sli: tf.Tensor;
                if (j === 0) {
                    sli = sub_m.slice([0, 1], [r - 1, r - 1])
                } else if (j === r - 1) {
                    sli = sub_m.slice([0, 0], [r - 1, r - 1])
                } else {
                    const [a, b, c] = tf.split(sub_m, [j, 1, r - (j + 1)], 1)
                    sli = tf.concat([a, c], 1)
                }
                dets.push(Math.pow(-1, (i + j)) * det(sli))
            }
        }
        let com = tf.tensor2d(dets, [r, r])
        let tr_com = com.transpose()
        let inv_m = tr_com.div(tf.scalar(d))
        return inv_m
    })
}

let solve = (a: tf.Tensor, rhs: tf.Tensor) => {
    if(rhs.rank === 1) {
        rhs = tf.reshape(rhs,[rhs.shape[0],1])
    }
    if(a.rank !== 2) {
        let message: string = 'Expected Rank of matrix a to be 2 but it is '+a.rank;
        throw new Error(message);
    }
    if(rhs.rank !== 2) {
        let message: string = 'Expected Rank of RHS to be 2 but it is '+rhs.rank;
        throw new Error(message);
    }
    return tf.tidy(()=>{
        const [r, c] = a.shape;
        if(r !== c) {
            let message: string = 'Matrix m should be square matrix';
            throw new Error(message);
        }
        let inverseMat = invertMatrix(a);
        let result = tf.matMul(inverseMat,rhs);
        return result;
    })
}

let is_lower = (a: tf.Tensor) => {
    const [n,m] = a.shape;
    if(n !== m) {
        throw new Error('Matrix is not sqaure matrix')
    }
    if(a.rank !== 2) {
        throw new Error('Matrix was expected but got a Tensor')
    }
    let get = (a:tf.Tensor,i:number,j:number) => {
        return a.dataSync()[n*i+j];
    }
    for(let i = 0; i<n; i++) {
        for(let j = i+1; j<m; j++) {
            if(get(a,i,j) !== 0)
                return false;
        }
    }
    return true;
}
let is_upper = (a: tf.Tensor) => {
    const [n,m] = a.shape;
    if(n !== m) {
        throw new Error('Matrix is not sqaure matrix')
    }
    if(a.rank !== 2) {
        throw new Error('Matrix was expected but got a Tensor')
    }
    let get = (a:tf.Tensor,i:number,j:number) => {
        return a.dataSync()[n*i+j];
    }
    for(let i = 2; i<n; i++) {
        for(let j = 0; j< Math.min(m,i-1); j++) {
            if(get(a,i,j) !== 0)
                return false;
        }
    }
    return true;
}

let eigvals = (mat: tf.Tensor,max_it = 10) => {
    if(is_upper(mat) || is_lower(mat)) {
        let res = diagonal(mat);
        
        return res;
    }
    return tf.tidy(()=>{
        let copy = mat.clone();
        let i = 0;
        while(i <= max_it) {  // stop loop when copy is right triangular matrix 
            const [a,b] = tf.linalg.qr(copy);
            copy = tf.matMul(b,a);
            i+=1;
        }
        let res = diagonal(copy);
        res = res.sort((a,b)=>{return b-a});
        return res;
    })
}

let slogdet = (a: tf.Tensor) => {
    let d = det(a);
    if(d == 0) {
        return [0,Infinity];
    }else if(d < 0 ) {
        return [-1,Math.log((-1*d))];
    } else {
        return [1,Math.log(d)];
    }
}

let covarianceMatrix = (mat: tf.Tensor) => {
    // using kalman filtering can also be calculated by (mat.transpose() * mat)/n-1
    if(mat.rank !== 2) {
        throw new Error('Matrix was expected but got a Tensor')
    }
    let [n,d] = mat.shape;
    return tf.tidy(()=>{
        // a = mat - (ones([n,n]) * mat)/n
        let a = tf.sub(mat,tf.div(tf.matMul(tf.ones([n,n]),mat),n));
        // res = (a.transpose*a)/n
        let res = tf.div(tf.matMul(a.transpose(),a),n);
        return res;
    })
}

let cholesky = (a: tf.Tensor) => {
    let get = (a:tf.Tensor,i:number,j:number) => {
        const [n,m] = a.shape;
        return a.dataSync()[m*i+j];
    }
    const [n,m] = a.shape;
    let ev = eigvals(a);
    for(let e of ev){
        if(isNaN(e) || e < 0){
            throw new Error("Matrix is not positive definite")
        }
    }
    let set = (a:tf.Tensor,val:number,i:number,j:number) => {
        const [n,m] = a.shape;
        a.dataSync()[m*i+j] = val;
    }
    let L = tf.zeros([n,n]);
    for(let i = 0;i<n;i++) {
        for(let j = 0; j<=i;j++) {
            let val = 0;
            if(i === j) {
                for(let k = 0; k < j; k++) {
                    val += Math.pow(get(L,j,k),2);
                }
                val = Math.sqrt(get(a,j,j) - val);
                set(L,val,j,j);
            } else {
                for(let k = 0; k < j; k++) {
                    val += (get(L,i,k)*get(L,j,k));
                }
                val = get(a,i,j) - val;
                val /= get(L,j,j);
                set(L,val,i,j);
            }
            if(isNaN(val)) {
                throw new Error("Matrix is not positive definite");
            }
        }
    }
    return L;
}


export { solve, invertMatrix, det, is_lower, is_upper, diagonal, eigvals, slogdet, covarianceMatrix, cholesky };
