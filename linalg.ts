import * as tf from '@tensorflow/tfjs';

let det = (m: tf.Tensor) => {
    return tf.tidy(() => { // cleans up intermediate tensors
        const [r, _] = m.shape
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
        const d = det(m)
        if (d === 0) {
            return
        }
        const [r, _] = m.shape
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

/*

Parameters:	

a : (…, M, M) array_like

    Coefficient matrix.
b : {(…, M,), (…, M, K)}, array_like

    Ordinate or “dependent variable” values.

Returns:	

x : {(…, M,), (…, M, K)} ndarray

    Solution to the system a x = b. Returned shape is identical to b.

*/

let solve = (a: tf.Tensor, rhs: tf.Tensor) => {
    if(det(a) == 0) {
        console.log('singular matrix'); // assertion
        return tf.tensor([0]);
    }
    let inverseMat = invertMatrix(a);
    let result = tf.matMul(inverseMat,rhs);
    return result;
}

export { solve,invertMatrix,det };