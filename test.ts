import {solve} from './linalg';
import * as tf from '@tensorflow/tfjs';
let a = tf.tensor([[1,1,-2],[1,2,1],[-2,1,1]]);
let b = tf.tensor([[1],[1],[1]]);
let x = solve(a, b);
x.print();
