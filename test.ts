import {solve} from './linalg';
import * as tf from '@tensorflow/tfjs';
let a = tf.tensor([[1,1,2],[0.5,5,0],[0,1,6]]);
let b = tf.tensor([[1,3],[4,1],[1,1]]);
let x = solve(a, b);
x.print();

