use ndarray::prelude::*;

fn main() {
    let arr = array![
        [0, 1, 2 ,3],
        [4, 5, 6, 7],
        [8, 9, 10, 11]
    ];
    println!("{0}",arr.slice(s![0.., 1..]));
    println!("{0}",arr.slice(s![1.., 1..]));
    println!("{0}",arr.slice(s![1..2, 1..2]));
}
