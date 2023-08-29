use std::{thread, time::Duration};

use onnx_runtime::threadpool_scheduless::ThreadPool;

fn main() {
    let threadpool = ThreadPool::new(10);

    for x in 0..100 {
        threadpool.execute(move || {
            println!("running task {}", x);
            thread::sleep(Duration::from_millis(1000))
        });
    }

    // just to keep the main thread alive
    loop {
        thread::sleep(Duration::from_millis(1000))
    }
}
