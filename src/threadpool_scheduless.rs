use std::sync::{mpsc, Arc, Mutex};
use std::thread;

enum Message<F: FnOnce() + Send + 'static> {
    NewJob(F),
    Terminate,
}

struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}
impl Worker {
    fn new<F>(id: usize, receiver: Arc<Mutex<mpsc::Receiver<Message<F>>>>) -> Worker
    where
        F: FnOnce() + Send + 'static,
    {
        let thread = thread::spawn(move || loop {
            let message = receiver.lock().unwrap().recv().unwrap();

            match message {
                Message::NewJob(job) => {
                    println!("Worker {} got a job; executing.", id);

                    job();
                }
                Message::Terminate => {
                    println!("Worker {} was told to terminate.", id);
                    break;
                }
            }
        });

        Worker {
            id,
            thread: Some(thread),
        }
    }
}

pub struct ThreadPool<F: FnOnce() + Send + 'static> {
    workers: Vec<Worker>,
    sender: mpsc::Sender<Message<F>>,
}

impl<F: FnOnce() + Send + 'static> ThreadPool<F> {
    pub fn new(size: usize) -> ThreadPool<F> {
        assert!(size > 0);

        let (sender, receiver) = mpsc::channel();

        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(size);

        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver)));
        }

        ThreadPool { workers, sender }
    }

    pub fn execute(&self, f: F) {
        let message = Message::NewJob(f);
        self.sender.send(message).unwrap();
    }
}

impl<F: FnOnce() + Send + 'static> Drop for ThreadPool<F> {
    fn drop(&mut self) {
        println!("Sending terminate message to all workers.");

        for _ in &mut self.workers {
            self.sender.send(Message::Terminate).unwrap();
        }

        println!("Shutting down all workers.");

        for worker in &mut self.workers {
            println!("Shutting down worker {}", worker.id);

            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}
