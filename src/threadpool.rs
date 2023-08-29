use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum ThreadPoolError {
    #[error("Worker not available, it is not listening for jobs")]
    WorkerNotAvailable,
}

enum JobMessage<F: FnOnce() + Send + 'static> {
    NewJob(F),
    JobDone,
}

struct Worker {
    id: i32,
    is_running: Arc<AtomicBool>,
}

impl Worker {
    pub fn new<F: FnOnce() + Send + 'static>(
        id: i32,
        job_receiver: Receiver<F>,
        job_done: Sender<JobMessage<F>>,
    ) -> Self {
        let is_running = Arc::new(AtomicBool::new(false));

        let is_running_clone = Arc::clone(&is_running);

        thread::spawn(move || loop {
            match job_receiver.recv() {
                Ok(job) => {
                    println!("executing in worker {}", id);
                    job();
                    job_done
                        .send(JobMessage::JobDone)
                        .expect("The threadpool cannot receive the job done message");
                    is_running_clone.store(false, Ordering::Release);
                }
                Err(_) => {
                    return;
                }
            }
        });

        Worker { id, is_running }
    }

    pub fn is_running_set(&self) -> bool {
        self.is_running
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }
}

pub struct ThreadPool<F: FnOnce() + Send + 'static> {
    job_schedule_sender: Sender<JobMessage<F>>,
}

impl<F: FnOnce() + Send + 'static> ThreadPool<F> {
    pub fn new(threads: usize) -> Self {
        let (job_schedule_sender, job_schedule_receiver) = channel::<JobMessage<F>>();

        let mut workers = Vec::new();
        let mut job_senders = Vec::new();

        for i in 0..threads {
            let (job_sender, job_receiver) = channel::<F>();
            let worker = Worker::new(i as i32, job_receiver, job_schedule_sender.clone());
            job_senders.push(job_sender);
            workers.push(worker);
        }

        // scheduler
        thread::spawn(move || {
            let mut jobs = VecDeque::<F>::new();

            loop {
                if let Ok(job) = job_schedule_receiver.recv() {
                    match job {
                        JobMessage::NewJob(job) => jobs.push_back(job),
                        JobMessage::JobDone => (),
                    }
                } else {
                    break;
                };

                if !jobs.is_empty() {
                    for (idx, w) in workers.iter().enumerate() {
                        if w.is_running_set() {
                            job_senders[idx]
                                .send(jobs.pop_front().unwrap())
                                .expect("A worker is not listening for jobs. The threadpool will shutting down and not attempt to recover");
                            break;
                        }
                    }
                }
            }
        });

        ThreadPool {
            job_schedule_sender,
        }
    }

    pub fn execute(&self, job: F) -> Result<(), ThreadPoolError> {
        self.job_schedule_sender
            .send(JobMessage::NewJob(job))
            .map_err(|_| ThreadPoolError::WorkerNotAvailable)
    }
}
