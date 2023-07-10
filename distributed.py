import os
import json
import tensorflow as tf
import utils

per_worker_batch_size = 64

# Loading the TF_CONFIG
tf_config = json.loads(os.environ["TF_CONFIG"])
print(f"tf_config: {tf_config}")

# Some print assignments for us to observe the workers and the variables
task_config = tf_config.get('task', {})
print(f"task_config: {task_config}")

task_type = task_config.get('type')
print(f"task_type: {task_type}")

task_index = task_config.get('index')
print(f"task_index: {task_index}")

num_workers = len(tf_config['cluster']['worker'])
print(f"number of workers: {num_workers}")

cluster_config = tf_config.get('cluster', {})
print(f"cluster_config: {cluster_config}")

worker_hosts = cluster_config.get('worker')
print(f"worker_host: {worker_hosts}")

worker_hosts_str = ','.join(worker_hosts)
print(f"worker_hosts_str: {worker_hosts_str}")

# Print the available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Available GPUs:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPUs found.")

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = utils.mnist_dataset(global_batch_size)

with strategy.scope():
    multi_worker_model = utils.build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset, epochs=10, steps_per_epoch=70)
