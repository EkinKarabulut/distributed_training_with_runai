# Distributed Training on Kubernetes with Run:ai

When it comes to training big models or handling large datasets, relying on a single node might not be sufficient and can lead to slow training processes. This is where multi-node training comes to the rescue. There are several incentives for teams to transition from single-node to multi-node training. Some common reasons include:

## Faster Experimentation: Speeding Up Training
In research and development, time is of the essence. Teams often need to accelerate the training process to obtain experimental results quickly. Employing multi-node training techniques, such as data parallelism, helps distribute the workload and leverage the collective processing power of multiple nodes, leading to faster training times.

## Large Batch Sizes: Data Parallelism
When the batch size required by your model is too large to fit on a single machine, data parallelism becomes crucial. It involves duplicating the model across multiple GPUs, with each GPU processing a subset of the data simultaneously. This technique helps distribute the workload and accelerate the training process.

## Large Models: Model Parallelism
In scenarios where the model itself is too large to fit on a single machine's memory, model parallelism is utilized. This approach involves splitting the model across multiple GPUs, with each GPU responsible for computing a portion of the model's operations. By dividing the model's parameters and computations, model parallelism enables training on machines with limited memory capacity.


In this tutorial, we will focus on data parallelism using PyTorch and Tensorflow on Kubernetes with Run:ai 
