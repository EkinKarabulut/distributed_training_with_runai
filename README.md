# Distributed Training on Kubernetes with Run:ai

When it comes to training big models or handling large datasets, relying on a single node might not be sufficient and can lead to slow training processes. This is where multi-node training comes to the rescue. There are several incentives for teams to transition from single-node to multi-node training. Some common reasons include:

## Faster Experimentation: Speeding Up Training
In research and development, time is of the essence. Teams often need to accelerate the training process to obtain experimental results quickly. Employing multi-node training techniques, such as data parallelism, helps distribute the workload and leverage the collective processing power of multiple nodes, leading to faster training times.

## Large Batch Sizes: Data Parallelism
When the batch size required by your model is too large to fit on a single machine, data parallelism becomes crucial. It involves duplicating the model across multiple GPUs, with each GPU processing a subset of the data simultaneously. This technique helps distribute the workload and accelerate the training process.

## Large Models: Model Parallelism
In scenarios where the model itself is too large to fit on a single machine's memory, model parallelism is utilized. This approach involves splitting the model across multiple GPUs, with each GPU responsible for computing a portion of the model's operations. By dividing the model's parameters and computations, model parallelism enables training on machines with limited memory capacity.


In this repo, we will introduce how to leverage data parallelism using PyTorch and Tensorflow on Kubernetes with Run:ai. For demo purposes, the training scripts that are presented are slightly modified versions of the example scripts that [Pytorch](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html) and [Tensorflow](https://www.tensorflow.org/guide/distributed_training) provide on their official websites. If you are new to distributed training, please refer to their documentation for more information.

## Prerequisites

Before diving into the distributed training setup, ensure that you have the following prerequisites in place:

### Run:ai environment (v2.10 and later)
Make sure you have access to a Run:ai environment with version 2.10 or a later release. Run:ai provides a comprehensive platform for managing and scaling deep learning workloads on Kubernetes.
### Two nodes with one GPU each 
For this tutorial, we will use a setup consisting of two nodes, each equipped with one GPU. However, you can scale up by adding more nodes or GPUs to suit your specific requirements.
### Image Registry (e.g., Docker Hub Account)
Prepare an image registry, such as a Docker Hub account, where you can store your custom Docker images for distributed training.

## Data parallelism with Pytorch (using `torchrun` and DDP)

You can find the related documents in [pytorch](https://github.com/EkinKarabulut/distributed_training_with-Run-ai/tree/main/pytorch) folder and the created Docker image [here](https://hub.docker.com/repository/docker/ekink/distributed_training_pytorch/general). If you want to create your own image, you can edit your code, create your image and push the image to your image registry with the following commands; 

```
docker build -t YOUR-USER-NAME/distributed_training_pytorch .
docker push YOUR-USER-NAME/distributed_training_pytorch
```

In order to submit a job, you will need to run the following command on your terminal:

```
runai submit-pytorch --name distributed-training-pytorch --workers=1 -g 1 \
        -i docker.io/YOUR-USER-NAME/distributed_training_pytorch
```

This will launch 2 pods - one is master node and the other one is a worker node. Each of them will have 1 GPU. Be aware that you automatically get these environment variables on every pod (master and workers); `$MASTER_ADDR`, `$MASTER_PORT`, `$WORLD_SIZE` and `$RANK`. So you can directly launch a job on `launch.sh` script passing these variables.

For more information about `runai submit-pytorch` command, please refer to the [documentation](https://docs.run.ai/v2.10/Researcher/cli-reference/runai-submit-pytorch/)

## Data parallelism with Tensorflow (using `MultiWorkerMirroredStrategy`)

You can find the related documents in [pytorch](https://github.com/EkinKarabulut/distributed_training_with-Run-ai/tree/main/tensorflow) folder and the created Docker image [here](https://hub.docker.com/repository/docker/ekink/distributed_training_tensorflow/general). If you want to create your own image, you can edit your code, create your image and push the image to your image registry with the following commands; 

```
docker build -t YOUR-USER-NAME/distributed_training_tensorflow .
docker push YOUR-USER-NAME/distributed_training_tensorflow
```

In order to submit a job, you will need to run the following command on your terminal:

```
runai submit-tf --name distributed-training-tf --workers=2 -g 1 \
        -i docker.io/YOUR-USER-NAME/distributed_training_tensorflow --no-master
```

This will launch 2 workers with 1 GPU on each. Be aware that the required envrionment variable `TF_CONFIG` is automatically populated when the pods are created, which means that you do not need to populate it yourself and just export it in the bash script and use it in your training scripts.

For more information about `runai submit-tf` command, please refer to the [documentation](https://docs.run.ai/v2.10/Researcher/cli-reference/runai-submit-tf/)
