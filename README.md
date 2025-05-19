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

You can find the related documents in [pytorch](https://github.com/EkinKarabulut/distributed_training_with-Run-ai/tree/main/pytorch) folder and the created Docker image [here](https://hub.docker.com/repository/docker/ekink/distributed_training_pytorch/general). 

### Modifying the Training Script for DDP Training

To prepare your code for DDP training, you need to make some modifications to the training script. In this tutorial, we will use a slightly different version of [the example script provided in the PyTorch documentation](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series). Please refer to the official PyTorch documentation for more details. You can find the modified training script, named distributed.py, and all other documents presented in this guide in this GitHub repository.

Here are some essential points to consider when modifying the training script for DDP:

- [Setup a Process Group](https://github.com/EkinKarabulut/distributed_training_with_runai/blob/409178fe291276e0ab870395747f7c50d5d70c76/pytorch/distributed.py#L26): Create a worker within a group of workers using the torch.distributed.init_process_group() function.
- [Choose a Communication Backend](https://github.com/EkinKarabulut/distributed_training_with_runai/blob/409178fe291276e0ab870395747f7c50d5d70c76/pytorch/distributed.py#L26): Select the appropriate communication backend that handles the communication between workers. For GPU-accelerated jobs, it is recommended to use "nccl." For CPU jobs, "gloo" is the recommended backend. In this tutorial, we will use "nccl."
- [Use DistributedSampler for Data Loading](https://github.com/EkinKarabulut/distributed_training_with_runai/blob/409178fe291276e0ab870395747f7c50d5d70c76/pytorch/distributed.py#L104): Wrap your dataset with the DistributedSampler to ensure that each GPU receives a different portion of the dataset automatically. This sampler helps distribute the data across the nodes efficiently.
- [Wrap Your Model with DDP](https://github.com/EkinKarabulut/distributed_training_with_runai/blob/409178fe291276e0ab870395747f7c50d5d70c76/pytorch/distributed.py#L50): Enclose your model with DistributedDataParallel (DDP) to enable data parallelism. DDP synchronizes gradients across each model replica and specifies the devices to synchronize, which is the entire world by default.

After making the required changes to the training script, we will create a bash script, [launch.sh](https://github.com/EkinKarabulut/distributed_training_with_runai/blob/main/pytorch/launch.sh), which will launch the training job.

We will execute distributed.py using torchrun on every node, as explained in the PyTorch documentation. The script includes various system-related arguments passed to the torchrun command. Here is an overview of what each variable does:

- `nproc_per_node`: The number of workers on each node. In our case, this value is set to 1.
- `nnodes`: The total number of nodes participating in the distributed training.
- `rdzv_endpoint`: The IP address and port on which the C10d rendezvous backend should be instantiated and hosted. It is recommended to select a node with high bandwidth for optimal performance.
- `rdzv_backend`: The backend of the rendezvous (e.g. c10d).
By following this approach, you won't need to recreate your Docker image if the master node changes.

> [!NOTE]
> During training, it is a best practice to save checkpoints frequently to mitigate the impact of network challenges. In this tutorial, we save checkpoints every 2 epochs. However, feel free to adjust this frequency based on your specific use case.

### Creating the Docker Image & Pushing to a Docker Registry

Now, let's create a Docker image to encapsulate our training environment. 

If you want to create your own image, you can edit your code, create your image and push the image to your image registry with the following commands; 

```
docker build -t YOUR-USER-NAME/distributed_training_pytorch .
docker push YOUR-USER-NAME/distributed_training_pytorch
```

### Launching the Distributed Training on Run:ai

In order to submit a job, you will need to run the following command on your terminal:

```
# For CLI versions below 2.13:
runai submit-pytorch --name distributed-training-pytorch --workers=1 -g 1 \
        -i docker.io/YOUR-USER-NAME/distributed_training_pytorch

# For CLI versions 2.13 and above:
runai submit-dist pytorch --name distributed-training-pytorch --workers=1 -g 1 \
        -i docker.io/YOUR-USER-NAME/distributed_training_pytorch       
```

This will launch 2 pods - one is master node and the other one is a worker node. Each of them will have 1 GPU. Be aware that you automatically get these environment variables on every pod (master and workers); `$MASTER_ADDR`, `$MASTER_PORT`, `$WORLD_SIZE` and `$RANK`. So you can directly launch a job on `launch.sh` script passing these variables.

For more information about `runai submit-pytorch` command, please refer to the [documentation](https://docs.run.ai/v2.10/Researcher/cli-reference/runai-submit-pytorch/)

## Data parallelism with Tensorflow (using `MultiWorkerMirroredStrategy`)

You can find the related documents in [tensorflow](https://github.com/EkinKarabulut/distributed_training_with-Run-ai/tree/main/tensorflow) folder and the created Docker image [here](https://hub.docker.com/repository/docker/ekink/distributed_training_tensorflow/general). If you want to create your own image, you can edit your code, create your image and push the image to your image registry with the following commands; 

```
docker build -t YOUR-USER-NAME/distributed_training_tensorflow .
docker push YOUR-USER-NAME/distributed_training_tensorflow
```

In order to submit a job, you will need to run the following command on your terminal:

```
runai submit-tf --name distributed-training-tf --workers=2 -g 1 \
        -i docker.io/YOUR-USER-NAME/distributed_training_tensorflow --no-master
```

This will launch 2 workers with 1 GPU on each. Be aware that the required envrionment variable `TF_CONFIG` is automatically populated when the pods are created, which means that you do not need to populate it yourself. You can export it in the bash script and use it in your training scripts.

For more information about `runai submit-tf` command, please refer to the [documentation](https://docs.run.ai/v2.10/Researcher/cli-reference/runai-submit-tf/)
