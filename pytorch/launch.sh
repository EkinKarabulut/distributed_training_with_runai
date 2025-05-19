#!/bin/bash

echo "Using configuration:"
echo "  MASTER_ADDR:   $MASTER_ADDR"
echo "  MASTER_PORT:   $MASTER_PORT"
echo "  WORLD_SIZE:    $WORLD_SIZE"
echo "  PET_NPROC_PER_NODE: $PET_NPROC_PER_NODE"

torchrun --nproc_per_node=1 \
  --nnodes="$WORLD_SIZE" \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_backend=c10d \
  distributed.py \
  --batch_size 8 30 2
