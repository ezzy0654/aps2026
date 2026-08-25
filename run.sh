#!/bin/bash

srun --partition aps --gres=gpu:1 --exclusive \
  ./main "$@"
