
module load gcc/10.3
module load cuda/12.1
module load ninja/1.12.1
module load cmake/3.22.0
module load cudnn/8.9.3.28_cuda12.x
module load nccl/2.19.3-1_RTX4090-cuda12.1
module load openmpi/4.1.5_ucx1.14.1_nvhpc23.5_cuda12.1


source /HOME/scw6doz/run/zly/python_virtual_environment/Megatron_test_env/bin/activate

export CUDA_PATH=/data/apps/cuda/12.1
export PATH=$CUDA_PATH/bin:$PATH
export CUDACXX=$CUDA_PATH/bin/nvcc
export CUDA_HOME=/data/apps/cuda/12.1
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

export CUDNN_PATH=/data/apps/cudnn/8.9.7_cuda12.x
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH
export CUDNN_INCLUDE_DIR=$CUDNN_PATH/include
export CUDNN_LIB_DIR=$CUDNN_PATH/lib

export OMPI_DIR=/data/apps/openmpi/4.1.5_ucx1.14.1_nvhpc23.5_cuda12.1
export PATH=$OMPI_DIR/bin:$PATH
export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_GID_INDEX=3




export NODE_RANK=0
ip addr show

export MASTER_ADDR=10.254.149.83
