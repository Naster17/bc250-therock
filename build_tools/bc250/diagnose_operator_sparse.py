#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Isolate the new sparse operator coverage in fresh processes with fault traces."""

import argparse
import faulthandler


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--layout", choices=("csr", "coo"), required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(91013)
    dtype = getattr(torch, args.dtype)
    layout = torch.sparse_csr if args.layout == "csr" else torch.sparse_coo
    dense = torch.randn(11, 13, dtype=dtype)
    dense[dense.abs() < 1] = 0
    right = torch.randn(13, 5, dtype=dtype)
    print("CREATE", args.dtype, args.layout, flush=True)
    sparse = dense.to_sparse(layout=layout)
    gpu = sparse.cuda()
    gpu_right = right.cuda()
    print("MATMUL", flush=True)
    actual = torch.sparse.mm(gpu, gpu_right)
    print("COPY", flush=True)
    torch.testing.assert_close(actual.cpu(), dense @ right, rtol=3e-4, atol=3e-4)
    print("SPARSE_OPERATOR_PASS", flush=True)


if __name__ == "__main__":
    main()
