"""
Functions for parsing configs.
"""
from typing import List

import torch
import logging as log

from jiant.tasks import ALL_GLUE_TASKS, ALL_SUPERGLUE_TASKS


def parse_task_list_arg(task_list: str) -> List[str]:
    """Parse task list argument into a list of task names.

    Parameters
    ----------
    task_list : str
        comma-delimited list of tasks.

    Returns
    -------
    List[str]
        List of tasks names.

    """
    task_names = []
    for task_name in task_list.split(","):
        if task_name == "glue":
            task_names.extend(ALL_GLUE_TASKS)
        elif task_name == "superglue":
            task_names.extend(ALL_SUPERGLUE_TASKS)
        elif task_name == "none" or task_name == "":
            continue
        else:
            task_names.append(task_name)
    return task_names


def parse_cuda_related_args(args):
    """
    Parse list of decives in args.cuda
    Resolve auto options of args.cuda and args.use_amp
    """
    result_cuda = []
    if args.cuda == "auto":
        result_cuda = list(range(torch.cuda.device_count()))
        if len(result_cuda) == 1:
            result_cuda = result_cuda[0]
        elif len(result_cuda) == 0:
            result_cuda = -1
    elif isinstance(args.cuda, int):
        result_cuda = args.cuda
    elif "," in args.cuda:
        result_cuda = [int(d) for d in args.cuda.split(",")]
    else:
        raise ValueError(
            "Your cuda settings do not match any of the possibilities in defaults.conf"
        )
    if torch.cuda.device_count() == 0 and result_cuda != -1:
        raise ValueError("You specified usage of CUDA but CUDA devices not found.")

    if args.use_amp == "auto":
        if result_cuda != -1:
            args.use_amp = 1
        else:
            args.use_amp = 0
    elif args.use_amp == 1 and result_cuda == -1:
        raise ValueError("use_amp requires CUDA")

    return result_cuda
