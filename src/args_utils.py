"Utils for reading CLI args"

import argparse

from file_utils import read_json
from metrics import get_metric
from models import get_loss
from models.decoder import get_decoder
from models.task import get_task


def read_tasks(args_tasks):
    """Reads a task argument list and convert to task objects"""
    task_list = []
    for task in args_tasks:
        Task = get_task(task["type"])
        Decoder = get_decoder(task["decoder"]["type"])
        metrics = {name: get_metric(name)() for name in task["metrics"]}
        loss = get_loss(task["loss"], **task.get("loss_params", {}))

        _task = Task(
            name=task["name"],
            decoder=Decoder,
            metrics=metrics,
            features=task["features"],
            channels=task["channels"],
            mask_feature=task.get("mask_feature"),
            decoder_args=task["decoder"]["args"],
            train_on_disparity=task.get("train_on_disparity", False),
            loss=loss,
            disp_metrics=task.get("disp_metrics", True),
        )

        task_list.append(_task)
    return task_list


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)
    parser.add_argument("--evaluate_path", type=str, default=None)
    parser.add_argument("--evaluate_config", type=str, default=None)
    parser.add_argument("--gpu", nargs="+", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--slurm", action="store_true")

    args = vars(parser.parse_args())
    if not args["evaluate_path"]:
        args.update(read_json(args["config"]))

    args["input_args"] = args.copy()
    if "tasks" in args:
        args["tasks"] = read_tasks(args["tasks"])

    return args
