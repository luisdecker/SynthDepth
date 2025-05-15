"Utils for evaluating a network"

import torch
import torchvision
from tqdm.rich import tqdm

from args_utils import read_tasks
from file_utils import read_json
from models.convNext import ConvNext
from models.losses import compute_scale_and_shift


def get_model(args):
    "Loads a model from a checkpoint file"

    DEVICE = int(args["gpu"][0])
    # Get model path
    model_path = args["evaluate_path"]

    # Locate experiment folder
    exp_path = args["evaluate_config"]

    # Load experiment args
    exp_args = read_json(exp_path)
    args.update(exp_args)
    args["tasks"] = read_tasks(args["tasks"])
    # Get model weights
    print("Loading model weights")
    model_weights = torch.load(model_path, map_location=torch.device(f"cuda:{DEVICE}"))

    print("Loading with load_state_dict")
    # Build a model and load weights
    # TODO: Get model from args
    model = ConvNext(
        tasks=args["tasks"],
        features=args["features"],
        pretrained_encoder=False,
        encoder_name=args.get("encoder_name"),
    ).to_gpu(DEVICE)
    model.load_state_dict(model_weights["state_dict"])
    model.eval()
    model.transforms = torchvision.transforms._presets.SemanticSegmentation(
        resize_size=None
    )
    return model


def evaluate_dataset(dataset, model, metrics, max_depth=None, squared_dataset=None):
    results = {}
    model.eval()
    for i in tqdm(range(len(dataset))):
        img, gt = dataset[i]
        gt = gt[0, 0]
        image = img.to(model.device)
        pred = model(image.unsqueeze(dim=1)).squeeze(dim=1)

        pred_ss = pred
        gt_disp = torch.zeros_like(gt)
        gt_disp[gt > 0] = 1 / gt[gt > 0]
        gt_disp_ss = gt_disp.unsqueeze(0).to(model.device)

        if squared_dataset:
            squared_input, squared_gt = squared_dataset[i]
            squared_input = squared_input.to(model.device)
            squared_gt = squared_gt.to(model.device)
            pred_ss = model(squared_input.unsqueeze(dim=1)).squeeze(dim=1)
            squared_gt[squared_gt > 0] = 1 / squared_gt[squared_gt > 0]
            gt_disp_ss = squared_gt.unsqueeze(0).to(model.device)[0, 0]

        scale, shift = compute_scale_and_shift(pred_ss[0], gt_disp_ss, gt_disp_ss > 0)
        pred = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
        pred = pred[0, 0, ...]
        if max_depth:
            pred = pred.clamp(min=(1 / max_depth))
        pred[pred > 0] = 1 / pred[pred > 0]

        sample_metrics = {}
        for metric_name, metric in metrics.items():
            # print(pred.shape, gt.shape)
            val = metric(pred, gt.to(model.device)).detach().cpu().numpy()
            sample_metrics[metric_name] = val
        results[i] = sample_metrics
    return results
