import argparse
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from utils import *


def run_benchmark(
    lr_scaler=1.0,
    lr_end_fraction=0.1,
    epochs=16,
    batch_size=512,
    ema_epochs=2,
    n_runs=1,
    warmup_fraction=5,
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Download and prepare dataset
    dataset = cifar10(root="./data/")
    dataset = map_nested(to(device), dataset)

    # Upload the mean and standard deviations to the GPU
    mean, std = [
        torch.tensor(x, device=device, dtype=torch.float16)
        for x in (CIFAR10_MEAN, CIFAR10_STD)
    ]

    train_set = preprocess(
        dataset["train"],
        [
            partial(pad, border=4),
            transpose,
            partial(normalise, mean=mean, std=std),
            to(torch.float16),
        ],
    )
    valid_set = preprocess(
        dataset["valid"],
        [transpose, partial(normalise, mean=mean, std=std), to(torch.float16)],
    )

    train_batches = partial(
        Batches,
        dataset=train_set,
        shuffle=True,
        drop_last=True,
        max_options=200,
        device=device,
    )

    valid_batches = partial(
        Batches, dataset=valid_set, shuffle=False, drop_last=False, device=device
    )

    # Data pre-processing
    timer = Timer(synch=torch.cuda.synchronize)
    eigen_values, eigen_vectors = compute_patch_whitening_statistics(train_set)

    # Run the training process n_runs times
    logs = []
    for run in range(n_runs):
        # Network construction
        channels = {"prep": 64, "layer1": 128, "layer2": 256, "layer3": 512}

        input_whitening_net = build_network(
            channels=channels,
            extra_layers=(),
            res_layers=("layer1", "layer3"),
            conv_pool_block=conv_pool_block_pre,
            prep_block=partial(
                whitening_block, eigen_values=eigen_values, eigen_vectors=eigen_vectors
            ),
            scale=1 / 16,
            types={
                nn.ReLU: partial(nn.CELU, 0.3),
                BatchNorm: partial(GhostBatchNorm, num_splits=16, weight=False),
            },
        )

        model = (
            Network(input_whitening_net, label_smoothing_loss(0.2)).half().to(device)
        )
        is_bias = group_by_key(
            ("bias" in k, v) for k, v in trainable_params(model).items()
        )
        loss = model.loss

        timer = Timer(torch.cuda.synchronize)

        # Data iterators
        transforms = (Crop(32, 32), FlipLR())
        tbatches = train_batches(batch_size, transforms)
        train_batch_count = len(tbatches)
        vbatches = valid_batches(batch_size)

        # Construct the learning rate, weight decay and momentum schedules
        opt_params = {
            "lr": lr_schedule(
                [0, epochs / warmup_fraction, epochs - ema_epochs],
                [0.0, lr_scaler * 1.0, lr_scaler * lr_end_fraction],
                batch_size,
                train_batch_count,
            ),
            "weight_decay": Const(5e-4 * lr_scaler * batch_size),
            "momentum": Const(0.9),
        }

        opt_params_bias = {
            "lr": lr_schedule(
                [0, epochs / warmup_fraction, epochs - ema_epochs],
                [0.0, lr_scaler * 1.0 * 64, lr_scaler * lr_end_fraction * 64],
                batch_size,
                train_batch_count,
            ),
            "weight_decay": Const(5e-4 * lr_scaler * batch_size / 64),
            "momentum": Const(0.9),
        }

        opt = SGDOpt(
            weight_param_schedule=opt_params,
            bias_param_schedule=opt_params_bias,
            weight_params=is_bias[False],
            bias_params=is_bias[True],
        )

        # Train the network
        model.train(True)
        epochs_log = []
        for epoch in range(epochs):
            activations_log = []
            for tb in tbatches:
                # Forward step
                out = loss(model(tb))
                model.zero_grad()
                out["loss"].sum().backward()
                opt.step()

                # Log activations
                activations_log.append(("loss", out["loss"].detach()))
                activations_log.append(("acc", out["acc"].detach()))

            # Compute the average over the activation logs for the last epoch
            res = map_values(
                (lambda xs: to_numpy(torch.cat(xs)).astype(float)),
                group_by_key(activations_log),
            )
            train_summary = mean_members(res)
            timer()

            # Evaluate the model
            model.eval()
            valid_summary = eval_on_batches(model, loss, vbatches)
            model.train()
            timer(update_total=False)
            time_to_epoch_end = timer.total_time

            epochs_log.append(
                {
                    "valid": valid_summary,
                    "train": train_summary,
                    "time": time_to_epoch_end,
                }
            )

            print(f"Epoch {epoch + 1}/{epochs}")
            print(
                f"Train acc {train_summary['acc']:.3f} loss {train_summary['loss']:.3f}"
            )
            print(
                f"Valid acc {valid_summary['acc']:.3f} loss {valid_summary['loss']:.3f}"
            )
            print(f"Time: {time_to_epoch_end:.3f}s")

        timer()

        if run == 0:
            save_log_to_tsv(epochs_log, path="timing_log.tsv")
            torch.save(model.state_dict(), "cifar10_model")

        logs.append(
            {
                "train_acc": train_summary["acc"],
                "train_loss": train_summary["loss"],
                "valid_acc": valid_summary["acc"],
                "valid_loss": valid_summary["loss"],
                "time": timer.total_time,
            }
        )

    # Compute the average accuracies and training times
    times = [d["time"] for d in logs]
    accuracies = [d["valid_acc"] for d in logs]
    print(f"Maximum training time {np.max(times):.3f}s median {np.median(times):.3f}s")
    print(
        f"Lowest accuracy {np.min(accuracies):.3f} median {np.median(accuracies):.3f}"
    )
    print(
        f"{np.count_nonzero(np.array(accuracies) >= 0.94)} runs reached 0.94 out of {n_runs}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr_scaler",
        type=float,
        default=1.5,
        help="Multiplicative scaling factor for the learning rate schedule",
    )
    parser.add_argument(
        "--lr_scaler_end_fraction",
        type=float,
        default=0.1,
        help="Fraction of the peak learning rate used for the final step",
    )
    parser.add_argument(
        "--epochs", type=int, default=18, help="Total number of training epochs"
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=5,
        help="Inverse of fraction of the epochs used to reach the peak learning rate",
    )
    parser.add_argument(
        "--ema_ep",
        type=float,
        default=2,
        help="Number of epochs (at the end of training) where the learning rate is to be maintained constant",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--runs", type=int, default=1, help="Number of replicas")
    args = parser.parse_args()

    run_benchmark(
        lr_scaler=args.lr_scaler,
        lr_end_fraction=args.lr_scaler_end_fraction,
        epochs=args.epochs,
        ema_epochs=args.ema_ep,
        n_runs=args.runs,
        batch_size=args.batch_size,
        warmup_fraction=args.warmup_fraction,
    )
