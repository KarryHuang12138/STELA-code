import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import wandb

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from data_provider.data_factory import data_provider
from models import STELA
from utils.tools import EarlyStopping, adjust_learning_rate, del_files, vali

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def build_parser():
    parser = argparse.ArgumentParser(description="STELA training")
    parser.add_argument("--is_training", type=int, required=True, default=1, help="training status flag")
    parser.add_argument("--model_id", type=str, required=True, default="test", help="model identifier")
    parser.add_argument(
        "--model_comment",
        type=str,
        required=True,
        default="none",
        help="suffix used when saving checkpoints and logs",
    )
    parser.add_argument("--root_path", type=str, default="./dataset", help="dataset root path")
    parser.add_argument("--data_path", type=str, default="NYC_subway.csv", help="dataset file name")
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help=(
            "time feature frequency, e.g. h, d, w, 15min, or 3h"
        ),
    )
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="checkpoint directory")
    parser.add_argument("--seq_len", type=int, default=12, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=3, help="decoder warm-up length")
    parser.add_argument("--pred_len", type=int, default=3, help="prediction length")
    parser.add_argument("--enc_in", type=int, default=167, help="number of input variables")
    parser.add_argument("--d_ff", type=int, default=32, help="feed-forward hidden dimension")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="return attention maps from the model",
    )
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="patch stride")
    parser.add_argument("--llm_model", type=str, default="LLAMA", help="LLM backbone name")
    parser.add_argument(
        "--llm_path",
        type=str,
        default="",
        help="Local path or Hugging Face model id for the selected LLM",
    )
    parser.add_argument("--llm_dim", type=int, default=4096, help="LLM hidden dimension")
    parser.add_argument("--llm_train", type=int, default=4, help="number of selected LLM layers to train")
    parser.add_argument("--num_workers", type=int, default=10, help="data loader worker count")
    parser.add_argument("--itr", type=int, default=1, help="number of experiment runs")
    parser.add_argument("--train_epochs", type=int, default=10, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="training batch size")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--lradj", type=str, default="type1", help="learning rate schedule")
    parser.add_argument("--pct_start", type=float, default=0.2, help="OneCycle warm-up ratio")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="enable automatic mixed precision training",
    )
    parser.add_argument("--llm_layers", type=int, default=16, help="number of LLM layers to load")
    parser.add_argument("--percent", type=int, default=100, help="dataset usage percentage")
    return parser


def build_accelerator():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="./ds_config_zero2.json")
    return Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


def compute_attention_entropy(attentions):
    """Compute the mean attention entropy for each transformer layer."""
    entropy_scores = []
    for attn in attentions:
        attn = attn.detach()
        entropy = -torch.sum(attn * torch.log(attn + 1e-9), dim=-1)
        mean_entropy = entropy.mean().item()
        entropy_scores.append(mean_entropy)
    return entropy_scores


def print_trainable_parameters(model):
    total, trainable = 0, 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        if param.requires_grad:
            trainable += num_params
            print(f"[Trainable] {name} ({num_params} parameters)")
    print(f"\nParameter summary | total: {total} | trainable: {trainable} | ratio: {trainable/total:.4f}")


def load_adjacency_tensor(args, accelerator):
    if args.data_path == "metr_la.csv":
        adj_matrix = sp.load_npz("dataset/metr_la_adj.npz")
        adj_tensor = torch.tensor(adj_matrix.toarray(), dtype=torch.float32).to(accelerator.device)
    elif args.data_path == "bw.csv":
        adj_matrix = pd.read_csv("dataset/bw_adj.csv", header=None).values
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(accelerator.device)
    elif args.data_path == "pems04.csv":
        adj_matrix = pd.read_csv("dataset/pems04_adj.csv", header=None).values
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(accelerator.device)
    elif args.data_path == "pems08.csv":
        adj_matrix = pd.read_csv("dataset/pems08_adj.csv", header=None).values
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(accelerator.device)
    else:
        adj_matrix = pd.read_csv("dataset/NYC_subway_adj.csv", header=None).values
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(accelerator.device)
    return adj_tensor


def select_trainable_llm_layers(model, train_loader, args, accelerator):
    sample_iter = iter(train_loader)
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(sample_iter)

    batch_x = batch_x.float().to(accelerator.device)
    batch_y = batch_y.float().to(accelerator.device)
    batch_x_mark = batch_x_mark.float().to(accelerator.device)
    batch_y_mark = batch_y_mark.float().to(accelerator.device)
    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(accelerator.device)

    model.eval()
    with torch.no_grad():
        _, attentions = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_attn=True)

    entropy_list = compute_attention_entropy(attentions)
    top_k = sorted(enumerate(entropy_list), key=lambda x: x[1])[:args.llm_train]
    top_k_ids = [idx for idx, _ in top_k]
    model.llm_model.set_finetune_layers(top_k_ids)
    accelerator.print(f"[INFO] Selected LLM layers to unfreeze: {top_k_ids}")


def build_scheduler(args, model_optim, train_steps):
    if args.lradj == "COS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    elif args.lradj == "PLATEAU":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model_optim, mode="min", factor=0.5, patience=10, verbose=True, min_lr=1e-8
        )
    elif args.lradj == "WARM":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            model_optim, T_0=10, T_mult=2, eta_min=1e-8
        )
    elif args.lradj == "STEP":
        scheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=10, gamma=0.5)
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )
    return scheduler


def main():
    parser = build_parser()
    args = parser.parse_args()
    accelerator = build_accelerator()

    # Store per-epoch metrics for the final summary.
    tl = []
    vl = []
    ttl = []
    tml = []
    trl = []
    tmal = []

    for ii in range(args.itr):
        setting = "{}_sl{}_ll{}_pl{}_df{}_patch{}_stride{}_llm{}_{}".format(
            args.model_id,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_ff,
            args.patch_len,
            args.stride,
            args.llm_model,
            ii,
        )

        train_data, train_loader = data_provider(args, "train")
        vali_data, vali_loader = data_provider(args, "val")
        test_data, test_loader = data_provider(args, "test")

        adj_tensor = load_adjacency_tensor(args, accelerator)
        model = STELA.Model(args, adj_tensor).float()
        model = model.to(dtype=torch.bfloat16)
        model = model.to(accelerator.device)

        select_trainable_llm_layers(model, train_loader, args, accelerator)

        path = os.path.join(args.checkpoints, setting + "-" + args.model_comment)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
        scheduler = build_scheduler(args, model_optim, train_steps)

        criterion = nn.L1Loss()
        mae_metric = nn.L1Loss()
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler
        )

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if accelerator.is_local_main_process:
            wandb.init(
                project="STELA",
                name=args.model_id + "_" + args.model_comment,
                config=vars(args),
                mode="disabled",
            )
            wandb.watch(model, log="all")

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    accelerator.device
                )

                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = outputs[:, -args.pred_len:, :]
                        batch_y = batch_y[:, -args.pred_len:, :].to(accelerator.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, -args.pred_len:, :]
                    batch_y = batch_y[:, -args.pred_len:, :]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    if accelerator.is_local_main_process:
                        wandb.log(
                            {
                                "train_loss": loss.item(),
                                "epoch": epoch + 1,
                                "iter": i + 1,
                                "grad_norm": torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_norm=1.0,
                                ),
                                "learning_rate": model_optim.param_groups[0]["lr"],
                            }
                        )

                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "Iteration {0}, epoch {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print(
                        "Speed: {:.4f}s/iter | estimated time left: {:.4f}s".format(speed, left_time)
                    )
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    if epoch == 0 and i == 5:
                        print_trainable_parameters(model.llm_model)
                    model_optim.step()

                if args.lradj == "TST":
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            accelerator.print("Epoch {0} finished in {1:.2f}s".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            if accelerator.is_local_main_process:
                wandb.log({"epoch_train_loss": train_loss, "epoch": epoch + 1})

            vali_loss, vali_mae_loss, vali_rmse_loss, vali_mape_loss = vali(
                args, accelerator, model, vali_data, vali_loader, criterion, mae_metric
            )
            test_loss, test_mae_loss, test_rmse_loss, test_mape_loss = vali(
                args, accelerator, model, test_data, test_loader, criterion, mae_metric
            )

            if accelerator.is_local_main_process:
                wandb.log(
                    {
                        "epoch_val_loss": vali_loss,
                        "epoch_test_loss": test_loss,
                        "epoch_val_mae": vali_mae_loss,
                        "epoch_test_mae": test_mae_loss,
                        "epoch_val_rmse": vali_rmse_loss,
                        "epoch_test_rmse": test_rmse_loss,
                        "epoch_val_mape": vali_mape_loss,
                        "epoch_test_mape": test_mape_loss,
                    }
                )

            tl.append(train_loss)
            vl.append(vali_loss)
            ttl.append(test_loss)
            tml.append(test_mae_loss)
            trl.append(test_rmse_loss)
            tmal.append(test_mape_loss)

            accelerator.print(
                "Epoch {0} | train loss: {1:.7f} | val loss: {2:.7f} | test loss: {3:.7f} | "
                "test MAE: {4:.7f} | test RMSE: {5:.7f} | test MAPE: {6:.7f}".format(
                    epoch + 1,
                    train_loss,
                    vali_loss,
                    test_loss,
                    test_mae_loss,
                    test_rmse_loss,
                    test_mape_loss,
                )
            )

            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping triggered.")
                break

            if args.lradj != "TST":
                if args.lradj == "COS":
                    scheduler.step()
                    accelerator.print("Learning rate: {:.10f}".format(model_optim.param_groups[0]["lr"]))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]["lr"]
                        accelerator.print("Learning rate: {:.10f}".format(model_optim.param_groups[0]["lr"]))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
            else:
                accelerator.print("Learning rate updated to {}".format(scheduler.get_last_lr()[0]))

    for i in range(len(ttl)):
        accelerator.print(
            "History epoch {0} | train loss: {1:.7f} | val loss: {2:.7f} | test loss: {3:.7f} | "
            "test MAE: {4:.7f} | test RMSE: {5:.7f} | test MAPE: {6:.7f}".format(
                i + 1,
                tl[i],
                vl[i],
                ttl[i],
                tml[i],
                trl[i],
                tmal[i],
            )
        )

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        model_path = f"{args.checkpoints}/final_model.pth"
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

        path = "./checkpoints"
        del_files(path)
        accelerator.print("Deleted checkpoint directory.")


if __name__ == "__main__":
    main()
