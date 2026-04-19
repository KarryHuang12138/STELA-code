# STELA

STELA is a spatio-temporal forecasting project that combines graph modules with a large language model backbone for traffic-style multivariate time-series prediction.

## Project Structure

- `run_main.py`: main training entrypoint
- `models/STELA.py`: STELA model definition
- `data_provider/`: dataset loading and split logic
- `layers/`: graph and transformer building blocks
- `dataset/`: example datasets and adjacency matrices

## Requirements

Install the Python dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Depending on your environment, you may also need a matching PyTorch install with CUDA support.

## Data Preparation

The current training script expects CSV files under `dataset/` and uses a matching adjacency matrix file for each supported dataset:

- `metr_la.csv` with `metr_la_adj.npz`
- `bw.csv` with `bw_adj.csv`
- `pems04.csv` with `pems04_adj.csv`
- `pems08.csv` with `pems08_adj.csv`
- `NYC_subway.csv` with `NYC_subway_adj.csv`

Each CSV is expected to contain a `date` column followed by node features.

## Pretrained LLM Setup

The model no longer uses hard-coded local paths. Instead, pass the backbone through `--llm_model` and `--llm_path`.

- For `GPT2`, `--llm_path` is optional. If omitted, the code uses `gpt2`.
- For `LLAMA`, `--llm_path` is required. It can point to:
  - a local Hugging Face model directory
  - a Hugging Face model id that your environment is allowed to access

Example:

```bash
--llm_model LLAMA --llm_path /path/to/your/llama-model
```

## Training

Example command for Metr-LA:

```bash
python run_main.py \
  --is_training 1 \
  --model_id Metr \
  --model_comment STELA-Metr-LA \
  --root_path ./dataset \
  --data_path metr_la.csv \
  --seq_len 12 \
  --label_len 3 \
  --pred_len 3 \
  --enc_in 207 \
  --d_ff 32 \
  --patch_len 16 \
  --stride 8 \
  --llm_model LLAMA \
  --llm_path /path/to/your/llama-model \
  --llm_dim 4096 \
  --llm_train 8 \
  --llm_layers 16 \
  --itr 1 \
  --train_epochs 500 \
  --batch_size 4 \
  --patience 10 \
  --learning_rate 1e-5 \
  --lradj COS
```

For a fast smoke test, reduce `--train_epochs` and `--batch_size`.

## Notes

- The current data loader keeps an internal `max_data_len = 8000` cap to match the present experimental setup.
- Checkpoints are written under `./checkpoints/` during training and cleaned up at the end of a run.
- Weights & Biases is initialized in disabled mode in the current script.

## License

See `LICENSE` for licensing information.
