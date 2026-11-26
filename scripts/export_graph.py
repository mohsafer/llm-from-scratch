import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.tb_utils import TensorBoardLogger


def main():
    ap = argparse.ArgumentParser(description="Write the MiniMind model graph to a TensorBoard log dir")
    ap.add_argument('--hidden_size', type=int, default=768, help="hidden size")
    ap.add_argument('--num_hidden_layers', type=int, default=8, help="number of layers")
    ap.add_argument('--use_moe', type=int, default=0, choices=[0, 1], help="0=dense, 1=MoE")
    ap.add_argument('--seq_len', type=int, default=32, help="dummy sequence length for tracing")
    ap.add_argument('--log_dir', type=str, default='runs',
                    help="TensorBoard log dir (same one you pass to `tensorboard --logdir`)")
    ap.add_argument('--project', type=str, default='MiniMind-Graph',
                    help="run name / sub-folder under log_dir")
    args = ap.parse_args()

    config = MiniMindConfig(hidden_size=args.hidden_size,
                            num_hidden_layers=args.num_hidden_layers,
                            use_moe=bool(args.use_moe),
                            flash_attn=False)
    model = MiniMindForCausalLM(config)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    dummy = torch.randint(0, config.vocab_size, (1, args.seq_len), device=device)

    writer = TensorBoardLogger(log_dir=args.log_dir, project=args.project)
    writer.add_graph(model, dummy)
    writer.finish()

    print(f'Graph written -> {args.log_dir}/{args.project}/')
    print(f'View with:')
    print(f'  tensorboard --logdir {args.log_dir}')
    print('  -> open the "Graph" tab in the TensorBoard UI')


if __name__ == '__main__':
    main()
