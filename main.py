import torch
import argparse
from train import train
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np
from dataset import GrooveDataset
from model import MusicVAE
import random
import argparse
from data_prepare import get_file_path, data_processing, transform_to_midi

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument('--enc_input_size', type=int, default=512)
    parser.add_argument('--enc_latent_dim', type=int, default=512)
    parser.add_argument('--enc_hidden_size', type=int, default=1024)
    parser.add_argument('--con_hidden_size', type=int, default=512)
    parser.add_argument('--dec_hidden_size', type=int, default=1024)
    parser.add_argument('--dec_output_size', type=int, default=512)
    parser.add_argument('--bar_units', type=int, default=16)
    parser.add_argument('--model_path', type=str, default='trained_model/check_point.pth')
    args = parser.parse_args()
    # print(torch.cuda.is_available())
    # prepare data

    # model load
    model = MusicVAE(args)
    model.to(args.device)

    # train mode
    if args.mode == "train":
        train_file, valid_file, test_file = get_file_path('groove')
        train_data, valid_data = data_processing(train_file), data_processing(valid_file)
        train_dataset = GrooveDataset(train_data)
        valid_dataset = GrooveDataset(valid_data)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        valid_dataLoader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(args, train_dataLoader, valid_dataLoader, model, optimizer)
    else:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model'])
        sample = model.generate()
        midi_sample = transform_to_midi(sample, fs=16000)
        midi_sample.write('sample.mid')
        