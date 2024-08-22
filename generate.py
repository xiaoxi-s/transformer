import argparse
import torch
import wandb
import torch.nn as nn
import numpy as np

from models.transformer import Transformer
from hyperparams import *
from constants import *
from data.utils import get_artifacts_name
from initialize import initialize_storage


def generate_contents(
    model, vocab_to_ind, ind_to_vocab, tokenizer, device="cpu", max_num_of_tokens=1000
):
    """Generate contents from the model."""

    output = None
    token_indx = [0]
    with torch.no_grad():
        for i in range(max_num_of_tokens):
            input = torch.tensor(token_indx).unsqueeze(0).to(device)
            output = model(input, input)
            output = output[:, -1, :]
            output = torch.softmax(output, dim=-1)  # [1, vocab_size]
            output = torch.multinomial(output, num_samples=1)
            token_indx.append(output.item())
            if tokenizer == "word" and output.item() == vocab_to_ind["<stop>"]:
                break
            print(ind_to_vocab[output.item()], end="")


if __name__ == "__main__":
    np.random.seed(7777)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-name", "--model-name", type=str)
    argparser.add_argument("-m", "--max-token", type=int, default=1000)
    argparser.add_argument(
        "-p", "--parallel", default="true", type=str
    )  # option that takes a value
    argparser.add_argument("-t", "--tokenizer", default="char", type=str)
    argparser.add_argument("-d", "--dataset", default="default", type=str)
    argparser.add_argument("-l", "--location", default="local", type=str)

    parser = argparser.parse_args()
    model_name = parser.model_name
    max_token = parser.max_token
    tokenizer = parser.tokenizer
    dataset = parser.dataset
    location = parser.location
    parallel = parser.parallel.lower()

    model_artifact_name, dataset_artifact_name, vocab_artifact_name = get_artifacts_name(tokenizer, dataset)
    # init storage & load the vocab
    vocab_name = f"vocab-{tokenizer}-for-dataset-{dataset}.pth"
    wandb.init(project=wandb_project)
    storage = initialize_storage(
        location,
        wandb_project,
        model_artifact_name,
        dataset_artifact_name,
        vocab_artifact_name,
    )  # these parameters are constants for the project
    vocab_to_ind = storage.load_vocab(vocab_name)  # assume the vocab exists
    ind_to_vocab = {v: k for k, v in vocab_to_ind.items()}

    # load the model
    model = Transformer(
        len(vocab_to_ind),
        dropout=dropout,
        block_size=block_size,
        num_of_decoder_layers=num_of_decoder_layers,
        num_of_encoder_layers=num_of_encoder_layers,
        dmodel=dmodel,
    ).to(device)
    if parallel == "true" or parallel == "t":
        model = nn.DataParallel(model)
    model.load_state_dict(storage.load_model(model_name))
    model.eval()

    generate_contents(
        model,
        vocab_to_ind,
        ind_to_vocab=ind_to_vocab,
        tokenizer=tokenizer,
        device=device,
        max_num_of_tokens=max_token,
    )
