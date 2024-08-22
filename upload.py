import argparse
import wandb
from hyperparams import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='shakespear-training',
                    description='pretrain shakespeare transformer')
    parser.add_argument('-n', '--model-name', type=str)
    parser.add_argument('-d', '--dataset', default='default', type=str)
    parser.add_argument('-p', '--parallel', default="true", type=str)      # option that takes a value
    parser.add_argument('-t', '--tokenizer', default='char', type=str)

    args = parser.parse_args()
    model_name = args.model_name
    tokenizer = args.tokenizer
    dataset = args.dataset
    parallel = args.parallel.lower()
    model_artifact_name = f'model-with-{tokenizer}-tokenizer-on-dataset-{dataset}'

    wandb.init(
        project="shakespear-transformer",
        name=f"Upload model: {model_name}"
    )

    print("Begin uploading...")
    art = wandb.Artifact(model_artifact_name, type='model')
    art.add_file(f'data/{model_name}')
    wandb.log_artifact(art)
    print(f"Uploaded model {model_name} to artifact {model_artifact_name}")
