# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import argparse
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Embedding size. For complex embeddings this is the size "
        "of the complex vector",
    )
    parser.add_argument(
        "--scoring-function",
        type=str,
        default="TransE",
        help="Scoring function to be used with model",
    )
    parser.add_argument("--scoring-norm", type=int, default=1, choices=[1, 2])
    parser.add_argument("--normalize-entities", type=bool, default=True)
    parser.add_argument("--embedding-height", type=int, default=1)
    parser.add_argument("--input-dropout", type=float, default=0.2)
    parser.add_argument("--batch-normalization", type=bool, default=True)
    parser.add_argument("--add-inverse-triples", action="store_true", default=False)
    parser.add_argument(
        "--loss-function",
        type=str,
        default="logsigmoid",
        choices=("logsigmoid", "margin", "sampled_softmax"),
        help="Loss function to be used with model",
    )
    parser.add_argument("--margin", type=float, default=12.0)
    parser.add_argument(
        "--neg", type=int, default=1, help="Number of negative samples per shard-pair"
    )
    parser.add_argument("--neg-adversarial-sampling", action="store_true")
    parser.add_argument(
        "-o", "--optimiser", type=str, default="adam", choices=("sgd", "adam")
    )

    # Execution parameters
    parser.add_argument(
        "-e", "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps per epoch"
    )
    parser.add_argument("--final-validation", action="store_true")
    parser.add_argument("--validation-epochs", type=int, default=5)
    parser.add_argument(
        "--validation-topk",
        type=int,
        default=100,
        help="number of predictions to keep",
    )
    parser.add_argument(
        "--validation-triples",
        type=int,
        default=1000,
        help="number of triples for interleaved validation",
    )
    parser.add_argument(
        "--inference-device",
        type=str,
        default="ipu",
        choices=("ipu", "cpu"),
        help="device type to be used for inference",
    )
    parser.add_argument(
        "--filter-test",
        action="store_true",
        help="filter out false negatives for final validation/test",
    )
    parser.add_argument(
        "--return-all-scores",
        action="store_true",
        help="return all scores during inference",
    )
    parser.add_argument(
        "--return-topk",
        action="store_true",
        help="return TopK predictions during inference",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr-scheduler", type=str, default=None, help="Learning rate scheduler"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum or Adam beta_1"
    )
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta_2")
    parser.add_argument("--wd", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument(
        "-s",
        "--shards",
        type=int,
        default=4,
        help="Number of shards for the entity embedding table" "(=number of IPUs)",
    )
    parser.add_argument(
        "--loss-scale", type=float, default=1.0, help="Loss scaling factor"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=240, help="Batch size per device"
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=256,
        help="Batch size for validation/test",
    )
    parser.add_argument(
        "--inference-window-size",
        type=int,
        default=500,
        help="Number of tails to simultaneously compare to a batch during validation/test",
    )
    parser.add_argument(
        "--accum-factor", type=int, default=6, help="Gradient accumulation factor"
    )
    parser.add_argument("--device-iter", type=int, default=8, help="Device iterations")
    parser.add_argument(
        "--device-iter-inf", type=int, default=8, help="Device iterations for inference"
    )
    parser.add_argument("--half", action="store_true", help="Using half precision")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    parser.add_argument(
        "--wandb", action="store_true", help="Log results with weights and biases"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="research",
        help="Weights and Biases entity",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="kge-experimentation",
        help="Weights and Biases project",
    )
    parser.add_argument("--logging-dir", type=str, default=None)
    parser.add_argument(
        "--store-predictions", action="store_true", help="Store top-k predictions"
    )
    parser.add_argument("--profile", action="store_true", help="Profile training")
    parser.add_argument("--profile-dir", type=str, default=None)

    # Dataset parameters
    parser.add_argument(
        "--data",
        type=str,
        help="Data directory. Should contain 'triples.pt' and,  optionally, "
        "'entity_dict.pkl', 'relation_dict.pkl' and 'type_offset.pkl",
    )
    parser.add_argument(
        "--manual-data-split",
        type=str,
        help="Pickle file with a dictionary of triples in train/valid/test split",
        default=None,
    )
    parser.add_argument(
        "--test-relations",
        nargs="+",
        default=[],
        help="If specified, sample valid/test triples only from this list of relations",
    )
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument(
        "--test-on",
        type=str,
        choices=("valid", "test"),
        default="valid",
    )

    return parser.parse_args()


def to_yaml(args: argparse.Namespace, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(vars(args)), file
