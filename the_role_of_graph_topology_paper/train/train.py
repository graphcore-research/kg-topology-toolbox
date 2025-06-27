# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import json
import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import poptorch
import torch
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

from besskge import bess
import besskge.batch_sampler as bess_batch_sampler
import besskge.dataset as bess_dataset
import besskge.loss as bess_loss
import besskge.metric as bess_metric
import besskge.negative_sampler as bess_neg_sampler
import besskge.pipeline as bess_pipeline
import besskge.scoring as bess_scoring
import besskge.sharding as bess_sharding
import params
import utils


def train(model, dataloader, steps=None):
    """Trains the model for one epoch"""
    loss = 0
    epoch_triples = 0
    for n_batch, batch in enumerate(dataloader):
        triple_mask = batch.pop("triple_mask")
        res = model(**{k: v.flatten(end_dim=1) for k, v in batch.items()})
        epoch_triples += triple_mask.numel()
        if torch.any(torch.isnan(res["loss"])):
            # Preemptively kill run if nans in batch loss
            sys.exit(f"NaNs in loss for batch {n_batch}")
        loss += float(torch.sum(res["loss"])) / triple_mask[-1].numel()
        if steps and n_batch >= steps:
            break
    return {"loss": loss / (n_batch + 1), "epoch_triples": epoch_triples}


def test(
    score_fn, triples, evaluation, args, pos_triples=None, filter=None, device="ipu"
):
    """Performs inference on IPU or CPU"""
    if device == "ipu":
        if not isinstance(triples, bess_sharding.PartitionedTripleSet):
            raise TypeError(
                "`triples` has to be a `PartitionedTripleSet` if inference is performed on IPU"
            )
        candidate_sampler = bess_neg_sampler.PlaceholderNegativeSampler(
            corruption_scheme="t", seed=args.seed
        )
        batch_sampler = bess_batch_sampler.RigidShardedBatchSampler(
            partitioned_triple_set=triples,
            negative_sampler=candidate_sampler,
            shard_bs=args.inference_batch_size,
            batches_per_step=args.device_iter_inf,
            seed=args.seed,
            return_triple_idx=True,
        )

        if args.filter_test or args.return_all_scores:
            # the less efficient `AllScoresBESS` module has to be used in this case

            pipeline = bess_pipeline.AllScoresPipeline(
                batch_sampler,
                corruption_scheme="t",
                score_fn=score_fn,
                evaluation=evaluation,
                filter_triples=pos_triples,
                return_scores=args.return_all_scores,
                return_topk=args.return_topk,
                k=args.validation_topk,
                window_size=args.inference_window_size,
            )
            results = pipeline()
            del pipeline.dl

            metrics = results["metrics_avg"]
            predictions = dict(ranks=results["ranks"], triple_ids=results["triple_idx"])
            if args.return_all_scores:
                predictions.update(scores=results["scores"])
            if args.return_topk:
                predictions.update(predicted_ids=results["topk_global_id"])
        else:
            # `TopKQueryBessKGE` can be used for maximal efficiency

            options_inference = poptorch.Options()
            options_inference._Popart.set("saveInitializersToFile", "weights_inf.onnx")
            options_inference.replication_factor = score_fn.sharding.n_shard
            options_inference.deviceIterations(args.device_iter_inf)
            options_inference.outputMode(poptorch.OutputMode.All)

            model = bess.TopKQueryBessKGE(
                k=args.validation_topk,
                candidate_sampler=candidate_sampler,
                score_fn=score_fn,
                evaluation=evaluation,
                window_size=args.inference_window_size,
            )
            poptorch_model = poptorch.inferenceModel(model, options=options_inference)
            poptorch_model.entity_embedding.replicaGrouping(
                poptorch.CommGroupType.NoGrouping,
                0,
                poptorch.VariableRetrievalMode.OnePerGroup,
            )
            dataloader = batch_sampler.get_dataloader(
                options=options_inference,
                shuffle=False,
                num_workers=3,
            )

            n_val_queries = 0
            batch_metrics = []
            ranks = []
            predicted_ids = []
            triple_ids = []
            for n_batch, batch in enumerate(dataloader):
                triple_mask = batch["triple_mask"]
                n_val_queries += triple_mask.sum()
                triple_idx = batch.pop("triple_idx", None)
                res = poptorch_model(
                    **{k: v.flatten(end_dim=1) for k, v in batch.items()}
                )
                batch_metrics.append(
                    {
                        k: v.sum()
                        for k, v in zip(
                            evaluation.metrics.keys(),
                            res["metrics"].T,
                        )
                    }
                )
                ranks.append(res["ranks"][triple_mask.flatten()])
                predicted_ids.append(res["topk_global_id"][triple_mask.flatten()])
                if triple_idx is not None:
                    triple_ids.append(triple_idx[triple_mask])
            del dataloader

            metrics = {
                metric: sum([metrics[metric] for metrics in batch_metrics])
                / n_val_queries
                for metric in evaluation.metrics.keys()
            }
            predictions = dict(
                ranks=torch.concatenate(ranks, dim=0),
                predicted_ids=torch.concatenate(predicted_ids, dim=0),
                triple_ids=(
                    torch.concatenate(triple_ids, dim=0)
                    if len(triple_ids) > 0
                    else None
                ),
            )

    else:  # device="cpu"
        # Unshard entity embedding table
        ent_table = score_fn.entity_embedding.detach()[
            score_fn.sharding.entity_to_shard, score_fn.sharding.entity_to_idx
        ].to(torch.float32)

        rel_dtype = score_fn.relation_embedding.dtype
        if rel_dtype == torch.float16:
            score_fn.float()

        if filter is not None:
            if pos_triples is not None:
                raise ValueError(
                    "Only one of `pos_triples` and `filter` should be passed"
                )
            filter = filter.split(args.inference_batch_size or triples.shape[0], dim=0)

        predictions = []
        ranks = []
        batches = triples.split(args.inference_batch_size or triples.shape[0], dim=0)
        for batch_id, triples_batch in tqdm(enumerate(batches), total=len(batches)):
            filter_batch = None
            if filter is not None:
                filter_batch = filter[batch_id]
            if pos_triples is not None:
                filter_batch = get_triple_filter(pos_triples, triples_batch)

            # Score query (h,r,?) against all entities in the knowledge graph and select top-k scores
            scores = score_fn.score_tails(
                ent_table[triples_batch[:, 0]],
                triples_batch[:, 1],
                ent_table.unsqueeze(0),
            )
            if filter_batch is not None:
                # set scores of filtered triples to -inf and re-insert scores of query triples
                true_scores = scores[torch.arange(scores.shape[0]), triples_batch[:, 2]]
                scores[filter_batch[:, 0], filter_batch[:, 1]] = -torch.inf
                scores[torch.arange(scores.shape[0]), triples_batch[:, 2]] = true_scores
            top_k = torch.topk(
                scores,
                dim=-1,
                k=scores.shape[1] if args.return_all_scores else args.validation_topk,
            )
            all_preds = top_k.indices.squeeze()
            rks = evaluation.ranks_from_indices(triples_batch[:, 2], all_preds)
            predictions.append(all_preds[:, : args.validation_topk])
            ranks.append(rks)

        # Use evaluation.ranks_from_indices to rank the ground truth, if present, among the predictions
        predictions = torch.concatenate(predictions)
        ranks = torch.concatenate(ranks)
        metrics = {
            k: float(v / triples.shape[0])
            for k, v in evaluation.dict_metrics_from_ranks(ranks).items()
        }
        predictions = dict(
            ranks=ranks,
            predictions=predictions,
        )
        if rel_dtype == torch.float16:
            score_fn.half()
    return metrics, predictions


def get_triple_filter(pos_triples, test_triples):
    relations = test_triples[:, 1:2]
    relation_filter = (pos_triples[:, 1:2]).view(1, -1) == relations

    entities = test_triples[:, 0:1]
    entity_filter_test = (pos_triples[:, 0:1]).view(1, -1) == entities

    filters = (entity_filter_test & relation_filter).nonzero(as_tuple=False)
    filters[:, 1] = pos_triples[:, 2:3].view(1, -1)[:, filters[:, 1]]

    return filters


def main(args):
    if args.wandb:
        assert wandb is not None
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=args)
        args = wandb.config

    logger = utils.create_logger()

    if args.profile:
        args.steps = 2
        args.epochs = 1
        args.device_iter = 1
        args.accum_factor = min(2, args.accum_factor)
        args.validation_epochs = None
        profile_dir = args.profile_dir or os.path.join(
            ".", "profiles", f"{datetime.now().strftime( '%Y-%m-%d_%H-%M-%S')}"
        )
        params.to_yaml(
            args,
            os.path.join(
                profile_dir,
                f"config_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.yaml",
            ),
        )
        eng_opts = json.loads(os.environ.get("POPLAR_ENGINE_OPTIONS", "{}"))
        eng_opts.setdefault("autoReport.all", "true")
        eng_opts.setdefault("debug.allowOutOfMemory", "true")
        eng_opts.setdefault("autoReport.directory", profile_dir)
        os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(eng_opts)
        logger.info(f"Writing profile to {profile_dir}")

    entity_dict = utils.load_or_none(os.path.join(args.data, "entity_dict.pkl"))
    relation_dict = utils.load_or_none(os.path.join(args.data, "relation_dict.pkl"))

    if args.manual_data_split is not None:
        if args.test_relations:
            raise ValueError(
                "`test_relations` is not compatible with `manual_data_split`"
            )
        with open(
            os.path.join(os.path.join(args.data, args.manual_data_split)), "rb"
        ) as f_in:
            triples = pickle.load(f_in)
        total_triple_count = sum([len(v) for v in triples.values()])
        valid_split = len(triples.get("valid", [])) / total_triple_count
        test_split = len(triples.get("test", [])) / total_triple_count
        dataset = bess_dataset.KGDataset(
            n_entity=max([v[:, [0, 2]].max() for v in triples.values()]) + 1,
            n_relation_type=max([v[:, 1].max() for v in triples.values()]) + 1,
            entity_dict=entity_dict,
            relation_dict=relation_dict,
            triples=triples,
            original_triple_ids={k: np.arange(v.shape[0]) for k, v in triples.items()},
        )
    else:
        triples = torch.load(os.path.join(args.data, "triples.pt"))
        valid_split, test_split = args.validation_split, args.test_split
        if args.test_relations:
            # Sample valid/test triples only with specific relation types
            trip = {
                "train": np.empty((0, 3), dtype=np.int64),
                "valid": np.empty((0, 3), dtype=np.int64),
                "test": np.empty((0, 3), dtype=np.int64),
            }
            rel_triple_id = []
            for label in args.test_relations:
                if relation_dict is None or label not in relation_dict:
                    raise ValueError(f"Label {label} not in relation dictionary")
                rel_id = np.argwhere(relation_dict == label).item()
                filtered_triple_id = np.argwhere(triples[:, 1] == rel_id).flatten()
                rel_triple_id.append(filtered_triple_id)
                num_triple_train = int(
                    len(filtered_triple_id) * (1 - valid_split - test_split)
                )
                num_triple_valid = int(len(filtered_triple_id) * valid_split)
                np.random.default_rng(seed=args.seed).shuffle(filtered_triple_id)
                label_triples_train, label_triples_valid, label_triples_test = np.split(
                    triples[filtered_triple_id],
                    (num_triple_train, num_triple_train + num_triple_valid),
                    axis=0,
                )
                trip["train"] = np.concatenate([trip["train"], label_triples_train])
                trip["valid"] = np.concatenate([trip["valid"], label_triples_valid])
                trip["test"] = np.concatenate([trip["test"], label_triples_test])
            # Add triples of all other types to train set
            trip["train"] = np.concatenate(
                [
                    trip["train"],
                    triples[
                        np.setdiff1d(
                            np.arange(triples.shape[0]), np.concatenate(rel_triple_id)
                        )
                    ],
                ]
            )
            dataset = bess_dataset.KGDataset(
                n_entity=triples[:, [0, 2]].max() + 1,
                n_relation_type=triples[:, 1].max() + 1,
                entity_dict=entity_dict,
                relation_dict=relation_dict,
                triples=trip,
            )
        else:
            # Random split
            dataset = bess_dataset.KGDataset.from_triples(
                triples,
                entity_dict=entity_dict,
                relation_dict=relation_dict,
                split=(1 - (valid_split + test_split), valid_split, test_split),
                seed=args.seed,
            )
    data_partitions = ["train"]
    if valid_split > 0:
        data_partitions += ["valid"]
    if test_split > 0:
        data_partitions += ["test"]

    torch_triples = {
        part: torch.from_numpy(dataset.triples[part]) for part in data_partitions
    }

    logger.info("Dataset loaded\n")
    logger.info(f"Number of entities: {dataset.n_entity:,}\n")
    logger.info(f"Number of relation types: {dataset.n_relation_type}\n")
    logger.info(
        f"Number of triples:\n"
        f"   training: {dataset.triples.get('train', torch.empty(0)).shape[0]:,}\n"
        f"   validation {dataset.triples.get('valid', torch.empty(0)).shape[0]:,}\n"
        f"   test {dataset.triples.get('test', torch.empty(0)).shape[0]:,}\n"
    )

    sharding = bess_sharding.Sharding.create(
        n_entity=dataset.n_entity,
        n_shard=args.shards,
        seed=args.seed,
        type_offsets=None,
    )

    sharded_triples = {
        part: bess_sharding.PartitionedTripleSet.create_from_dataset(
            dataset=dataset,
            part=part,
            sharding=sharding,
            partition_mode="ht_shardpair" if part == "train" else "h_shard",
            add_inverse_triples=args.add_inverse_triples if part == "train" else False,
        )
        for part in data_partitions
    }

    # Sample train and valid queries for interleaved validation
    val_triple_subset = torch_triples["valid"][
        np.random.default_rng(seed=args.seed).choice(
            torch_triples["valid"].shape[0], args.validation_triples
        )
    ]
    train_triple_subset = torch_triples["train"][
        np.random.default_rng(seed=args.seed).choice(
            torch_triples["train"].shape[0], args.validation_triples
        )
    ]

    neg_sampler_train = bess_neg_sampler.RandomShardedNegativeSampler(
        n_negative=args.neg,
        sharding=sharding,
        seed=args.seed,
        corruption_scheme="t" if args.scoring_function == "ConvE" else "ht",
        local_sampling=False,
        flat_negative_format=False,
    )

    batch_sampler_train = bess_batch_sampler.RigidShardedBatchSampler(
        partitioned_triple_set=sharded_triples["train"],
        negative_sampler=neg_sampler_train,
        shard_bs=args.batch_size,
        batches_per_step=args.device_iter * args.accum_factor,
        seed=args.seed,
        return_triple_idx=False,
    )

    options_train = poptorch.Options()
    options_train._Popart.set("saveInitializersToFile", "weights.onnx")
    options_train.replication_factor = sharding.n_shard
    options_train.deviceIterations(args.device_iter)
    options_train.Training.gradientAccumulation(args.accum_factor)
    options_train._popart.setPatterns(dict(RemoveAllReducePattern=True))
    options_train.Precision.enableStochasticRounding(args.half)

    dataloader_train = batch_sampler_train.get_dataloader(
        options=options_train,
        shuffle=True,
        num_workers=3,
        persistent_workers=False,
    )

    if args.loss_function.lower() == "logsigmoid":
        loss_fn = bess_loss.LogSigmoidLoss(
            margin=args.margin,
            negative_adversarial_sampling=args.neg_adversarial_sampling,
            loss_scale=args.loss_scale,
        )
    elif args.loss_function.lower() == "margin":
        loss_fn = bess_loss.MarginRankingLoss(
            margin=args.margin,
            negative_adversarial_sampling=args.neg_adversarial_sampling,
            loss_scale=args.loss_scale,
        )
    elif args.loss_function.lower() == "sampled_softmax":
        loss_fn = bess_loss.SampledSoftmaxCrossEntropyLoss(
            n_entity=dataset.n_entity,
            loss_scale=args.loss_scale,
        )
    else:
        raise ValueError(f"Loss function {args.loss_function} not supported")

    scoring_args = dict(
        negative_sample_sharing=True,
        sharding=sharding,
        n_relation_type=dataset.n_relation_type,
        embedding_size=args.dim,
        inverse_relations=args.add_inverse_triples,
    )
    if (
        bess_scoring.DistanceBasedScoreFunction
        in getattr(bess_scoring, args.scoring_function).__bases__
    ):
        scoring_args.update(scoring_norm=args.scoring_norm)
    if args.scoring_function in ["PairRE", "TripleRE", "InterHT"]:
        scoring_args.update(normalize_entities=args.normalize_entities)
    if args.scoring_function in ["ConvE"]:
        scoring_args.update(embedding_height=args.embedding_height)
        if args.dim % args.embedding_height == 0:
            scoring_args.update(embedding_width=args.dim // args.embedding_height)
        else:
            ew = args.dim // args.embedding_height
            scoring_args.update(embedding_width=ew)
            scoring_args.update(embedding_size=args.embedding_height * ew)
        scoring_args.update(input_dropout=args.input_dropout)
        scoring_args.update(batch_normalization=args.batch_normalization)
    score_fn = getattr(bess_scoring, args.scoring_function)(**scoring_args)

    model = bess.EmbeddingMovingBessKGE(
        negative_sampler=neg_sampler_train,
        score_fn=score_fn,
        loss_fn=loss_fn,
        augment_negative=True,
    )

    logger.info(f"Model created. # model parameters: {model.n_embedding_parameters:,}")

    if args.half:
        model.half()

    if args.optimiser.lower() == "sgd":
        opt = poptorch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            velocity_accum_type=torch.float16 if args.half else torch.float32,
        )
    elif args.optimiser.lower() == "adam":
        opt = poptorch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.momentum, args.beta2),
            first_order_momentum_accum_type=(
                torch.float16 if args.half else torch.float32
            ),
            second_order_momentum_accum_type=torch.float32,
        )
    else:
        raise ValueError(f"Optimiser {args.optimiser} not supported")

    scheduler = utils.get_lr_scheduler(opt, args.lr_scheduler, num_epochs=args.epochs)

    poptorch_model = poptorch.trainingModel(model, options=options_train, optimizer=opt)

    poptorch_model.entity_embedding.replicaGrouping(
        poptorch.CommGroupType.NoGrouping,
        0,
        poptorch.VariableRetrievalMode.OnePerGroup,
    )

    evaluation = bess_metric.Evaluation(
        ["mrr", "hits@1", "hits@3", "hits@10"],
        worst_rank_infty=True,
        reduction="sum",
        return_ranks=True,
    )

    filter_triples = None
    filter_interleaved_train = None
    filter_interleaved_valid = None
    if args.filter_test:
        filter_triples = torch.concatenate(
            [torch_triples["train"], torch_triples["valid"]]
        )
        if args.test_on == "test":
            filter_triples = torch.concatenate([filter_triples, torch_triples["test"]])
        if args.validation_epochs:
            filter_interleaved_train = get_triple_filter(
                torch_triples["train"], train_triple_subset
            )
            filter_interleaved_valid = get_triple_filter(
                filter_triples, val_triple_subset
            )

    # Start training
    if args.wandb:
        wandb.log({"param_count": model.n_embedding_parameters})
    cumulative_triples = 0
    for epoch in range(0, args.epochs, 2 if args.add_inverse_triples else 1):
        t_start = time.time()
        results = train(poptorch_model, dataloader_train, steps=args.steps)
        results.update(loss=results["loss"] / args.loss_scale)
        results.update(duration_train=time.time() - t_start)
        cumulative_triples += results.pop("epoch_triples")
        results["num_triples"] = cumulative_triples
        results.update(learning_rate=opt.param_groups[0]["lr"])
        if args.validation_epochs and (
            epoch % args.validation_epochs == 0 or epoch == args.epochs - 1
        ):
            train_sample_eval, _ = test(
                score_fn,
                train_triple_subset,
                evaluation,
                args,
                filter=filter_interleaved_train,
                device="cpu",
            )
            results.update({"train_" + k: v for k, v in train_sample_eval.items()})

            valid_sample_eval, _ = test(
                score_fn,
                val_triple_subset,
                evaluation,
                args,
                filter=filter_interleaved_valid,
                device="cpu",
            )
            results.update({"valid_" + k: v for k, v in valid_sample_eval.items()})

        logger.info(f"Epoch {epoch}")
        for k, v in results.items():
            logger.info(
                "   {0}: {1:.{2}f}".format(k, v, 5 if isinstance(v, float) else 0)
            )
        if args.wandb:
            wandb.log(results, step=epoch)

        if scheduler:
            # Update lr scheduler after logging results
            scheduler.step()
            poptorch_model.setOptimizer(opt)

    poptorch_model.detachFromDevice()
    del dataloader_train

    # Final validation
    if args.final_validation:
        logger.info(f"Performing {args.test_on} on {args.inference_device}")

        t_start = time.time()
        test_metrics, test_predictions = test(
            score_fn,
            (
                sharded_triples[args.test_on]
                if args.inference_device == "ipu"
                else torch_triples[args.test_on]
            ),
            evaluation,
            args,
            pos_triples=filter_triples if filter_triples is None else [filter_triples],
            device=args.inference_device,
        )
        validation_dur = time.time() - t_start
        test_predictions.update(
            dict(
                triples=dataset.triples[args.test_on][
                    sharded_triples[args.test_on].triple_sort_idx[
                        test_predictions["triple_ids"]
                    ]
                ],
                filter_triples=filter_triples,
            )
        )
        test_predictions.update(
            dict(
                original_triple_ids=dataset.original_triple_ids[args.test_on][
                    sharded_triples[args.test_on].triple_sort_idx[
                        test_predictions["triple_ids"]
                    ]
                ],
                filter_triples=filter_triples,
            )
        )

        test_metrics = {f"{args.test_on}_{k}": v for k, v in test_metrics.items()}
        test_metrics.update(duration_inference=validation_dur)

        for k, v in test_metrics.items():
            logger.info(f"   {k}: {v:.5f}")
        if args.wandb:
            wandb.log(test_metrics, step=args.epochs - 1)
        if args.store_predictions:
            if args.wandb:
                np.savez(
                    os.path.join(wandb.run.dir, "predictions.npz"), **test_predictions
                )
            if args.logging_dir:
                os.makedirs(args.logging_dir, exist_ok=True)
                np.savez(
                    os.path.join(args.logging_dir, "predictions.npz"),
                    **test_predictions,
                )


if __name__ == "__main__":
    args = params.parse_args()
    main(args)
