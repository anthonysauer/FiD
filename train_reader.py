# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import wandb
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_f1, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, _, context_ids, context_mask) = batch

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.accumulation_steps == 0:
                log = f"{step} / {opt.total_steps} |"
                log += f"train: {curr_loss / opt.accumulation_steps:.3f} |"
                log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                logger.info(log)
                wandb.log({"loss": curr_loss/opt.accumulation_steps, "global_step": step})

                if tb_logger is not None:
                    tb_logger.add_scalar("Training", curr_loss / (opt.accumulation_steps), step)

                curr_loss = 0.

            if step % opt.eval_freq == 0:
                dev_f1, dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_f1 > best_dev_f1:
                        best_dev_f1 = dev_f1
                        src.util.save(model, optimizer, scheduler, step, best_dev_f1,
                                      opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"evaluation: {dev_f1:.2f}F1 |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)

                    wandb.log({"f1": dev_f1, "ems": 100*dev_em, "global_step": step})

                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_f1, step)

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_f1,
                              opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    total = 0
    total_multi = 0
    f1_scores = []
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=False)
                for special_token in tokenizer.all_special_tokens:
                    if special_token != '<extra_id_0>':
                        ans = ans.replace(special_token, '')
                if '<extra_id_0>' in ans:
                    total_multi += 1
                ans_list = ans.split('<extra_id_0>')
                ans_list_stripped = [s.strip() for s in ans_list]
                if total < 10:
                    logger.info('Sample answers: ' + '; '.join(ans_list_stripped))

                annotations = dataset.get_example(idx[k])['answers']

                max_f1 = 0
                max_ems = 0
                for annotation in annotations:
                    # iterate each annotation and take the maximum metrics
                    if annotation['type'] == 'singleAnswer':
                        f1 = src.evaluation.get_f1([annotation['answer']], ans_list_stripped)
                        max_f1 = max(max_f1, f1)

                        ems = src.evaluation.get_exact_match(annotation['answer'], ans_list_stripped)
                        max_ems = max(max_ems, ems)
                    elif annotation['type'] == 'multipleQAs':
                        max_f1 = max(max_f1,
                                     src.evaluation.get_f1([answer['answer'] for answer in annotation['qaPairs']],
                                                           ans_list_stripped))

                        ems = src.evaluation.get_exact_match([answer['answer'] for answer in annotation['qaPairs']],
                                                             ans_list_stripped)
                        max_ems = max(max_ems, ems)

                total += 1
                exactmatch.append(max_ems)
                f1_scores.append(max_f1)

    logger.info('Total number of multi answers: ' + str(total_multi))
    f1_scores, total = src.util.weighted_average(np.mean(f1_scores), total, opt)
    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return f1_scores, exactmatch

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_f1 = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_f1 = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_f1 = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    wandb.init(
        # set the wandb project where this run will be logged
        project="multi-answer-ir",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "dataset": "ambignq",
            "total_steps": 6000,
            "warmup_steps": 500
        }
    )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_f1,
        checkpoint_path
    )

    wandb.finish()
