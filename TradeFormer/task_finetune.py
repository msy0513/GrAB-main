import argparse
from loguru import logger
from pathlib import Path
from torch_geometric.loader import DataLoader
from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model
from TradeFormer.dataset.mini_dataset import create_miniDataset
from TradeFormer.model.grit_model import GritTransformer
from task_trainer import TaskTrainer
from torchinfo import summary


def get_dataloaders(config):
    train_set, val_set, test_set = create_miniDataset(config)

    train_dataloader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, test_dataloader


def main(args, config):
    set_random_seed(args.seed)
    log_GPU_info()
    nout = 2

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(config.data)
    model = GritTransformer(
        nout, **config.model.grit, ksteps=config.data.pos_enc_rrwp.ksteps
    )

    summary(model)

    if args.ckpt is not None:
        model = load_model(model, args.ckpt)
        model = model.to("cuda")
    trainer = TaskTrainer(model, args.output_dir, **config.train)

    if args.ckpt is not None and args.eval is not None:
        logger.info(f"Start evaluating")
        for epoch in range(1, 2):
            set_random_seed(epoch)
            train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
                config.data
            )
            trainer.eval_epoch(epoch - 1, model, test_dataloader)
        logger.info(f"Evaluating finished")
        exit()

    logger.info(f"Start training")
    trainer.fit(train_dataloader, val_dataloader, test_dataloader)
    logger.info(f"Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task_finetune.yaml")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--ckpt", default=None, type=str)
    parser.add_argument("--eval", default=True, type=bool)
    parser.add_argument("--ckpt_cl", default=None, type=str)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = parse_config(args.config)
    main(args, config)
