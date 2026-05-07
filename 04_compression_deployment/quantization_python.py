import pytorch_lightning as pl
from pathlib import Path
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
from neural_compressor.quantization import fit
from base_flower import FlowerLightModule, FlowerDataModule

ckpt_path = "./logs/checkpoints/checkpoint_base_epoch=29_val_acc=0.9756.ckpt"

datamodule = FlowerDataModule()
datamodule.prepare_data()
datamodule.setup(stage="fit")

pl_model = FlowerLightModule.load_from_checkpoint(ckpt_path)
pl_model.eval()

trainer = pl.Trainer(
    accelerator="cpu",
    devices=1,
    logger=False,
    enable_checkpointing=False,
)

def eval_func(q_model):
    pl_model.model = q_model
    result = trainer.validate(model=pl_model, dataloaders=datamodule.val_dataloader(), verbose=False)
    return float(result[0]["accuracy"])

accuracy_criterion = AccuracyCriterion(tolerable_loss=0.05)
tuning_criterion = TuningCriterion(max_trials=200)

conf = PostTrainingQuantConfig(
    approach="static",
    backend="default",
    tuning_criterion=tuning_criterion,
    accuracy_criterion=accuracy_criterion,
)

q_model = fit(
    model=pl_model.model,
    conf=conf,
    calib_dataloader=datamodule.train_dataloader(),
    eval_func=eval_func,
)

Path("./saved_model").mkdir(parents=True, exist_ok=True)
q_model.save("./saved_model")
print("PTQ done.")