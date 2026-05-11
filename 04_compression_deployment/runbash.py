# run base model and export pth
python3.10 base_flower.py fit -c base.yaml
python3.10 save_base_model.py --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt

# pruning base model

python3.10 pruning_l1structured.py --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt --sparsity 0.3 --finetune_epochs 10
python3.10 pruning_l1unstructured.py --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt --sparsity 0.3 --finetune_epochs 10
python3.10 pruning_physical_channel.py --ckpt logs/checkpoints/checkpoint_base_epoch=15_val_acc=0.9788.ckpt --pruning_ratio 0.15 --global_pruning --finetune_epochs 10


# benchmark pruning models
python3.10 benchmark.py --model_path logs/base_models/base_epoch=15_val_acc=0.9788.pth --run_name base
python3.10 benchmark.py --model_path logs/pruned/efficientnet_b0_pruned_l1_unst_30.pth --run_name l1_unstructured
python3.10 benchmark.py --model_path logs/pruned/efficientnet_b0_pruned_l1_st_30.pth --run_name l1_structured
python3.10 benchmark.py --model_path logs/pruned/efficientnet_b0_pruned_physical_15.pth --run_name l1_structured_remove

# quantize model
python3.10 export_and_quant_ort.py 

# benchmark quantized model
python3.10 benchmark_onnx.py --model_path flower_efficientnet_basemodel.onnx --run_name base
python3.10 benchmark_onnx.py --model_path flower_efficientnet_quantization.onnx --run_name quantized

#mlflow
python3.10 -m mlflow server --backend-store-uri sqlite:///mlflow.db