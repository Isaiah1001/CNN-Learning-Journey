python3.10 save_base_model.py --ckpt logs/checkpoints/checkpoint_pruned_l1unstructured_30%_epoch=02_val_acc=0.9984.ckpt
python3.10 save_base_model.py --ckpt logs/checkpoints/checkpoint_pruned_l1structured_30%_epoch=09_val_acc=0.9292.ckpt
python3.10 save_base_model.py --ckpt logs/checkpoints/pruned_physical_30%_epoch=09_val_acc=0.8436.ckpt
python3.10 save_base_model.py --ckpt logs/checkpoints/pruned_physical_30%_epoch=09_val_acc=0.9218.ckpt


python3.10 benchmark.py --model_path logs/base_models/base_epoch=29_val_acc=0.9756.pth --run_name base
python3.10 benchmark.py --model_path logs/pruned/efficientnet_b0_pruned_l1_unst_30.pth --run_name l1_unstructured
python3.10 benchmark.py --model_path logs/pruned/efficientnet_b0_pruned_l1_st_30.pth --run_name l1_structured
python3.10 benchmark.py --model_path logs/pruned/efficientnet_b0_pruned_physical_30.pth --run_name l1_structured_remove
