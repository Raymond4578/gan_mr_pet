# Bidirectional Biomedical Image Translation Between MRI and PET

## Adversrial U-Net

Adversrial U-Net has been updated.


```
cd adv_unet
```

If you want to train the model with the [Pix2pix code](https://github.com/phillipi/pix2pix), run the following command.

```
python3 train.py --epochs 100 --niter 50 --use_dropout True -- input mr --device 0
```

If you want to train the model with my refined Adversarial U-Net, run the following command.

```
python3 train_old.py --epochs 100 --niter 50 -- input mr --device 0
```



## RevGAN

RevGAN has been updated. Loss 0 for Pix2pix loss, Loss 1 for CycleGAN loss.

```
cd revgan
```

If you want to train the model with the [original code](https://github.com/tychovdo/RevGAN), run the following command.

```
python3 train.py --epochs 100 -- loss 0 --device 0
```

If you want to train the model with my refined RevGAN, run the following command.

```
python3 train_old.py --epochs 100 -- loss 0 --device 0
```


## BPGAN

BPGAN model has been updated.

```
cd bpgan
```

Train the model by the following command.

```
python3 train.py --epochs 100 --niter 50 --use_dropout True -- input mr --device 0
```
