# Bidirectional Biomedical Image Translation Between MRI and PET

## RevGAN

RevGAN has been updated.

```
cd bpgan
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
