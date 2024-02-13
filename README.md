# Bidirectional Biomedical Image Translation Between MRI and PET

## BPGAN

BPGAN model has been updated.

```
cd bpgan
```

Train the model by the following command.

```
python3 train.py --epochs 100 --niter 50 --use_dropout True -- input mr --device 0
```
