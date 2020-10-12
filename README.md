# Cornell Birdcall Identification Competition

6th place solution to the [Cornell Birdcall Identification Challenge](https://www.kaggle.com/c/birdsong-recognition) hosted on Kaggle.

## Context

> In this competition, you will identify a wide variety of bird vocalizations in soundscape recordings. Due to the complexity of the recordings, they contain weak labels. There might be anthropogenic sounds (e.g., airplane overflights) or other bird and non-bird (e.g., chipmunk) calls in the background, with a particular labeled bird species in the foreground. Bring your new ideas to build effective detectors and classifiers for analyzing complex soundscape recordings!

## Evaluation

> The hidden test_audio directory contains approximately 150 recordings in mp3 format, each roughly 10 minutes long. They will not all fit in a notebook's memory at the same time. The recordings were taken at three separate remote locations in North America. Sites 1 and 2 were labeled in 5 second increments and need matching predictions, but due to the time consuming nature of the labeling process the site 3 files are only labeled at the file level. Accordingly, site 3 has relatively few rows in the test set and needs lower time resolution predictions.

Scores were evaluated based on their row-wise micro averaged F1 score.

## Solution

My approach was

* 3 stages of training to gradually remove noise in labels
* SED style training and inference as I introduced in [an introductory notebook](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection).
* Ensemble of 11 EMA models trained with the whole dataset / whole [extended dataset](https://www.kaggle.com/c/birdsong-recognition/discussion/159970)

**Details are described in [kaggle discussion](https://www.kaggle.com/c/birdsong-recognition/discussion/183204).**

### First stage

* single [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) model (`Cnn14_DecisionLevelAtt`)
* BCE on `clipwise_output` and also on `max(framewise_output, axis=0)` (max pool on time axis) and sum them up with the weight 1.0 and 0.5 each.
* Adam + CosineAnnealingLR, 55 epochs of training
* train with randomly cropped 30s chunk
* validate on randomly cropped 30s chunk
* Augmentations on raw waveform.
  * `NoiseInjection` (max noise amplitude 0.04)
  * `PitchShift` (max pitch level 3)
  * `RandomVolume` (max db level 4)

### Find missing label

After 5fold training of fast stage, I conducted SED inference on the whole train dataset and get time interval labels. Although this label was still too noisy, it was useful enough to find missing labels in the training dataset.

### Second stage

* SED model with ResNeSt50 encoder, attention pooling head of PANNs (`torch.clamp` -> `torch.tanh`)
* BCE on `clipwise_output` and also on `max(framewise_output, axis=0)` (max pool on time axis) and sum them up with the weight 1.0 and 0.5 each.
* Adam + CosineAnnealingLR, 55 epochs of training
* Add additional `secondary_labels` found after stage1
* 3 channels input - (normal log-melspectrogram, PCEN, `librosa.power_to_db(melspec ** 1.5)`)
* train with randomly cropped 20s chunk
* validate on randomly cropped 30s chunk
* Augmentations on raw waveform.
  * `NoiseInjection` (max noise amplitude 0.04)
  * `PitchShift` (max pitch level 3)
  * `RandomVolume` (max db level 4)

### Third stage

* SED model with ResNeSt50 encoder / EfficientNet-B0 encoder, attention pooling head of PANNs (`torch.clamp` -> `torch.tanh`)
* BCE on `clipwise_output` and also on `max(framewise_output, axis=0)` (max pool on time axis) and sum them up with the weight 1.0 and 0.5 each. For EfficientNet-B0 encoder, I used FocalLoss in the same way as I did with BCE.
* Adam + CosineAnnealingLR, 55 epochs of training
* Add additional `secondary_labels` found after stage1
* Correct labels using the prediction of stage2 model.
* 3 channels input - (normal log-melspectrogram, PCEN, `librosa.power_to_db(melspec ** 1.5)`)
* train with randomly cropped 20s chunk
* validate on randomly cropped 30s chunk
* Augmentations on raw waveform.
  * `NoiseInjection` (max noise amplitude 0.04)
  * `PitchShift` (max pitch level 3)
  * `RandomVolume` (max db level 4)

## To reproduce the result

**NOTE: This is from my stupidity, but I've been embedding a bug regarding the seed fixation for a long time and found that in the very late stage, which was too late to fix. What I mean is that, you may not get exactly the same result as I got, although it's highly likely to get similar result.**

### 0. Data preparation

Put the original dataset and extended dataset in `input/birdsong-recognition`.

Run `make prepare` command in this directory(the directory where this `README.md` is placed). This will perform resampling on the datasets, which takes a couple of hours.

### 1. training

`make train`

### (Optional) Stage 1 only

`make train-stage1`

### (Optional) Stage 1 and Stage 2

Since stage2 depends on the output of stage1, it cannot be run without stage1

```shell
make train-stage1
make train-stage2
```

## Kaggle Solution

* Inference Notebook: https://www.kaggle.com/hidehisaarai1213/birdcall-resnestsed-effnet-b0-ema-all-th04
* Solution: https://www.kaggle.com/c/birdsong-recognition/discussion/183204
