# Cornell Birdcall Identification Competition

6th place solution to the [Cornell Birdcall Identification Challenge](https://www.kaggle.com/c/birdsong-recognition) hosted on Kaggle.

## Context

> In this competition, you will identify a wide variety of bird vocalizations in soundscape recordings. Due to the complexity of the recordings, they contain weak labels. There might be anthropogenic sounds (e.g., airplane overflights) or other bird and non-bird (e.g., chipmunk) calls in the background, with a particular labeled bird species in the foreground. Bring your new ideas to build effective detectors and classifiers for analyzing complex soundscape recordings!

## Evaluation

> The hidden test_audio directory contains approximately 150 recordings in mp3 format, each roughly 10 minutes long. They will not all fit in a notebook's memory at the same time. The recordings were taken at three separate remote locations in North America. Sites 1 and 2 were labeled in 5 second increments and need matching predictions, but due to the time consuming nature of the labeling process the site 3 files are only labeled at the file level. Accordingly, site 3 has relatively few rows in the test set and needs lower time resolution predictions.

Scores were evaluated based on their row-wise micro averaged F1 score.

## WIP

I'm currently on cleaning the training code which was quite messy.
