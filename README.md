# ChordCutterBot

ChordCutter is a system that utilizes Long Short-Term Memory (LSTM) networks to predict chord progressions directly from audio files.

## Data

MusicBench Dataset:
https://huggingface.co/datasets/amaai-lab/MusicBench

## Code Overview (chordBot.ipynb)

### Data Loading

Consists of library imports, loading the train/test MusicBench dataset, and preprocessing of audio files into chromas as well as other processing techniques such as denoising.

### Library Functions

Contains most functions for chord extraction and simplification, training loops, loss functions, chord tokenization, model checkpointing, Tonnetz embedding, and evaluation.

### Sequence-to-Sequence Models

A variety of model class definitions for varying LSTM models and their hyperparameters, training and results.

### Model Comparisons

Evaluates the varying LSTM models side-by-side for comparisons.

## Installation Instructions

1. `git clone` this repo
2. You must download and unzip the MusicBench.tar.gz tar file from the dataset linked above.
3. Run `python prep.py` to precompute the chromas on the dataset

Note: This may take a while (possibly 1+ hrs) and will use ~25 GB of memory. You only need the tar file and the precomputed chromas if you intend on running the model on anything from the dataset. Alternatively, you can just load the library functions, the model of your choosing (I recommend "Bidirectional LSTM" or "Large LSTM - Simplified" as they have the best performance), and load the appropriate checkpoint file stored in `checkpoints/`. Then, run the `predict_chords()` function, passing in the filepath of your audio file, your model instance, and the other parameters (just run the cells above the model section EXCEPT for the `train_model()` cells). The model will process the audio file and give you a chord progression along with corresponding timestamps. If you wish to play the output, then run `play_audio_from_progression`, passing in the chord progression, timestamps, and sampling rate you just received. Playing the progression audio is currently only supported for models using simplified chords (36 classes instead of 174).

Enjoy!
