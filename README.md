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
