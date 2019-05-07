# Intro

Radio ads are annoying. 
Skipping through every channel until you find something you don't mind listening to is more annoying.     
What if you could know which channels have music playing and which ones have ads playing?
The goal of this project is to make radio bearable.
We explore the idea of identifying what's playing at a radio station using machine learning. 


We propose to achieve this task using machine learning to classify audio streams into two categories -- Ads and music. 

## First stage
- Build dataset from music samples and radio advert samples. 
- Feature extraction from these samples. 
    - Currently using only [MFCC](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) for features. 
- Build and train classifier that categorizes an audio sample into either 'ad' or 'music'.

## Second stage
Some find the discussions on radio streams to be interesting. These sections should have their own label and our classifier should be able to extend to this.
- Add 'talk' label to the classifier. 
    - Train on generate conversation patterns perhaps?
- Recognize song titles
    - Should be straigtforward to use a audio fingerprinting API to figure out which song the classified sample belongs to.
   
   
## Working on this project

- Always work on a branch and then send a pull request.
- Review outstanding PRs. If you do not feel comfortable merging PRs, comment with a "+1" to signal your co-collaborators that it's passed your review.
- Feel free to push directly to master only for micro-fixes such as typos.

## Directory structure
Adapted from [here](https://github.com/drivendata/cookiecutter-data-science). Let's modify this structure as we add files. Can delete once we have a working pipeline.
```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- CSV files.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- model outputs 
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── download_dataset.py
|   |   └── make_dataset.py... 
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
```


## Usage
### Downloading the dataset
Downloading audio files from youtube videos/playlists
```
./src/data/download_dataset.py input_file1 input_file2 ... input_fileN
```

input_file is a list of youtube playlist/video links. Can use # for comments within the file.
For examples, see [here](https://github.com/srinathos/slightlyBetterRadio/tree/rao_dataset_download/data/raw/playlists)
