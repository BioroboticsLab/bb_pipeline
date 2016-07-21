#! /usr/bin/env bash

set -e
decoder_weights='https://github.com/BioroboticsLab/bb_decoder_models/raw/master/decoder_weights.h5'
decoder_model='https://raw.githubusercontent.com/BioroboticsLab/bb_decoder_models/master/decoder_architecture.json'
localizer_weights='https://github.com/nebw/saliency-localizer-models/raw/master/season_2015/saliency_weights-wobn'
tagSimilarityEncoder_model='https://raw.githubusercontent.com/lkairies/bb_tagSimilarityEncoder_model/master/tagSimilarityEncoder_architecture.json'
tagSimilarityEncoder_weights='https://github.com/lkairies/bb_tagSimilarity_Encoder_model/raw/master/tagSimilarityEncoder_weights.h5'

decoder_dir="test/models/decoder"
localizer_dir="test/models/localizer"
tagSimilarityEncoder_dir="test/models/tagSimilarityEncoder"

mkdir -p  $localizer_dir
mkdir -p  $decoder_dir
mkdir -p  $tagSimilarityEncoder_dir

decoder_weights_fname="$decoder_dir/decoder_weights.hdf5"
decoder_model_fname="$decoder_dir/decoder_architecture.json"
localizer_weights_fname="$localizer_dir/saliency-weights.hdf5"
tagSimilarityEncoder_weights_fname="$tagSimilarityEncoder_dir/tagSimilarityEncoder_weights.h5"
tagSimilarityEncoder_model_fname="$tagSimilarityEncoder_dir/tagSimilarityEncoder_architecture.json"

if [ ! -e "$decoder_weights_fname" ]; then
    echo "Downloading decoder weights"
    curl -L $decoder_weights > $decoder_weights_fname
fi

if [ ! -e "$decoder_model_fname" ]; then
    echo "Downloading decoder model"
    curl -L $decoder_model > $decoder_model_fname
fi

if [ ! -e "$localizer_weights_fname" ]; then
    echo "Downloading localizer weights"
    curl -L $localizer_weights > $localizer_weights_fname
fi

if [ ! -e "$tagSimilarityEncoder_weights_fname" ]; then
    echo "Downloading tagSimilarityEncoder weights"
    curl -L $tagSimilarityEncoder_weights > $tagSimilarityEncoder_weights_fname
fi
