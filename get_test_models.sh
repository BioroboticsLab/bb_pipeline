#! /usr/bin/env bash

set -e
decoder_weights='https://github.com/BioroboticsLab/bb_decoder_models/raw/master/decoder_weights.h5'
decoder_model='https://raw.githubusercontent.com/BioroboticsLab/bb_decoder_models/master/decoder_architecture.json'
localizer_weights='https://github.com/nebw/saliency-localizer-models/raw/master/season_2015/saliency_weights-wobn'

decoder_dir="test/models/decoder"
localizer_dir="test/models/localizer"


mkdir -p  $localizer_dir
mkdir -p  $decoder_dir

decoder_weights_fname="$decoder_dir/decoder_weights.hdf5"
decoder_model_fname="$decoder_dir/decoder_architecture.hdf5"
localizer_weights_fname="$localizer_dir/saliency-weights.hdf5"

if [ ! -e "$decoder_weights_fname" ]; then
    curl -L $decoder_weights > $decoder_weights_fname
fi

if [ ! -e "$decoder_model_fname" ]; then
    curl -L $decoder_model > $decoder_model_fname
fi

if [ ! -e "$localizer_weights_fname" ]; then
    curl -L $localizer_weights > $localizer_weights_fname
fi
