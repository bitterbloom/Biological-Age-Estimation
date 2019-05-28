#!/bin/bash
mkdir -p data
cd data

if [ ! -f wiki_crop.tar ]; then
    wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
fi

if [ ! -d wiki_crop ]; then
    tar xf wiki_crop.tar
fi

cd ../
if [ ! -f Model_Age.pth ]; then
    wget https://www.dropbox.com/s/ijoirmbwdf7orcf/Model_Age.pth?dl=1
    mv Model_Age.pth\?dl\=1 ./Model_Age.pth
fi
