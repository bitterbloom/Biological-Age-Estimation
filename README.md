# Biological-Age-Estimation
Biological Age Estimation in the IMDB-WIKI dataset for the final project of the course IBIO4490 at the Universidad de los Andes

#### The download and database clean scripts are taken from the [age-gender-estimation by yu4u](https://github.com/yu4u/age-gender-estimation)

## Dependencies
- Python3.5+
- Pytorch 1.0.1.post2
- scipy, numpy, tqdm, PIL

### Create training data from the IMDB-WIKI dataset
First, download the dataset
The dataset is downloaded and extracted to the `data` directory by:

```sh
./download.sh
```
*This also downloads the model*

Second, clean the dataset.
The images that had gender label as NaN, age label lower than 0 or higher than 100, faces below a face-score of 1 and images with more than one face were removed by:

```sh
python3 create_db.py --output data/wiki_db.mat --db wiki
```

### Test in Wiki dataset (38,138 images)

```sh
python3 eval.py 
```

### Demo in Wiki dataset

```sh
python3 demo.py 
```




