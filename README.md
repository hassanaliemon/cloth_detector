## Install Dependencies
Install dependencies by

`pip install -r requirements.txt`

## Download the model
Download the trained model from [here.](https://drive.google.com/file/d/158q3-on6aJE5yiwGzWCY9TxH4g7-Jk-r/view?usp=sharing)

move the model to `models` folder

## Run the Web UI
Run 

`python app.py`

Now go to `localhost:5000` from your browser. It will show two routes, one is `detect` and another is `health`. `detect` for detection and `health` to check api availability.

Detect route has image upload section. Upload the image and press `Detect Clothing` to get the detection of the image using the trained model

## Dataset Info
I have annotated a total of `87` images for the model of which `76` for train and `11` for valid. You can download the dataset from [here.](https://drive.google.com/file/d/19B-GxQPg0sxX8qb4rYLTRpUGzKnzzucs/view?usp=sharing) It also containes annotation file.

## Performance
Overall the model achived map 0.995 on IoU 0.5 (around) on the data when used augmentation like
- flip up down
- flip left right
- increase/decrease brightness(value of HSV)


## Limitation
- For simplicity and speed I can only annotate `87` images thus the model is not scalable
- No pre-processing or post-processing is used
