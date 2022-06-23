# Multimodal speech emotion recognition Using Audio and Text

Project for HSE Deep Learning course: emotions recognition from audio and text.

## Dataset
[IEMOCAMP](https://sail.usc.edu/iemocap/index.html) dataset with 4 largest emotion classes and united class "happy" and "excited". 
Preprocessed and splitted into train, test, validation as 8:1:1.

## Models architectures 
![Model](https://github.com/aapoliakova/SER_TASK/blob/master/photo_2022-06-23%2010.58.28.jpeg?raw=true)

___
+ ARE model: CNN

` baseline/baseline_audio.py `

+ TRE model: GRU

` baseline/baseline_text.py `

