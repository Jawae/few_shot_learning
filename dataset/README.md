## Datasets for few-shot learning

- Omniglot
    - `dataset/omniglot.py` will download the dataset automatically if it is not there.

- miniImagenet
    - Check the  `dataset/create_miniImagenet.py` folder on how to create one! (need to have the original ILSVRC-CLS 
    images ready)
    
    - Thrid-party instructions (OpenAI's Reptile) [here](https://github.com/openai/supervised-reptile/blob/master/fetch_data.sh).
  Google drive file [here](
https://drive.google.com/file/d/1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk/view) to directly download `images.tar.gz` file. 

    - The splits  `test.csv, train.csv, val.csv` (**already there if you clone our repo**) can be 
downloaded from [Ravi and Larochelle - splits](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet). 
For more information on how to obtain the images check the original source [Ravi and Larochelle - github](https://github.com/twitter/meta-learning-lstm)




- tierImagenet
    - Original info [here](https://github.com/renmengye/few-shot-ssl-public#tieredimagenet).
    To download the raw dataset from Google drive:
        ```
        python tools/download_gdrive.py 1hqVbS2nhHXa51R9_aB6QDXeC0P2LQG_u tier-imagenet.tar
        ```
    - create a symlink to the raw file: `ln -s /path/to/data dataset/tier-imagenet`

### Structure

In the `dataset/miniImagenet` folder, structure looks like this:
    
    few_shot_learning
    ├── ...
    └── dataset
       |__ data_loader.py
       |__ ...
       
       |__ miniImagenet                
          └── images
             ├── n0153282900000006.jpg
             ├── ...
             └── n1313361300001299.jpg
          |__ tes.csv
          |__ train.csv
          |__ val.csv
          |__ images.tar.gz   # created by dataset/create_miniImagenet.py
          
       |__ omniglot    # automatically download if missing
          |__ data
          |__ raw
          |__ splits


