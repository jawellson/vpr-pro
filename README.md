# The optimized res2net is used for voiceprint recognition
This project improves the Res2net network through the EFC module, GAM module and EDSC module to achieve feature fusion, attention optimization and lightweight respectively, thereby achieving voiceprint recognition optimization in complex scenarios.

# Install the relevant environment
 - The GPU version of Pytorch will be installed first, please skip it if you already have it installed.
```shell
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

 - Install ppvector.
 
Install it using pip with the following command:
```shell
python -m pip install mvector -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Download this project
```shell
git clone https://github.com/jawellson/vpr-pro.git
```

# Create Data
The author used [CN-Celeb](https://openslr.elda.org/resources/82) and [Vox-Celeb](https://openslr.elda.org/49/) for this tutorial. The previous dataset contains the voice data of approximately 3,000 people and over 650,000 voice data. The latter is a large-scale and text-independent voiceprint recognition dataset, containing 100,000 voices from 1,251 celebrities in YouTube videos. After downloading, you need to unzip the dataset to the 'dataset' directory. Also need to download [CN-Celeb Test](https://aistudio.baidu.com/aistudio/datasetdetail/233361). If you have other better datasets, you can mix them up, but it's best to use python's aukit tool module for audio processing, noise reduction, and de-muting.

The format of the data list is `<voice_file_path\tspeech_classification_label>`. The creation of this list is mainly for the convenience of later reading, but also for the convenience of reading and using other speech data sets. Speech classification label refers to the unique ID of the speaker. Put these data sets in the same data list.

# Modify the configuration file
This project mainly consists of two configuration files. One is augmentation.yml for speech enhancement, and the other is myres2net.yml for model configuration. By default, the speech enhancement methods of speech rate enhancement, volume enhancement, noise enhancement, reverberation enhancement and spectrum enhancement are used. Fbank is used as the feature extraction method, EFCRes2Net_GAM is used as the training model, AAMLoss is used as the loss function, Adam is used as the optimizer, and the training epoch is 60.




