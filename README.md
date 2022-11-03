# [scut-bds-lab DSTC11-Track1-SIMMC2.1 Submission](https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab)

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a>
    <a href="support os"><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href="https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab/graphs/contributors"><img src="https://img.shields.io/github/contributors/scutcyr/dstc11-simmc2.1-scut-bds-lab?color=9ea"></a>
    <a href="https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab/commits"><img src="https://img.shields.io/github/commit-activity/m/scutcyr/dstc11-simmc2.1-scut-bds-lab?color=3af"></a>
    <a href="https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab/issues"><img src="https://img.shields.io/github/issues/scutcyr/dstc11-simmc2.1-scut-bds-lab?color=9cc"></a>
    <a href="https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab/stargazers"><img src="https://img.shields.io/github/stars/scutcyr/dstc11-simmc2.1-scut-bds-lab?color=ccf"></a>
</p>

**Team**: scut-bds-lab

## Recent Update
- 👏🏻  2022.10.10: The repository `dstc11-simmc2.1-scut-bds-lab` for [DSTC11 Track1](https://github.com/facebookresearch/simmc2) is created.
- 👏🏻  2022.10.28: The model is public on huggingface, see the link [https://huggingface.co/scutcyr/dstc11-simmc2.1-scut-bds-lab](https://huggingface.co/scutcyr/dstc11-simmc2.1-scut-bds-lab) for detail.


## Overview
The [SIMMC2.1](https://github.com/facebookresearch/simmc2) challenge aims to lay the foundations for the real-world assistant agents that can handle multimodal inputs, and perform multimodal actions. It has 4 tasks: Ambiguous Candidate Identification, Multimodal Coreference Resolution, Multimodal Dialog State Tracking, Response Generation. We consider the joint input of textual context, tokenized objects and scene as multi-modal input, as well as compare the performance of single task training and multi task joint training.
As to subtask4, we also consider the system belief state (act and slot values) as the prombt for response generation. Non-visual metadata is also considered by adding the embedding to the object.


### About Folder and File
* [./evaluation_tools](./evaluation_tools): The evaluation module changed from [SIMMC2.1](https://github.com/facebookresearch/simmc2.git).
* [./models](./models): The Huggingface-Transformers-Like model module. Each sub folder stores a model class.
* [./results](./results): The predicted results.
* [./runs](./runs): The model checkpoint and training log.
* [./scripts](./scripts): The preprocessing or training scripts.
* [./utils](./utils): The dataset loading module.
* [./convert.py](./convert.py): The dataset preprocessing code. It will convert data to several line-by-line .txt files. 
* [./convert_simmc21_sysana_for_task4.py](./convert.py): The dataset preprocessing code. It will convert data to one line-by-line .txt file. 
* [./eval_model.py](./eval_model.py): The evaluating code.
* [./eval_model_args.py](./eval_model_args.py): The arguements for running [./eval_model.py](./eval_model.py).
* [./train_model.py](./train_model.py): The training code.
* [./train_model_args.py](./train_model_args.py): The arguements for running [./train_model.py](./train_model.py).



**Clone Our Project:**

```bash
cd ~
git clone https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab.git
```

## Requirements and Installation
You can create the python environment for running this project by using [conda](https://www.anaconda.com/):
```bash
conda create -n py38 python=3.8
conda activate py38
cd ~/dstc11-simmc2.1-scut-bds-lab
pip install -r requirements.txt
```

The python package for running this project are shown as:
```bash
python>=3.7
accelerate==0.11.0
attrs
chardet==5.0.0
datargs==0.11.0
future==0.18.2
gdown==4.5.1
imagesize==1.4.1
ipdb==0.13.9
matplotlib==3.5.2
nltk==3.7
notebook
opencv-python==4.6.0.66
opencv-python-headless==4.6.0.66
pandas==1.4.3
parlai==1.6.0
pytorch-ignite==0.4.8
sacremoses==0.0.53
scikit-learn==1.1.1
sentencepiece==0.1.96
setuptools==59.5.0
sklearn
tensorflow==2.10.0
torch==1.12.0
torchaudio==0.12.0
torchtext==0.13.0
torchvision==0.13.0
tqdm==4.62.3
transformers==4.22.2
```

## Dataset Preprocessing

### (1) Download SIMMC2.1
Download the dataset SIMMC2.1 from [https://github.com/facebookresearch/simmc2](https://github.com/facebookresearch/simmc2) by ```git```:
```bash
cd ~
git lfs install
git clone https://github.com/facebookresearch/simmc2.git
```

### (2) Copy the Dataset and Unzip the .zip file
There are five ```.zip``` file: 
- simmc2_scene_images_dstc10_public_part1.zip
- simmc2_scene_images_dstc10_public_part2.zip
- simmc2_scene_images_dstc10_teststd.zip
- simmc2_scene_jsons_dstc10_public.zip
- simmc2_scene_jsons_dstc10_teststd.zip
```bash
cp -rf ~/simmc2/data ~/dstc11-simmc2.1-scut-bds-lab/
cd ~/dstc11-simmc2.1-scut-bds-lab/data
ls # list the file
```

The files in ```~/dstc11-simmc2.1-scut-bds-lab/data```` are shown as follow:
```bash
fashion_prefab_metadata_all.json    simmc2.1_dials_dstc11_dev.json      simmc2_scene_images_dstc10_public_part1.zip  simmc2_scene_jsons_dstc10_teststd.zip
furniture_prefab_metadata_all.json  simmc2.1_dials_dstc11_devtest.json  simmc2_scene_images_dstc10_public_part2.zip
scut-bds-lab_PREPOCESS.md                simmc2.1_dials_dstc11_mini.json     simmc2_scene_images_dstc10_teststd.zip
README.md                           simmc2.1_dials_dstc11_train.json    simmc2_scene_jsons_dstc10_public.zip
```

Then unzip the .zip file to current path:
```bash
cd ~/dstc11-simmc2.1-scut-bds-lab/data
unzip simmc2_scene_images_dstc10_public_part1.zip  # --> ./simmc2_scene_images_dstc10_public_part1
unzip simmc2_scene_images_dstc10_public_part2.zip  # --> ./simmc2_scene_images_dstc10_public_part2
# Merge part1 and part2 files into ./simmc2_scene_images_dstc10_public
mkdir simmc2_scene_images_dstc10_public
cp simmc2_scene_images_dstc10_public_part1/* simmc2_scene_images_dstc10_public
cp simmc2_scene_images_dstc10_public_part2/* simmc2_scene_images_dstc10_public
rm -rf simmc2_scene_images_dstc10_public_part1
rm -rf simmc2_scene_images_dstc10_public_part2

unzip simmc2_scene_images_dstc10_teststd.zip       # --> ./simmc2_scene_images_dstc10_teststd
unzip simmc2_scene_jsons_dstc10_public.zip         # --> ./public
mkdir simmc2_scene_jsons_dstc10_public
cp public/* simmc2_scene_jsons_dstc10_public
rm -rf public

unzip simmc2_scene_jsons_dstc10_teststd.zip        # --> ./simmc2_scene_jsons_dstc10_teststd
```

### (3) Preprocess the dataset
* Open the [0_dataset_preprocessing.sh](./scripts/0_dataset_preprocessing.sh), [0_dataset_preprocessing_predict_with_sys_state.sh](./scripts/0_dataset_preprocessing_predict_with_sys_state.sh) and [0_dataset_preprocessing_for_task4.sh](./scripts/0_dataset_preprocessing_for_task4.sh); 
* Specify the path of the .bashrc file or remove ```source ~/.bashrc_cuda11``` and ```source ~/.bashrc```; 
* Specify the conda python environment, e.g. ```conda activate py38cu113``` or ```conda activate py38```; 
* Change the ```INPUT_DIR=~/dstc11-simmc2.1-scut-bds-lab/data``` and ```WORK_DIR=~/dstc11-simmc2.1-scut-bds-lab``` to the actual path you specified; 
* Then run the dataset preprocessing scripts:
```bash
cd ~/dstc11-simmc2.1-scut-bds-lab/scripts
./0_dataset_preprocessing.sh
./0_dataset_preprocessing_predict_with_sys_state.sh
./0_dataset_preprocessing_for_task4.sh
```
The above preprocessing scripts takes about **two days**, because it generates samples of different conversation rounds. You can simplify it.    
After preprocessing the dataset, you can find the preprocessed file in ```~/dstc11-simmc2.1-scut-bds-lab/data_convert```.

**Note**: We have two different preprocessed dataset file, Files like ```simmc2.1_dials_dstc11_train_ctxlen2_sysana_for_task4.txt``` gather all data and labels (split by Tab) required for training into one file. Another data form contains multiple line-by-line files.


## Model
All the models are defined in the folder [./models](./models).
We provide the following pre training model interface codes for fine-tuning on SIMMC2.1:    
**(a) Encoder-Decoder:**
* BART: [paper](https://arxiv.org/abs/1910.13461), [pretrained-model](https://huggingface.co/facebook/bart-large), [model code](./models/simmc21_bart)
* T5: [paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf), [pretrained-model](https://huggingface.co/t5-11b), [model code](./models/simmc21_t5)
* UL-2: [paper](https://arxiv.org/abs/2205.05131v1), [pretrained-model](https://huggingface.co/google/ul2), [model code](./models/simmc21_ul2)
* BlenderBot: [paper](https://arxiv.org/abs/1907.06616), [pretrained-model](https://huggingface.co/facebook/blenderbot-3B), [model code](./models/simmc21_blenderbot)

**(b) Multi-Modal Encoder**
* Flava: [paper](https://arxiv.org/abs/2112.04482), [pretrained-model](https://huggingface.co/facebook/flava-full), [model code](./models/simmc21_flava)

**(c) Multi-Modal Encoder-Decoder**
* OFA: [paper](https://arxiv.org/abs/2112.04482), [pretrained-model](https://huggingface.co/OFA-Sys/OFA-large), [model code](./models/simmc21_ofa)

**Note:** You can specify different pre training models for further fine-tuning by modifying arguments ```--model_type``` and ```--model_name_or_path```. For [T5-11B](https://huggingface.co/t5-11b) and [UL-2](https://huggingface.co/google/ul2), you can achieve model pipeline parallelism by specifying parameter ```--model_parallel```.

## Training
You can find the training script examples in the folder ```./scripts```. Before running the script, you need to modify some arguments, mainly including: ```source .bashrc path```, ```conda python environment```, ```WORK_DIR```, ```INIT_DATA_DIR```, ```PREPROCESS_DATA_DIR```, ```--model_name_or_path```. All the pretrained model can be downloaded from [🤗 Transformers - Hugging Face](https://huggingface.co/models).    
For example, you can download the [BART-large model](https://huggingface.co/facebook/bart-large) by using:
```bash
cd ~
mkdir pretraining_model
cd pretraining_model
git lfs install
git clone https://huggingface.co/facebook/bart-large
```
After modifying the above arguments and downloading the pretrained model, you can run the bash script to fine tune the model:
```bash
cd ~/dstc11-simmc2.1-scut-bds-lab/scripts
./run_train_model_simmc21_bart_20221017_1040.sh
```

View the training process of the model via ```Tensorboard```:
```bash
cd ~/dstc11-simmc2.1-scut-bds-lab
tensorboard --logdir=./runs --port=6666 --bind_all
```
Then use browser to open the url, according to the terminal prompts:
```bash
TensorBoard 2.10.1 at http://<your_server_ip_or_name>:6666/ (Press CTRL+C to quit)
```

**Note**: If you use the preprocessed data converted by [0_dataset_preprocessing_for_task4.sh](./scripts/0_dataset_preprocessing_for_task4.sh), you only need to appoint the ```--train_input_file``` and ```--eval_input_file```, e.g. [run_train_model_simmc21_ofa_20221013_0930.sh](./scripts/run_train_model_simmc21_ofa_20221013_0930.sh)


```bash
INIT_DATA_DIR=~/dstc11-simmc2.1-scut-bds-lab/data
PREPROCESS_DATA_DIR=~/dstc11-simmc2.1-scut-bds-lab/data_convert
CONTEXT_LENGTH=6 # 2,4,6,8
# Single file input format
    --train_input_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_ctxlen${CONTEXT_LENGTH}_sysana_for_task4.txt \
    --eval_input_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_ctxlen${CONTEXT_LENGTH}_sysana_for_task4.txt \
# Multiple files input format
    --train_input_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_predict_ctxlen${CONTEXT_LENGTH}.txt \
    --train_target_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_target_ctxlen${CONTEXT_LENGTH}.txt  \
    --disambiguation_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_disambiguation_label.txt \
    --response_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_response.txt \
    --eval_input_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_predict_ctxlen${CONTEXT_LENGTH}.txt \
    --eval_target_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_target_ctxlen${CONTEXT_LENGTH}.txt \
# IF the model need images
    --train_image_path_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_train_scene_name.txt \
    --train_image_dir=$INIT_DATA_DIR/simmc2_scene_images_dstc10_public \
    --eval_image_path_file=$PREPROCESS_DATA_DIR/simmc2.1_dials_dstc11_devtest_scene_name.txt \
    --eval_image_dir=$INIT_DATA_DIR/simmc2_scene_images_dstc10_public \
```




## Evaluation
We provide the code [eval_model.py](./eval_model.py) for evaluating the model.

You can download our model from [https://huggingface.co/scutcyr/dstc11-simmc2.1-scut-bds-lab](https://huggingface.co/scutcyr/dstc11-simmc2.1-scut-bds-lab) by using follow scrips:
```bash
cd ~
mkdir pretrained_model
cd pretrained_model
git lfs install
git clone https://huggingface.co/scutcyr/dstc11-simmc2.1-scut-bds-lab
```

Then change the ```--model_dir``` to specify model, such as:
```sh
--model_dir=~/pretrained_model/dstc11-simmc2.1-scut-bds-lab/mt-bart/checkpoint-12
```
in the bash script file [run_test_model_simmc21_bart_20221020_2000_use_focalloss_exp3.sh](./scripts/run_test_model_simmc21_bart_20221020_2000_use_focalloss_exp3.sh)

or 

```sh
--model_dir=~/pretrained_model/dstc11-simmc2.1-scut-bds-lab/mt-bart-sys/checkpoint-11
```
in the bash script file [run_infer_model_simmc21_bart_20221020_1800.sh](./scripts/run_infer_model_simmc21_bart_20221020_1800.sh)

or


```sh
--model_dir=~/pretrained_model/dstc11-simmc2.1-scut-bds-lab/mt-bart-sys-nvattr/checkpoint-15
```
in the bash script file [run_test_model_simmc21_bart_sys_state_attr_ctxlen6_20221025_0100_use_focalloss.sh](./scripts/run_test_model_simmc21_bart_sys_state_attr_ctxlen6_20221025_0100_use_focalloss.sh)





## Results

### devtest result

| Model | Subtask-1 Amb. Candi. F1 | Subtask-2 MM Coref F1 | Subtask-3 MM DST Slot F1 | Subtask-3 MM DST Intent F1 | Subtask-4 Response Gen. BLEU-4 |
|:----:|:----:|:----:|:----:|:----:|:----:|
| mt-bart-ensemble | 0.68466 | 0.77860 | 0.91816 | 0.97828 | 0.34496 |
| mt-bart-dstcla | 0.67589 | 0.78407 | 0.92013 | 0.97468 |  |
| mt-bart-dstcla-ensemble | 0.67777 | 0.78640 | 0.92055 | 0.97456 |  |
| mt-bart-sys |  |  |  |  | 0.39064 |
| mt-bart-sys-2 |  |  |  |  | 0.3909 |
| mt-bart-sys-ensemble |  |  |  |  | 0.3894 |
| mt-bart-sys-nvattr |  |  |  |  | 0.38995 |

### teststd result
The teststd result is provided in the [./results/teststd-result](./results/teststd-result). One subfolder corresponds to one model.





## References

```
@inproceedings{kottur-etal-2021-simmc,
    title = "{SIMMC} 2.0: A Task-oriented Dialog Dataset for Immersive Multimodal Conversations",
    author = "Kottur, Satwik  and
      Moon, Seungwhan  and
      Geramifard, Alborz  and
      Damavandi, Babak",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.401",
    doi = "10.18653/v1/2021.emnlp-main.401",
    pages = "4903--4912",
}

@inproceedings{lee-etal-2022-learning,
    title = "Learning to Embed Multi-Modal Contexts for Situated Conversational Agents",
    author = "Lee, Haeju  and
      Kwon, Oh Joon  and
      Choi, Yunseon  and
      Park, Minho  and
      Han, Ran  and
      Kim, Yoonhyung  and
      Kim, Jinhyeon  and
      Lee, Youngjune  and
      Shin, Haebin  and
      Lee, Kangwook  and
      Kim, Kee-Eung",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.61",
    doi = "10.18653/v1/2022.findings-naacl.61",
    pages = "813--830",
}
```


## Acknowledge
* We would like to express our gratitude to the authors of [Hugging Face's Transformers🤗](https://huggingface.co/) and its open source community for the excellent design on pretrained models usage.
* We would like to express our gratitude to [Meta Research | Facebook AI Research](https://github.com/facebookresearch) for the SIMMC2.1 dataset and the baseline code.
* We would like to express our gratitude to [KAIST-AILab](https://github.com/KAIST-AILab/DSTC10-SIMMC) for the basic research framework on SIMMC2.0.

## License
The project is provided under the [Apache-2.0 License](https://github.com/scutcyr/dstc11-simmc2.1-scut-bds-lab/blob/main/LICENSE).

<p align="center">
  <img src="./figure/scut-bds-lab_logo.jpg" />
</p>

[dstc11]:https://dstc11.dstc.community/
[dstc10]:https://sites.google.com/dstc.community/dstc10/home
[simmc1]:https://github.com/facebookresearch/simmc
[simmc2_arxiv]:https://arxiv.org/pdf/2104.08667.pdf
[simmc_arxiv]:https://arxiv.org/abs/2006.01460


