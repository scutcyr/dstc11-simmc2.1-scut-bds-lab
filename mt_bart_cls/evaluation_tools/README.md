# DSTC11 SIMMC2.1 Evaluation Tool (DSTC-11 SIMMC2.1 测评工具)

* Updated by Yirong Chen 
* Mail: [yrchen5@iflytek.com](mailto:yrchen5@iflytek.com) or [eeyirongchen@mail.scut.edu.cn](mailto:eeyirongchen@mail.scut.edu.cn)
* Date: 2022/08/04

## About the Code
This code is changed from [https://github.com/facebookresearch/simmc2.git](https://github.com/facebookresearch/simmc2.git) and used for evaluation on the Third Situated Interactive Multimodal Conversations (SIMMC 2.1) Track page for [DSTC11][dstc11] 2022.

## File

* ```convert.py```: original file is from [simmc2/blob/main/model/mm_dst/gpt2_dst/utils/convert.py](https://github.com/facebookresearch/simmc2/blob/main/model/mm_dst/gpt2_dst/utils/convert.py)


* ```evaluate.py```: original file is from [simmc2/blob/main/model/mm_dst/gpt2_dst/scripts/evaluate.py](https://github.com/facebookresearch/simmc2/blob/main/model/mm_dst/gpt2_dst/scripts/evaluate.py)


* ```evaluate_dst.py```: original file is from [simmc2/blob/main/model/mm_dst/utils/evaluate_dst.py](https://github.com/facebookresearch/simmc2/blob/main/model/mm_dst/utils/evaluate_dst.py)


* ```evaluate_response.py```: original file is from [simmc2/blob/main/model/mm_dst/gpt2_dst/scripts/evaluate_response.py](https://github.com/facebookresearch/simmc2/blob/main/model/mm_dst/gpt2_dst/scripts/evaluate_response.py)





## Important Links

* [SIMMC 2.1 Challenge Proposal][simmc2.1_proposal] (DSTC11)
* [Task Description Paper][simmc2_arxiv] (EMNLP 2021)
* [Data Formats](data/README.md)

## Track Description

### Tasks and Metrics

For this edition of the challenge, we focus on four sub-tasks primarily aimed at replicating human-assistant actions in order to enable rich and interactive shopping scenarios.

For more detailed information on the new SIMMC 2.1 dataset and the instructions, please refer to the [DSTC11 challenge proposal document][simmc2.1_proposal].

| Sub-Task #1 | [Ambiguous Canaidate Identification (New)](model/mm_dst) |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal | Given ambiguous object mentions, to resolve referent objects to thier canonical ID(s). |
| Input | Current user utterance, Dialog context, Multimodal context |
| Output |  Canonical object IDs |
| Metrics | Object Identification F1 / Precision / Recall |

| Sub-Task #2 | [Multimodal Coreference Resolution](model/mm_dst) |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal | To resolve referent objects to thier canonical ID(s) as defined by the catalog. |
| Input | Current user utterance, Dialog context, Multimodal context |
| Output |  Canonical object IDs |
| Metrics |  Coref F1 / Precision / Recall |

| Sub-Task #3 | [Multimodal Dialog State Tracking (MM-DST)](model/mm_dst) |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal | To track user belief states across multiple turns |
| Input | Current user utterance, Dialogue context, Multimodal context |
| Output | Belief state for current user utterance |
| Metrics | Slot F1, Intent F1 |

| Sub-Task #4 | [Multimodal Dialog Response Generation](model/mm_dst)  |
|---------|---------------------------------------------------------------------------------------------------------------------------------------|
| Goal | To generate Assistant responses  |
| Input | Current user utterance, Dialog context, Multimodal context, (Ground-truth API Calls) |
| Output | Assistant response utterance |
| Metrics | BLEU-4 |


Please check the [task input](./TASK_INPUTS.md) file for a full description of inputs
for each subtask.

### Baseline Results

We will provide the baselines for all the four tasks to benchmark their models.
Feel free to use the code to bootstrap your model.

| Subtask | Name | Baseline Results | 
| :--: | :--: | :--: |
| #1 | Ambiguous Candidate Identification | [Link][subtask1_results] |
| #2 | Multimodal Coreference Resolution | [Link][subtask2_results] |
| #3 | Multimodal Dialog State Tracking (MM-DST) | [Link][subtask3_results] |
| #4 | Multimodal Dialog Response Generation | [Link][subtask4_results] |


## How to Download Datasets and Code

* Git clone our repository to download the datasets and the code. You may use the provided baselines as a starting point to develop your models.
```
$ git lfs install
$ git clone https://github.com/facebookresearch/simmc2.git
```

* Also please feel free to check out other open-sourced repositories from the previous SIMMC 2.0 challenge [here](CHALLENGE_RESULTS.md).
 

## Contact

### Questions related to SIMMC Track, Data, and Baselines
Please contact simmc@fb.com, or leave comments in the Github repository.

### DSTC Mailing List
If you want to get the latest updates about DSTC10, join the [DSTC mailing list](https://groups.google.com/a/dstc.community/forum/#!forum/list/join).


## Citations

If you want to publish experimental results with our datasets or use the baseline models, please cite the following articles:

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
```
**NOTE**: The [paper][simmc2_arxiv] (EMNLP 2021) above describes in detail the datasets, the collection process, and some of the baselines we provide in this challenge. 

## License

SIMMC 2 is released under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode), see [LICENSE](LICENSE) for details.


[dstc11]:https://dstc11.dstc.community/
[dstc10]:https://sites.google.com/dstc.community/dstc10/home
[simmc1]:https://github.com/facebookresearch/simmc
[simmc2_arxiv]:https://arxiv.org/pdf/2104.08667.pdf
[simmc_arxiv]:https://arxiv.org/abs/2006.01460
[subtask1_results]:./model/disambiguate#performance-on-simmc-20
[subtask2_results]:./model/mm_dst#results
[subtask3_results]:./model/mm_dst#results
[subtask4_results]:./model/mm_dst#results
[simmc2.0]:./dstc10/CHALLENGE_RESULTS.md
[simmc2.1_proposal]:https://drive.google.com/file/d/1_Tdl7CXm71gqlWutbOe0e8O1hhiycsQf/view
