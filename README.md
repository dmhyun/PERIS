<h1 align="center">
<span> Beyond Learning from Next Item: Sequential Recommendation via Personalized Interest Sustainability
</span>
</h1>

<p align="center">
    <a href="https://www.cikm2022.org/" alt="Conference">
        <img src="https://img.shields.io/badge/CIKM'22-Full%20paper-brightgreen" /></a>   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>   
</p>

<p align="center">
<span>Official implementation of </span>
<a href="https://arxiv.org/pdf/2209.06644.pdf">CIKM'22 paper</a>
</p>

## Overview
### Sequential Recommender Systems
There have been two groups of existing sequential models. The user-centric models capture personalized interest drift based on each user's sequential consumption history, but do not explicitly consider whether users' interest in items sustains beyond the training time, i.e., interest sustainability. On the other hand, the item-centric models consider whether users' general interest sustains after the training time, but it is not personalized. In this work, we propose a recommender system taking advantages of the models in both categories.

<p align="center"><img src="images/intro_comparison.png" alt="graph" width="45%"></p>

### Personalized Interest Sustainability
Our proposed model (PERIS) captures personalized interest sustainability, indicating whether each user's interest in items will sustain beyond the training time or not. We first formulate a task that requires to predict which items each user will consume in the recent period of the training time based on users' consumption history.

<p align="center"><img src="images/pisp.png" alt="graph" width="45%"></p>

### Supplementation schems for users' sparse history
It is non-trivial to predict items that each user is likely to consume in the recent period of the training time because most users have insufficient consumption history per item. We hence devise simple yet effective schemes to supplement users’ sparse consumption history in both intrinsic and extrinsic manners.

<p align="center"><img src="images/supp.png" alt="graph" width="45%"></p>

### Recommendation Performance
PERIS significantly outperforms the baseline models including the general, user-centric, and item-centric sequential models on the 11 real-world datasets. This result indicates the effectiveness of incorporating the PIS, i.e., whether each user’s interest in items will sustain beyond the training time, over various domains.
<p align="center"><img src="images/performance.png" alt="graph" width="65%"></p>


## Major Requirements
* Python
* Pytorch
* Numpy

## Preprocessing Data
1. Download user-item consumption data (and extract the compressed file) into `./data/`.
    * [Amazon](http://jmcauley.ucsd.edu/data/amazon/)
      <pre>[Example] <code>wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Cell_Phones_and_Accessories.csv</code></pre>
    * [Yelp](https://www.yelp.com/dataset)
    * [Google Maps](cseweb.ucsd.edu/~jmcauley/datasets.html)
    * Other data you want

    :exclamation: *Please make sure your data in the same csv format of Amazon data.*
    
    
 
 2. Split your data into training/validation/test data in `./data/`.
   <pre><code>python split_data.py your_decompressed_file.csv</code></pre>
 
 3. Build a dataset for training a recommender syetem with using the splitted data.
 <pre><code>python build_recdata.py generated_directory </code></pre>
 
 ## Training    
Train the proposed recommender system (PERIS).
 <pre><code>python train.py --dataset your_dataset --learning_rate 1e-3 --lamb 0.5 --mu 0.3 --K 128 </code></pre>
 
## Citation
If you use this repository for your work, please consider citing [our paper](https://arxiv.org/pdf/2209.06644):

<pre><code>@article{2209.06644,
Author = {Dongmin Hyun, Chanyoung Park, Junsu Cho, and Hwanjo Yu},
Title = {Beyond Learning from Next Item: Sequential Recommendation via Personalized Interest Sustainability},
Year = {2022},
Eprint = {arXiv:2209.06644},
Doi = {10.1145/3511808.3557415},
}
</code></pre>
