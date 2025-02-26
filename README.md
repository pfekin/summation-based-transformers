## Random vector search 
Before creating the document vectors, it is necessary to download the 
[Simple English Wikipedia Cirrus JSON dump]( https://dumps.wikimedia.org/other/cirrussearch/20250106/simplewiki-20250106-cirrussearch-content.json.gz).
Then, move _simplewiki-20250106-cirrussearch-content.json.gz_ to _/data_  
and convert the Wikiepedia dump in JSON format to text by running the following script.
```
python json2text.py -source ../data/simplewiki-20250106-cirrussearch-content.json.gz -target ../data/wiki.txt -lower 1
```
Next, create the actual doculent vectors.
```
python make_wiki_model.py
```
Finally, run the model
```
python search_demo.py
```
## Image classification with fixed output random vectors   
The 5 datasets benchmarked are [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html), 
[LFW](https://www.kaggle.com/datasets/atulanandjha/lfwpeople), [EuroSat](https://github.com/phelber/eurosat), and [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).  

You can install the dependencies via pip or conda. To execute the benchmark one simply needs to enter:
```
python net.py 
```
Modify parameters accordingly before running net.py.

```
dataset_name = 'cifar100' # cifar100, cifar10, lfw, eurosat, oxford_flowers102

model_type = 'cnn' # cnn, resnet, cct

is_sdr = True # True is Random vector, False defaults to softmax classifier for benchmarking purposes  
```
   

    
**Benchmark results**.

![Benchmark](https://github.com/user-attachments/assets/3b2f35e0-0ece-4f62-88a6-16dca7f275ab)
 Notice how random vector representations performs noticebly better than softmax on the
 [Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/atulanandjha/lfwpeople) dataset.

