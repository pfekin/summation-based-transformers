## Image classification with fixed output random vectors   
The 5 datasets benchmarked are [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html), 
[LFW](https://www.kaggle.com/datasets/atulanandjha/lfwpeople), [EuroSat](https://github.com/phelber/eurosat), and [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).  

You can install the dependencies via pip or conda. To execute the benchmark one simply needs to enter:
```
python net.py 
```
You might want to modify parameters accordingly before running net.py.

```
dataset_name = 'cifar100' # cifar100, cifar10, lfw, eurosat, oxford_flowers102

model_type = 'cnn' # cnn, resnet, cct

is_sdr = True # True is Random vector, False defaults to softmax classifier for benchmarking purposes  
```
   

    
**Benchmark results**.

![Benchmark](https://github.com/user-attachments/assets/3b2f35e0-0ece-4f62-88a6-16dca7f275ab)

## Random vector search 
Before creating the document vectors, it is necessary to download the 
[Simple English Wikipedia Cirrus JSON dump]( https://dumps.wikimedia.org/other/cirrussearch/20250106/simplewiki-20250106-cirrussearch-content.json.gz).
Then, move _simplewiki-20250106-cirrussearch-content.json.gz_ to _/data_  
and convert the Wikiepedia dump from JSON format to Unicode text by running the following script.
```
python json2text.py -source ../data/simplewiki-20250106-cirrussearch-content.json.gz -target ../data/wiki.txt -lower 1
```
Next, create the actual document vectors.
```
python make_wiki_model.py
```
Finally, run the model
```
python search_demo.py

Load model

Search: apple
[0.45746]   Apple (disambiguation)
[0.45796]   Apple TV
[0.49158]   Apple Maps
[0.52688]   Apple Corps
[0.54919]   Apple Watch
[0.56049]   Apple Records
[0.62137]   IBook
[0.62521]   Apple juice
[0.63462]   Adam's apple (disambiguation)
[0.65961]   Apple Inc.
[0.66889]   Apple A4
[0.67015]   The Apple Dumpling Gang
[0.67613]   Apple Intel transition
[0.67633]   Fiona Apple
[0.69011]   Apple pie
[0.70617]   Ronald Wayne
[0.71797]   Pond-apple
[0.72091]   MacOS
[0.72208]   FaceTime
[0.72422]   Baldwin apple

Search: marlon brando
[0.54094]   Anna Kashfi
[0.54247]   Movita Castaneda
[0.56732]   Marlon Brando
[0.58934]   Julius Caesar (movie)
[0.69778]   Marlon de Souza Lopes
[0.70550]   A Streetcar Named Desire
[0.70823]   On the Waterfront
[0.70896]   Marlon Wayans
[0.72329]   Candy (1968 movie)
[0.73589]   Last Tango in Paris
[0.73672]   The Love of Captain Brando
[0.73901]   Marlon Jackson
[0.74242]   Tim Brando
[0.76223]   The Wild One
[0.76231]   Dream Street (Janet Jackson album)
[0.76399]   Maria Schneider
[0.76581]   Mário Brandão da Silveira
[0.76853]   Superman Returns (movie)
[0.78955]   Miiko Taka
[0.79664]   Requiem for a Dream
```

