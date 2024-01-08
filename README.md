# Adversarially Robust Few-Shot Image Classification: Exploring Implicit Model Uncertainty via Task Embeddings
TODO: here goes the abstract of the paper

![Forward path](uncertainty/figures/fewshotforwardpath.svg)

*Architecture of n-way classifier: Depicts the forward path when doing few-shot classification on support set $\mathcal{S}$ and query set $\mathcal{Q}$. $\hat{\tau}_{1}^\mathcal{S} \cdots \hat{\tau}_{n}^\mathcal{S}$ are the adapted task embeddings for each of the n binary one-vs-rest classifiers.*

## Setup
### VC v14+ runtime
The package learn2learn (currently) requires the VCv 14+ runtime library to be installed on your PC. If you install Visual Studio with workload "Desktop development with C++", you will have all the necessary packages installed.

### Conda
```
conda create --name dl python=3.9
conda activate dl

pip install -r requirements.txt
```

## Reproduce the Results
### Ours
#### Training
Note how the following commands are meant for Omniglot but you can change the scenario to miniimagenet. Also you want might want to change the task embedding size (taskembsize).
To train the binary classifiers you can run:
```
python uncertainty.py train-binary-classifiers --scenario omniglot --taskembsize 256
```
Next you need to compute the embeddings using either mean
```
python uncertainty.py train-meta-embedding --scenario omniglot --method mean --taskembsize 256 --modelfile "uncertainty/models/omniglot/best-binary-classifiers-256.pt"
```
or maml method:
```
python uncertainty.py train-meta-embedding --scenario omniglot --method maml --taskembsize 256 --modelfile "uncertainty/models/omniglot/best-binary-classifiers-256.pt"
```

#### Evaluations
Requires the models to be trained first!

To see how the binary classifiers perform directly after training you can run:
```
python uncertainty.py val-original --scenario omniglot --taskembsize 256 --modelfile "uncertainty/models/omniglot/best-binary-classifiers-256.pt"
```

To see how good the binary classifiers can be adapted to classes it has seen during training and new classes you can run:
```
python uncertainty.py val-binary-adaptability --scenario omniglot --taskembsize 256 --modelfile "uncertainty/models/omniglot/best-binary-classifiers-256.pt" --metaembfile "uncertainty/models/omniglot/best-meta-embedding-maml-256.pt" --shots 5 --adaptationsteps 5 --lr 0.75
```
Note that you can select the corresponding method (mean vs maml) by the meta-embedding you specify.

To measure how well the one-vs-rest classifiers can work together to build a n-way classifier, run:
```
python uncertainty.py val-fewshot --scenario omniglot --taskembsize 256 --modelfile "uncertainty/models/omniglot/best-binary-classifiers-256.pt" --metaembfile "uncertainty/models/omniglot/best-meta-embedding-maml-256.pt" --ways 5 --shots 5 --adaptationsteps 5 --lr 0.75 --iterations 100
```

To measure the effect of modelling uncertainty on adversarial accuracy and standard accuracy, run: 
```
python uncertainty.py val-defense --scenario omniglot --taskembsize 256 --modelfile "uncertainty/models/omniglot/best-binary-classifiers-256.pt" --metaembfile "models/omniglot/best-meta-embedding-maml-256.pt" --ways 5 --shots 5 --adaptationsteps 5 --lr 0.75 --noise 0.5 --samples 10 --pgdepsilon 0.0314 --pgdstepsize 0.008 --pgdsteps 7 --iterations 250
```
Note that you can run without defense if you choose noise = 0. Add the --noisepublic flag to make the noise known to the attacker.

To run the competitive evaluation with different attack strengths etc, run:
```
python evaluate.py --name "uncertainty"
```

### Adversarial Querying
#### Training
To train Adversarial Querying using MAML, run:
```
python adversarial_querying/aq.py
```
You can change the dataset {omniglot, mini-imagenet} and the number of ways {1, 5} in the same file.

#### Evaluations
**AQ (MAML):**  
To evaluate Adversarial Querying change the path to the folder containing the trained models in the file `aq_baseline.py` and run:
```
python evaluate.py --name "AQ"
```

**AQ (MAML) + Noise:** 
To evaluate Adversarial Querying with noise change the path to the folder containing the trained models in the file `aq_baseline.py` and run:
```
python evaluate.py --name "AQ" --add_noise True 
```

### Hypershot
#### Training
Training has been done entirely in Jupyter Notebooks. There are 2 of them:
resnet-hypershot-mini (for mini-image net)
resnet-hypershot-omni (for omniglot)
They can be run as any Jupyter notebook and they will store results in a folder called models at each epoch.

#### Evaluations
Evaluation of the Hypershot models are handled a bit separately due to some specificity of the data format used during training.
In order to run the experiments, you need to run the evaluate.py file inside the hypershot main folder. If you want to run another model than the ones we ran,
you will need to provide the according .pth file in the hypershot_baseline.py file. Also, for the omniglot dataset, you can comment in or out the
corresponding line as indicated in the file to run with or without adversarial training.


#### GR 

You can see the latest version of the code for our attempt at reproducing the GR baseline in the gen_baseline folder.

