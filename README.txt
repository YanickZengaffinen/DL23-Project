This code is an attempt at reporducing the baseline of the paper:
https://openaccess.thecvf.com/content/CVPR2022/papers/Dong_Improving_Adversarially_Robust_Few-Shot_Image_Classification_With_Generalizable_Representations_CVPR_2022_paper.pdf

The aim behind this paper is to train a model that outputs feature embeddings that can generalize well
at infererence when facing unseen classes. It is also supposed to be robust because of the way it is trained.

Important takeaways from the paper:

There are 4 main components to the architecture:
    1. The main model (feature extractor) which outputs feature embeddings.
    2. A binary classifier module which using the feature embeddings as input, classifies if the input
        is from a real or adversarial example. The loss of this module is named Laa.
    3. A re-weighting module for adversarial examples. It aims to re-weight the loss of adversarial examples 
        fed through the feature extractor and classification head based on variables determined during adversarial
        generation (PGD). The loss of this modules is named Lar.
    4. A feature purifier module which aims to 'purify' feature embeddings of adversarial examples. 
        The loss of this module is named Lfp. 


Code info: 

- For the datasets I just followed the code of Yannick. Omniglot has 1100 classes for training. MiniImageNet has 64.

- To generate adversarial examples (for training), the paper uses PGD with eps=8/255, alpha=2/255 and steps = 7. 
    - To be able to retreive some variables for the adversarial re-weighting, I had to modify the PGD file. 
        Consequently, please make sure to use the my_pgd.py file. (It should be imported at the top).
        my_pgd.py needs the attack.py file as well so I put them in the same directory. 
    
- The def "train_model" is where everything happens. It's relatively simple, it takes as input
    which module to use for feature extraction, the size of the embeddings of the model output
    the train dataset, the train epochs, the number of classifiable classes for training, and 
    the learning rate for the optimizers.

    - After variable definitions, it generates adversarial examples of the batch.

    - Then the 3 losses are computed for each batch and it backpropagates.

    - Repeat for all batches. 

    - Next epoch. 

    Notes:
    1. All optimizers have the same parameters. This is because in the paper, they only specify 
        one optimizer's settings.

    2. For generation of adversarial examples, and for classifying the adversarial examples 
        for Lar, their is a need for a classification head. This head is a simple linear layer + softmax
        and it is concatenated to the feature extractor.

        For testing we could use this head or just add a new linear layer to the feature extractor.
        (The ouput of this function is the feature extractor only at the moment).

    3. Right now there is no validation implemented.

    4. Depending on how long it takes to train, I think it may be wise to reduce the number of epochs.
        The paper does 100 epochs on the 38400 images for training which could take considerable 
        amount of time.

    5. The architecture of the linear purifier can also be chnaged. Right now its just a single
        linear layer.

    6. I have not put code to use cuda in here yet. 

