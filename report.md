### Training Strategy

We use a custom training strategy that combines cutmix and cosine annealing.

We applied Label Smoothing (0.1) and Exponential Moving Average (EMA) with decay $\beta$ to improve generalization and stabilize training. EMA maintains a smoothed version of model weights, reducing overfitting.


### Data Augmentation

We started with a basic data augmentation strategy like random horizontal flip, random crop, color jitter, random rotation and an AutoAugment with CIFAR10 policy.

Then we applied several advanced augmentation strategies to further enhance the model's robustness and generalization performance, as follows:

Random Augmentation

RandomCutout:

We applied RandomCutout when building the training data loader.

In this process, we randomly remove a rectangular region of each image, with the size ranging from 8 to 16 pixels. By learning to recognize objects despite missing information, the model becomes more resilient to occlusions and improves its generalization performance on unseen data.

Cutmix:

We applied CutMix augmentation starting after 300 epochs to further enhance generalization during training. We gradually increase the CutMix probability from 0 to 0.5, following the pace of the cosine annealing scheduler restart proxy.

During training, for each batch, we randomly permute the images to create paired samples. For each pair, we cut a random region from one image and paste it onto another. The corresponding labels are combined proportionally based on the area of the patch.

This encourages the model to learn discriminative features from multiple images simultaneously, making it more robust and less prone to overfitting. By forcing the model to predict soft labels for mixed samples, CutMix effectively regularizes the training and improves generalization to unseen data.




We used a learning rate of 0.001 and a batch size of 128.

We used a cosine annealing scheduler with a restart of 100 epochs.



We 