# Improving DGA based Malicious Domain Classifiers for Malware Defense with Adversarial Machine Learning

I first reported on the successful application of a DGA-based malicious domain classifier using the Long Short-Term Memory (LSTM) method with a novel feature engineering technique and compared the model's performance with a previous model where our results have higher level of effectiveness. Second, I proposed a new method in using adversarial machine learning in order to generate new malware-related domain families and successfully be used against our highly trained model to show shortcomings of ML algorithms. Finally, I augmented the training dataset such that the retrained ML models are more effective in detecting never-before-seen malicious domain name variants with improved accuracy.

Contributions:
- I used a machine learning technique based on the Long Short-Term Memory (LSTM) model for automatic detection of malicious domains using a DGA classifier that analyses character by character over a massive labeled dataset
- To the best of my knowledge it is the first study to propose the generation of malicious domain names using a data perturbation approach in order to expand the training dataset with adversarial samples.
- I showed that, as expected, the LSTM model fails to recognize newly introduced adversarial samples in augmented training dataset.
- I used adversarial training to retrain the model with correct labelling of the adversarial samples in the training dataset to develop the generalization and solidity of the model.
- i showED that the augmented training dataset can help the LSTM model to detect, not only never-seen-before DGAs, but also new DGA families.
