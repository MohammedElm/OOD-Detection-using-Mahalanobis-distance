# OOD-Detection-using-Mahalanobis-distance

In this project, we  reproduced  the  main  results  presented  in  [1].    Thus,  we  built  from  scratch (using PyTorch) the Mahalanobis-based score with Input pre-processing and feature ensembling as suggested in [1].  Also, we re-trained from scratch a ResNet34 neural network [3] on CIFAR-10 [5], SVNH [6] and CIFAR-100 [5] datasets and evaluated the performance of the Mahalanobis-based score on different In-Distribution (ID) /Out-of-Distribution (OOD) setups. We achieve very comparable results to [1]. We have also conducted an ablation study to understand the correlation between validation accuracy of the NN and the performance ofthe mahalanobis score OOD detector. We found that the more well-trained a NN, the best is the performance of the score.  We studied the contribution of each layer of the ResNet to the overall performance and concluded that, counter-intuitively, the first layers are not necessary bad predictors but are actually good calibrators to the overall performance. Finally, we extended this score beyond Images-classification tasks by applying it to text data using a pretrained BERT BASE[7] model. In these experiments, we achieved good baseline performances. These experiments could be the ground for a more text-adapted confidence score. Future work could also explore how gradient-based input preprocessing could be implemented in a language model, such as BERT.

References:
[1] Kimin Lee et al.A Simple Unified Framework for Detecting Out-of-Distribution Samples and AdversarialAttacks. 2018. arXiv:1807.03888 [stat.ML].
[2] Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich.Deep Anomaly Detection with Outlier Exposure.1952019. arXiv:1812.04606 [cs.LG].
[3] Kaiming He et al.Deep Residual Learning for Image Recognition. 2015. arXiv:1512.03385 [cs.CV].
[4] Apoorv Vyas et al. “Out-of-Distribution Detection Using an Ensemble of Self Supervised Leave-outClassifiers”. In: (Sept. 2018).
[5] Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. “CIFAR (Canadian Institute for Advanced Re-200search)”. In: ().url:http://www.cs.toronto.edu/~kriz/cifar.html.
[6] Yuval Netzer et al. “Reading Digits in Natural Images with Unsupervised Feature Learning”. In:NIPSWorkshop on Deep Learning and Unsupervised Feature Learning 2011. 2011.url:http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf.
[7] Jacob Devlin et al.BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.2052019. arXiv:1810.04805 [cs.CL].
[8] Andrew L. Maas et al. “Learning Word Vectors for Sentiment Analysis”. In:Proceedings of the 49thAnnual Meeting of the Association for Computational Linguistics: Human Language Technologies. Portland,Oregon, USA: Association for Computational Linguistics, 2011, pp. 142–150.url:http://www.aclweb.org/anthology/P11-1015.210
[9] Xiang Zhang, Junbo Zhao, and Yann LeCun. “Character-Level Convolutional Networks for Text Classifi-cation”. In:arXiv:1509.01626 [cs](Sept. 2015). arXiv:1509.01626 [cs].DD2412 - Advanced Deep Learning11