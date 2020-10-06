# Improving Deep Neural Networks

___
<br>
<br>


## Things to keep in mind

	1. Number of Layers
	2. Number of hidden units
	3. Learing rate
	4. Activation functions

	
As I build machine learngin application, keep in mind that it is a highly iterative process. Come up with an idea, code it, and experiment with it alot. 

<br>

### Train/Dev/Test Sets
-

	Depending on how much data I have think about having a train, dev and test
	set. Use a 60/20/20,  80/10/10...If not stick with 80/20 rule. 
	
	For deep learning (big data) 98/1/1 with 1 Mil. traning examples.
	
	Also make sure that my dev and test sets come from the same distribution.
	
	What if I don't have a test set. Not having a dev set might be ok if I don't 
	need a completely unbiased estimate of the performance of the algorith. 
<br>
<br>

### Bias - Variance
-

**Bias**

	Bias occurs when the model is undefitting the data. The predictions are far
	off from the model. When the bias is very high we can say that the model is
	very simple and does not fit the complexity of the data. In other words it
	pays little attention to the training data and leads to high error on
	training and testing sets.

**Variance**

	When we take a look at the validation set, this pretty much tells me how 
	scattered the data is from the actual values. High variance means that the
	model has trained with a lot of noise and is overfitting the data. Models
	with high variance do not generalize well on the data it has not seen and 
	perform well on the training set.

**Irreducible nois**
	
	Error that can't be reduced by creating good models. It measures the
	amount of noise present in the data. No matter how good out model is
	there will always be irrreducible error that can't be removed.
	

**Hoe does it affect the model**


	Both influence the outcome of the model. To improve bias/variance we can 
	get more data, add more features, increase the complexity or reduce the 
	regularization term.
	

**Bias/variance Trade-off**

	Explain..

**Mathematically**

	![image goes here]
































































	