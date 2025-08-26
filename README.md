# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

To apply the same strategy in the new market, we need a neural network classification model that predicts the correct customer segment (A–D) for new customers based on their features.
## Neural Network Model Explanation
1. Architecture

The model is a fully connected feedforward neural network built using PyTorch. It consists of:

Input Layer

Number of neurons = number of input features in the dataset (e.g., age, income, spending score, etc.).

Accepts customer information as input.

Hidden Layer 1

A dense (Linear) layer with 64 neurons.

Activation: ReLU → helps the network learn non-linear relationships.

Hidden Layer 2

A dense (Linear) layer with 32 neurons.

Activation: ReLU.

Hidden Layer 3

A dense (Linear) layer with 16 neurons.

Activation: ReLU.

Hidden Layer 4

A dense (Linear) layer with 8 neurons.

Activation: ReLU.

Output Layer

A dense (Linear) layer with 4 neurons (since there are 4 customer classes: A, B, C, D).

Produces raw output scores (logits) for each class.

Flow of data:

Input → Linear(input → 64) → ReLU 
      → Linear(64 → 32) → ReLU 
      → Linear(32 → 16) → ReLU 
      → Linear(16 → 8) → ReLU 
      → Linear(8 → 4) → Output (logits)

2. Activation Functions

ReLU (Rectified Linear Unit) in all hidden layers:

ReLU(x)=max(0,x)

Introduces non-linearity.

Prevents vanishing gradients.

Softmax (at prediction time):

Although not included in the model’s forward method, PyTorch’s CrossEntropyLoss internally applies LogSoftmax.

During evaluation, softmax can be applied to obtain class probabilities:

probs = F.softmax(outputs, dim=1)

3. Loss Function

CrossEntropyLoss is used.

Standard choice for multi-class classification.

Compares predicted class logits with actual labels.

4. Optimizer

Adam optimizer is chosen.

Adaptive learning rate optimization algorithm.

Provides faster convergence and works well for classification tasks.

5. Training & Evaluation

### Training:

Forward pass → Compute loss → Backpropagation → Update weights.

Loss decreases over epochs, showing improved learning.

### Evaluation:

Predictions made on the test set.

Metrics used: Accuracy, Confusion Matrix, Classification Report.

## DESIGN STEPS
### STEP 1:

Import the required libraries for data handling and neural networks.

### STEP 2:

Load the dataset and explore its structure.

### STEP 3:

Clean the dataset and handle missing values if present.

### STEP 4:

Encode categorical variables into numerical format.

### STEP 5:

Normalize or scale the numerical features.

### STEP 6:

Split the dataset into training and testing sets.

### STEP 7:

Define the neural network architecture (64 → 32 → 16 → 8 → 4).

### STEP 8:

Select CrossEntropyLoss as the loss function and Adam as the optimizer.

### STEP 9:

Train the model using forward pass, loss calculation, backpropagation, and weight updates.

### STEP 10:

Evaluate the model using accuracy, confusion matrix, and classification report.
## PROGRAM

### Developed By: K MADHAVA REDDY
### Register Number: 212223240064

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=self.fc5(x)
        return x
```
```python
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')
```
```python
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

train_model(model,train_loader,criterion,optimizer,epochs=100)
```




## Dataset Information

<img width="989" height="662" alt="image" src="https://github.com/user-attachments/assets/a1511f04-f6f0-4112-8688-2e53b3178245" />

## OUTPUT



### Confusion Matrix

<img width="694" height="578" alt="image" src="https://github.com/user-attachments/assets/1f35a64a-f232-4e89-8357-cc2934185447" />

### Classification Report

<img width="598" height="430" alt="image" src="https://github.com/user-attachments/assets/95177d21-81f6-4112-bc94-4eda98f4379b" />


### New Sample Data Prediction

<img width="1022" height="339" alt="image" src="https://github.com/user-attachments/assets/8b2b221e-685c-4936-b80c-1f54b7024649" />

## RESULT:
The neural network classification model was successfully developed and trained to predict customer segments (A, B, C, D). The model achieved good accuracy on the test dataset, demonstrating its effectiveness in classifying new customers into the correct groups.

## RESULT
Include your result here
