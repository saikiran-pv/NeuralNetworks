# NeuralNetworks
 
BASIC CNN: 
 
Architecture: 
 
Convolution layers: 
 
      nn.Conv2d(1,32,3,padding=1),       
      nn.ReLU(),       
      nn.MaxPool2d(kernel_size=2,stride=2),       
      nn.Conv2d(32,64,3, padding=1),       
      nn.ReLU(),       
      nn.MaxPool2d(kernel_size=2,stride=2),       
      nn.Conv2d(64,64,3, padding=1),       
      nn.ReLU(), 
      nn.MaxPool2d(kernel_size=2,stride=2), 
 
Fully connected layers: 
 
        nn.Linear(64 * 8 * 8, 128),         
        nn.ReLU(),         
        nn.Linear(128, 128),         
        nn.ReLU(),         
        nn.Linear(128, n),         
        nn.Sigmoid(), 
        nn.ReLU(), 
 
 
I have used 3 convolution layers and sigmoid and ReLU activation functions. 
 
The first convolution network takes 1 input (grey scale image) and maps it to 32 feature maps. And the second layer takes the 32 feature maps as input and maps to 64 feature maps. The third layer maps these 64 to another 64 features. 
 
We apply ReLU activations after each layer. 
 
After the convolution layers I am flattening to give it as input to the fully connected layers. 
I have used 3 fully connected linear layers. 
 
I have applied the ReLU activation after each fully connected linear layer. 
 
Loss and Accuracy: 
 
Training loss for basic cnn: 1.9016 Validation loss for basic cnn: 1.9321 
 
Test loss for basic cnn: 1.95574 Accuracy for basic cnn: 0. 38350 
 
 
 
 
ALL CNN: 
 
Architecture: 
 
    self.cl1 = nn.Conv2d(1,64,3,padding=1)     
    self.cl2 = nn.Conv2d(64,128,3, padding=1)     
    self.cl3 = nn.Conv2d(128, 128, 3)     
    self.cl4 = nn.Conv2d(128, 192, 3)     
    self.cl5 = nn.Conv2d(192, 192, 3)     
    self.cl6 = nn.Conv2d(192, 192, 3)     
    self.cl7 = nn.Conv2d(192, 192, 3)     
    self.cl8 = nn.Conv2d(192, 10, 3) 
    
   def forward(self, x): 
      x = F.relu(self.cl1(x))     
      x = F.relu(self.cl2(x))     
      x = F.relu(self.cl3(x))     
      x = F.relu(self.cl4(x))     
      x = F.relu(self.cl5(x))     
      x = F.relu(self.cl6(x))     
      x = F.relu(self.cl7(x))     
      x = F.relu(self.cl8(x))     
      x = F.max_pool2d(x, x.size()[2:])     
      x = x.view(x.size(0), -1)     
      x = F.log_softmax(x, dim=1) 
 
 
for all CNN  
I have used 8 convolution networks with the relu activations. 
 
•	After each convolution layer relu activation is applied. 
•	The first layer takes the input 1 and maps it to 64 features. And this count increases progressively. 
 
I have used the average pool to do the global average and followed by it I have used the SoftMax activation function. 
 
Loss and Accuracy: 
 
Training loss: 1.9112 
Validation loss: 1.8596 
Test loss: 1.9317 Accuracy: 0.337485 
 
From basic cnn and the all cnn accuracy and loss results we can say that they are close to each other. But in my case the basic cnn performed well compared to the all cnn.  
 
Basic cnn got accuracy around 38% where as the all cnn accuracy is around 33% 
 
Basic CNN with Regularization: 
 
I have used the basic cnn to apply the regularization. I have used the dropout functionality to do regularization. 
 
I have used the dropout value of 0.4 after each fully connected linear layer. 
 
Loss and Accuracy: 
 
Training loss: 1.9578 
Validation loss: 1.9253 
Test loss: 1.954804 Accuracy: 0.377225 
 
Comparison: 
 
The accuracy is decreased a little after applying the regularization to the basic cnn. It was 38% initially and got reduced to 37.7% after adding the regularization. 
 
There is no big difference in the accuracy. 
 
The Loss values are almost similar. 
 
 
 
 
CIFAR FROM SCRATCH: 
 
I have used the basic cnn which is the first model to train the cifar from the scratch. 
 
Loss and Accuracy: 
 
Training loss: 1.9533 Validation loss: 1.9490 
 
Test loss : 1.9570034742355347 
Accuracy: 0.3386 
 
Train and Validation Losses: 
 
  <img width="288" alt="image" src="https://github.com/user-attachments/assets/c5b8c79c-4659-4fae-b28e-289f5717dbb4" />

 
CIFAR USING PRETRAINED MODEL: 
 
I have loaded the pretrained the basic cnn model from the checkpoint and tuned it using the cifar data using the TuneBasicCNN class. 
 
 
Loss and Accuracy: 
 
Training loss:: 1.9528 
Validation los: 1.9556 
 
Test loss: 1.9584 Accuracy: 0.3297 
 
Training and Validation Losses plots: 
 
  <img width="288" alt="image" src="https://github.com/user-attachments/assets/d99025f2-f8ed-4286-bbb7-73e71916b74d" />

 
 
 
