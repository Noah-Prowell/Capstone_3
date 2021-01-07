# South Park GANs

Welcome to my capstone 3 for Galvanize data science immersive!  A little background on the project.  I first started this project as an idea to create a chatbot using South Park script data.  I wanted to be able to have you interact with the characters as if you were actually there in the town of South Park!  I quickly realized that this was way out of the scope of a week project simply for the reason of getting the computer to "understand language".  So I moved to generating script data!  

## What is a GAN?
  - A GAN is a Generative Adversarial Network.
  - They use Neural Network architechure to generate all kinds of things such as images and text!
  - GANs do not have labels or targets therefore they are a unsupervised machine learning technique.
  - Once trained on text data or images they are ableto generate something that could come close to being in the original dataset.

## GANs used in this project
  - I used two different GAN architectures to generate my text.
  - Both architectures were trained on the same individual character South Park script data
  - The first architecture I used was a simple LSTM neural network.
  - The second was a retrained simple-GPT2
  
## LSTM NN
   <p align="left">
    <img src="Screenshot from 2021-01-07 11-51-26.png" width='700'/>
    </p>
  - Pictured above is the architecture for my LSTM generative neural network
  - To get the neural network to actually produce text I used a text generation class that took the predicted encoded text and decoded it into abtual words
  
## Simple-GPT2
  - What is GPT2?
    - GPT2 is an open source text generative model that was made and trained by OPENAI.
  - Using a google collab notebook with some premade opensource code I was able to further train a GPT2 model on my South Park characters.

## So Which is Better?
  - To tackle this question I built a discriminator that used the generated text from both models and the real script data and trained to predict whether the text given was real or generated!
  - I used a Random Forest classifier because it is quite fast at trainging since it can work in parallel, and it is quite accurate without much need for tuning.
  - The GPT2 generated text was able to be distinguished roughly 60% of the time vs. the LSTM generated text was able to be distinguished 100% of the time.
  - This shows that the GPT2 model was able to generate more realistic text since the accuracy of the random forest classifier was lower than that of the LSTM model.  
