#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import re
import cv2
from PIL import Image
import time
import json
from keras.models import Model , Sequential , load_model
from keras.layers import *
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50 , preprocess_input , decode_predictions


# In[2]:


def readTextFile(file_path):
    with open(file_path) as f:
        return f.read()

doc = readTextFile("/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt")

lines = doc.split("\n")[:-1]    # since last line is an empty string
print(len(lines))


# In[3]:


lines[0]


# In[4]:


descriptions = {}

for line in lines:
    
    tokens = line.split('\t')
    
    # take the first token as image id, the rest as caption
    img_name , caption = tokens[0] , tokens[1]
    img_name = img_name.split('.')[0]
    
    # if image id is not present in dictionary then initialize it with empty list
    if descriptions.get(img_name) is None:
        descriptions[img_name] = []
        
    descriptions[img_name].append(caption)


# In[5]:


descriptions['1000268201_693b08cb0e']


# In[6]:


len(descriptions)   # each image id has corresponding 5 captions


# In[8]:


img_path = "/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Images/"
img = cv2.imread(img_path+'1000268201_693b08cb0e'+'.jpg')
img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis('off')
plt.show()


# In[7]:


def clean_text(sentence):
    
    # convert to lower case
    sentence = sentence.lower()
    
    # remove punctuation and numbers from each token
    sentence = re.sub('[^a-z]+' , ' ' , sentence)
    
    # remove single characters like 's' and 'a'
    sentence = [word for word in sentence.split() if len(word)>1]
    sentence = " ".join(sentence)
    
    return sentence


for img_id , caption_list in descriptions.items():
    for caption in range(len(caption_list)):
        caption_list[caption] = clean_text(caption_list[caption])

descriptions["1000268201_693b08cb0e"]


# In[8]:


with open("descriptions.txt" , "w") as f:
    f.write(str(descriptions))


# In[9]:


with open("descriptions.txt" , "r") as f:
    descriptions = f.read()

type(descriptions)


# In[10]:


json_acceptable_str = descriptions.replace("'" , "\"")

# loads() method is used to parse a valid JSON string and convert it into a Python Dictionary
descriptions = json.loads(json_acceptable_str)
type(descriptions)


# In[11]:


descriptions['1000268201_693b08cb0e']


# In[12]:


#Create a vocab


# In[13]:


total_words = []

for key, caption_list in descriptions.items():
    [total_words.append(word) for caption in caption_list for word in caption.split()]

print("Total no of words across all image captions: %d" %(len(total_words)))


# In[14]:


unique_words = set()

for key in descriptions.keys():
    [unique_words.update(caption.split()) for caption in descriptions[key]]

print("Total no of unique words across all image captions: %d" %(len(unique_words))) 


# In[15]:


# Consider only words which occur at least 10 times in the corpus

word_count_threshold = 10
freq_counts = {}

for word in total_words:
    freq_counts[word] = freq_counts.get(word , 0) + 1 

print(len(freq_counts.keys()))


# In[16]:


# Sort the dictionary according to the frequency count

sorted_freq_count = sorted(freq_counts.items() , reverse = True , key = lambda x:x[1])


# In[17]:


print(sorted_freq_count[:10])


# In[18]:


sorted_freq_count = [x for x in sorted_freq_count if x[1] > word_count_threshold]

vocabulary = [x[0] for x in sorted_freq_count]
print("Final Vocab Size: %d" %len(vocabulary))


# In[19]:


train_file_data = readTextFile("/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt")
test_file_data = readTextFile("/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt")


# In[20]:


train_id = [id.split(".")[0] for id in train_file_data.split("\n")[:-1]]


# In[21]:


print("No of Image Id's in Train Data: %d" %len(train_id))


# In[22]:


print("No of Image Id's in Train Data: %d" %len(train_id))


# In[23]:


test_file_data[-1]


# In[24]:


test_id = [id.split(".")[0] for id in test_file_data.split("\n")[:-1]]
print("No of Image Id's in Test Data: %d" %len(test_id))


# In[25]:


# Prepare descriptions for the training data
# Tweak - Add 'start' and 'end' token to our training data
train_descriptions = {}

for img_id in train_id:
    train_descriptions[img_id] = []
    
    for caption in descriptions[img_id]:
        caption_to_append = "startseq "+caption+" endseq"
        train_descriptions[img_id].append(caption_to_append)

print("Train Descriptions: %d" %len(train_descriptions))    


# In[26]:


train_descriptions = {}

for img_id in train_id:
    train_descriptions[img_id] = descriptions.get(img_id)

for key , caption_list in train_descriptions.items():
    for caption in range(len(caption_list)):
        caption_list[caption] = 'startseq ' + caption_list[caption] + ' endseq' 


# In[27]:


train_descriptions['1000268201_693b08cb0e']


# In[28]:


from transformers import ViTImageProcessor, ViTModel


# In[29]:


# Load the ViT processor and model
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# Define Image Path
IMG_PATH = '/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Images/'


# In[30]:


# Function to preprocess images for ViT
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")  # Ensure RGB format
    inputs = image_processor(images=image, return_tensors="pt")  # Convert to tensor
    return inputs

# Function to encode image features using ViT
def encode_image(img_path):
    inputs = preprocess_image(img_path)
    with torch.no_grad():  # No gradient calculation needed
        outputs = model(**inputs)  
    feature_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Extract CLS token features
    return feature_vector


# In[32]:


# Extract Features for Training Images
encoded_train_data = {}
start = time.time()

for ix, img_id in enumerate(train_id):
    img_path = IMG_PATH + img_id + ".jpg"
    encoded_train_data[img_id] = encode_image(img_path)
    
    if ix % 500 == 0:
        print(f"Encoding in Progress Step {ix}")

end = time.time()
print("Time Elapsed: ", end - start)


# In[33]:


with open("encoded_train_img_features_vit.pkl" , "wb") as file:
    pickle.dump(encoded_train_data , file)


# In[46]:


train_features = pickle.load(open("encoded_train_img_features_vit.pkl", "rb"))
print('Photos: train=%d' % len(train_features))


# In[32]:


encoded_test_data = {}
start = time.time()

for ix, img_id in enumerate(test_id):
    img_path = IMG_PATH + img_id + ".jpg"
    encoded_test_data[img_id] = encode_image(img_path)
    
    if ix%500 == 0:
        print(f"Encoding in Progress Step {ix}")
    
end = time.time()
print("Time Elapsed: " , end-start) 


# In[33]:


with open("encoded_test_img_features_vit.pkl" , "wb") as file:
    pickle.dump(encoded_test_data , file)


# In[34]:


with open("encoded_test_img_features_vit.pkl" , "rb") as f:
    encoded_test_data = pickle.load(f)


# In[35]:


len(vocabulary)


# In[33]:


word_to_idx = {}
idx_to_word = {}

for i, word in enumerate(vocabulary):
    word_to_idx[word] = i+1
    idx_to_word[i+1] = word


# In[34]:


print(word_to_idx['dog'])
print(idx_to_word[6])


# In[35]:


word_to_idx['startseq'] = 1846
idx_to_word[1846] = 'startseq'

word_to_idx['endseq'] = 1847
idx_to_word[1847] = 'endseq'


# vocab size is len+1 because we will append 0's as well  ;  0th index will be reserved for zero padding
vocab_size = len(word_to_idx) + 1             
print("Final Vocab Size: %d" %vocab_size)


# In[36]:


max_len = 0

for key in descriptions.keys():
    for caption in descriptions[key]:
        max_len = max(max_len , len(caption.split()))
        
print('Max Description Length: %d' %max_len)


# In[40]:


# train_descriptions : dictionary with maps image id with its captions for training data
# encoding_train_data : how particular image is mapped to 2048 dimensional feature vector
# word_to_idx : how to convert any given word to index in vocabulary
# max_len : max_len for any sequence in training data
# batch size : how many training examples we want to include in a particular batch

def data_generator(train_descriptions , encoded_train_data , word_to_idx , max_len , batch_size):
    
    """X1.append(image vector) , X2.append(partial-captions-input_seq) , y.append(output_seq)"""
    
    x1 , x2 , y = [] , [] , []
    n = 0
    
    while True:
        for key, caption_list in train_descriptions.items():
            n += 1
            
             # retrieve the photo feature
            photo = encoded_train_data[key]
            
            for caption in caption_list:
                # encode the sequence
                seq = [word_to_idx[word] for word in caption.split() if word in word_to_idx]
                
                for i in range(1 , len(seq)):
                    # split into input and output pair
                    input_seq , output_seq = seq[0:i] , seq[i]
                    
                    # zero pad input sequence - # accepts a 2-D list and returns a 2-D matrix
                    input_seq = pad_sequences([input_seq], maxlen=max_len, value = 0, padding = 'post')[0] 
                    
                    # encode output sequence
                    output_seq = to_categorical([output_seq] , num_classes =  vocab_size)[0]
                    
                    # append training point one by one
                    x1.append(photo)      # 2048 dim
                    x2.append(input_seq)  # 33 dim
                    y.append(output_seq)  # one hot vector of size vocab_size = 1848
                    
            # yield the batch data - generator remembers the state where the function was in previous call        
            if n == batch_size:
                
                yield ((np.array(x1) , np.array(x2)) , np.array(y))   # generator yields a tuple
                
                # for next function call, when the control comes back again to this generator function, x1, x2 and y 
                # will be initialized with empty list because we do not want to add examples of the prev batch           
                x1 , x2 , y = [] , [] , [] 
                n = 0


# In[41]:


file = open("/Users/prayashdas/Downloads/glove/glove.6B.50d.txt" , encoding = "utf8")


# In[42]:


for line in file:
    print(line)
    break


# In[43]:


embeddings_index = {}   # embeddings for 6 billion words

for line in file:
    values = line.split()
    word = values[0]
    word_embedding = np.array(values[1:] , dtype = 'float')
    embeddings_index[word] = word_embedding
    
file.close()


# In[44]:


def get_embedding_matrix():
    emb_dim = 50
    embedding_matrix = np.zeros((vocab_size , emb_dim))
    
    for idx, word in idx_to_word.items():
        embedding_vector = embeddings_index.get(word)
        
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
            
    return embedding_matrix


# In[45]:


embeddings_matrix = get_embedding_matrix()
embeddings_matrix.shape


# In[63]:


# Define captioning parameters
vocab_size = 1848  # Adjust based on actual vocab size
max_len = 33  # Maximum caption length

# Image feature extractor input (from ViT)
input_img_features = Input(shape=(768,))  # ViT CLS token output is 768-d
fe1 = Dropout(0.3)(input_img_features)
fe2 = Dense(256, activation='relu')(fe1)  # Reduce dimensionality

# Caption sequence model
input_captions = Input(shape=(max_len,))
se1 = Embedding(vocab_size, 50, mask_zero=True)(input_captions)  # Word embedding (50-d)
se2 = Dropout(0.3)(se1)
se3 = LSTM(256)(se2)  # LSTM extracts text features (256-d)

# Decoder combining Image & Text Features
decoder1 = add([fe2, se3])  # Merge Image & Text embeddings
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)  # Predict next word in sequence

# Define the Model
model = Model(inputs=[input_img_features, input_captions], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Summary of the Model
model.summary()


# In[64]:


model.layers[2].set_weights([embeddings_matrix])
model.layers[2].trainable = False


# In[65]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')


# In[66]:


len(train_descriptions)


# In[48]:


epochs = 10
batch_size = 3
steps_per_epoch = len(train_descriptions)//batch_size


# In[68]:


for i in range(epochs):
    generator = data_generator(train_descriptions , encoded_train_data , word_to_idx , max_len , batch_size)
    model.fit(generator , steps_per_epoch = steps_per_epoch , epochs = 1)


# In[69]:


j = 11
for i in range(epochs):
    generator = data_generator(train_descriptions , encoded_train_data , word_to_idx , max_len , batch_size)
    model.fit(generator , steps_per_epoch = steps_per_epoch , epochs = 1)
    model.save('model_weights_vit'+str(j)+'.h5')
    j += 1


# In[70]:


model.optimizer.lr = 0.0001
epochs = 5
batch_size = 6
steps_per_epoch = len(train_descriptions)//batch_size


# In[71]:


for i in range(epochs):
    generator = data_generator(train_descriptions , encoded_train_data , word_to_idx , max_len , batch_size)
    model.fit(generator , steps_per_epoch = steps_per_epoch , epochs = 1)
    model.save('model_weights_vit'+str(j)+'.h5')
    j += 1


# In[78]:


from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, add
import numpy as np

# Define captioning parameters
vocab_size = 1848  # Adjust based on actual vocab size
max_len = 33  # Maximum caption length

# Image feature extractor input (from ViT)
input_img_features = Input(shape=(768,))  # ViT CLS token output is 768-d
fe1 = Dropout(0.3)(input_img_features)
fe2 = Dense(256, activation='relu')(fe1)  # Reduce dimensionality

# Caption sequence model
input_captions = Input(shape=(max_len,))
se1 = Embedding(vocab_size, 50, mask_zero=True)(input_captions)  # Word embedding (50-d)
se2 = Dropout(0.3)(se1)
se3 = LSTM(256)(se2)  # LSTM extracts text features (256-d)

# Decoder combining Image & Text Features
decoder1 = add([fe2, se3])  # Merge Image & Text embeddings
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)  # Predict next word in sequence

# Define the Model
model = Model(inputs=[input_img_features, input_captions], outputs=outputs)

# Load the trained weights
model.load_weights('model_weights_vit25.h5')

print("Model loaded successfully!")


# In[51]:


def predict_caption_using_greedySearch(photo):
    
    inp_text = 'startseq'
    for i in range(max_len):
        sequence = [word_to_idx[word] for word in inp_text.split() if word in word_to_idx]
        sequence = pad_sequences([sequence] , maxlen = max_len , padding = 'post')
        
        #pred_label = model.predict([photo , sequence])
        photo_features = torch.tensor(photo).unsqueeze(0)  # Convert NumPy array to tensor and add batch dimension
        sequence = torch.tensor(sequence) # Convert sequence to tensor

        pred_label = model.predict([photo , sequence])

            
        #pred_label = pred_label.numpy()  # Convert back to NumPy
        pred_label = pred_label.argmax()      # Greedy Sampling : Word with max probability always
        pred_word = idx_to_word[pred_label]   # retreiving the word
    
        inp_text += " " + pred_word    # adding it to the sequence

        # if <e>/end sequence is encountered
        if pred_word == "endseq":
            break

    final_caption = inp_text.split(' ')[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption



# In[52]:


from keras.models import load_model

# Define the model architecture (same as the one used for training)
model.load_weights('model_weights_vit25.h5')  # Load trained weights
print("Model loaded successfully!")


# In[53]:


IMG_PATH = '/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Images/'

for i in range(15):
    idx = np.random.randint(0, 1000)
    
    all_test_images = list(encoded_test_data.keys())
    test_img_id = all_test_images[idx]
    test_img_vec = encoded_test_data[test_img_id].reshape((1, 768))  # batch size x feature vector
  
    output_caption = predict_caption_using_greedySearch(test_img_vec)

    img = plt.imread(IMG_PATH + test_img_id + ".jpg")
    plt.imshow(img)
    plt.title("Predicted Caption: " + output_caption)
    plt.axis('off')
    plt.show()


# In[66]:


import numpy as np
import matplotlib.pyplot as plt
import json

IMG_PATH = '/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Images/'
generated_captions = {}  # Dictionary to store generated captions

for i in range(15):  # Generate captions for 15 random images
    idx = np.random.randint(0, 1000)
    
    all_test_images = list(encoded_test_data.keys())
    test_img_id = all_test_images[idx]
    test_img_vec = encoded_test_data[test_img_id].reshape((1, 768))  # batch size x feature vector
  
    output_caption = predict_caption_using_greedySearch(test_img_vec)

    # Store the generated caption
    generated_captions[test_img_id] = output_caption

    # Display the image with the predicted caption
    img = plt.imread(IMG_PATH + test_img_id + ".jpg")
    plt.imshow(img)
    plt.title("Predicted Caption: " + output_caption)
    plt.axis('off')
    plt.show()

# Save the generated captions to a JSON file
with open("generated_captions.json", "w") as file:
    json.dump(generated_captions, file, indent=4)

print("Generated captions saved to 'generated_captions.json' successfully!")


# In[54]:


test_descriptions = {}

for img_id in test_id:  # test_id contains image IDs of test images
    test_descriptions[img_id] = descriptions.get(img_id, [])

print(f"Test Descriptions: {len(test_descriptions)} images")


# In[55]:


sample_img_id = list(test_descriptions.keys())[0]
print(f"Image ID: {sample_img_id}")
print(f"Captions: {test_descriptions[sample_img_id]}")


# In[56]:


from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import numpy as np


# In[57]:


import random
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

def evaluate_bleu_subset(test_descriptions, encoded_test_data, subset_size=100):
    bleu1, bleu2, bleu3, bleu4 = [], [], [], []

    # Select a random subset of test images
    subset_images = random.sample(list(test_descriptions.keys()), min(subset_size, len(test_descriptions)))

    for img_id in subset_images:
        reference_captions = [caption.split() for caption in test_descriptions[img_id]]  # Tokenized references
        test_img_vec = encoded_test_data[img_id].reshape((1, 768))  # Reshape for model input
        
        predicted_caption = predict_caption_using_greedySearch(test_img_vec).split()  # Generate caption & tokenize

        # Compute BLEU scores
        bleu1.append(sentence_bleu(reference_captions, predicted_caption, weights=(1, 0, 0, 0)))
        bleu2.append(sentence_bleu(reference_captions, predicted_caption, weights=(0.5, 0.5, 0, 0)))
        bleu3.append(sentence_bleu(reference_captions, predicted_caption, weights=(0.33, 0.33, 0.33, 0)))
        bleu4.append(sentence_bleu(reference_captions, predicted_caption, weights=(0.25, 0.25, 0.25, 0.25)))

    print(f"Evaluating on a subset of {len(subset_images)} images")
    print(f"BLEU-1: {np.mean(bleu1):.4f}")
    print(f"BLEU-2: {np.mean(bleu2):.4f}")
    print(f"BLEU-3: {np.mean(bleu3):.4f}")
    print(f"BLEU-4: {np.mean(bleu4):.4f}")


# In[59]:


evaluate_bleu_subset(test_descriptions, encoded_test_data, subset_size=100)  # Change subset_size as needed


# In[60]:


#Meteor


# In[67]:


with open("generated_captions.json", "r") as file:
    generated_captions = json.load(file)

print("Loaded generated captions:", generated_captions)


# In[71]:


import json

with open("generated_captions.json", "r") as file:
    generated_captions = json.load(file)

if not generated_captions:
    print("Error: No generated captions found.")


# In[72]:


print(f"Total reference images: {len(reference_dict)}")
print(f"Sample reference captions: {list(reference_dict.items())[:5]}")


# In[73]:


import json

with open("generated_captions.json", "r") as file:
    generated_captions = json.load(file)

print(f"Total generated captions: {len(generated_captions)}")
print(f"Sample generated captions: {list(generated_captions.items())[:5]}")


# In[74]:


# Fix generated caption keys by appending ".jpg"
generated_captions = {k + ".jpg": v for k, v in generated_captions.items()}

# Find matching keys
common_keys = set(reference_dict.keys()) & set(generated_captions.keys())

print(f"Total matching images after fix: {len(common_keys)}")  # Should be 15


# In[76]:


import nltk
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

scores = {}

for image in common_keys:  # Only use matching images
    references = [word_tokenize(ref) for ref in reference_dict[image]]
    hypothesis_tokens = word_tokenize(generated_captions[image])

    # Compute METEOR Score
    score = meteor_score(references, hypothesis_tokens)
    scores[image] = score

# Prevent ZeroDivisionError
if len(scores) == 0:
    print("No valid METEOR scores computed. Check filenames and data.")
else:
    avg_meteor = sum(scores.values()) / len(scores)
    print(f"Average METEOR Score: {avg_meteor:.4f}")


# In[35]:


#Attention Mechanism


# In[50]:


# Load ViT processor and model
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs

def encode_image_spatial(img_path):
    inputs = preprocess_image(img_path)
    with torch.no_grad():
        outputs = vit_model(**inputs)
    # outputs.last_hidden_state has shape (1, 197, 768); remove CLS token (index 0)
    patch_features = outputs.last_hidden_state[:, 1:, :]  # shape: (1, 196, 768)
    return patch_features.squeeze(0).numpy()  # shape: (196, 768)

# Example usage:
IMG_PATH = '/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Images/'
sample_img_path = IMG_PATH + '1000268201_693b08cb0e.jpg'
sample_patch_features = encode_image_spatial(sample_img_path)
print("Sample patch features shape:", sample_patch_features.shape)  # Expected: (196, 768)


# In[51]:


import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Concatenate, RepeatVector, Activation, Lambda, add
from keras.models import Model

# Parameters
vocab_size = 1848   # Adjust as needed
max_len = 35        # Maximum caption length
num_patches = 196   # Number of image patches (from ViT)
feature_dim = 768   # Dimensionality of each patch's feature
attn_units = 256    # Units for intermediate attention layer

# --- Image Feature Input (Spatial) ---
input_img_features = Input(shape=(num_patches, feature_dim))  # (batch, 196, 768)
img_dropout = Dropout(0.3)(input_img_features)
# Reduce dimensionality of each patch to 256
img_features = Dense(256, activation='relu')(img_dropout)       # (batch, 196, 256)

# --- Caption Input ---
input_captions = Input(shape=(max_len,))  # (batch, 33)
embeddings = Embedding(vocab_size, 50, mask_zero=True)(input_captions)  # (batch, 33, 50)
cap_dropout = Dropout(0.3)(embeddings)
# Use LSTM with return_state=True to get hidden state for attention
lstm_out, state_h, state_c = LSTM(256, return_state=True)(cap_dropout)  # lstm_out: (batch, 256)

# --- Attention Mechanism ---
# Repeat the decoder's hidden state to align with image patches
hidden_with_time = RepeatVector(num_patches)(state_h)  # (batch, 196, 256)
# Concatenate image features and hidden state along the feature dimension
attn_input = Concatenate(axis=-1)([img_features, hidden_with_time])  # (batch, 196, 512)
# Compute intermediate attention scores
attn_dense = Dense(attn_units, activation='tanh')(attn_input)  # (batch, 196, attn_units)
attn_scores = Dense(1)(attn_dense)  # (batch, 196, 1)
attn_scores = Lambda(lambda x: tf.squeeze(x, axis=-1))(attn_scores)  # (batch, 196)
# Softmax to get attention weights
attn_weights = Activation('softmax', name='attn_weights')(attn_scores)  # (batch, 196)

# Compute context vector as weighted sum of image features
context = Lambda(lambda x: tf.reduce_sum(x[0] * tf.expand_dims(x[1], -1), axis=1))([img_features, attn_weights])
# context has shape: (batch, 256)

# --- Decoder ---
# Combine context with the LSTM output
decoder_input = add([context, lstm_out])  # (batch, 256)
decoder_dense = Dense(256, activation='relu')(decoder_input)
outputs = Dense(vocab_size, activation='softmax')(decoder_dense)  # final word prediction

# Define the model: output both predictions and attention weights
model = Model(inputs=[input_img_features, input_captions], outputs=[outputs, attn_weights])
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()


# In[38]:


import pickle
import time
import numpy as np

# Assuming IMG_PATH and train_id are already defined.
# And that encode_image_spatial() is defined to return spatial features (e.g., shape: (196, 768))
encoded_train_data_spatial = {}
start = time.time()

for ix, img_id in enumerate(train_id):
    img_path = IMG_PATH + img_id + ".jpg"
    # Extract spatial features from the image (excluding the CLS token)
    encoded_train_data_spatial[img_id] = encode_image_spatial(img_path)  # Expected shape: (num_patches, feature_dim)
    
    if ix % 500 == 0:
        print(f"Encoding in progress, step {ix}")

end = time.time()
print("Time Elapsed for spatial feature extraction:", end - start)

# Save the spatial features to a pickle file
with open("encoded_train_img_features_spatial.pkl", "wb") as file:
    pickle.dump(encoded_train_data_spatial, file)

print("Spatial training features saved successfully!")


# In[52]:


model.compile(
    loss=['categorical_crossentropy', lambda y_true, y_pred: 0.0],
    loss_weights=[1.0, 0.0],
    optimizer='adam'
)


# In[53]:


import itertools
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def spatial_data_generator(train_descriptions, encoded_train_data, word_to_idx, max_len, batch_size):
    """
    Yields batches:
      x1: image spatial features with shape (batch, num_patches, feature_dim)
      x2: partial caption sequences with shape (batch, max_len)
      y: a tuple containing:
         - y_word: one-hot encoded next word (batch, vocab_size)
         - dummy_attn: dummy zeros with shape (batch, num_patches)
    """
    x1, x2, y_word = [], [], []
    n = 0
    # Use itertools.cycle to loop indefinitely over all training items
    for key, caption_list in itertools.cycle(train_descriptions.items()):
        n += 1
        # Get the spatial features for the image (expected shape: (num_patches, feature_dim))
        photo = encoded_train_data[key]
        for caption in caption_list:
            # Convert caption to sequence of word indices
            seq = [word_to_idx[word] for word in caption.split() if word in word_to_idx]
            # Only process if the sequence has at least 2 tokens
            if len(seq) > 1:
                for i in range(1, len(seq)):
                    # Create the partial input sequence and pad/truncate to max_len
                    input_seq = pad_sequences([seq[:i]], maxlen=max_len, padding='post')[0]
                    # One-hot encode the next word
                    output_seq = to_categorical([seq[i]], num_classes=vocab_size)[0]
                    x1.append(photo)       # photo shape: (num_patches, feature_dim)
                    x2.append(input_seq)   # shape: (max_len,)
                    y_word.append(output_seq)  # shape: (vocab_size,)
        if n == batch_size:
            # Convert lists to NumPy arrays and then to Tensors
            x1_arr = tf.convert_to_tensor(np.array(x1, dtype=np.float32))
            x2_arr = tf.convert_to_tensor(np.array(x2, dtype=np.int32))
            y_word_arr = tf.convert_to_tensor(np.array(y_word, dtype=np.float32))
            # Create dummy attention labels: zeros with shape (batch_size, num_patches)
            dummy_attn = tf.zeros((y_word_arr.shape[0], num_patches), dtype=tf.float32)
            yield ((x1_arr, x2_arr), (y_word_arr, dummy_attn))
            x1, x2, y_word = [], [], []
            n = 0


# In[57]:


import pickle

with open("encoded_train_img_features_spatial.pkl", "rb") as f:
    encoded_train_data_spatial = pickle.load(f)

print("Spatial training features loaded:", len(encoded_train_data_spatial))


# In[54]:


output_signature = (
    (
        tf.TensorSpec(shape=(None, num_patches, feature_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None, max_len), dtype=tf.int32)
    ),
    (
        tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32),
        tf.TensorSpec(shape=(None, num_patches), dtype=tf.float32)
    )
)

def gen():
    return spatial_data_generator(train_descriptions, encoded_train_data_spatial, word_to_idx, max_len, batch_size)

dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


# In[68]:


model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=1)


# In[70]:


additional_epochs = 9
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint_{epoch}.weights.h5',
    save_weights_only=True,
    save_best_only=True,
    monitor='loss',  # or 'val_loss'
    mode='min'
)


model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=additional_epochs, callbacks=[checkpoint_callback])



# In[59]:


model.load_weights("model_checkpoint_6.weights.h5") #10+6+10
print("Model weights loaded successfully!")


# In[60]:


additional_epochs = 10
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_checkpoint_{epoch}.weights.h5',
    save_weights_only=True,
    save_best_only=True,
    monitor='loss',  # or 'val_loss'
    mode='min'
)

model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=additional_epochs, callbacks=[checkpoint_callback])


# In[47]:


# Compute maximum caption length and average length
caption_lengths = []
max_len_found = 0

for key, captions in train_descriptions.items():
    for caption in captions:
        tokens = caption.split()
        caption_lengths.append(len(tokens))
        if len(tokens) > max_len_found:
            max_len_found = len(tokens)

print("Max Caption Length:", max_len_found)
print("Average Caption Length:", np.mean(caption_lengths))


# In[64]:


import pickle
import time

# Assuming test_id is a list of test image IDs (without the .jpg extension)
# and IMG_PATH is defined (the folder where your images are stored)

encoded_test_data_spatial = {}
start = time.time()

for ix, img_id in enumerate(test_id):
    img_path = IMG_PATH + img_id + ".jpg"
    try:
        spatial_features = encode_image_spatial(img_path)  # shape: (196, 768)
        encoded_test_data_spatial[img_id] = spatial_features
    except Exception as e:
        print(f"Error processing {img_id}: {e}")
    
    if ix % 100 == 0:
        print(f"Processed {ix} images...")

end = time.time()
print("Time elapsed:", end - start)

# Save the spatial features to a pickle file
with open("encoded_test_img_features_spatial.pkl", "wb") as f:
    pickle.dump(encoded_test_data_spatial, f)

print("Spatial test features saved successfully!")


# In[65]:


import pickle

with open("encoded_test_img_features_spatial.pkl", "rb") as f:
    encoded_test_data_spatial = pickle.load(f)

print("Loaded spatial test features for", len(encoded_test_data_spatial), "images")


# In[68]:


#Visualize attention
import matplotlib.pyplot as plt
from PIL import Image

def visualize_attention(image_path, caption, attn_weights, grid_size=(14,14)):
    """
    Visualizes the attention map over the image for each word in the caption.
    
    image_path: path to the image file.
    caption: list of predicted words.
    attn_weights: list of attention weight vectors (one per time step), each of shape (num_patches,)
    grid_size: tuple indicating the spatial grid dimensions (e.g., (14, 14) for 196 patches).
    """
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    
    num_words = len(caption)
    plt.figure(figsize=(10, 3 * num_words))
    for t in range(num_words):
        ax = plt.subplot(num_words, 1, t + 1)
        ax.set_title(f"Word: {caption[t]}")
        
        # Get the attention vector for this word and reshape to grid dimensions
        attn_map = np.array(attn_weights[t])
        if attn_map.size != grid_size[0] * grid_size[1]:
            # If grid_size doesn't match, try to infer a square grid
            side = int(np.sqrt(attn_map.size))
            grid_size = (side, side)
        attn_map = attn_map.reshape(grid_size)
        
        # Normalize and upscale the attention map to the image size
        attn_map = attn_map / attn_map.max() if attn_map.max() > 0 else attn_map
        attn_map = Image.fromarray(np.uint8(255 * attn_map))
        attn_map = attn_map.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
        attn_map = np.array(attn_map)
        
        ax.imshow(image)
        ax.imshow(attn_map, cmap='jet', alpha=0.6)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage:
visualize_attention(sample_img_path, predicted_caption, attention_maps, grid_size=(14,14))


# In[ ]:


#Inference


# In[69]:


import random
import matplotlib.pyplot as plt

# Get all test image IDs from the spatial features dictionary
all_test_images = list(encoded_test_data_spatial.keys())

# Randomly select 15 images
random_image_ids = random.sample(all_test_images, 15)

for img_id in random_image_ids:
    image_path = IMG_PATH + img_id + ".jpg"
    # Generate caption with attention using your inference function
    caption, attn_weights = predict_caption_with_attention(image_path, max_len=35)
    print("Image ID:", img_id)
    print("Generated Caption:", " ".join(caption))
    visualize_attention(image_path, caption, attn_weights, grid_size=(14,14))
    plt.pause(1)
    plt.close('all')


# In[70]:


#BLUE AND METEOR


# In[72]:


# Assuming 'descriptions' is already defined (from Flickr8k.token.txt) and contains all captions.
# Also, assuming you have a test file from which you can extract test image IDs.

def readTextFile(file_path):
    with open(file_path) as f:
        return f.read()

test_file_data = readTextFile("/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt")
test_ids = [id.split(".")[0] for id in test_file_data.split("\n") if id.strip() != ""]

# Build test_descriptions dictionary:
test_descriptions = {}
for img_id in test_ids:
    # Get captions from your descriptions dictionary (or an empty list if not found)
    test_descriptions[img_id] = descriptions.get(img_id, [])

print("Test descriptions loaded for", len(test_descriptions), "images")


# In[73]:


import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Define how many test images to evaluate
subset_size = 100  # for example, evaluate on 100 test images
# Randomly select a subset of test image IDs from test_descriptions
subset_images = random.sample(list(test_descriptions.keys()), min(subset_size, len(test_descriptions)))

bleu_scores = []
meteor_scores = []

for img_id in subset_images:
    # Build the full path to the image file
    image_path = IMG_PATH + img_id + ".jpg"
    
    # Generate a caption using your inference function
    generated_caption_tokens, _ = predict_caption_with_attention(image_path, max_len=35)
    generated_caption = " ".join(generated_caption_tokens)
    
    # Tokenize the generated caption (convert to lowercase for consistency)
    hypothesis_tokens = word_tokenize(generated_caption.lower())
    
    # Get the reference captions for this image (make sure they are preprocessed similarly)
    references = test_descriptions[img_id]  # this should be a list of strings
    reference_tokens = [word_tokenize(ref.lower()) for ref in references]
    
    # Compute BLEU score for this image (using sentence_bleu for a single sentence)
    bleu = sentence_bleu(reference_tokens, hypothesis_tokens)
    bleu_scores.append(bleu)
    
    # Compute METEOR score for this image
    meteor = meteor_score(reference_tokens, hypothesis_tokens)
    meteor_scores.append(meteor)

# Calculate average metrics
avg_bleu = np.mean(bleu_scores)
avg_meteor = np.mean(meteor_scores)

print("Average BLEU score: {:.4f}".format(avg_bleu))
print("Average METEOR score: {:.4f}".format(avg_meteor))


# In[85]:


def beam_search_caption_with_attention(image_features, beam_width=3, max_len=35, num_patches=196):
    """
    Generates a caption using beam search and collects attention weights for each time step
    of the best sequence.
    
    Args:
      image_features: A numpy array of shape (1, num_patches, feature_dim) representing the image features.
      beam_width: The beam width.
      max_len: Maximum caption length.
      num_patches: Expected size of the attention vector (number of image patches).
    
    Returns:
      caption_tokens: List of generated word tokens (excluding start/end tokens).
      best_score: Cumulative log-probability of the best sequence.
      attn_weights_list: List of attention vectors (each of shape (num_patches,))
    """
    start_idx = word_to_idx['startseq']
    end_idx = word_to_idx['endseq']
    
    # Initialize beam: a list of tuples (sequence, cumulative log-probability)
    beam = [([start_idx], 0.0)]
    
    for _ in range(max_len):
        new_beam = []
        for seq, score in beam:
            if seq[-1] == end_idx:
                new_beam.append((seq, score))
                continue
            
            input_seq = pad_sequences([seq], maxlen=max_len, padding='post')
            # Model returns (word_probs, attn_weights); we ignore attention here.
            pred_probs, _ = model.predict([image_features, input_seq])
            pred_probs = pred_probs[0]  # shape: (vocab_size,)
            
            top_indices = np.argsort(pred_probs)[-beam_width:]
            for idx in top_indices:
                new_score = score + np.log(pred_probs[idx] + 1e-10)
                new_seq = seq + [idx]
                new_beam.append((new_seq, new_score))
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[-1] == end_idx for seq, _ in beam):
            break

    best_seq, best_score = beam[0]
    caption_tokens = [idx_to_word[idx] for idx in best_seq if idx not in (start_idx, end_idx)]
    
    # Now, collect attention weights for each time step of the best sequence.
    attn_weights_list = []
    for t in range(1, len(best_seq)):
        partial_seq = best_seq[:t]
        input_seq = pad_sequences([partial_seq], maxlen=max_len, padding='post')
        # Obtain both predictions and attention weights.
        _, attn = model.predict([image_features, input_seq])
        attn = np.array(attn)
        # If attn has more than one dimension, take the first element.
        if attn.ndim > 1:
            attn = attn[0]
        else:
            # If it's a scalar, replace with zeros.
            attn = np.zeros(num_patches)
        # If the attention vector size doesn't match, adjust by creating a zero vector.
        if attn.size != num_patches:
            attn = np.zeros(num_patches)
        attn_weights_list.append(attn)
    
    return caption_tokens, best_score, attn_weights_list


# In[83]:


import random
import matplotlib.pyplot as plt

# Assuming:
# - IMG_PATH is defined (path to your images)
# - encoded_test_data_spatial is a dictionary mapping image IDs to spatial features (shape: (num_patches, feature_dim))
# - test_descriptions is defined, mapping image IDs to lists of reference captions.

# Get all test image IDs from your spatial features dictionary
all_test_images = list(encoded_test_data_spatial.keys())

# Randomly select 15 image IDs
random_image_ids = random.sample(all_test_images, 15)

for img_id in random_image_ids:
    image_path = IMG_PATH + img_id + ".jpg"
    
    # Get the spatial features for this image and add a batch dimension.
    image_features = np.expand_dims(encoded_test_data_spatial[img_id], axis=0)  # shape: (1, num_patches, feature_dim)
    
    # Generate caption using beam search
    caption_tokens, beam_score = beam_search_caption(image_features, beam_width=3, max_len=35)
    generated_caption = " ".join(caption_tokens)
    
    print("Image ID:", img_id)
    print("Generated Caption:", generated_caption)
    print("Beam Score:", beam_score)
    
    # Optionally, display the image along with the caption.
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title("Caption: " + generated_caption)
    plt.axis('off')
    plt.show()


# In[79]:


#Evaluate METEOR for 


# In[78]:


import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# Smoothing function for BLEU
cc = SmoothingFunction()

# Evaluate on a subset of test images
subset_size = 100  # Evaluate on 100 test images
subset_images = random.sample(list(test_descriptions.keys()), min(subset_size, len(test_descriptions)))

bleu_scores = []
meteor_scores = []

for img_id in subset_images:
    image_path = IMG_PATH + img_id + ".jpg"
    
    # Retrieve spatial features and generate caption using beam search
    image_features = np.expand_dims(encoded_test_data_spatial[img_id], axis=0)
    caption_tokens, _ = beam_search_caption(image_features, beam_width=3, max_len=35)
    generated_caption = " ".join(caption_tokens)
    
    # Tokenize generated caption
    hypothesis_tokens = word_tokenize(generated_caption.lower())
    
    # Get and tokenize reference captions
    references = test_descriptions[img_id]  # list of strings
    reference_tokens = [word_tokenize(ref.lower()) for ref in references]
    
    # Compute BLEU score with smoothing
    bleu = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=cc.method1)
    bleu_scores.append(bleu)
    
    # Compute METEOR score
    meteor = meteor_score(reference_tokens, hypothesis_tokens)
    meteor_scores.append(meteor)
    
    print(f"Image ID: {img_id}")
    print("Generated Caption:", generated_caption)
    print(f"BLEU: {bleu:.4f}  METEOR: {meteor:.4f}\n")

avg_bleu = np.mean(bleu_scores)
avg_meteor = np.mean(meteor_scores)

print("Average BLEU score: {:.4f}".format(avg_bleu))
print("Average METEOR score: {:.4f}".format(avg_meteor))


# In[80]:


#Inference with visualization


# In[86]:


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_attention(image_path, caption, attn_weights, grid_size=(14,14)):
    """
    Overlays attention maps for each word on the original image.
    
    Args:
      image_path: Path to the image.
      caption: List of generated words.
      attn_weights: List of attention vectors (one per time step).
      grid_size: Tuple for reshaping attention vector (e.g., (14,14) for 196 patches).
    """
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    num_words = len(caption)
    
    plt.figure(figsize=(10, 3 * num_words))
    for t in range(num_words):
        ax = plt.subplot(num_words, 1, t + 1)
        ax.set_title(f"Word: {caption[t]}")
        
        # Use the attention weight if available; otherwise, use zeros.
        if t < len(attn_weights):
            attn_map = np.array(attn_weights[t])
        else:
            attn_map = np.zeros(grid_size[0] * grid_size[1])
        
        # Ensure attn_map can be reshaped to grid_size.
        if attn_map.size != grid_size[0] * grid_size[1]:
            side = int(np.sqrt(attn_map.size))
            grid_size = (side, side)
        attn_map = attn_map.reshape(grid_size)
        attn_map = attn_map / (attn_map.max() if attn_map.max() > 0 else 1)
        attn_map = Image.fromarray(np.uint8(255 * attn_map))
        attn_map = attn_map.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
        attn_map = np.array(attn_map)
        
        ax.imshow(image)
        ax.imshow(attn_map, cmap='jet', alpha=0.6)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[87]:


import random

# Pick a random test image ID from your spatial test data
all_test_images = list(encoded_test_data_spatial.keys())
sample_img_id = random.choice(all_test_images)
sample_img_path = IMG_PATH + sample_img_id + ".jpg"
print("Sample Image ID:", sample_img_id)

# Get the spatial features for the sample image and add a batch dimension
image_features = np.expand_dims(encoded_test_data_spatial[sample_img_id], axis=0)

# Generate caption and attention weights using beam search with attention
generated_caption_tokens, beam_score, attention_weights = beam_search_caption_with_attention(image_features, beam_width=3, max_len=35, num_patches=196)
print("Generated Caption:", " ".join(generated_caption_tokens))
print("Beam Score:", beam_score)

# Visualize the attention maps over the image
visualize_attention(sample_img_path, generated_caption_tokens, attention_weights, grid_size=(14,14))


# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


# Implement Contrastive Dual-Encoder (Karpathy-style)


# In[9]:


import os
import re
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import ViTModel, ViTImageProcessor


# In[1]:


# %% [markdown]
# 📦 Step 1: Load and Read Caption File

# %% [code]
import re

def read_text_file(file_path):
    with open(file_path, "r") as f:
        return f.read()

doc = read_text_file("/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt")
lines = doc.strip().split("\n")
print(f"Total lines in caption file: {len(lines)}")

# %% [markdown]
# 🧠 Step 2: Parse and Store Descriptions per Image ID

# %% [code]
descriptions = {}
for line in lines:
    img_token, caption = line.split('\t')
    img_id = img_token.split('.')[0]
    descriptions.setdefault(img_id, []).append(caption)

print(f"Total unique image IDs: {len(descriptions)}")

# %% [markdown]
# 🧼 Step 3: Clean Captions

# %% [code]
def clean_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z]+', ' ', sentence)
    sentence = " ".join([word for word in sentence.split() if len(word) > 1])
    return sentence

for img_id in descriptions:
    descriptions[img_id] = [clean_text(caption) for caption in descriptions[img_id]]

# %% [markdown]
# 📁 Step 4: Save Cleaned Descriptions

# %% [code]
with open("descriptions.txt", "w") as f:
    f.write(str(descriptions))


# In[3]:


# %% [markdown]
# 🔤 Step 5: Build Vocabulary (with threshold)

# %% [code]
total_words = [word for captions in descriptions.values() for caption in captions for word in caption.split()]
print(f"Total words across all captions: {len(total_words)}")

unique_words = set(total_words)
print(f"Total unique words: {len(unique_words)}")

# Frequency count
freq_counts = {}
for word in total_words:
    freq_counts[word] = freq_counts.get(word, 0) + 1

# Filter words by frequency threshold
word_count_threshold = 10
sorted_freq_count = sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)
vocabulary = [word for word, count in sorted_freq_count if count > word_count_threshold]
print(f"Final vocabulary size (threshold={word_count_threshold}): {len(vocabulary)}")


# In[4]:


# %% [markdown]
# 🧾 Step 6: Add start/end tokens to captions

# %% [code]
# Load train IDs
train_ids_file = "/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt"
with open(train_ids_file, 'r') as f:
    train_id = [line.strip().split('.')[0] for line in f.readlines() if line.strip()]

train_descriptions = {}
for img_id in train_id:
    train_descriptions[img_id] = ['startseq ' + caption + ' endseq' for caption in descriptions.get(img_id, [])]

print(f"Prepared train descriptions for {len(train_descriptions)} images")


# In[5]:


# %% [markdown]
# 🔡 Step 7: Build word_to_idx and idx_to_word dictionaries

# %% [code]
word_to_idx = {word: idx + 1 for idx, word in enumerate(vocabulary)}  # idx+1 to reserve 0 for padding
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Add special tokens
special_tokens = {'startseq': len(word_to_idx) + 1, 'endseq': len(word_to_idx) + 2}
word_to_idx.update(special_tokens)
idx_to_word.update({v: k for k, v in special_tokens.items()})

# Final vocab size
vocab_size = len(word_to_idx) + 1  # +1 for padding (index 0)
print(f"Final vocab size (with padding): {vocab_size}"


# In[6]:


# %% [markdown]
# 📏 Step 8: Calculate Max Caption Length

# %% [code]
max_len = max(len(caption.split()) for captions in descriptions.values() for caption in captions)
print(f"Maximum caption length: {max_len}")


# In[10]:


class Flickr8kDataset(Dataset):
    def __init__(self, images_dir, train_descriptions, image_ids, word_to_idx, max_len=35, transform=None):
        self.images_dir = images_dir
        self.descriptions = train_descriptions
        self.image_ids = image_ids
        self.word_to_idx = word_to_idx
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, img_id + ".jpg")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Randomly select one cleaned caption
        caption = np.random.choice(self.descriptions[img_id])
        tokens = caption.strip().split()
        seq = [self.word_to_idx.get(token, 0) for token in tokens]

        # Pad or truncate
        if len(seq) < self.max_len:
            seq += [0] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]

        return image, torch.tensor(seq, dtype=torch.long)


# In[11]:


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = Flickr8kDataset(
    images_dir="/Users/prayashdas/Downloads/archive-12/Flickr_Data/Flickr_Data/Images",
    train_descriptions=train_descriptions,
    image_ids=train_id,
    word_to_idx=word_to_idx,
    max_len=33,  # based on your max caption length
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)


# In[12]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTImageProcessor

# Image Encoder using ViT + Linear Projection
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.projection = nn.Linear(768, 256)

    def forward(self, images):
        # Input: (B, 3, 224, 224)
        inputs = {"pixel_values": images}
        with torch.no_grad():  # Freeze ViT backbone
            outputs = self.vit(**inputs)
        cls_token = outputs.last_hidden_state[:, 0]  # (B, 768)
        embed = self.projection(cls_token)           # (B, 256)
        embed = F.normalize(embed, dim=1)
        return embed

# Text Encoder using Embedding + LSTM + Projection
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, 256)

    def forward(self, captions):
        embedded = self.embedding(captions)          # (B, T, embed_dim)
        _, (hidden, _) = self.lstm(embedded)         # hidden: (1, B, hidden_dim)
        hidden = hidden.squeeze(0)                   # (B, hidden_dim)
        embed = self.projection(hidden)              # (B, 256)
        embed = F.normalize(embed, dim=1)
        return embed


# In[13]:


# -------------------------------
# 3. Define the Contrastive Loss (InfoNCE)
# -------------------------------

def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    # Normalize again just in case
    image_embeds = F.normalize(image_embeds, dim=1)
    text_embeds = F.normalize(text_embeds, dim=1)

    logits = image_embeds @ text_embeds.T  # (B, B)
    logits /= temperature

    batch_size = logits.size(0)
    labels = torch.arange(batch_size).to(logits.device)

    # InfoNCE loss: both image-to-text and text-to-image
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


# In[14]:


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate encoders
img_encoder = ImageEncoder().to(device)
txt_encoder = TextEncoder(vocab_size=len(word_to_idx) + 1).to(device)  # +1 for padding

# Optimizer and learning rate scheduler
optimizer = torch.optim.Adam(
    list(img_encoder.parameters()) + list(txt_encoder.parameters()), lr=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# In[15]:


# Training function
def train_epoch(loader, img_encoder, txt_encoder, optimizer, device):
    img_encoder.train()
    txt_encoder.train()
    total_loss = 0.0

    for images, captions in loader:
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()
        img_embeds = img_encoder(images)       # (B, 256)
        txt_embeds = txt_encoder(captions)     # (B, 256)
        loss = contrastive_loss(img_embeds, txt_embeds)
        loss.backward()

        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(list(img_encoder.parameters()) + list(txt_encoder.parameters()), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# Start training
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    loss = train_epoch(train_loader, img_encoder, txt_encoder, optimizer, device)
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")


# In[16]:


additional_epochs = 15
start_epoch = num_epochs  # num_epochs = 10 from before

for epoch in range(start_epoch, start_epoch + additional_epochs):
    loss = train_epoch(train_loader, img_encoder, txt_encoder, optimizer, device)
    scheduler.step()
    print(f"Epoch {epoch+1}/{start_epoch + additional_epochs}, Loss: {loss:.4f}")


# In[17]:


torch.save(img_encoder.state_dict(), "image_encoder_final.pt")
torch.save(txt_encoder.state_dict(), "text_encoder_final.pt")


# In[18]:


#  Step-by-Step Retrieval Evaluation Plan


# In[19]:


def extract_embeddings(loader, img_encoder, txt_encoder, device):
    img_encoder.eval()
    txt_encoder.eval()

    all_img_embeds = []
    all_txt_embeds = []

    with torch.no_grad():
        for images, captions in loader:
            images = images.to(device)
            captions = captions.to(device)

            img_embed = img_encoder(images)
            txt_embed = txt_encoder(captions)

            all_img_embeds.append(img_embed)
            all_txt_embeds.append(txt_embed)

    # Concatenate all batches
    all_img_embeds = torch.cat(all_img_embeds, dim=0)
    all_txt_embeds = torch.cat(all_txt_embeds, dim=0)
    return all_img_embeds, all_txt_embeds


# In[20]:


def compute_recall_at_k(image_embeds, text_embeds, k_vals=[1, 5, 10]):
    sim_matrix = image_embeds @ text_embeds.T  # (N, N)
    N = sim_matrix.size(0)

    recalls = {'i2t': {}, 't2i': {}}

    # Image → Text
    ranks_i2t = []
    for i in range(N):
        sorted_indices = sim_matrix[i].argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        ranks_i2t.append(rank)
    for k in k_vals:
        recalls['i2t'][f"R@{k}"] = 100.0 * sum(r < k for r in ranks_i2t) / N

    # Text → Image
    ranks_t2i = []
    for i in range(N):
        sorted_indices = sim_matrix[:, i].argsort(descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        ranks_t2i.append(rank)
    for k in k_vals:
        recalls['t2i'][f"R@{k}"] = 100.0 * sum(r < k for r in ranks_t2i) / N

    recalls['i2t']["Median Rank"] = int(np.median(ranks_i2t))
    recalls['t2i']["Median Rank"] = int(np.median(ranks_t2i))

    return recalls


# In[21]:


img_embeds, txt_embeds = extract_embeddings(train_loader, img_encoder, txt_encoder, device)
recall_results = compute_recall_at_k(img_embeds, txt_embeds)

print("🔁 Image → Text Retrieval:")
for k, v in recall_results['i2t'].items():
    print(f"{k}: {v:.2f}")

print("\n📝 Text → Image Retrieval:")
for k, v in recall_results['t2i'].items():
    print(f"{k}: {v:.2f}")


# In[22]:


# Visualize embeddings with PCA


# In[23]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


# In[25]:


img_np = img_embeds.cpu().numpy()
txt_np = txt_embeds.cpu().numpy()


# In[26]:


# Reduce to 2D using PCA
pca = PCA(n_components=2)
img_2d = pca.fit_transform(img_np)
txt_2d = pca.fit_transform(txt_np)

# Plot
plt.figure(figsize=(10, 5))

# Image Embeddings
plt.subplot(1, 2, 1)
plt.scatter(img_2d[:, 0], img_2d[:, 1], c='blue', label='Image Embeds', alpha=0.5)
plt.title("Image Embeddings (PCA)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()

# Text Embeddings
plt.subplot(1, 2, 2)
plt.scatter(txt_2d[:, 0], txt_2d[:, 1], c='green', label='Text Embeds', alpha=0.5)
plt.title("Text Embeddings (PCA)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()

plt.tight_layout()
plt.show()


# In[27]:


# Stack both together
combined = np.vstack([img_np, txt_np])
pca = PCA(n_components=2)
combined_2d = pca.fit_transform(combined)

plt.figure(figsize=(8, 6))
plt.scatter(combined_2d[:len(img_np), 0], combined_2d[:len(img_np), 1], c='blue', label='Images', alpha=0.5)
plt.scatter(combined_2d[len(img_np):, 0], combined_2d[len(img_np):, 1], c='green', label='Captions', alpha=0.5)
plt.title("Image vs Text Embeddings (PCA Projection)")
plt.legend()
plt.show()


# In[ ]:




