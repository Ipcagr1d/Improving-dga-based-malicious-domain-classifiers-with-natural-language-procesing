# coding: utf-8

# In[3]:


# coding: utf-8

# ### Imports

# In[34]:


# Essential Imports

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

import itertools
import os

import gensim

# get_ipython().run_line_magic('matplotlib', 'inline')
USE_CUDA = False

# ### Read real dataset

# In[6]:


TEST_SIZE = 0.1
# Read real dataset with pandas read_csv methods
real = pd.read_csv('majestic_million.csv', usecols=['Domain'])
# train_size = int(real.size * (1 - TEST_SIZE))
train_size = int(135000 * (1 - TEST_SIZE))
real_train = real[:train_size]
real_test = real[train_size:135000]
real.head()  # show sample rows

# ### Shape of real data

# In[7]:


# Shape of real data
real.shape

# ### Add label to real data

# In[8]:


# Add Label(0-fake, 1-real)
real_train['label'] = 1
real_test['label'] = 1

# ### Read fake data files

# In[10]:
currentDir = os.getcwd()
dstasetlist = os.listdir(currentDir + '/Dataset')
# loop dataset directory

fake_train = pd.DataFrame()
fake_test = pd.DataFrame()
fake_testWithClass = []
totoal_malicious_count = 0;
for f in dstasetlist:
    faketemp = pd.read_csv(currentDir + '/Dataset/' + f, header=None, usecols=[0], names=['Domain'])
    faketemp = faketemp.drop([0], axis=0)
    totoal_malicious_count += faketemp.size;
    train_size = int(faketemp.size * (1 - TEST_SIZE))
    fake_train = pd.concat([fake_train, faketemp[:train_size]])
    fake_test = pd.concat([fake_test, faketemp[train_size:]])
    fake_test_class = faketemp[train_size:]
    fake_test_class['label'] = 0
    fake_testWithClass.append([f, fake_test_class])

# ### Shape of fake data

# In[11]:


# Shape of fake data

# ### Adding label to fake data

# In[12]:


# Add Label (0-fake, 1-real)
fake_train['label'] = 0
fake_test['label'] = 0

# ### Stack up real and fake data

# In[19]:


# Merge both real and fake to make the dataset
frames_train = [real_train, fake_train]  # dataframes to be stacked up
frames_test = [real_test, fake_test]
data_train = pd.concat(frames_train)  # concat
data_test = pd.concat(frames_test)  # concat
data_train.reset_index(inplace=True)  # update the index of concated dataframe
data_test.reset_index(inplace=True)

# ### Print shape of the data

# In[20]:


# Shape of merged data

# ### Print Class balance

# In[21]:


# Plot class label counts
data_train['label'].value_counts().plot(kind="bar", rot=0)
plt.show()  # show plot

data_test['label'].value_counts().plot(kind="bar", rot=0)
plt.show()

# ### Tokenize domain name and domain types

# In[23]:


# tokenize the domain names
data_train['domain_name'] = data_train['Domain'].str.partition('.', expand=True)[[0]]
data_train['domain_type'] = data_train['Domain'].str.partition('.', expand=True)[[2]]
data_test['domain_name'] = data_test['Domain'].str.partition('.', expand=True)[[0]]
data_test['domain_type'] = data_test['Domain'].str.partition('.', expand=True)[[2]]

# ### Word2Vector model

# In[24]:


# create word2vec model with tokenized domains data
tokens_train = [[token] for token in data_train['domain_name'].tolist()]
tokens_test = [[token] for token in data_test['domain_name'].tolist()]

# In[19]:


max_length = max([max([len(x[0]) for x in tokens_train]), max([len(x[0]) for x in tokens_test])])

pad_token = 256
# ### Create train and test splits


# In[29]:


# Train/Test split with a test size of 33%
# X_train, X_test, y_train, y_test = train_test_split(tokens, data['label'],
#                                                     test_size=0.1, random_state=42)
X_train = tokens_train
X_test = tokens_test
y_train = data_train['label']
y_test = data_test['label']

X_train_temp = []
seq_length_train = []
# convert to ASCII and
for x_train in X_train:
    ascii_v = [ord(x) for x in list(x_train)[0]]
    seq_length_train.append(len(ascii_v))
    padded_x = np.ones((max_length)) * pad_token
    padded_x[:seq_length_train[-1]] = ascii_v
    X_train_temp.append(padded_x)

X_test_temp = []
seq_length_test = []
# convert to ASCII and
for x_test in X_test:
    ascii_v = [ord(x) for x in list(x_test)[0]]
    seq_length_test.append(len(ascii_v))
    padded_x = np.ones(max_length) * pad_token
    padded_x[:seq_length_test[-1]] = ascii_v
    X_test_temp.append(padded_x)

X_train = np.array(X_train_temp)
X_test = np.array(X_test_temp)
seq_length_train = np.array(seq_length_train)
seq_length_test = np.array(seq_length_test)

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VALID = 128
BATCH_SIZE_TEST = 128

torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train.values).type(torch.LongTensor)  # data type is long
torch_seq_length_train = torch.from_numpy(seq_length_train).type(torch.LongTensor)

# create feature and targets tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test.values).type(torch.LongTensor)  # data type is long
torch_seq_length_test = torch.from_numpy(seq_length_test).type(torch.LongTensor)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train, torch_seq_length_train)
test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test, torch_seq_length_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE_TEST, shuffle=True)
valid_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE_VALID, shuffle=True)


# ### MLP class

# In[31]:

class CharLSTM(nn.ModuleList):
    def __init__(self, sequence_len, vocab_size, embed_size, num_layers, hidden_dim, batch_size):
        super(CharLSTM, self).__init__()

        # init the meta parameters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        padding_idx = 256
        # embedding look up for characters
        self.embeds = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=padding_idx)

        # first layer lstm cell
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_dim, num_layers=self.num_layers, batch_first=True)

        # dropout layer for the output of the second layer cell
        self.dropout = nn.Dropout(p=0.5)

        # fully connected layer to connect the output of the LSTM cell to the output
        self.fc = nn.Linear(in_features=hidden_dim, out_features=2)

    def init_hidden(self, x):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim)
        hidden_b = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim)

        if x.is_cuda:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, x, x_lengths):
        """
            x: input to the model
                *  x[t] - input of shape (batch, input_size) at time t

            hc: hidden and cell states
                *  tuple of hidden and cell state
        """

        # pass the hidden and the cell state from one lstm cell to the next one
        # we also feed the output of the first layer lstm cell at time step t to the second layer cell
        # init the both layer cells with the zero hidden and zero cell states
        self.hidden = self.init_hidden(x)

        embedding = self.embeds(x)
        embedding_p = torch.nn.utils.rnn.pack_padded_sequence(embedding, x_lengths, batch_first=True,
                                                              enforce_sorted=False)
        h_1, c_1 = self.lstm(embedding_p, self.hidden)
        h_1, _ = torch.nn.utils.rnn.pad_packed_sequence(h_1, batch_first=True)
        idx = (x_lengths - 1).view(-1, 1).expand(
            len(x_lengths), h_1.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        if h_1.is_cuda:
            idx = idx.cuda(h_1.data.get_device())
        # Shape: (batch_size, rnn_hidden_dim)
        out_h = h_1.gather(
            time_dimension, Variable(idx)).squeeze(time_dimension)
        # form the output of the fc
        output = self.fc(self.dropout(out_h))

        # return the output sequence
        return output, embedding
		#return F.log_softmax(output.view((x.shape[0], -1)),dim=1), embedding


# ### FIt method to train the MLP

# In[36]:


def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # , weight_decay=0.001)#, betas=(0.9,0.999))

    error = nn.CrossEntropyLoss().cuda()
    EPOCHS = 6
    # model.train()

    for epoch in range(EPOCHS):
        correct = 0
        model.train()
        for batch_idx, (X_batch, y_batch, len_batch) in enumerate(train_loader):
            # var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            var_len_batch = Variable(len_batch)
            var_X_batch = torch.tensor(X_batch, dtype=torch.long)
            if USE_CUDA:
                var_X_batch = var_X_batch.cuda()
                var_y_batch = var_y_batch.cuda()
                var_len_batch = var_len_batch.cuda()

            optimizer.zero_grad()
            output, _ = model(var_X_batch, var_len_batch)
            loss = error(output, var_y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for p in model.parameters():
                p.data.add_(-0.001, p.grad.data)
            optimizer.step()

        correct_valid = 0
        num_samples = 0
        model.eval()
        for batch_val_idx, (X_batch_valid, y_batch_valid, len_batch_val) in enumerate(valid_loader):
            var_X_batch_val = torch.tensor(X_batch_valid, dtype=torch.long)
            var_y_batch_val = Variable(y_batch_valid)
            var_len_batch_val = Variable(len_batch_val)
            num_samples += len(var_y_batch_val)

            if USE_CUDA:
                var_X_batch_val = var_X_batch_val.cuda()
                var_y_batch_val = var_y_batch_val.cuda()
                var_len_batch_val = var_len_batch_val.cuda()
            output, _ = model(var_X_batch_val, var_len_batch_val)
            predicted = torch.max(output.data, 1)[1]
            correct_valid += (predicted == var_y_batch_val).sum()
        print('Validation - Epoch : {} [{}/{} ({:.0f}%)]\t\t Accuracy:{:.3f}%'.format(
            epoch, BATCH_SIZE_VALID * (batch_val_idx + 1),
            len(test_loader.dataset),
                   100. * batch_val_idx / len(test_loader),
                   float(correct_valid * 100) / float(num_samples)))



# ### Create Multi-Layer-Perceptron and fit the train data

# In[37]:


net = CharLSTM(sequence_len=128, vocab_size=257, embed_size=256, num_layers=2, hidden_dim=512, batch_size=128)

pretrained_model = "model.pth"
#if os.path.exists(pretrained_model):
#    net.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

print(net)
if USE_CUDA:
    net = net.cuda()
fit(net, train_loader)
torch.save(net.state_dict(), pretrained_model)


# ### Carry out FGSM attack

# In[38]:


# FGSM attack code
def fgsm_attack(domain, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed domain
    perturbed_domain = domain + epsilon * sign_data_grad
    # Return the perturbed domain
    return perturbed_domain


# ### Testing function

# In[39]:


def test(model, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    num_samples = 0
    adv_examples = []
    model.train()
    # Loop over all examples in test set
    for batch_test_idx, (data, target, seq_len) in enumerate(test_loader):

        data = torch.tensor(data, dtype=torch.long)
        target = Variable(target)
        seq_len = Variable(seq_len)
        num_samples += len(target)

        if USE_CUDA:
            data = data.cuda()
            target = target.cuda()
            seq_len = seq_len.cuda()

        # Set requires_grad attribute of tensor. Important for Attack
        # data.requires_grad = True

        # Forward pass the data through the model
        output, embedding = model(data, seq_len)
        init_pred = torch.max(output, 1)[1]  # get the index of the max log-probability

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        embedding_grad = [p for p in model.parameters()]
        embedding_grad = embedding_grad[0]
        embedding_grad = embedding_grad.grad.data
        embedding_grad = embedding_grad[data]
        # Call FGSM Attack
        perturbed_embedding = fgsm_attack(embedding, epsilon, embedding_grad)
        embeddings_all = model.embeds.weight
        embeddings_all = F.normalize(embeddings_all, p=2, dim=1)
        perturbed_embedding = F.normalize(perturbed_embedding, p=2, dim=1)
        perturbed_data = torch.bmm(perturbed_embedding,
                                   embeddings_all.transpose(0, 1).unsqueeze(0).repeat(perturbed_embedding.shape[0], 1,
                                                                                      1))
        perturbed_data = perturbed_data.argmax(dim=2)
        # perturbed_data = perturbed_data.unsqueeze(dim=0)
        # Re-classify the perturbed image
        output, _ = model(perturbed_data, seq_len)

        # Check for success
        final_pred = torch.max(output, 1)[1]  # get the index of the max log-probability

        correct += (final_pred == target).sum()

        if len(adv_examples) < 1:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append(adv_ex)
            adv_seq_len = seq_len

    # Calculate final accuracy for this epsilon
    final_acc = float(correct * 100) / num_samples
    print("Epsilon: {}\tTest Accuracy  = {}".format(epsilon, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, adv_seq_len


# ### Generating adversarial domains

# In[40]:


import random


def generate_adv_examples(examples, seq_lengths):
    dom_list = []
    examples = examples[0]
    for idx in range(len(seq_lengths)):
        ex = examples[idx].tolist()
        seq_len = seq_lengths[idx]
        domain_len = random.randint(8, 40)
        gen_ex = []
        for i in range(int(seq_len)):
            gen_ex.append(chr(ex[i]))
        gen_ex = ''.join(gen_ex)
        dom_list.append(gen_ex + '.' + data_train['domain_type'][domain_len])
    return dom_list


# ### Adversarial Domains generated

# In[41]:


def write_adv_examples(examples, epsilon):
    with open('adversarial_examples_eps' + str(epsilon) + '.txt', 'w') as f:
        for adv_dom in set(examples):
            f.writelines(str(adv_dom.encode("utf-8")))
            f.writelines('\n')


# ### Testing the NN for various epsilon values

# In[ ]:

#
# epsilons = [0, .5, 1.0, 2.5, 5, 7.5, 10]
# accuracies = []
# # Run test for each epsilon
# for eps in epsilons:
#     acc, examples, seq_lengs = test(net, test_loader, eps)
#     accuracies.append(acc)
#     adv_ex = generate_adv_examples(examples, seq_lengs)
#     write_adv_examples(adv_ex, eps)
#
# # ### Plot Accuracy vs. Epsilon curve
# plt.figure(figsize=(5,5))
# plt.plot(epsilons, accuracies, "*-")
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, 10, step=0.5))
# plt.title("Accuracy vs Epsilon")
# plt.xlabel("Epsilon")
# plt.ylabel("Accuracy")
# plt.show()

# In[ ]:


def test_without_ep(model, test_loader):
    # Accuracy counter
    correct = 0
    num_samples = 0
    adv_examples = []
    model.train()
    # Loop over all examples in test set
    for batch_test_idx, (data, target, seq_len) in enumerate(test_loader):

        # fake_idx = np.argwhere(target == 0)
        # data = data[fake_idx].squeeze(dim=0)
        # seq_len = seq_len[fake_idx].squeeze(dim=0)
        # target = target[fake_idx].squeeze(dim=0)
        num_samples += len(target)
        #
        # data = torch.tensor(data, dtype=torch.long)
        # seq_len = Variable(seq_len)
        target_batch = Variable(target)
        seq_len_batch = Variable(seq_len)
        data_batch = torch.tensor(data, dtype=torch.long)

        if USE_CUDA:
            data_batch = data.cuda()
            target_batch = target.cuda()
            seq_len_batch = seq_len.cuda()

        # Set requires_grad attribute of tensor. Important for Attack
        # data.requires_grad = True

        # Forward pass the data through the model
        output, embedding = model(data_batch, seq_len_batch)
        predicted = torch.max(output.data, 1)[1]  # get the index of the max log-probability
        correct += (predicted == target_batch).sum()

    # Calculate final accuracy for this epsilon
    final_acc = float(correct * 100) / num_samples
    #print("Test Accuracy  = {}".format(final_acc))

    # Return the accuracy and an adversarial example
    return final_acc


accuracies = []
names = []
for fake in fake_testWithClass:
    data = fake[1]
    f = fake[0]
    data['domain_name'] = data['Domain'].str.partition('.', expand=True)[[0]]
    data['domain_type'] = data['Domain'].str.partition('.', expand=True)[[2]]

    tokens = [[token] for token in data['domain_name'].tolist()]

    max_length = max([len(x[0]) for x in tokens])

    pad_token = 256

    X_test_temp = []
    seq_length_test = []
    # convert to ASCII and
    for x_test in tokens:
        ascii_v = [ord(x) for x in list(x_test)[0]]
        seq_length_test.append(len(ascii_v))
        padded_x = np.ones(max_length) * pad_token
        padded_x[:seq_length_test[-1]] = ascii_v
        X_test_temp.append(padded_x)

    X_test = np.array(X_test_temp)

    seq_length_test = np.array(seq_length_test)

    BATCH_SIZE_TEST = 2048

    torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
    torch_y_test = torch.from_numpy(data['label'].values).type(torch.LongTensor)  # data type is long
    torch_seq_length_test = torch.from_numpy(seq_length_test).type(torch.LongTensor)

    test_tensor = torch.utils.data.TensorDataset(torch_X_test, torch_y_test, torch_seq_length_test)

    test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=BATCH_SIZE_TEST, shuffle=False)
    acc = test_without_ep(net, test_loader)
    accuracies.append(acc)
    names.append(f)
    print("dga: {}\tTest Accuracy  = {}".format(f,  acc))

# plt.figure(figsize=(5,5))
# plt.plot(names, accuracies, "*-")
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(names)
# plt.title("Accuracy vs DGA")
# plt.xlabel("DGA")
# plt.ylabel("Accuracy")
# plt.show()