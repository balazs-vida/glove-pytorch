from nltk.tokenize import word_tokenize
#from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import progressbar
progressbar.streams.wrap_stderr() 

#from sklearn.datasets import fetch_20newsgroups

#PARAMETERS
EMBEDDING_SIZE = 50

# "Unless otherwise noted, we use a context of ten words to the left and
# ten words to the right."
CONTEXT_SIZE = 10

# "For all our experiments, we set x_max = 100, alpha = 3/4"
X_MAX = 100
ALPHA = 0.75

BATCH_SIZE = 32

# "initial learning rate of 0.05"
LEARNING_RATE = 0.05

# "we run 50 iterations for vectors smaller than 300 dimensions..."
EPOCHS = 50


#DATA PROCESSING
# Open and read in text
text_file = open('short_story.txt', 'r')
raw_text = text_file.read().lower()
text_file.close()

#https://scikit-learn.org/0.19/modules/generated/sklearn.datasets.fetch_20newsgroups.html#sklearn.datasets.fetch_20newsgroups
#https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
#cats = ['alt.atheism', 'sci.space']
#newsgroup = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'), subset='train', categories=cats)
#raw_text = ' '.join(e for e in newsgroup.data)

# Create tokenized text (list) and vocabulary (set of unique words)
token_text = word_tokenize(raw_text)
len_token_text = len(token_text)
vocab = set(token_text)
vocab_size = len(vocab)

# Create word to index and index to word mapping
word_to_ix = {word: ind for ind, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

# Construct co-occurence matrix
co_occ_mat = np.zeros((vocab_size, vocab_size))
for i in range(len_token_text):
	for dist in range(1, CONTEXT_SIZE + 1):
		ix = word_to_ix[token_text[i]]
		if i - dist > 0:
			left_ix = word_to_ix[token_text[i - dist]]
			co_occ_mat[ix, left_ix] += 1.0 / dist
		if i + dist < len_token_text:
			right_ix = word_to_ix[token_text[i + dist]]
			co_occ_mat[ix, right_ix] += 1.0 / dist

# Non-zero co-occurrences
co_occs = np.transpose(np.nonzero(co_occ_mat))


# MODEL
class Glove(nn.Module):

    def __init__(self, vocab_size, comat, embedding_size, x_max, alpha):
        super(Glove, self).__init__()
        
        # embedding matrices
        self.embedding_V = nn.Embedding(vocab_size, embedding_size) # embedding matrix of center words (aka. input vector matrix)
        self.embedding_U = nn.Embedding(vocab_size, embedding_size) # embedding matrix of context words (aka. output/target vector matrix)

        # biases
        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)
        
        # initialize all params
        for params in self.parameters():
            nn.init.uniform_(params, a = -0.5, b = 0.5)
            
        #hyperparams
        self.x_max = x_max
        self.alpha = alpha
        self.comat = comat
    
    
    def forward(self, center_word_lookup, context_word_lookup):
        # indexing into the embedding matrices
        center_embed = self.embedding_V(center_word_lookup)
        target_embed = self.embedding_U(context_word_lookup)

        center_bias = self.v_bias(center_word_lookup).squeeze(1)
        target_bias = self.u_bias(context_word_lookup).squeeze(1)

        # elements of the co-occurence matrix
        co_occurrences = torch.tensor([self.comat[center_word_lookup[i].item(), context_word_lookup[i].item()] for i in range(BATCH_SIZE)])
        
        # weight_fn applied to non-zero co-occurrences
        weights = torch.tensor([self.weight_fn(var) for var in co_occurrences])

        # the loss as described in the paper
        loss = torch.sum(torch.pow((torch.sum(center_embed * target_embed, dim=1)
            + center_bias + target_bias) - torch.log(co_occurrences), 2) * weights)
        
        return loss
        
    def weight_fn(self, x):
        # the proposed weighting fn
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1
        
    def embeddings(self):
        # "we choose to use the sum W + W_tilde as our word vectors"
        return self.embedding_V.weight.data + self.embedding_U.weight.data
        
        
# Batch sampling function
def gen_batch(model, batch_size=BATCH_SIZE):
    """
    picks random indices for lookup in the embedding matrix
    "stochastically sampling non-zero elements from X [ie. the co-occurrence matrix]"
    """	
    sample = np.random.choice(np.arange(len(co_occs)), size=batch_size, replace=False)
    v_vecs_ix, u_vecs_ix = [], []
    
    for chosen in sample:
        ind = tuple(co_occs[chosen])     
        lookup_ix_v = ind[0]
        lookup_ix_u = ind[1]
        
        v_vecs_ix.append(lookup_ix_v)
        u_vecs_ix.append(lookup_ix_u) 
        
    return torch.tensor(v_vecs_ix), torch.tensor(u_vecs_ix)

# TRAINING
def train_glove(comat):
    losses = []
    model = Glove(vocab_size, comat, embedding_size=EMBEDDING_SIZE, x_max=X_MAX, alpha=ALPHA)
    optimizer = optim.Adagrad(model.parameters(), lr = LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = int(len_token_text/BATCH_SIZE)
        print("Beginning epoch %d" %epoch)
        progress_bar = progressbar.ProgressBar()
        for batch in progress_bar(range(num_batches)):
            model.zero_grad()
            data = gen_batch(model, BATCH_SIZE)
            loss = model(*data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        print('Epoch : %d, mean loss : %.02f' % (epoch, np.mean(losses)))
    return model, losses 

model, losses = train_glove(co_occ_mat)


# Plot loss fn
def plot_loss_fn(losses, title):
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.figure()

plot_loss_fn(losses, "GloVe loss function")


# Visualize embeddings
if EMBEDDING_SIZE == 2:
	# Pick some random words
	word_inds = np.random.choice(np.arange(len(vocab)), size=10, replace=False)
	for word_ind in word_inds:
		w_embed = model.embeddings()[word_ind].numpy()
		x, y = w_embed[0], w_embed[1]
		plt.scatter(x, y)
		plt.annotate(ix_to_word[word_ind], xy=(x, y), xytext=(5, 2),
			textcoords='offset points', ha='right', va='bottom')
	plt.savefig("glove.png")


## Save embeddings
#f = open("embeddings_with_glove.txt","w+", encoding="utf-8")
#for word in vocab:
#  input = torch.autograd.Variable(torch.LongTensor([word_to_ix[word]])) 
#  vector = model.embeddings()[input.item()]
#  weight_list = vector.numpy().tolist()
#  f.write(word + ' ' + ' '.join(str(e) for e in weight_list) + '\n')
#f.close()

# TESTS
# word similarity, word analogies
def get_word(word, model, word_to_ix):
    """
    returns the embedding that belongs to the given word 
    
    word (str)
    """
    return model.embeddings()[word_to_ix[word]]


def closest(vec, word_to_ix, n=10):
    """
    Find the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word(w, model, word_to_ix))) for w in word_to_ix]
    return sorted(all_dists, key=lambda t: t[1])[:n]


def print_tuples(tuples):
    for tuple in tuples:
        print('(%.4f) %s' % (tuple[1], tuple[0]))

# In the form w1 : w2 :: w3 : ?
def analogy(w1, w2, w3, n=5, filter_given=True):
    print('\n[%s : %s :: %s : ?]' % (w1, w2, w3))
   
    # w2 - w1 + w3 = w4
    closest_words = closest(get_word(w2, model, word_to_ix) - get_word(w1, model, word_to_ix) + get_word(w3, model, word_to_ix), word_to_ix)
    
    # Optionally filter out given words
    if filter_given:
        closest_words = [t for t in closest_words if t[0] not in [w1, w2, w3]]
        
    print_tuples(closest_words[:n])