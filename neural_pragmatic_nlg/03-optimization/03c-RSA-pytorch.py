import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

##################################################
## data to fit
##################################################

object_names     = ['blue_circle', 'green_square', 'blue_square']
utterance_names  = ['blue', 'circle', 'green', 'square']
semantic_meaning = torch.tensor(
    # blue circle, green square, blue square
    [[1, 0, 1],  # blue
     [1, 0, 0],  # circle
     [0, 1, 0],  # green
     [0, 1, 1]],  # square,
    dtype= torch.float32
    )

##################################################
## data to fit
##################################################

salience_prior = F.normalize(torch.tensor([71,139,30],
                                          dtype = torch.float32),
                             p = 1, dim = 0)

# matrix of number of utterance choices for each state
# (rows: objects, columns: utterances)
production_data = torch.tensor([[9, 135, 0, 0],
                                [0, 0, 119, 25],
                                [63, 0, 0, 81]])

# matrix of number of object choices for each ambiguous utterance
# (rows: utterances, columns: objects)
interpretation_data = torch.tensor([[66, 0, 115],   # "blue"
                                    [0, 117, 62]])  # "square"

##################################################
## RSA model (forward pass)
##################################################

def RSA(alpha, cost_adjectives):
    costs = torch.tensor([1.0, 0, 1.0, 0]) * cost_adjectives
    literal_listener   = F.normalize(semantic_meaning, p = 1, dim = 1)
    pragmatic_speaker  = F.normalize(torch.t(literal_listener)**alpha *
                                     torch.exp(-alpha * costs), p = 1, dim = 1)
    pragmatic_listener = F.normalize(torch.t(pragmatic_speaker) * salience_prior, p = 1, dim = 1)
    return({'speaker': pragmatic_speaker, 'listener': pragmatic_listener})

print("speaker predictions:\n", RSA(1, 1.6)['speaker'])

##################################################
## model parameters to fit
##################################################

alpha           = torch.tensor(1.0, requires_grad=True) # soft-max parameter
cost_adjectives = torch.tensor(0.0, requires_grad=True) # differential cost of 'adjectives'

##################################################
## optimization
##################################################

opt = torch.optim.SGD([alpha, cost_adjectives], lr = 0.0001)

# output header
print('\n%5s %24s %15s %15s' %
      ("step", "loss", "alpha", "cost") )

for i in range(4000):

    RSA_prediction      = RSA(alpha, cost_adjectives)
    speaker_pred        = RSA_prediction['speaker']
    Multinomial_speaker = torch.distributions.multinomial.Multinomial(144, probs = speaker_pred)
    logProbs_speaker    = Multinomial_speaker.log_prob(production_data)

    listener_pred          = RSA_prediction['listener']
    Multinomial_listener_0 = torch.distributions.multinomial.Multinomial(181,probs = listener_pred[0,])
    logProbs_listener_0    = Multinomial_listener_0.log_prob(interpretation_data[0,])
    Multinomial_listener_1 = torch.distributions.multinomial.Multinomial(179,probs = listener_pred[3,])
    logProbs_listener_1    = Multinomial_listener_1.log_prob(interpretation_data[1,])

    loss = -torch.sum(logProbs_speaker) - logProbs_listener_0 - logProbs_listener_1

    loss.backward()

    if (i+1) % 250 == 0:
        print('%5d %24.5f %15.5f %15.5f' %
              (i + 1, loss.item(), alpha.item(),
               cost_adjectives.item()) )

    opt.step()
    opt.zero_grad()
