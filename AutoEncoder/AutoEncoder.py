import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import torch.multiprocessing as mp

import time, datetime
import pickle
import numpy as np
from collections import OrderedDict

# model architecture and training

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        if kwargs['linear']:
            self.activation = nn.Identity()
        else:
            self.activation = nn.ReLU()

        encoder = OrderedDict()
        for idx,(i,o) in enumerate(zip(kwargs['layers'][:-1],
                                       kwargs['layers'][1:])):
            encoder["encoder-"+str(idx)] = nn.Sequential(nn.Linear(in_features=i,
                                                                   out_features=o,
                                                                   bias=kwargs['bias']),
                                                         self.activation)
        self.encoder = nn.Sequential(encoder)


        decoder = OrderedDict()
        for idx,(i,o) in enumerate(zip(list(reversed(kwargs['layers']))[:-1],
                                       list(reversed(kwargs['layers']))[1:])):
            decoder["decoder-"+str(idx)] = nn.Sequential(nn.Linear(in_features=i,
                                                                   out_features=o,
                                                                   bias=kwargs['bias']),
                                                         self.activation)

        # change last layer to linear if params['last-linear'] == True
        enc = len(kwargs['layers'])-2
        if kwargs['last-linear']:
            decoder["decoder-"+str(enc)] = nn.Sequential(nn.Linear(in_features=kwargs['layers'][1],
                                                                   out_features=kwargs['layers'][0],
                                                                   bias=kwargs['bias']))

        self.decoder = nn.Sequential(decoder)


    def forward(self, x):
        shape = x.shape
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.reshape(x, shape)
        return x

    def activations(self, x, rank_only=True):
        self.eval()
        Phi = []
        with torch.no_grad():
            shape = x.shape
            x = torch.flatten(x, start_dim=1)

            for l in self.encoder:
                x = l(x)
                if rank_only:
                    Phi.append( np.linalg.matrix_rank(np.matrix(x)) )
                else:
                    Phi.append(x)

            for l in self.decoder:
                x = l(x)
                if rank_only:
                    Phi.append( np.linalg.matrix_rank(np.matrix(x)) )
                else:
                    Phi.append(x)

        return Phi

def train(model, train_loader, q, device, verbose=False, **kwargs):
    optimizer=  kwargs['optimizer'](model.parameters(), lr=kwargs['lr'])
    model.train()
    hist = []

    for epoch in range(kwargs['epochs']):
        loss = torch.zeros((), device = device)

        for batch_features, _ in train_loader:
            batch_features = batch_features.to(device)
            optimizer.zero_grad()

            outputs = model(batch_features)
            train_loss = kwargs['criterion'](outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        loss = loss / len(train_loader)
       	hist.append(loss.item())
        if verbose:
            print("{};{:.6f}".format(epoch + 1, loss))
    q.put(hist)

def get_loader(set0, batch_size):
    return torch.utils.data.DataLoader(set0, batch_size = batch_size, shuffle = True)


def cuda_memory() -> str:
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = c-a  # free inside cache
    return str("reserved cache {:.0f}/{:.0f} MB.".format(c/1000**2,t/1000**2))


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# hyperparams
	params = {
		# model
		"linear": False,
		"bias": False,
		"last-linear": False,
		"layers": [784,10,1],

		# training
		"lr": 1e-3,
		"epochs": 40,
		"iterations": 5,
		"optimizer": optim.Adam,
		"criterion": nn.MSELoss(reduction='mean'),
		"hidden_sizes": [[500],[1000],[2000],[1000,500],[2000,1000],[2000,1000,500],[2000,1000,500,300],[2000,1000,500,300,100]],
		"code_lengths": [1,2,3,4,5],
		"batch_sizes": [10,100,500,1000],
		"activation_types": ["ReLU","linear"],

		# misc
		"verbose": False,
		"device": device,
		"model_save": False,
	}

	# load data
	set0 = tv.datasets.MNIST("../MNIST/",
			                 download=True,
			                 train = True,
			                 transform = tv.transforms.ToTensor())
	loader0 = get_loader(set0, len(set0))
	dataiter = iter(loader0)
	image, _ = dataiter.next()

	params.update({
		"code_lengths": [1,2,3],
		"batch_sizes": [100],
		"activation_types": ["ReLU"],
		#"hidden_sizes": [[10],[20]],
		#"epochs": 3,
		#"iterations": 2,
	})

	verbose = True
	if verbose:
		print(device)
		cuda_memory()

	start = round(time.time())

	mp.set_start_method('spawn', force=True)
	q = mp.Queue()

	for activation_type in params['activation_types']:
		if verbose: print(activation_type)
		params['linear'] = (activation_type == "linear")
		for batch in params['batch_sizes']:
		    loader = get_loader(set0, batch)
		    if verbose:
		        print("Batch size: {}".format(batch))

		    for code in params['code_lengths']:
		        for hidden in params['hidden_sizes']:
		            if verbose: print(hidden + [code])
		            params['layers'] = [784] + hidden + [code]
		            name = "AE-{}-Adam-L{}-B{}".format(activation_type, params['layers'][1:], batch)
		            Phi = [[] for i in range(params['iterations'])]
		            Hist = [[] for i in range(params['iterations'])]

		            for i in range(params['iterations']):
		                model = AE(**params).to(device)
		                model.share_memory()
		                p = mp.Process(target=train, args=(model, loader, q), kwargs=params)
		                p.start()
		                p.join()
		                del p

		                model.to(torch.device("cpu"))
		                Phi[i] = model.activations(image)
		                Hist[i] = q.get()
		            print(cuda_memory())

		            # saving the results
		            if params['model_save'] == True:
			            torch.save(model.state_dict(),"models/"+name+"_model")
		            with open("models/"+name+"_activations", "wb") as fp:
		                pickle.dump(Phi, fp)
		            with open("models/"+name+"_history", "wb") as fp:
		                pickle.dump(Hist, fp)

		            del model, Phi
		print("\n")


	end = round(time.time())
	print("\nelapsed in {}".format(datetime.timedelta(seconds=(end - start))))


if __name__ == '__main__':
	main()
