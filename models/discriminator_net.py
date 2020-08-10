from torch import nn

class Discriminator(nn.Module):
	def __init__(self, hidden_size=512):
		super(Discriminator,self).__init__()

		self.model = nn.Sequential(
			nn.Linear(hidden_size, hidden_size//2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(hidden_size//2, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
			# nn.Softmax()
			nn.Sigmoid()
		)

	def forward(self, vqg, questions, qlengths):         # pass object of vqg
		encoder_hidden = vqg.encode_questions_discriminator(questions, qlengths)
		x = encoder_hidden.view(-1, vqg.hidden_size)
		verdict = self.model(x)
		return verdict