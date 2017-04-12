from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import model
import dataset, arguments

# Training settings

args = args.get_arguments()

model = model.Segmenter(2)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
training = make_trainer(args)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(training.loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(training.dataset),
                100. * batch_idx / len(training.loader), loss.data[0]))

# def test(epoch):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += F.nll_loss(output, target).data[0]
#         pred = output.data.max(1)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()
#
#     test_loss = test_loss
#     test_loss /= len(test_loader) # loss function already averages over batch size
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
