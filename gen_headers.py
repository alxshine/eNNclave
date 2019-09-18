import torch

state = torch.load("mnist_cnn.pt")
f = open('state.h', 'w+')
f.write('#ifndef STATE_H\n#define STATE_H\n')

w1 = state['fc1.weight'].detach().numpy().T
w1_w = w1.shape[0]
w1_h = w1.shape[1]
w1 = w1.flatten()

f.write('int w1_w = %d;\n' % w1_w)
f.write('int w1_h = %d;\n' % w1_h)
f.write('float w1[] = {\n')
for i, v in enumerate(w1):
    f.write('%f, ' % v)
    if (i + 1) % 8 == 0:
        f.write('\n')
f.write('};\n')

b1 = state['fc1.bias'].detach().numpy()
b1_h = b1.shape[0]
b1 = b1.flatten()

f.write('\n')

f.write('int b1_h = %d;\n' % b1_h)
f.write('float b1[] = {\n')
for i, v in enumerate(b1):
    f.write('%f, ' % v)
    if (i + 1) % 8 == 0:
        f.write('\n')
f.write('};\n')

f.write('\n')

w2 = state['fc2.weight'].detach().numpy().T
w2_w = w2.shape[0]
w2_h = w2.shape[1]
w2 = w2.flatten()

f.write('int w2_w = %d;\n' % w2_w)
f.write('int w2_h = %d;\n' % w2_h)
f.write('float w2[] = {\n')
for i, v in enumerate(w2):
    f.write('%f, ' % v)
    if (i + 1) % 8 == 0:
        f.write('\n')
f.write('};\n')

b2 = state['fc2.bias'].detach().numpy()
b2_h = b2.shape[0]
b2 = b2.flatten()

f.write('\n')

f.write('int b2_h = %d;\n' % b2_h)
f.write('float b2[] = {\n')
for i, v in enumerate(b2):
    f.write('%f, ' % v)
    if (i + 1) % 8 == 0:
        f.write('\n')
f.write('};\n')

f.write('\n')
f.write('#endif\n')
