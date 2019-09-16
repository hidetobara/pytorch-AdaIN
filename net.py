import torch
import torch.nn as nn
import torch.nn.functional as F

from function import adaptive_instance_normalization as adain
from function import calc_mean_std

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class Abstracter(nn.Module):
    def __init__(self, size=3):
        super(Abstracter, self).__init__()
        self.size = size
        self.down = nn.Linear(512*(self.size**2), 8)
        self.up = nn.Linear(8, 512*(self.size**2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def execute(self, input):
        i = nn.functional.upsample(input, size=(self.size,self.size), mode="bilinear", align_corners=True)
        i = i.view(-1, 512*(self.size**2))
        o = self.relu(self.up(self.down(i)))
        mean = o.mean(dim=1)
        std = o.var(dim=1).sqrt() + 0.001
        o = (o - mean) / std + mean
        o = o.view(-1, 512, self.size, self.size)
        return nn.functional.upsample(o, size=input.size()[2:], mode="bilinear", align_corners=True)

    def forward(self, input):
        i = nn.functional.upsample(input, size=(self.size,self.size), mode="bilinear", align_corners=True)
        i = i.view(-1, 512*(self.size**2))
        if self.training:
            i = self.dropout(i)
        o = self.relu(self.up(self.down(i)))
        #print("size=", input.size(), i.size(), o.size())
        return i, o

    def save(self, path):
        state = self.state_dict()
        for key in state.keys():
            state[key] = state[key].to('cpu')
        torch.save(state, path)
    def load(self, path):
        self.load_state_dict(torch.load(path))

class Abstracter2(nn.Module):
    def __init__(self):
        super(Abstracter2, self).__init__()
        self.prepare = nn.Sequential(
            nn.AdaptiveMaxPool2d( (32,32) ))
        self.down = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 128, (3, 3)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            )
        self.up = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 128, (3, 3)),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 512, (3, 3)),
            nn.ReLU()
            )

    def execute(self, input):
        size = input.size()[2:]
        i = self.prepare(input)
        m = self.down(i)
        o = self.up(m)
        #print("exe=", i, m, o)
        o = nn.functional.upsample(o, size=size, mode='bilinear', align_corners=True)
        #o = nn.functional.upsample(o, size=size, mode='nearest')
        return o.view(1, 512, size[0], size[1])
    def forward(self, input):
        i = self.prepare(input)
        o = self.up(self.down(i))
        return i, o

    def save(self, path):
        state = self.state_dict()
        for key in state.keys():
            state[key] = state[key].to('cpu')
        torch.save(state, path)
    def load(self, path):
        self.load_state_dict(torch.load(path))

class Corrector(nn.Module):
    def __init__(self):
        super(Corrector, self).__init__()
        self.down = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU())
        self.correct = nn.Sequential(
            nn.Linear(512, 32),
            nn.Linear(32, 2))
        self.cel = torch.nn.CrossEntropyLoss()
    def execute(self, input):
        size = input.size()
        N, C, H, W = size
        downed = self.down(input)
        corrected = self.correct(downed.transpose(1,3).transpose(1,2).contiguous().view(-1, C))
        cross = F.softmax(corrected * 0.2, dim=1)
        #print(corrected, cross[:,1])
        return cross[:,1].view(N, 1, H, W)
    def forward(self, input):
        size = input.size()
        N, C, H, W = size
        downed = self.down(input)
        tr = downed.transpose(1,3).transpose(1,2).contiguous()
        return self.correct(tr.view(N*H*W, C))

    def save(self, path):
        state = self.state_dict()
        for key in state.keys():
            state[key] = state[key].to('cpu')
        torch.save(state, path)
    def load(self, path):
        self.load_state_dict(torch.load(path))

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
