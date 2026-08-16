import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.Doconv import DOConv2d

class FuzzyLearning(nn.Module):
    def __init__(self, in_channel, fuzzynum, fuzzychannel, T=20) -> None:
        super(FuzzyLearning, self).__init__()
        self.n = fuzzynum
        self.T = T

        self.conv1 = nn.Conv2d(in_channel, fuzzychannel, 3, padding=1)
        self.conv2 = nn.Conv2d(3 * fuzzychannel, in_channel, 3, padding=1)

        self.mu = nn.Parameter(torch.randn((fuzzychannel, self.n)))
        self.sigma = nn.Parameter(torch.randn((fuzzychannel, self.n)))

    def forward(self, x):
        x = self.conv1(x)
        feat = x.permute((0, 2, 3, 1))

        member = feat.unsqueeze(-1).expand(-1, -1, -1, -1, self.n)
        member = torch.exp(-((member - self.mu) / self.sigma) ** 2)

        sample = torch.randn_like(member) * self.sigma + self.mu

        member_and = torch.sum(
            sample * torch.softmax((1 - member) * self.T, dim=4),
            dim=4).permute((0, 3, 1, 2))

        member_or = torch.sum(
            sample * torch.softmax(member * self.T, dim=4),
            dim=4).permute((0, 3, 1, 2))

        feat = torch.cat([x, member_and, member_or], dim=1)
        feat = self.conv2(feat)
        return feat


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h, dct_w,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16):
        super(MultiFrequencyChannelAttention, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)

        # fixed DCT init
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx),
                                 self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x

        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        for name, params in self.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq

        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        multi_spectral_attention_map = F.sigmoid(
            multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y,
                                                                                                            mapper_y,
                                                                                                            tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
        

class SEBlock(nn.Module):
    def __init__(self, 
            in_channels: int,
            out_channels: int
        ):
        super(SEBlock, self).__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels//16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.extract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor):
        weight = self.se(x)
        x = x * weight
        res = self.extract(x)
        return res

class EEM(nn.Module):
    def __init__(self, in_channel, fuzzynum, fuzzychannel, dct_h, dct_w, frequency_branches=16,
                 frequency_selection='top', reduction=16):
        super(EEM, self).__init__()
        self.flm = FuzzyLearning(in_channel, fuzzynum, fuzzychannel)
        self.mfca = MultiFrequencyChannelAttention(in_channel, dct_h, dct_w, frequency_branches, frequency_selection,
                                                   reduction)
        """
        self.res = nn.Sequential(
            DOConv2d(in_channels=in_channel*3,out_channels=in_channel,kernel_size=3),
            nn.ReLU(),
            nn.Sigmoid()
        )"""
        self.se = SEBlock(in_channel*3,in_channel)
    def forward(self, x):
        x1 = self.flm(x)
        x2 = self.mfca(x)
        concatenated_feature = torch.cat([x, x1, x2], dim=1)
        res = self.se(concatenated_feature)
        return res

"""
if __name__ == '__main__':
    input = torch.randn(2, 32, 512, 512)
    eca = EEM(32, 16, 32, 32, 256)
    output = eca(input)
    print(output.shape)
"""