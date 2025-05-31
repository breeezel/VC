# Файл: src/models/hifi_gan_components.py
# Содержит упрощенные компоненты для симуляции архитектуры HiFi-GAN генератора.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

# Простой класс для доступа к ключам словаря как к атрибутам
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# Типичные остаточные блоки HiFi-GAN
class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=self.get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=self.get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=self.get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(self.init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=self.get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=self.get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=self.get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0.0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l_convs in [self.convs1, self.convs2]:
            for l in l_convs:
                remove_weight_norm(l)

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2


class ResBlock2(torch.nn.Module):
    # Другой тип остаточного блока, если используется в архитектуре
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=self.get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=self.get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0.0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2


class Generator(torch.nn.Module):
    # Упрощенный HiFi-GAN Генератор
    def __init__(self, h): # h - это AttrDict с конфигурацией HiFi-GAN
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Начальный сверточный слой для мел-спектрограммы (вход: 80 каналов)
        self.conv_pre = weight_norm(nn.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        # Блоки повышающей дискретизации (Upsampling Blocks)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                   h.upsample_initial_channel // (2 ** (i + 1)),
                                   k, u, padding=(k - u) // 2)))

        # Остаточные блоки (MRF - Multi-Receptive Field Fusion)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k_res, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                if h.resblock == '1': # Тип ResBlock1
                    self.resblocks.append(ResBlock1(ch, k_res, d))
                elif h.resblock == '2': # Тип ResBlock2
                    self.resblocks.append(ResBlock2(ch, k_res, d))
                else:
                    raise ValueError(f"Неизвестный тип ResBlock: {h.resblock}")

        # Финальный сверточный слой
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(self.init_weights)
        self.conv_post.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            m.weight.data.normal_(0.0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # x: входная мел-спектрограмма (batch, num_mels, frames)
        x = self.conv_pre(x) # (batch, upsample_initial_channel, frames)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x) # Повышение дискретизации
            # Применение MRF
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x) # (batch, 1, samples)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        # print('Удаление весовой нормализации...') # Закомментировано
        for l_ups in self.ups: remove_weight_norm(l_ups)
        for l_resblock_list in self.resblocks: l_resblock_list.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
