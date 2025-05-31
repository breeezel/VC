import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Вспомогательные Модули ---

class SpeakerAdaptationMLP(nn.Module):
    # MLP для адаптации эмбеддинга диктора к параметрам стиля AdaIN
    def __init__(self, speaker_embedding_dim, adain_style_dim, num_layers=2, hidden_dim=256):
        super(SpeakerAdaptationMLP, self).__init__()
        layers = [nn.Linear(speaker_embedding_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1): layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, adain_style_dim * 2)); # Для scale и bias
        self.mlp = nn.Sequential(*layers)
        self.adain_style_dim = adain_style_dim
        # print(f"SpeakerAdaptationMLP: input_dim={speaker_embedding_dim}, output_dim={adain_style_dim*2}") # Закомментировано для чистоты логов

    def forward(self, speaker_embedding):
        # speaker_embedding: (Batch, speaker_embedding_dim)
        style_params = self.mlp(speaker_embedding) # (Batch, adain_style_dim * 2)
        # Разделение на scale и bias для AdaIN
        return style_params[:, :self.adain_style_dim], style_params[:, self.adain_style_dim:]

def adaptive_instance_normalization(content_feat, style_scale, style_bias):
    # Применяет Adaptive Instance Normalization.
    # content_feat: (Batch, Channels, Time) - признаки контента
    # style_scale: (Batch, Channels) - параметры масштабирования от SpeakerAdaptationMLP
    # style_bias: (Batch, Channels) - параметры смещения от SpeakerAdaptationMLP
    assert content_feat.size(0) == style_scale.size(0) == style_bias.size(0)
    assert content_feat.size(1) == style_scale.size(1) == style_bias.size(1)
    epsilon = 1e-5
    mean = content_feat.mean(dim=2, keepdim=True)
    std = content_feat.std(dim=2, keepdim=True) + epsilon
    normalized_feat = (content_feat - mean) / std
    return normalized_feat * style_scale.unsqueeze(2) + style_bias.unsqueeze(2)

class ResidualBlock(nn.Module):
    # Остаточный блок, используемый в "бутылочном горлышке" генератора
    def __init__(self, channels, adain_style_dim, kernel_size=3, dilation=1):
        super(ResidualBlock, self).__init__()
        # Использование GLU подразумевает, что выход conv слоя = 2*channels, затем GLU делит его пополам.
        self.conv1_glu = nn.Conv1d(channels, channels * 2, kernel_size, padding=dilation, dilation=dilation)
        self.glu1 = nn.GLU(dim=1) # GLU применяется по оси каналов
        self.conv2_glu = nn.Conv1d(channels, channels * 2, kernel_size, padding=dilation, dilation=dilation)
        self.glu2 = nn.GLU(dim=1)
        self.adain_style_dim = adain_style_dim
        # Комментарий: Применение AdaIN здесь концептуальное. Полная реализация
        # заменила бы InstanceNorm или была бы явным слоем. В этом скелете
        # предполагается, что style_scale/bias используются (непоказанным) механизмом AdaIN
        # внутри концептуального потока данных этого блока, вероятно, перед активациями.
        # print(f"ResidualBlock: channels={channels}, adain_style_dim_expected={adain_style_dim}") # Закомментировано

    def forward(self, x, style_scale, style_bias): # style_scale/bias предназначены для AdaIN
        residual = x
        # Слой 1
        out = self.conv1_glu(x)
        # Здесь должно быть применение AdaIN к 'out' или его компонентам перед GLU.
        # В данном скелете это опущено для простоты; предполагается, что AdaIN интегрирован.
        # Например, если AdaIN применяется к первой половине для GLU:
        # if out.size(1) // 2 == self.adain_style_dim:
        #    out_part1_adain = adaptive_instance_normalization(out[:, :out.size(1)//2, :], style_scale, style_bias)
        #    out_part2 = out[:, out.size(1)//2:, :]
        #    out_for_glu = torch.cat((out_part1_adain, out_part2), dim=1)
        # else: out_for_glu = out # Или ошибка, если размерности не совпадают
        out = self.glu1(out)
        # Слой 2
        out = self.conv2_glu(out)
        # Аналогичное концептуальное применение AdaIN здесь
        out = self.glu2(out)
        return out + residual

class Generator(nn.Module):
    # Архитектура Генератора StarGAN-VC
    def __init__(self, config): # Принимает объект конфигурации (config['model'])
        super(Generator, self).__init__()

        model_config = config
        gen_config = model_config['generator']

        self.num_speakers = model_config['num_speakers']
        self.speaker_embedding_dim = model_config['speaker_embedding_dim']
        mel_channels = gen_config['mel_channels']
        hidden_channels = gen_config['hidden_channels']
        bottleneck_channels = gen_config['bottleneck_channels']
        adain_style_dim = bottleneck_channels # Размерность стиля AdaIN = кол-ву каналов в bottleneck
        num_downsample_blocks = gen_config['num_downsample_blocks']
        num_residual_blocks = gen_config['num_residual_blocks']
        num_upsample_blocks = gen_config['num_upsample_blocks']

        # MLP для адаптации эмбеддинга диктора к параметрам AdaIN
        self.speaker_mlp = SpeakerAdaptationMLP(self.speaker_embedding_dim, adain_style_dim)

        # 1. Энкодер (понижение размерности)
        encoder_layers = [
            nn.Conv1d(mel_channels, hidden_channels * 2, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm1d(hidden_channels * 2, affine=True),
            nn.GLU(dim=1) # Выход: hidden_channels
        ]
        current_channels = hidden_channels
        for _ in range(num_downsample_blocks):
            encoder_layers.extend([
                nn.Conv1d(current_channels, current_channels * 2, kernel_size=4, stride=2, padding=1), # Stride 2 для понижения
                nn.InstanceNorm1d(current_channels*2, affine=True),
                nn.GLU(dim=1) # Выход: current_channels (т.к. GLU делит *2 каналы)
            ])
        self.encoder = nn.Sequential(*encoder_layers)

        # 2. "Бутылочное горлышко" (остаточные блоки с AdaIN)
        self.bottleneck_pre_conv = nn.Conv1d(current_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1)
        current_channels = bottleneck_channels

        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(current_channels, adain_style_dim))

        self.bottleneck_post_conv = nn.Conv1d(current_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        current_channels = hidden_channels

        # 3. Декодер (повышение размерности)
        decoder_layers = []
        for _ in range(num_upsample_blocks):
            decoder_layers.extend([
                nn.ConvTranspose1d(current_channels, current_channels * 2, kernel_size=4, stride=2, padding=1), # Повышение размерности
                nn.InstanceNorm1d(current_channels*2, affine=True),
                nn.GLU(dim=1)
            ])
        decoder_layers.append(nn.Conv1d(current_channels, mel_channels, kernel_size=7, stride=1, padding=3)) # Финальный слой к мел-каналам
        self.decoder = nn.Sequential(*decoder_layers)

        # print(f"Генератор инициализирован из конфигурации: кол-во дикторов={self.num_speakers}, мел-кан.={mel_channels}, скрыт.кан.={hidden_channels}")

    def forward(self, mel_spectrogram_source, speaker_embedding_target):
        # mel_spectrogram_source: (Batch, mel_channels, Time_Frames) - исходный контент
        # speaker_embedding_target: (Batch, speaker_embedding_dim) - целевой стиль диктора

        encoded_content = self.encoder(mel_spectrogram_source)
        bottleneck_input = self.bottleneck_pre_conv(encoded_content)

        # Получение параметров стиля из эмбеддинга целевого диктора для AdaIN
        style_scale, style_bias = self.speaker_mlp(speaker_embedding_target)

        x = bottleneck_input
        for block in self.residual_blocks:
            # Forward метод ResidualBlock должен принимать style_scale и style_bias
            # и применять AdaIN внутри.
            x = block(x, style_scale, style_bias)

        x = self.bottleneck_post_conv(x)
        output_mel_spectrogram = self.decoder(x)
        return output_mel_spectrogram

class Discriminator(nn.Module):
    # Архитектура Дискриминатора StarGAN-VC
    # (Можно также рефакторить для приема config, если параметры станут сложнее)
    def __init__(self, num_speakers, mel_channels=80, hidden_channels=256, num_conv_blocks=4, downsample_every_n=1):
        super(Discriminator, self).__init__()
        self.num_speakers = num_speakers; layers = []; current_channels = mel_channels
        for i in range(num_conv_blocks):
            out_channels = min(hidden_channels * (2**(i // downsample_every_n)), 1024) # Ограничение каналов
            layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size=3,
                          stride=2 if (i + 1) % downsample_every_n == 0 else 1, padding=1), # Понижение размерности каждый N-ый блок
                nn.LeakyReLU(0.2, inplace=True)])
            current_channels = out_channels
        self.conv_blocks = nn.Sequential(*layers)
        # Выход для оценки подлинности (patch-based)
        self.out_adv = nn.Conv1d(current_channels, 1, kernel_size=3, padding=1)
        # Выход для классификации диктора
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1) # Пулинг до (Batch, current_channels, 1)
        self.out_cls = nn.Linear(current_channels, num_speakers)
        # print(f"Дискриминатор инициализирован: кол-во дикторов={num_speakers}, мел-кан.={mel_channels}")

    def forward(self, mel_spectrogram):
        # mel_spectrogram: (Batch, mel_channels, Time_Frames)
        features = self.conv_blocks(mel_spectrogram)
        adv_score_patch = self.out_adv(features) # (B, 1, Time_downsampled_final) - оценка для каждого патча
        cls_logits = self.out_cls(self.adaptive_pool(features).squeeze(2)) # (B, num_speakers) - логиты классификации
        return adv_score_patch, cls_logits
