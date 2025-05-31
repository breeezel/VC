import torch
import torch.nn as nn
import torch.nn.functional as F

# (SpeakerAdaptationMLP, adaptive_instance_normalization, ResidualBlock definitions remain the same as previous version)
class SpeakerAdaptationMLP(nn.Module):
    def __init__(self, speaker_embedding_dim, adain_style_dim, num_layers=2, hidden_dim=256):
        super(SpeakerAdaptationMLP, self).__init__()
        layers = [nn.Linear(speaker_embedding_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1): layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, adain_style_dim * 2)); self.mlp = nn.Sequential(*layers)
        self.adain_style_dim = adain_style_dim
        print(f"SpeakerAdaptationMLP: input_dim={speaker_embedding_dim}, output_dim={adain_style_dim*2}")

    def forward(self, speaker_embedding):
        style_params = self.mlp(speaker_embedding)
        return style_params[:, :self.adain_style_dim], style_params[:, self.adain_style_dim:]

def adaptive_instance_normalization(content_feat, style_scale, style_bias):
    assert content_feat.size(0) == style_scale.size(0) == style_bias.size(0)
    assert content_feat.size(1) == style_scale.size(1) == style_bias.size(1)
    epsilon = 1e-5; mean = content_feat.mean(dim=2, keepdim=True); std = content_feat.std(dim=2, keepdim=True) + epsilon
    normalized_feat = (content_feat - mean) / std
    return normalized_feat * style_scale.unsqueeze(2) + style_bias.unsqueeze(2)

class ResidualBlock(nn.Module):
    def __init__(self, channels, adain_style_dim, kernel_size=3, dilation=1):
        super(ResidualBlock, self).__init__()
        # Using GLU implies conv output is 2*channels, then GLU halves it.
        self.conv1_glu = nn.Conv1d(channels, channels * 2, kernel_size, padding=dilation, dilation=dilation)
        self.glu1 = nn.GLU(dim=1)
        self.conv2_glu = nn.Conv1d(channels, channels * 2, kernel_size, padding=dilation, dilation=dilation)
        self.glu2 = nn.GLU(dim=1)
        self.adain_style_dim = adain_style_dim # To verify compatibility with style_scale/bias
        # Comment: AdaIN application is conceptual here. A full implementation
        # would replace InstanceNorm or be an explicit layer. For this skeleton,
        # we assume style_scale/bias are used by an (unshown) AdaIN mechanism
        # within this block's conceptual dataflow, likely before activations.
        print(f"ResidualBlock: channels={channels}, adain_style_dim_expected={adain_style_dim}")

    def forward(self, x, style_scale, style_bias): # style_scale/bias are for AdaIN
        residual = x
        # Layer 1
        out = self.conv1_glu(x)
        # Conceptual AdaIN application to 'out' or its components before GLU
        # For example, if AdaIN is applied to the first half for GLU:
        # if out.size(1) // 2 == self.adain_style_dim: # Check if style dim matches half channels for GLU
        #    out_part1_adain = adaptive_instance_normalization(out[:, :out.size(1)//2, :], style_scale, style_bias)
        #    out_part2 = out[:, out.size(1)//2:, :]
        #    out_for_glu = torch.cat((out_part1_adain, out_part2), dim=1)
        # else:
        #    out_for_glu = out # Or raise error if dimensions mismatch for AdaIN
        # For this skeleton, we pass through, assuming AdaIN integrated into GLU conceptually or not explicitly shown.
        out = self.glu1(out)
        # Layer 2
        out = self.conv2_glu(out)
        # Similar conceptual AdaIN application here
        out = self.glu2(out)
        return out + residual

class Generator(nn.Module):
    def __init__(self, config): # Changed to accept config object
        super(Generator, self).__init__()

        model_config = config # Assuming 'config' passed is config['model']
        gen_config = model_config['generator']

        self.num_speakers = model_config['num_speakers']
        self.speaker_embedding_dim = model_config['speaker_embedding_dim']
        mel_channels = gen_config['mel_channels']
        hidden_channels = gen_config['hidden_channels']
        bottleneck_channels = gen_config['bottleneck_channels']
        adain_style_dim = bottleneck_channels # AdaIN style dim matches bottleneck channels
        num_downsample_blocks = gen_config['num_downsample_blocks']
        num_residual_blocks = gen_config['num_residual_blocks']
        num_upsample_blocks = gen_config['num_upsample_blocks']

        self.speaker_mlp = SpeakerAdaptationMLP(self.speaker_embedding_dim, adain_style_dim)

        encoder_layers = [
            nn.Conv1d(mel_channels, hidden_channels * 2, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm1d(hidden_channels * 2, affine=True),
            nn.GLU(dim=1)
        ]
        current_channels = hidden_channels
        for _ in range(num_downsample_blocks):
            encoder_layers.extend([
                nn.Conv1d(current_channels, current_channels * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm1d(current_channels*2, affine=True),
                nn.GLU(dim=1)
            ])
        self.encoder = nn.Sequential(*encoder_layers)

        self.bottleneck_pre_conv = nn.Conv1d(current_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1)
        current_channels = bottleneck_channels

        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(current_channels, adain_style_dim))

        self.bottleneck_post_conv = nn.Conv1d(current_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        current_channels = hidden_channels

        decoder_layers = []
        for _ in range(num_upsample_blocks):
            decoder_layers.extend([
                nn.ConvTranspose1d(current_channels, current_channels * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm1d(current_channels*2, affine=True),
                nn.GLU(dim=1)
            ])
        decoder_layers.append(nn.Conv1d(current_channels, mel_channels, kernel_size=7, stride=1, padding=3))
        self.decoder = nn.Sequential(*decoder_layers)

        print(f"Generator initialized from config: num_speakers={self.num_speakers}, mel_ch={mel_channels}, hidden_ch={hidden_channels}")

    def forward(self, mel_spectrogram_source, speaker_embedding_target):
        encoded_content = self.encoder(mel_spectrogram_source)
        bottleneck_input = self.bottleneck_pre_conv(encoded_content)
        style_scale, style_bias = self.speaker_mlp(speaker_embedding_target)
        x = bottleneck_input
        for block in self.residual_blocks:
            x = block(x, style_scale, style_bias)
        x = self.bottleneck_post_conv(x)
        output_mel_spectrogram = self.decoder(x)
        return output_mel_spectrogram

class Discriminator(nn.Module): # Definition remains largely same, but ensure it can be init with config too if needed
    def __init__(self, num_speakers, mel_channels=80, hidden_channels=256, num_conv_blocks=4, downsample_every_n=1):
        super(Discriminator, self).__init__()
        self.num_speakers = num_speakers; layers = []; current_channels = mel_channels
        for i in range(num_conv_blocks):
            out_channels = min(hidden_channels * (2**(i // downsample_every_n)), 1024)
            layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size=3,
                          stride=2 if (i + 1) % downsample_every_n == 0 else 1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)])
            current_channels = out_channels
        self.conv_blocks = nn.Sequential(*layers)
        self.out_adv = nn.Conv1d(current_channels, 1, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.out_cls = nn.Linear(current_channels, num_speakers)
        print(f"Discriminator initialized: num_speakers={num_speakers}, mel_ch={mel_channels}")

    def forward(self, mel_spectrogram):
        features = self.conv_blocks(mel_spectrogram)
        adv_score_patch = self.out_adv(features)
        cls_logits = self.out_cls(self.adaptive_pool(features).squeeze(2))
        return adv_score_patch, cls_logits
