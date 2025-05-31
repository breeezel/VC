import torch
import torch.nn.functional as F

# --- Adversarial Losses ---
def calculate_generator_adv_loss(fake_discriminator_output):
    """
    Calculates adversarial loss for the generator (e.g., LSGAN).
    Aims to make the discriminator output 1 (or target_real_label) for fake samples.
    Args:
        fake_discriminator_output (torch.Tensor): Discriminator's output for fake samples.
    Returns:
        torch.Tensor: Generator's adversarial loss.
    """
    # Example: LSGAN loss - (D(G(z)) - 1)^2
    loss = torch.mean((fake_discriminator_output - 1.0)**2)
    # print(f"Loss G_adv: {loss.item()}") # For debugging
    return loss

def calculate_discriminator_adv_loss(real_discriminator_output, fake_discriminator_output):
    """
    Calculates adversarial loss for the discriminator (e.g., LSGAN).
    Aims to make discriminator output 1 for real samples and 0 for fake samples.
    Args:
        real_discriminator_output (torch.Tensor): Discriminator's output for real samples.
        fake_discriminator_output (torch.Tensor): Discriminator's output for fake samples.
    Returns:
        torch.Tensor: Discriminator's adversarial loss.
    """
    # Example: LSGAN loss - (D(x) - 1)^2 + (D(G(z)))^2
    loss_real = torch.mean((real_discriminator_output - 1.0)**2)
    loss_fake = torch.mean(fake_discriminator_output**2)
    total_loss = loss_real + loss_fake
    # print(f"Loss D_adv_real: {loss_real.item()}, D_adv_fake: {loss_fake.item()}, Total: {total_loss.item()}") # For debugging
    return total_loss

# --- Reconstruction Loss (formerly Cycle Consistency) ---
def calculate_reconstruction_loss(original_mel, reconstructed_mel):
    """
    Calculates L1 reconstruction loss.
    Used for StarGAN-VC's source reconstruction: G(source_mel, source_speaker_emb) vs source_mel.
    Args:
        original_mel (torch.Tensor): The original mel-spectrogram.
        reconstructed_mel (torch.Tensor): The mel-spectrogram reconstructed by the generator.
    Returns:
        torch.Tensor: L1 reconstruction loss.
    """
    loss = F.l1_loss(original_mel, reconstructed_mel)
    # print(f"Loss Recon: {loss.item()}") # For debugging
    return loss

# --- Identity Mapping Loss ---
def calculate_identity_mapping_loss(original_mel, identity_reconstructed_mel):
    """
    Calculates L1 identity mapping loss.
    Ensures G(target_mel, target_speaker_emb) is close to target_mel.
    Args:
        original_mel (torch.Tensor): Mel-spectrogram of the target speaker.
        identity_reconstructed_mel (torch.Tensor): Output of G(original_mel, target_speaker_emb, target_speaker_emb).
    Returns:
        torch.Tensor: L1 identity mapping loss.
    """
    loss = F.l1_loss(original_mel, identity_reconstructed_mel)
    # print(f"Loss Identity: {loss.item()}") # For debugging
    return loss

# --- Speaker Classification Losses ---
def calculate_speaker_classification_loss_generator(target_speaker_labels, predicted_speaker_logits_on_fake):
    """
    Calculates speaker classification loss for the generator.
    Generator tries to make the discriminator classify fake samples as the target speaker.
    Args:
        target_speaker_labels (torch.Tensor): True labels of the target speakers for the fake samples.
        predicted_speaker_logits_on_fake (torch.Tensor): Discriminator's speaker classification logits for fake samples.
    Returns:
        torch.Tensor: Generator's speaker classification loss.
    """
    loss = F.cross_entropy(predicted_speaker_logits_on_fake, target_speaker_labels)
    # print(f"Loss G_cls: {loss.item()}") # For debugging
    return loss

def calculate_speaker_classification_loss_discriminator(real_speaker_labels, predicted_speaker_logits_on_real):
    """
    Calculates speaker classification loss for the discriminator.
    Discriminator tries to correctly classify the speaker of real samples.
    Args:
        real_speaker_labels (torch.Tensor): True labels of the real speakers.
        predicted_speaker_logits_on_real (torch.Tensor): Discriminator's speaker classification logits for real samples.
    Returns:
        torch.Tensor: Discriminator's speaker classification loss.
    """
    loss = F.cross_entropy(predicted_speaker_logits_on_real, real_speaker_labels)
    # print(f"Loss D_cls: {loss.item()}") # For debugging
    return loss

# --- Keep old placeholders if they were different and are still needed ---
# Based on instructions, the above are the primary implementations.
# Old placeholders like calculate_cycle_consistency_loss (if it was a distinct concept) are now removed or renamed.
