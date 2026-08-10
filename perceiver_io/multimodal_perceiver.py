from typing import Sequence

import torch.nn as nn
import torch

from perceiver_io.io_processors.postprocessors import AudioPostprocessor, ProjectionPostprocessor, \
    ClassificationPostprocessor
from perceiver_io.io_processors.preprocessors import AudioPreprocessor, ImagePreprocessor, OneHotPreprocessor,FeaturePreprocessor
from perceiver_io.output_queries import FourierQuery, TrainableQuery
from perceiver_io.perceiver import PerceiverIO
from perceiver_io.position_encoding import PosEncodingType


class MultiModalPerceiver(nn.Module):
    """
    MultiModalPerceiver: Perceiver for auto-encoding video data.
    Args:
        img_size (Sequence[int]): Size of the image. Default: (224, 224)
        img_channels (int): Number of channels of the image. Default: 3
        num_frames (int): Number of frames to use for the video. Default: 16
        num_classes (int): Number of possible classes. Default: 700
        audio_samples_per_frame (int): Number of audio samples per video frame. Default: 128
        audio_samples_per_patch (int): Number of audio samples that are combined as a patch. Default: 16
        num_self_attends_per_block (int): Number of self attends per block. Default: 8
        num_blocks (int): Number of blocks. All blocks share weights. Default: 1
        num_latents (int): Number of latent variables. Default: 784
        num_latent_channels (int): Number of channels for latent variables. Default: 512
    """

    def __init__(
            self,
            img_size: Sequence[int] = (224, 224),
            img_channels: int = 3,
            num_frames: int = 16,
            num_classes: int = 700,
            audio_samples_per_frame: int = 48000 // 25,
            audio_samples_per_patch: int = 16,
            num_self_attends_per_block: int = 8,
            num_blocks: int = 1,
            num_latents: int = 28 * 28 * 1,
            num_latent_channels: int = 512,
            dropout_prob: float = 0.3,
            mask_prob: float = 0.3,
            is_gated: bool = False,
            is_skip_connection: bool = False,
            module: str = "multi",  # new: control input modality: 'audio', 'visual', or 'multi'
        ):

        super().__init__()

        self.model = module.lower()
        self.H, self.W = img_size
        self.num_classes = num_classes
        self.audio_samples_per_frame = audio_samples_per_frame
        self.audio_samples_per_patch = audio_samples_per_patch

        n_audio_samples = num_frames * audio_samples_per_frame

        input_preprocessors = {}

        if self.model in ["audio", "multi"]:
            input_preprocessors["audio"] = AudioPreprocessor(
                samples_per_batch=n_audio_samples,
                position_encoding_type=PosEncodingType.FOURIER,
                fourier_position_encoding_kwargs=dict(
                    num_bands=192,
                    max_resolution=(n_audio_samples,),
                    sine_only=False,
                    concat_pos=True,
                ),
                n_extra_pos_mlp=0,
                prep_type="patches",
                samples_per_patch=audio_samples_per_patch)
        if self.model in ["visual", "multi"]:
            input_preprocessors["image"] = ImagePreprocessor(
                img_size=(self.H, self.W),
                input_channels=img_channels,
                num_frames=num_frames,
                position_encoding_type=PosEncodingType.FOURIER,
                fourier_position_encoding_kwargs=dict(
                    num_bands=32,
                    max_resolution=(num_frames, self.H // 4, self.W // 4),
                    sine_only=False,
                    concat_pos=True,
                ),
                n_extra_pos_mlp=0,
                prep_type="patches",
                spatial_downsample=4,
                temporal_downsample=1)

        # Always include label preprocessor
        input_preprocessors["label"] = OneHotPreprocessor(
            input_channels=num_classes,
        )

        output_postprocessors = {
            "label": ClassificationPostprocessor(
                num_input_channels=num_latent_channels,
                num_classes=num_classes),
        }

        label_out_query = TrainableQuery(
            output_index_dims=(10,),
            concat_preprocessed_input=False,
            num_channels=num_latent_channels,
            init_scale=0.02)

        output_queries = {
            "label": label_out_query }

        self.perceiver = PerceiverIO(
            num_self_attends_per_block=num_self_attends_per_block,
            num_blocks=num_blocks,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            input_preprocessors=input_preprocessors,
            output_postprocessors=output_postprocessors,
            output_queries=output_queries,
            input_padding_channels=4,
            output_query_padding_channels=2,
            input_mask_probs={"image": mask_prob, "audio": mask_prob, "label": 1},
            perceiver_encoder_kwargs={'dropout_prob': dropout_prob},
            is_gated=is_gated,
            is_skip_connection = is_skip_connection)

    def forward(self, images: torch.Tensor, audio: torch.Tensor, label: torch.Tensor):
        """"""
        reconstruction = {}
        inputs = {}
        inputs["label"] = label
        if self.model in ["audio", "multi"]:
            inputs["audio"] = audio
        if self.model in ["visual", "multi"]:
            inputs["image"] = images

        output = self.perceiver(
           inputs)
        reconstruction["label"] = output["label"]
        return reconstruction
