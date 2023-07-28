import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *
from diffusers import KandinskyV22PriorPipeline


@threestudio.register("kandinsky-prompt-processor")
class KandinskyPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.pipe = KandinskyV22PriorPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path
        ).to(self.device)

    def destroy_text_encoder(self) -> None:
        del self.pipe
        cleanup()

    @torch.no_grad()
    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B SEQ_LEN 1280"], Float[Tensor, "B SEQ_LEN 1280"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        text_embeddings, uncond_text_embeddings = self.pipe(prompt, negative_prompt, num_inference_steps=1000).to_tuple()
        return text_embeddings, uncond_text_embeddings

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        pipe = KandinskyV22PriorPipeline.from_pretrained(
            pretrained_model_name_or_path
        ).to("cuda:0")

        with torch.no_grad():
            text_embeddings = pipe(
                prompts, num_inference_steps=1000
            ).to_tuple()[0].cpu()

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del pipe
