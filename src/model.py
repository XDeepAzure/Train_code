
from typing import Any, Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
import os
import json

from transformers import (M2M100ForConditionalGeneration,
                          M2M100Config,
                          M2M100Model,)
from transformers.modeling_outputs import Seq2SeqLMOutput

class PureFFN(nn.Module):
    """
    投影的前馈网络，在forward中会返回loss
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(PureFFN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # self.model = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size),
        #                            nn.ReLU(),
        #                            nn.Linear(in_features=hidden_size, out_features=output_size))
        
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x, y, y_mask):
        """
        给distilled model的represent (y) 和model对应的represent (x) ，计算二者的loss
        """

        # y_hat = self.model(x)
        def _todevice(x, y):
            return y.to(device=x.device)
        y_mask = _todevice(y, y_mask!=0)                # 不为0表示不被mask
        y_hat = x
        y_hat = _todevice(y, y_hat)
        
        y_hat_mask = y_hat[y_mask]                      # 取出未被mask的部分, 
        y1_mask = y[y_mask]

        # 计算蒸馏loss
        loss = self.loss_fn(input=y_hat_mask, target=y1_mask)
        return loss

    @staticmethod
    def from_pretrained(path):
        return None

    def save_pretrained(self, path):
        config = {"input_size":self.input_size,
                  "hidden_size": self.hidden_size,
                  "output_size": self.output_size}
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        # state_dict = self.model.state_dict()
        # torch.save(state_dict, os.path.join(path, "pytorch_model.bin"))
        pass


class PureM2M100(M2M100ForConditionalGeneration):
    
    def __init__(self, config: M2M100Config, distilled_model_path=None):
        super().__init__(config)
        input_size = 1024
        hidden_size = 1024
        output_size = 1024
        self.ffn = PureFFN(input_size=input_size,
                           hidden_size=hidden_size,
                           output_size=output_size)
        
        self.distilled_model = M2M100Model.from_pretrained("/data/hyxu/codes/LLMPruner/model/nllb-200-distilled-1.3B")
        self.distilled_model = self.distilled_model.encoder
        # self.distilled_model = None
    # @staticmethod
    # def from_pretrained(path):
    #     model = super().from_pretrained(path)
    #     ffn_path = os.path.join(path, "ffn.bin")
    #     if os.path.exists(ffn_path):
    #         ffn = torch.load(ffn_path)
    #         model.ffn.load_state_dict(ffn)
    #     else:
    #         model.ffn = PureFFN(1024,1024,1024)
    #     return model

    def save_pretrained(self, save_directory: str | os.PathLike, is_main_process: bool = True, state_dict: dict | None = None, save_function: Callable[..., Any] = torch.save, push_to_hub: bool = False, max_shard_size: int | str = "10GB", safe_serialization: bool = False, variant: str | None = None, **kwargs):
        ffn_path = os.path.join(save_directory, "ffn.bin")
        torch.save(self.ffn.state_dict(), ffn_path)
        return super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, variant, **kwargs)

    def distilled_fn(self, input_ids, attention_mask, student_encoder_hidden_states):
        if input_ids.device != self.distilled_model.device:
            input_ids = input_ids.to(self.distilled_model.device)
            attention_mask = attention_mask.to(self.distilled_model.device)
        teacher_encoder_hidden_states = self.distilled_model(input_ids = input_ids,
                                                             attention_mask = attention_mask,
                                                             output_hidden_states=True).hidden_states
        loss = self.ffn(x=student_encoder_hidden_states[-1],
                        y=teacher_encoder_hidden_states[-1],
                        y_mask=attention_mask)
        # loss += self.ffn(x=student_encoder_hidden_states[-2],
        #                 y=teacher_encoder_hidden_states[-2],
        #                 y_mask=attention_mask)
        return loss

    def forward(
        self,
        distilled_input_ids = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,                         #! None -> Ture
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """

        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if labels is not None:
        #     if decoder_input_ids is None:
        #         decoder_input_ids = shift_tokens_right(
        #             labels, self.config.pad_token_id, self.config.decoder_start_token_id
        #         )
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  decoder_input_ids =decoder_input_ids,
                                  decoder_attention_mask = decoder_attention_mask,
                                  head_mask = head_mask,
                                  decoder_head_mask = decoder_head_mask,
                                  cross_attn_head_mask = cross_attn_head_mask,
                                  encoder_outputs = encoder_outputs,
                                  past_key_values = past_key_values,
                                  inputs_embeds = inputs_embeds,
                                  decoder_inputs_embeds = decoder_inputs_embeds,
                                  labels = labels,
                                  use_cache = use_cache,
                                  output_attentions = output_attentions,
                                  output_hidden_states=output_hidden_states,
                                  return_dict=return_dict)

        masked_loss = None
        if labels is not None:
            # 得到蒸馏的loss
            ## ! debug 看看encoder_hidden_size
            distilled_loss = self.distilled_fn(distilled_input_ids,
                                               attention_mask,
                                               outputs.encoder_hidden_states)
            distilled_loss = distilled_loss.to(device=outputs.loss.device)
            masked_loss = 0.5 * distilled_loss + 0.5 * outputs.loss 

        return Seq2SeqLMOutput(
            loss=masked_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            # distilled_loss = distilled_loss,
            # translate_loss = outputs.loss,
        )
