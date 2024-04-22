import argparse
import time
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from minigpt4.common.registry import registry

import numpy as np

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    
CONV_VISION_Vicuna0 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_LLama2 = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("<s>[INST] ", " [/INST] "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)



class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000, do_sample=True, num_return_sequences=1):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )
        
        if num_return_sequences == 1:
            output_token = outputs[0]
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_text = output_text.strip('</s>')   # remove the end '</s>'
            output_text = output_text.strip()
            conv.messages[-1][1] = output_text
            return output_text, output_token.cpu().numpy()
        else:
            output_text_all = []
            for i in range(outputs.shape[0]):
                output_token = outputs[i]
                #output_token = output_token[output_token!=0]  # remove <unk> in the output tokens
                if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                    output_token = output_token[1:]
                if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                    output_token = output_token[1:]
                output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
                output_text = output_text.split('###')[0]  # remove the stop sign '###'
                output_text = output_text.split('Assistant:')[-1].strip()
                output_text = output_text.strip('</s>')   # remove the end '</s>'
                output_text = output_text.strip()
                output_text_all.append(output_text)
            return output_text_all
    
    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt = prompt.strip()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs


class ChatInference:
    def __init__(
        self, 
        model:torch.nn.Module, 
        vis_processor:object, 
        device:str,
        annotate:bool=False,
        conv_rec:int=20
    )->None:
        self.device = device
        self.model = model
        self.annotate = annotate
        self.conv_rec = conv_rec
        
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.default_prompt = [
            'Given the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img>',
            '</Img> '
        ]
        self.img_list = [None]

    def classify(
        self,
        embs:torch.Tensor,
        out_path:str=None,
    )->torch.Tensor:
            inputs = BatchEncoding({
                "input_ids":torch.argmax(torch.nn.Softmax()(embs), dim=-1).to(embs.device),
                "attention_mask":torch.ones(embs.shape[1]).unsqueeze(0).to(embs.device)
            })
            with torch.no_grad(), self.model.maybe_autocast():
                outputs = self.model.llama_model(**inputs)
                pred = self.model.classifier(outputs.logits)[:,-1]
                if out_path is not None:
                    file = out_path.split("/")
                    torch.save(outputs.logits.cpu(), out_path.replace(".jpg",".pt"))
                    torch.save(pred,os.path.join("/".join(file[:-2]),"preds",file[-1].replace(".jpg",".pt")))
                return pred

    def answer(
        self, 
        question:str,
        max_new_tokens:int=300, 
        num_beams:int=1, 
        min_length:int=1, 
        top_p:float=0.9,
        repetition_penalty:float=1.0, # change this for less repitition
        length_penalty:int=1, 
        temperature:float=1.0, 
        max_length:int=2000,
        inference:bool=False,
        out_path:str=None,
    )->Tuple[str,np.ndarray]:
        embs = self.get_context_emb(question, self.img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]
        # print(f"MIXEMB IDS - {torch.argmax(embs, dim=-1)} {embs.shape}")

        if inference:
            return self.classify(embs,out_path=out_path)
        
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        # print(f"OUTPUT TOKEN - {output_token}")

        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        self.default_prompt[-1] += ' ' + output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(
        self, 
        image:Image,
    )->str:            
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        image_emb, _ = self.model.encode_img(image)
        self.img_list[0] = image_emb
        self.default_prompt = [*self.default_prompt[:2]]
        return "Received."

    def get_context_emb(
        self, 
        text:str, 
        img_list:list
    )->torch.Tensor:
        """
        PROMPT SEGS -  
            [
            'Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.
            ###Human: <Img>', '</Img> Is this an image of a cat?
            ###Assistant: The image is an orange and white cat standing in a grassy field with its tail flicking back and forth. It appears to be a small kitten.
            ###Human: Is this an image of a dog?
            ###Assistant: The image is an orange and white cat standing in a grassy field with its tail flicking back and forth. It appears to be a small kitten.
            ###Human: What is this an image of?
            ###Assistant:'
        ]
        
        """
        text = f'{text}###Assistant:'
        
        if self.annotate or len(self.default_prompt) > self.conv_rec + 1:
            self.default_prompt.pop(3)
            
        self.default_prompt.append(text)
        prompts = (self.default_prompt[0],''.join(self.default_prompt[1:]),)
        # print(f"PROMPT - {prompts}")
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device)
            # only add bos to the first seg
            for i, seg in enumerate(prompts)
        ]
        # attn_masks = [seg_t.attention_mask for seg_t in seg_tokens]
        # print("ATTN MASK - ",torch.cat(attn_masks,dim=1).shape)
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t.input_ids) for seg_t in seg_tokens]
        # print("SEG EMBS - ",torch.cat(seg_embs,dim=1).shape)
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        # print("MIXED EMBS - ",mixed_embs.shape)
        return mixed_embs