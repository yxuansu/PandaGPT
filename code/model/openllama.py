from header import *
import torch.nn.functional as F
from .ImageBind import *
from .ImageBind import data
from .modeling_llama import LlamaForCausalLM

import torch
from torch.nn.utils import rnn
def build_one_instance(tokenizer, conversation):
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = '</Img> ' + turn['value'] + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

PROMPT_START = '### Human: <Img>'
class OpenLLAMAPEFTModel(nn.Module):

    '''LoRA for LLaMa model'''

    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        imagebind_ckpt_path = args['imagebind_ckpt_path']
        vicuna_ckpt_path = args['vicuna_ckpt_path']
        max_tgt_len = args['max_tgt_len']
        stage = args['stage']

        print (f'Initializing visual encoder from {imagebind_ckpt_path} ...')
        self.visual_encoder, self.visual_hidden_size = \
        imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print ('Visual encoder initialized.')

        print (f'Initializing language decoder from {vicuna_ckpt_path} ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print ('Language decoder initialized.')

        self.llama_proj = nn.Linear(
            self.visual_hidden_size, self.llama_model.config.hidden_size
        )

        self.max_tgt_len = max_tgt_len
        self.device = torch.cuda.current_device()

    def encode_audio(self, audio_paths):
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO] # bsz x 1024
        inputs_llama = self.llama_proj(audio_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'] # bsz x 1024
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = img_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim

        # create targets
        empty_targets = (
            torch.ones([batch_size, 1+p_before_embeds.size()[1]+1], # 1 (bos) + s1 + 1 (image vector)
                       dtype=torch.long).to(self.device).fill_(-100)  
        ) # bsz x (1 + s1 + 1)
        targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
        assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
        return inputs_embeds, targets, attention_mask 

    def forward(self, inputs):
        image_paths = inputs['image_paths']
        img_embeds, _ = self.encode_image(image_paths)

        output_texts = inputs['output_texts']
        input_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, output_texts, self.max_tgt_len)
        inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        # calculate the token accuarcy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]    # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)    # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask    # [B*S]
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    def extract_multimodal_feature(self, inputs):
        mode = inputs['mode']
        if mode == 'vision':
            image_embeds, _ = self.encode_image(inputs['image_paths'])
            return image_embeds
        elif mode == 'audio':
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            return audio_embeds
        elif mode == 'vision+audio':
            image_embeds, _ = self.encode_image(inputs['image_paths'])
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            return image_embeds + audio_embeds
        else:
            raise Exception('Wrong inference mode.')

    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        feature_embeds = self.extract_multimodal_feature(inputs)
        batch_size = feature_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        text = '</Img> ' + prompt + '\n### Assistant:'
        p_after_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
        return inputs_embeds

    def generate(self, inputs):
        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length
            }
        '''
        input_embeds = self.prepare_generation_embedding(inputs)
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
        )
        output_text = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text, outputs[0]

    '''
    def generate(self, inputs):
        image_paths = inputs['image_paths']
        if image_paths[0].endswith('.wav'):
            img_embeds, atts_img = self.encode_audio(image_paths)
        else:
            img_embeds, atts_img = self.encode_image(image_paths)
        prompt = random.choice(self.prompt_list)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
        img_embeds, atts_img = img_embeds.cuda(), atts_img.cuda()

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=img_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_tgt_len,
        )
        generated = self.llama_tokenizer.decode(outputs[0, 1:], add_special_tokens=False)    # [B, S, V]
        if '###' in generated:
            generated = generated[:generated.index('###')]
        return generated, prompt

    def composition_generate(self, inputs):
        image_paths = inputs['image_paths']
        audio_paths = inputs['audio_paths']
        audio_embeds, atts_img = self.encode_audio(audio_paths)
        img_embeds, atts_img = self.encode_image(image_paths)
        img_embeds += audio_embeds
        img_embeds = F.normalize(img_embeds, dim=-1, p=2)
        prompt = random.choice(self.prompt_list)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
        img_embeds, atts_img = img_embeds.cuda(), atts_img.cuda()

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1], dtype=torch.long, device=img_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=self.max_tgt_len,
        )
        generated = self.llama_tokenizer.decode(outputs[0, 1:], add_special_tokens=False)    # [B, S, V]
        if '###' in generated:
            generated = generated[:generated.index('###')]
        return generated, prompt
    '''

