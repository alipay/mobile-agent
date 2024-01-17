from vllm import LLM, SamplingParams
import torch
import sys
import logging
import json
import os
from peft import LoraConfig
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer
from peft import PeftModel
import argparse
_INFER_BATCH=1000

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",type=str)
parser.add_argument("--lora_dir",type=str)
parser.add_argument("--test_file__dir",type=str)

def infer_with_label(llm, sampling_params, json_dir):
    acc_map={}
    file_no=0
    prompt_pool=[]
    instruct_map={}
    line_count = int(os.popen('wc -l %s' % json_dir).read().split()[0])
    strong_sampling_params = SamplingParams(temperature=0, use_beam_search=True, n=2, best_of=2, top_k=-1, max_tokens=4096)
    with open(json_dir, 'r') as file:
        content = file.readline()
        intent=""
        while(content):
            file_no+=1
            line=""
            try:
                line=json.loads(content, strict=False)
            except:
                print(content)
            keys = ['instruction', 'output', 'id']
            prompt, output, id = list(map(lambda key: sub_data.get(key), keys))
            instruct_map[prompt]=id
            if(intent!=id or intent==""):
                if(acc_map.get(id)):
                    acc_map[id][prompt]={"output":output}
                else:
                    acc_map[id]={prompt:{"output":output}}
                prompts = list(acc_map[intent].keys())
                prompt_pool+=prompts
                if(len(prompt_pool)>_INFER_BATCH):
                    outputs = llm.generate(prompt_pool, sampling_params)
                    prompt_pre=""
                    for predict in outputs:
                        prompt_pre = predict.prompt
                        intent_pre = instruct_map[prompt_pre]
                        generated_text = predict.outputs[0].text
                        if(len(generated_text)<5 and len(predict.prompt_token_ids)<4096):
                            new_output=llm.generate([prompt_pre], strong_sampling_params)
                            new_output_list=[new_output[0].outputs[i].text for i in range(2)]
                            generated_text= max(new_output_list, key=len)
                        acc_map[intent_pre][prompt_pre]["predict"]=generated_text
                        generated_text=generated_text[generated_text.find("action_type"):]
                        label=0
                        if(generated_text.lower()==acc_map[intent_pre][prompt_pre]["output"].lower()):
                            label=1
                        acc_map[intent_pre][prompt_pre]["label"]=label
                    prompt_pool=[]
                intent=id
            else:
                acc_map[id][prompt]={"output":output}
            content = file.readline()
    return acc_map

def calculate_acc(acc_map):
    num_query=0
    num_prompt=0
    acc=0
    num_sample=0
    for x in acc_map.keys():
        a_num=0
        b_acc=0
        for y in acc_map[x].keys():
            num_sample+=1
            if("label" in acc_map[x][y].keys()):
                a_num+=1 
                b_acc+=acc_map[x][y]["label"]
        if(a_num!=0):
            num_query+=1
            num_prompt+=a_num
            acc+=(b_acc/(a_num+0.00001))
    return num_sample,num_query,num_prompt,acc,acc/(num_query+0.0001)


if __name__ == '__main__':
    tokenizer = LlamaTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir,trust_remote_code=True)
    llm = PeftModel.from_pretrained(model, lora_dir)
    sampling_params = SamplingParams(temperature=0.2, top_p=0.95, n=1, top_k=20, max_tokens=2048)
    acc_map=infer_with_label(llm,sampling_params,test_file__dir)
    num_sample,num_query,num_prompt,acc,avg_acc=calculate_acc(acc_map)
    print(f"Total prompt amount: {num_sample} \n Total Task instruction amount: {num_query}\n The average ACC : {avg_acc}")