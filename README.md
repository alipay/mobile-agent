<div align="center">

# MobileAgent
<div align="left">

# 1. Introduction
Agents centered around Large Language Models (LLMs) are now capable of automating mobile device operations for users. After fine-tuning to learn a user's mobile operations, these agents can adhere to high-level user instructions online. They execute tasks such as goal decomposition, sequencing of sub-goals, and interactive environmental exploration, until the final objective is achieved. However, privacy concerns related to personalized user data arise during mobile operations, requiring user confirmation. Moreover, users' real-world operations are exploratory, with action data being complex and redundant, posing challenges for agent learning. To address these issues, in our practical application, we have designed interactive tasks between agents and humans to identify sensitive information and align with personalized user needs. Additionally, we integrated Standard Operating Procedure (SOP) information within the model's in-context learning to enhance the agent's comprehension of complex task execution.

paper : https://arxiv.org/abs/2401.04124

Model : https://huggingface.co/tineding/ACT-SOP

# 2. Data
Our research involved an investigation of renowned datasets and their statistical characteristics within the domain of large model control devices. In the field of web control, the Mind2Web dataset is particularly notable. Similarly, in the realm of mobile control, the AitW dataset stands out as a prominent resource.
- Examples of SOP data related to AIA and AitW scenarios in our paper can be found in the *\<data>* folder.

|  Dataset/URL   | Platform  | Human demos|APPs or websites|Task steps|Observation format|Screen features| Real| High-level instruction| 
|  ----  | ----  | ----| ----  | ----  | ----| ----  | ----  | ----|
| [MiniWoB](https://miniwob.farama.org/)  |  web |  100  |  100  |  3.6  |  DOM  |  ×  |  ×  |  ×  |
| [WebShop](https://webshop-pnlp.github.io/)  |  web |  12,000  |  1  |  11.3  |  DOM  |  ×  |  ×  |  ✔️  | 
| [RUSS](https://paperswithcode.com/dataset/russ-dataset)  |  web |  80 |  22  |  5.4  |  DOM  |  ×  |  ✔️  |  ✔️  | 
| [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/)  |  web |  2,350 |  137  |  7.3  |  DOM  |  ×  |  ✔️  |  ✔️  | ✔️  |
|  ----  | ----  | ----| ----  | ----  | ----| ----  | ----  | ----|
| [RicoSCA](https://paperswithcode.com/dataset/ricosca)  | Android(apps) |  0  |  n/a  |  1.0  |  VH,screen  |  ×  |  ×  |  ×  | 
| [UIBert](https://github.com/google-research-datasets/uibert)  | Android(apps) |  16,660  |  n/a  |  1.0  |  VH,screen  |  ×  |  ✔️  |  ×  | 
| [PixelHelp](https://paperswithcode.com/dataset/pixelhelp)  | Android(apps) |  187  |  4  |  4.2  |  VH,screen  |  ×  |  ✔️  |  ✔️  | 
| [META-GUI](https://x-lance.github.io/META-GUI-Leaderboard/)  | Android(apps) |  1125  |  11  |  4.3  |  VH,screen  |  ×  |  ✔️  |  ✔️  | 
| [UGIF](https://paperswithcode.com/dataset/ugif)  | Android(apps) |  523  |  12  |  5.3  |  VH,screen  |  ×  |  ✔️  |  ✔️  | 
| [MoTIF](https://github.com/aburns4/MoTIF)  | Android(apps) |  4,707  |  125  |  4.5  |  VH,screen  |  ×  |  ✔️  |  ✔️  |
| [AitW](https://github.com/google-research/google-research/blob/master/android_in_the_wild/README.md#android-in-the-wild-aitw)  | Android(apps+web) |  715,142(5,689,993Example)  |  357+  |  6.5  |  screen  |  ✔️  |  ✔️  |  ✔️  |

- Mobile 
   -   **Android in the Wild (AitW)** is a large-scale dataset for mobile device control that contains human-collected demonstrations of natural language instructions, user interface (UI) screens, and actions for a variety of human tasks.
- Web
   -   **Mind2Web** is a dataset for developing and evaluating generalist agents for the web that can follow language instructions to complete complex tasks on any website. Mind2Web contains 2,350 tasks from 137 websites spanning 31 domains that: Reflect diverse and practical use cases on the web. Provide challenging yet realistic environments with real-world websites.Test generalization ability across tasks and environments.

# 3. DashBoard for AitW
We utilized the AitW dataset as a standard test set to evaluate the performance of various team models, including our model developed based on the SOP mechanism.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-style:solid;border-width:0px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;
  padding:10px 5px;word-break:normal;}
.tg th{border-style:solid;border-width:0px;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-di1h{border-color:#656565;text-align:center;vertical-align:middle}
.tg .tg-x38u{border-color:#656565;color:#fe0000;text-align:center;vertical-align:middle}
.tg .tg-fmsv{border-color:#656565;color:#333333;text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-di1h"> Type </th>
    <th class="tg-di1h">Action Entity</th>
    <th class="tg-di1h">Tuning</th>
    <th class="tg-di1h">Research Organization</th>
    <th class="tg-di1h"> Model </th>
    <th class="tg-di1h"> Overall </th>
    <th class="tg-di1h">General </th>
    <th class="tg-di1h">Install </th>
    <th class="tg-di1h">GoogleApps </th>
    <th class="tg-di1h">Single </th>
    <th class="tg-di1h">WebShopping </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-di1h" rowspan="7"> <br>LLM </td>
    <td class="tg-di1h" rowspan="3"> <br>Point-Coordinates </td>
    <td class="tg-di1h"> <br>No </td>
    <td class="tg-di1h" rowspan="3"> SJTU </td>
    <td class="tg-di1h"> <br>ChatGPT-CoT (5-shot) </td>
    <td class="tg-di1h"> <br>7.72 </td>
    <td class="tg-di1h"> <br>5.93 </td>
    <td class="tg-di1h"> <br>4.38 </td>
    <td class="tg-di1h"> <br>10.47 </td>
    <td class="tg-di1h"> <br>9.39 </td>
    <td class="tg-di1h"> <br>8.42 </td>
  </tr>
  <tr>
    <td class="tg-di1h" rowspan="6"> <br>Yes </td>
    <td class="tg-di1h">PaLM </td>
    <td class="tg-di1h"> 39.6 </td>
    <td class="tg-di1h"> - </td>
    <td class="tg-di1h"> - </td>
    <td class="tg-di1h"> - </td>
    <td class="tg-di1h"> - </td>
    <td class="tg-di1h"> - </td>
  </tr>
  <tr>
    <td class="tg-di1h">Llama2 (1% data) </td>
    <td class="tg-di1h"> 28.4 </td>
    <td class="tg-di1h"> 28.56 </td>
    <td class="tg-di1h"> 35.18 </td>
    <td class="tg-di1h"> 30.99 </td>
    <td class="tg-di1h"> 27.35 </td>
    <td class="tg-di1h"> 19.92 </td>
  </tr>
  <tr>
    <td class="tg-di1h" rowspan="7"> <br>Enlarged-Frame </td>
    <td class="tg-di1h" rowspan="4"> AntFin </td>
    <td class="tg-di1h">Llama 2 </td>
    <td class="tg-di1h">65.43</td>
    <td class="tg-di1h"> 55.3 </td>
    <td class="tg-di1h"> 73.65 </td>
    <td class="tg-di1h"> 62.33 </td>
    <td class="tg-di1h"> 74.82 </td>
    <td class="tg-di1h"> 61.07 </td>
  </tr>
  <tr>
    <td class="tg-di1h">Llama 2+plan </td>
    <td class="tg-di1h">62.08</td>
    <td class="tg-di1h"> 52.1 </td>
    <td class="tg-di1h"> 71.65 </td>
    <td class="tg-di1h"> 56.23 </td>
    <td class="tg-di1h"> 74.18 </td>
    <td class="tg-di1h"> 56.22 </td>
  </tr>
  <tr>
    <td class="tg-di1h">Llama 2+plan+state </td>
    <td class="tg-di1h">62.86</td>
    <td class="tg-di1h"> 53.77 </td>
    <td class="tg-di1h"> 69.1 </td>
    <td class="tg-di1h"> 61.19 </td>
    <td class="tg-di1h"> 73.51 </td>
    <td class="tg-di1h"> 56.74 </td>
  </tr>
  <tr>
    <td class="tg-di1h"> Llama 2+SOP </td>
    <td class="tg-x38u"><span style="color:#FE0000">66.92 </span></td>
    <td class="tg-x38u"><span style="color:#FE0000"> 55.8 </span></td>
    <td class="tg-x38u"><span style="color:#FE0000"> 74.98 </span></td>
    <td class="tg-x38u"><span style="color:#FE0000"> 63.95 </span></td>
    <td class="tg-x38u">76.27 </td>
    <td class="tg-x38u"> <span style="color:#FE0000"> 63.61 </span><br></td>
  </tr>
  <tr>
    <td class="tg-di1h" rowspan="7"> <br>Multi-Modal </td>
    <td class="tg-di1h" rowspan="3"> No </td>
    <td class="tg-di1h" rowspan="3"> Microsoft/UC </td>
    <td class="tg-di1h"> GPT-4V ZS +text </td>
    <td class="tg-di1h"> 50.54  </td>
    <td class="tg-di1h"> 41.66  </td>
    <td class="tg-di1h"> 42.64  </td>
    <td class="tg-di1h"> 49.82  </td>
    <td class="tg-di1h"> 72.83  </td>
    <td class="tg-di1h"> 45.73 </td>
  </tr>
  <tr>
    <td class="tg-di1h"> GPT-4V ZS image-only </td>
    <td class="tg-di1h"> 51.92  </td>
    <td class="tg-di1h"> 42.44  </td>
    <td class="tg-di1h"> 49.18  </td>
    <td class="tg-di1h"> 48.26  </td>
    <td class="tg-di1h"> 76.34  </td>
    <td class="tg-di1h"> 43.35 </td>
  </tr>
  <tr>
    <td class="tg-di1h"> GPT-4V ZS +history </td>
    <td class="tg-di1h"> 52.96  </td>
    <td class="tg-di1h"> 43.01  </td>
    <td class="tg-di1h"> 46.14  </td>
    <td class="tg-di1h"> 49.18  </td>
    <td class="tg-di1h"> <span style="color:#FE0000"> 78.29 </span> </td>
    <td class="tg-di1h"> 48.18 </td>
  </tr>
  <tr>
    <td class="tg-di1h" rowspan="4"> Point-Coordinates </td>
    <td class="tg-di1h" rowspan="4"> Yes </td>
    <td class="tg-di1h" rowspan="2"> Google </td>
    <td class="tg-di1h"> BC-single </td>
    <td class="tg-di1h"> 68.7 </td>
    <td class="tg-di1h"> - </td>
    <td class="tg-di1h"> - </td>
    <td class="tg-di1h"> - </td>
    <td class="tg-di1h"> - </td>
    <td class="tg-di1h"> - </td>
  </tr>
  <tr>
    <td class="tg-di1h"> BC-history </td>
    <td class="tg-di1h"> 73.1 </td>
    <td class="tg-di1h"> 63.7 </td>
    <td class="tg-di1h"> 77.5 </td>
    <td class="tg-di1h"> 75.7 </td>
    <td class="tg-di1h"> 80.3 </td>
    <td class="tg-di1h"> 68.5 </td>
  </tr>
  <tr>
    <td class="tg-di1h" rowspan="2"> SJTU </td>
    <td class="tg-di1h"> Auto-UIseparate </td>
    <td class="tg-di1h"> 74.07 </td>
    <td class="tg-di1h"> 65.94 </td>
    <td class="tg-di1h"><span style="color:#FE0000"> 77.62 </span></td>
    <td class="tg-di1h"><span style="color:#FE0000"> 76.45 </span></td>
    <td class="tg-di1h"> 81.39 </td>
    <td class="tg-di1h"> 69.72 </td>
  </tr>
  <tr>
    <td class="tg-di1h"> Auto-UIunified </td>
    <td class="tg-fmsv"><span style="color:#FE0000"> 74.27 </span></td>
    <td class="tg-di1h"><span style="color:#FE0000"> 68.24 </span></td>
    <td class="tg-di1h"> 76.89 </td>
    <td class="tg-di1h"> 71.37 </td>
    <td class="tg-di1h"><span style="color:#FE0000"> 84.58 </span></td>
    <td class="tg-di1h"><span style="color:#FE0000"> 70.26 </span></td>
  </tr>
</tbody>
</table>

**Paper**

- [Android in the Wild: A Large-Scale Dataset for Android Device Control](https://arxiv.org/abs/2307.10088)
- [you only look at screens: multimodal chain-of-action agents](https://arxiv.org/pdf/2309.11436.pdf)
- [GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation](https://arxiv.org/pdf/2311.07562.pdf)

# 4. Our Code
We have provided the basic processing code for AITW for everyone to train their own large models. This set of code includes data processing, model training, and model evaluation.
## Mobile-AITW
### data process
For more information, please refer to [AitW](https://github.com/google-research/google-research/blob/master/android_in_the_wild/README.md#android-in-the-wild-aitw) and [Auto-UI](https://github.com/cooelf/Auto-UI).
- Base enlarged-Frame
```bash
> pip install jax,jaxlib
> python code/data_process.py --input_dir [aitw_file_path] --output_dir [output_file_path]
-- for example: python code/data_process.py --input_dir /mntnlp/peisu/data/autoui/google_apps_blip_test.obj --output_dir /mntnlp/data/autoui/output_name.json

```

*For example:*

**Input:**
```
{
	'episode_id': '16150593985172894737', 
	'data': 
			[{
			'goal': "What's the top post on reddit today?", 
			'step_id': 1, 
			'image': tensor([-4.1089e-01,  3.0527e+00, -6.8378e-04,  ...,  5.6299e-01,
         					1.2676e+00, -9.7266e-01], dtype=torch.float16), 
         	'ui_positions': array([[0.05131579, 0.09722222, 0.02171053, 0.27777779],
					       [0.58684212, 0.31111112, 0.06447368, 0.14166667],
					       [0.60855263, 0.83333331, 0.025     , 0.02916667],
					       [0.67171055, 0.77916664, 0.01447368, 0.14166667],
					       [0.67302632, 0.33888888, 0.01315789, 0.09027778],
					       [0.67302632, 0.5625    , 0.01315789, 0.10972222],
					       [0.77302629, 0.12083333, 0.04605263, 0.05555556],
					       [0.77565789, 0.82499999, 0.04342105, 0.04305556],
					       [0.77894735, 0.34999999, 0.03618421, 0.05833333],
					       [0.81776315, 0.40277779, 0.01513158, 0.19583334],
					       [0.87828946, 0.84027779, 0.0381579 , 0.03472222],
					       [0.87894738, 0.12083333, 0.03618421, 0.04027778],
					       [0.95657897, 0.19166666, 0.02894737, 0.02916667],
					       [0.95657897, 0.48055556, 0.02828947, 0.03194445],
					       [0.95723683, 0.77083331, 0.02697368, 0.02916667]]), 
			'ui_text': ['Mon, Oct 10', 'M', '', 'YouTube', 'Gmail', 'Photos', '', '', '', 'Preferences', '', 'G', '', '', ''], 
			'ui_type': ['TEXT', 'TEXT', 'ICON_PLAY', 'TEXT', 'TEXT', 'TEXT', 'ICON_CALL', 'ICON_LOCATION', 'ICON_CHAT', 'TEXT', 'ICON_MIC', 
                        'ICON_GOOGLE', 'ICON_V_BACKWARD', 'ICON_NAV_BAR_CIRCLE', 'ICON_NAV_BAR_RECT'], 
			'result_touch_yx': [0.8917460441589355, 0.4927879273891449], 
            'result_lift_yx': [0.8917460441589355, 0.4927879273891449], 
			'result_action': ['DUAL_POINT', '']
			}]
}
```
**output**
```
"id":"2##What's the news in Chile?",

"instruction":
"Given a mobile screen and a question, provide the action based on the screeninformation.

Previous Actions:
step_id:0 action_type:PRESS_HOME
step_id:1 action_type:DUAL_POINT ui_text: ui_type:ICON_MIC
step_id:2 action_type:DUAL_POINT ui_text:abcnews.go.Com ui_type:TEXT
step_id:3 action_type:TYPE typed_text:What's the news in Chile?
step_id:4 action_type:TYPE typed_text:
step_id:5 action_type:PRESS_ENTER

Screen:
id:0 ui_text: ui_type:ICON_HOME
id:1 ui_text: ui_type:ICON_THREE_DOTS
id:2 ui_text:google.com/search?q ui_type:TEXT
id:3 ui_text:Google ui_type:TEXT
id:4 ui_text:= ui_type:ICON_THREE_BARS
id:5 ui_text: ui_type:ICON_MIC
id:6 ui_text:Q ui_type:ICON_MAGNIFYING_GLASS
id:7 ui_text:What's the news in Chile? 
ui_type:TEXT\nid:8 ui_text:Al ui_type:TEXT
id:9 ui_text:News ui_type:TEXT
id:10 ui_text:Images ui_type:TEXT
id:11 ui_text:Videos ui_type:TEXT
id:12 ui_text:Maps ui_type:TEXT
id:13 ui_text: ui_type:ICON_THREE_DOTS
id:14 ui_text:4 ui_type:TEXT
id:15 ui_text:https://www.aljazeera.com> where ui_type:TEXT
id:16 ui_text:Chile | Today's latest from Al ui_type:TEXT
id:17 ui_text:Jazeera ui_type:TEXT
id:18 ui_text:Stay on top of Chile latest developments on ui_type:TEXT
id:19 ui_text:the ground with Al ui_type:TEXT
id:20 ui_text:Jazeera's fact-based ui_type:TEXT
id:21 ui_text:news, ui_type:TEXT
id:22 ui_text:exclusive video footage, ui_type:TEXT
id:23 ui_text:photos ui_type:TEXT
id:24 ui_text:and ui_type:TEXT
id:25 ui_text:updated... ui_type:TEXT
id:26 ui_text: ui_type:ICON_THREE_DOTS
id:27 ui_text:https://www.reuters.com» archive ui_type:TEXT
id:28 ui_text:Chile News Headlines |Reuters ui_type:TEXT
id:29 ui_text:Chile permanently closes ui_type:TEXT
id:30 ui_text:mining areas ui_type:TEXT
id:31 ui_text:Chile files ui_type:TEXT
id:32 ui_text:connected to giant sinkhole ui_type:TEXT
id:33 ui_text:charges against mining company for giant... ui_type:TEXT
id:34 ui_text: ui_type:ICON_THREE_DOTS
id:35 ui_text:https://www.independent.co.uk> topic ui_type:TEXT
id:36 ui_text:Chile ui_type:TEXT
id:37 ui_text:latest news, ui_type:TEXT
id:38 ui_text:breaking ui_type:TEXT
id:39 ui_text:- ui_type:TEXT
id:40 ui_text:stories and comMent - The ui_type:TEXT
id:41 ui_text: ui_type:ICON_NAV_BAR_CIRCLE
id:42 ui_text: ui_type:ICON_NAV_BAR_RECT
id:43 ui_text: ui_type:ICON_V_BACKWARD

Instruction:What's the news in Chile?
Answer:",

"input":"",

"output":"action_type:DUAL_POINT ui_text:Jazeera ui_type:TEXT id:17"
```
### Data-prepair
To add a AitW dataset the following steps need to be performed.

- Create a dataset configuration after the schema described above. Examples can be found in ***code/llama-recipes-main/src/llama_recipes/configs/datasets.py***.
```
      @dataclass
      class aitw_dataset:
         dataset: str = "aitw_dataset"
         train_split: str = "/mntnlp/peisu/data/autoui/train.json" 
         test_split: str = "/mntnlp/peisu/data/autoui/val.json"
```
- Create a preprocessing routine which loads the data and returns a PyTorch style dataset. The signature for the preprocessing function needs to be (dataset_config, tokenizer, split_name) where split_name will be the string for train/validation split as defined in the dataclass.Examples can be found in ***code/llama-recipes-main/src/llama_recipes/datasets/aitw_dataset.py and \_\_init\_\_.py***
```
      class aitw(Dataset):
         def __init__(self, tokenizer, json_name=None,):
         ...
      def get_dataset(dataset_config, tokenizer, json_name=None):
         """cover function for handling loading the working dataset"""
         ...
```
- Register the dataset name and preprocessing function by inserting it as key and value into the DATASET_PREPROC dictionary in ***code/llama-recipes-main/src/llama_recipes/utils/dataset_utils.py***
```
      from llama_recipes.datasets import (
         get_grammar_dataset,
         get_alpaca_dataset,
         get_samsum_dataset,
         get_aitw_dataset,
      )
      DATASET_PREPROC = {
         "alpaca_dataset": partial(get_alpaca_dataset),
         "grammar_dataset": get_grammar_dataset,
         "samsum_dataset": get_samsum_dataset,
         "custom_dataset": get_custom_dataset,
         "aitw_dataset":get_aitw_dataset,
      }
```
- Set dataset field in training config to dataset name or use ***--dataset*** option.
```
      --dataset aitw_dataset
```
### Fine-tuning
Fine-tuning used Llama2-7b. For more information, please refer to [llama-recipes](https://github.com/facebookresearch/llama-recipes/tree/main)
```bash
> mkdir /mntnlp && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-23-tpc3.cn-heyuan-alipay.nas.aliyuncs.com:/ /mntnlp
> cd code & $pip install -r llama-recipes-main/requirements.txt
> torchrun --nnodes 1 \
         --nproc_per_node 1 llama-recipes-main/examples/finetuning.py \
         --enable_fsdp \
         --use_peft \
         --peft_method lora \
         --model_name /mntnlp/common_base_model/llama2-7b \
         --dataset aitw_dataset \
         --output_dir /mntnlp/tine/temp \
         --use_fast_kernels \
         --run_validation False \
         --batch_size_training 4 \
         --num_epochs 10 \
         --quantization False \

```
Here we make use of Parameter Efficient Methods (PEFT) as described in the next section. 

To run the command above make sure to pass the peft_method arg which can be set to lora, llama_adapter or prefix.

Once the model training is complete, the log will output information such as loss, tokenizer, LoRA storage path, and comprehensive training process details.

```
Epoch 10: train_perplexity=1.3190, train_epoch_loss=0.2768, epoch time 21.28177928365767s
Key: avg_train_prep, Value: 1.6622886657714844
Key: avg_train_loss, Value: 0.49041351675987244
Key: avg_epoch_time, Value: 21.60782462991774
Key: avg_checkpoint_time, Value: 0
```
### Inference And Evaluate
```bash
> python code/infer_vllm.py --model_dir [llama path] --lora_dir [lora path] --test_file__dir [test file path]
``` 
This code loads LoRA weights into the base model, performs predictions and evaluations on the test set.
Finally, it will return the model's prediction results, including the number of prompts in the test samples, the corresponding user task instruction count, and the task complete score.

``` 
Prompt count: 489436 
Task instruction count: 62493
Task complete score : 0.5622314184930866
``` 

- Task complete score =Average(Number Of Correct Actions / Episode Length)