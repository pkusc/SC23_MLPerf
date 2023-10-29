# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import array
import json
import os
import sys
import time
import ray
from typing import List
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

loadgen_path = "/home/intlsy/CM/repos/local/cache/fa9180d70a5e4e1f/"
inference_bert_path = "/home/intlsy/CM/repos/local/cache/8e90af0a933b4703/inference/language/bert"

# We set env vars here to let ray workers find the loadgen library
ray.init(runtime_env = {
    "env_vars": {
        "LD_LIBRARY_PATH": os.path.join(loadgen_path, 'install', 'lib'),
        "DYLD_FALLBACK_LIBRARY_PATH": os.path.join(loadgen_path, 'install', 'lib'),
        "PYTHONPATH": os.path.join(loadgen_path, 'install', 'python'),
        "C_INCLUDE_PATH": os.path.join(loadgen_path, 'install', 'include'),
        "CPLUS_INCLUDE_PATH": os.path.join(loadgen_path, 'install', 'include'),
    }
})

# ray.init(runtime_env={"conda": "mlperf"})
import mlperf_loadgen as lg

import numpy as np
import torch
import transformers
from transformers import BertConfig, BertForQuestionAnswering, MobileBertForQuestionAnswering
from squad_QSL import get_squad_QSL

num_gpus = 4
# model_name = 'origin_pytorch_model'
# model_name = 'mrm8488/mobilebert-uncased-finetuned-squadv1'
# model_name = 'yujiepan/internal.mobilebert-uncased-12blks-squadv1-int8-quantize-embedding'
# model_name = 'yujiepan/test.mobilebert-uncased-12blks-squadv1-int8-f1-88.77'
# model_name = 'yujiepan/mobilebert-uncased-squadv1-14blocks-structured39.8-int8'
# model_name = 'yujiepan/test.mobilebert-uncased-12blks-squadv1-int8-f1-90.31'
# model_name = 'neuralmagic/mobilebert-uncased-finetuned-squadv1'
model_name = 'csarron/mobilebert-uncased-squad-v1'
# model_name = 'google/mobilebert-uncased'

@ray.remote(num_cpus=1, num_gpus=1)
class BERT_PyTorch_Worker():
    def __init__(self, worker_id, bert_config, max_examples: int):
        torch.cuda.set_device(0)
        print(f"Initializing worker {worker_id} on GPU {torch.cuda.get_device_name()}")
        self.dev = torch.device("cuda:0")
        os.chdir(inference_bert_path)   # We chdir() here or the QSL cannot find the vocabuary path

        print("Loading PyTorch model...")
        print(f'Using {model_name}')
        if model_name == 'origin_pytorch_model':
            self.model = BertForQuestionAnswering(bert_config)
            self.model.to(self.dev)
            self.model.eval()
            model_file = os.environ.get("ML_MODEL_FILE_WITH_PATH", "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch")
            self.model.load_state_dict(torch.load(model_file), strict=False)
        else:
            # We set proxy here to let transformers download the model from HuggingFace
            # Thanks Yuanhang Sun for the proxy!
            # Fuck GFW.
            # proxy_addr = "http://115.27.161.58:7890"
            proxy_addr = "http://10.129.196.146:20171"
            os.environ["http_proxy"] = proxy_addr
            os.environ["https_proxy"] = proxy_addr
            os.environ["all_proxy"] = proxy_addr
            self.model = MobileBertForQuestionAnswering.from_pretrained(model_name)
            self.model.to(self.dev)
        
        # Initialize QSL (Query Sample Library)
        self.qsl = get_squad_QSL(max_examples)
    
    def init_ready(self) -> int:
        return 1
    
    def issue_queries(self, indexes: List[int]) -> torch.tensor:
        with torch.no_grad():
            input_ids = []
            attention_mask = []
            token_type_ids = []

            for index in indexes:
                eval_features = self.qsl.get_features(index)
                input_ids.append(eval_features.input_ids)
                attention_mask.append(eval_features.input_mask)
                token_type_ids.append(eval_features.segment_ids)
            
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=self.dev)
            attention_mask = torch.tensor(attention_mask, dtype=torch.int32, device=self.dev)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.int32, device=self.dev)

            model_output = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            start_scores = model_output.start_logits
            end_scores = model_output.end_logits
            output = torch.stack([start_scores, end_scores], axis=-1).to(torch.float16).cpu()   # We send float16 to save network bandwidth
            return output


class BERT_PyTorch_SUT():
    def __init__(self, args):
        # Read BERT config
        print("Loading BERT configs...")
        with open("bert_config.json") as f:
            config_json = json.load(f)

        bert_config = BertConfig(
            attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
            hidden_act=config_json["hidden_act"],
            hidden_dropout_prob=config_json["hidden_dropout_prob"],
            hidden_size=config_json["hidden_size"],
            initializer_range=config_json["initializer_range"],
            intermediate_size=config_json["intermediate_size"],
            max_position_embeddings=config_json["max_position_embeddings"],
            num_attention_heads=config_json["num_attention_heads"],
            num_hidden_layers=config_json["num_hidden_layers"],
            type_vocab_size=config_json["type_vocab_size"],
            vocab_size=config_json["vocab_size"])
        
        # Initialize workers
        self.workers = []
        for i in range(num_gpus):
            self.workers.append(BERT_PyTorch_Worker.remote(i, bert_config, args.max_examples))
        for worker in self.workers:
            ray.get(worker.init_ready.remote())

        # Initialize SUT (System Under Test)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        num_query_samples = len(query_samples)

        load_data_start_time = time.time()
        indexes = []
        for i in range(num_query_samples):
            indexes.append(query_samples[i].index)
        load_data_end_time = time.time()

        forward_time_start = time.time()
        model_output_handlers = []
        for worker_id in range(len(self.workers)):
            handler = self.workers[worker_id].issue_queries.remote(indexes[worker_id::num_gpus])
            model_output_handlers.append(handler)
        model_outputs = ray.get(model_output_handlers)
        forward_time_end = time.time()

        output = np.concatenate([x.to(torch.float32).numpy() for x in model_outputs], axis=0)

        submit_time_start = time.time()

        for i in range(num_query_samples):
            response_array = array.array("B", output[i].tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])
        submit_time_end = time.time()

        print(f"Time to load data: {str(load_data_end_time - load_data_start_time)} s")
        print(f"Time to forward: {str(forward_time_end - forward_time_start)} s")
        print(f"Time to submit: {str(submit_time_end - submit_time_start)} s")

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
