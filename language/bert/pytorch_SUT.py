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
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import torch
import transformers
from transformers import BertConfig, BertForQuestionAnswering, MobileBertForQuestionAnswering
from squad_QSL import get_squad_QSL

# model_name = 'origin_pytorch_model'
model_name = 'mrm8488/mobilebert-uncased-finetuned-squadv1'
# model_name = 'yujiepan/internal.mobilebert-uncased-12blks-squadv1-int8-quantize-embedding'
# model_name = 'yujiepan/test.mobilebert-uncased-12blks-squadv1-int8-f1-88.77'
# model_name = 'yujiepan/mobilebert-uncased-squadv1-14blocks-structured39.8-int8'
# model_name = 'yujiepan/test.mobilebert-uncased-12blks-squadv1-int8-f1-90.31'
# model_name = 'neuralmagic/mobilebert-uncased-finetuned-squadv1'

class BERT_PyTorch_SUT():
    def __init__(self, args):
        print("Loading BERT configs...")
        with open("bert_config.json") as f:
            config_json = json.load(f)

        config = BertConfig(
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

        self.dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.version = transformers.__version__

        print("Loading PyTorch model...")
        print(f'Using {model_name}')
        if model_name == 'origin_pytorch_model':
            self.model = BertForQuestionAnswering(config)
            self.model.to(self.dev)
            self.model.eval()
            model_file = os.environ.get("ML_MODEL_FILE_WITH_PATH", "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch")
            self.model.load_state_dict(torch.load(model_file), strict=False)
        else:
            self.model = MobileBertForQuestionAnswering.from_pretrained(model_name)
            self.model.to(self.dev)

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        with torch.no_grad():
            num_query_samples = len(query_samples)

            load_data_start_time = time.time()
            input_ids = []
            attention_mask = []
            token_type_ids = []

            for i in range(num_query_samples):
                eval_features = self.qsl.get_features(query_samples[i].index)
                input_ids.append(eval_features.input_ids)
                attention_mask.append(eval_features.input_mask)
                token_type_ids.append(eval_features.segment_ids)
            
            input_ids = torch.tensor(input_ids, device=self.dev, dtype=torch.int64)
            attention_mask = torch.tensor(attention_mask, device=self.dev, dtype=torch.int64)
            token_type_ids = torch.tensor(token_type_ids, device=self.dev, dtype=torch.int64)
            load_data_end_time = time.time()

            forward_time_start = time.time()
            model_output = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            torch.cuda.synchronize()
            forward_time_end = time.time()

            submit_time_start = time.time()
            if self.version >= '4.0.0':
                start_scores = model_output.start_logits
                end_scores = model_output.end_logits
            else:
                start_scores, end_scores = model_output
            output = torch.stack([start_scores, end_scores], axis=-1).cpu().numpy()

            for i in range(num_query_samples):
                response_array = array.array("B", output[i].tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
                lg.QuerySamplesComplete([response])
            submit_time_end = time.time()

            print(f"Time to load data: {str(load_data_end_time - load_data_start_time)} s")
            print(f"Time to forward: {str(forward_time_end - forward_time_start)} s")
            print(f"Time to submit: {str(submit_time_end - submit_time_start)} s")

        # with torch.no_grad():
        #     for i in range(len(query_samples)):
        #         print("Processing query " + str(i) + " of " + str(len(query_samples)))
        #         eval_features = self.qsl.get_features(query_samples[i].index)
        #         print(eval_features.input_ids)
        #         model_output = self.model.forward(input_ids=torch.tensor(eval_features.input_ids, device=self.dev, dtype=torch.int64).unsqueeze(0),
        #             attention_mask=torch.tensor(eval_features.input_mask, device=self.dev, dtype=torch.int64).unsqueeze(0),
        #             token_type_ids=torch.tensor(eval_features.segment_ids, device=self.dev, dtype=torch.int64).unsqueeze(0))
        #         if self.version >= '4.0.0':
        #             start_scores = model_output.start_logits
        #             end_scores = model_output.end_logits
        #         else:
        #             start_scores, end_scores = model_output
        #         output = torch.stack([start_scores, end_scores], axis=-1).squeeze(0).cpu().numpy()

        #         response_array = array.array("B", output.tobytes())
        #         bi = response_array.buffer_info()
        #         response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
        #         lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
