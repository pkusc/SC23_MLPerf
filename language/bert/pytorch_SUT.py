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
import pickle
from typing import List
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

import ray
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

proxy_addr = "http://10.129.196.146:20171"
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
        "http_proxy": proxy_addr,
        "https_proxy": proxy_addr,
        "all_proxy": proxy_addr,
        "NM_DISABLE_ANALYTICS": "1",
    }
})

# ray.init(runtime_env={"conda": "mlperf"})
import mlperf_loadgen as lg

import numpy as np
import torch
import transformers
from transformers import BertConfig, BertForQuestionAnswering, MobileBertForQuestionAnswering
from squad_QSL import get_squad_QSL
from squad_QSL_deepsparse import get_squad_QSL_deepsparse

from deepsparse import Engine, Scheduler
from deepsparse.utils import generate_random_inputs, model_to_path, override_onnx_input_shapes

MAX_SEQ_LEN = 384

num_deepsparse_workers = 2
deepsparse_worker_max_batchsize = 128
deepsparse_model_path = '/home/intlsy/.cache/sparsezoo/2088879f-4211-4c25-8a70-daf64d43c6ca/model.onnx'
num_cpus_per_deepsparse_worker = 56

num_pytorch_workers = 4
pytorch_worker_max_batchsize = 16384

deepsparse_pytorch_work_ratio = 650 / 2300

# model_name = 'origin_pytorch_model'
# model_name = 'mrm8488/mobilebert-uncased-finetuned-squadv1'
# model_name = 'yujiepan/internal.mobilebert-uncased-12blks-squadv1-int8-quantize-embedding'
# model_name = 'yujiepan/test.mobilebert-uncased-12blks-squadv1-int8-f1-88.77'
# model_name = 'yujiepan/mobilebert-uncased-squadv1-14blocks-structured39.8-int8'
# model_name = 'yujiepan/test.mobilebert-uncased-12blks-squadv1-int8-f1-90.31'
# model_name = 'neuralmagic/mobilebert-uncased-finetuned-squadv1'
model_name = 'csarron/mobilebert-uncased-squad-v1'
# model_name = 'google/mobilebert-uncased'

# For auto-testing on different models
if os.environ.get("INTLSYS_SCRIPT_MODEL_NAME", None) is not None:
    model_name = os.environ.get("INTLSYS_SCRIPT_MODEL_NAME")
    cast_to_fp16 = False
else:
    cast_to_fp16 = True

@ray.remote(num_cpus=num_cpus_per_deepsparse_worker)
class BERT_DeepSparse_Worker():
    def __init__(self, worker_id, bert_config, max_examples: int):
        print(f"Worker {worker_id} initializing")
        self.worker_id = worker_id
        self.model_path = deepsparse_model_path
        self.batch_size = deepsparse_worker_max_batchsize
        self.scenario = "Offline"
        self.max_examples = max_examples
        self.sequence_lengths = [MAX_SEQ_LEN]
        print(f"Worker {self.worker_id} initialized")
    
    def load_engine(self):
        print(f"Worker {self.worker_id} loading engine...")
        def scenario_to_scheduler(scenario):
            if scenario == "SingleStream":
                return Scheduler.single_stream
            elif scenario == "MultiStream":
                return Scheduler.single_stream
            elif scenario == "Offline":
                return Scheduler.single_stream
            elif scenario == "Server":
                return Scheduler.multi_stream
            else:
                raise Exception(scenario)
            
        os.environ["NM_BIND_THREADS_TO_CORES"] = "1"

        os.chdir(inference_bert_path)   # We chdir() here or the QSL cannot find the vocabuary path
        self.qsl = get_squad_QSL_deepsparse(total_count_override=self.max_examples, unpadding_lengths=self.sequence_lengths)

        print(f"Creating engine for seq_len={MAX_SEQ_LEN} and batch_size={self.batch_size}...")
        self.engine = Engine(
            deepsparse_model_path,
            batch_size=self.batch_size,
            scheduler=scenario_to_scheduler(self.scenario),
            input_shapes=[[self.batch_size, MAX_SEQ_LEN]],
            num_cores=num_cpus_per_deepsparse_worker
        )

        print("Warming up engine...")
        with override_onnx_input_shapes(self.model_path, input_shapes=[[self.batch_size, MAX_SEQ_LEN]]) as model_path:
            warmup_inputs = generate_random_inputs(model_path, self.batch_size)
            for i in range(5):
                self.engine.run(warmup_inputs, val_inp=False)

    def init_ready(self) -> int:
        return 1
    
    def issue_queries(self, indexes: List[int]) -> torch.tensor:
        def pad_to_batch(x):
            x_pad = np.pad(x, ((0,self.batch_size-x.shape[0]), (0,0)))
            return x_pad
        def process_batch(batched_features):
            pad_func = lambda x: pad_to_batch(x) if len(batched_features) != self.batch_size else x
            fd = [
                pad_func(np.stack(
                    np.asarray([f.unpadded_input_ids for f in batched_features]).astype(np.int64)[np.newaxis, :])[0, :, :]),
                pad_func(np.stack(
                    np.asarray([f.unpadded_input_mask for f in batched_features]).astype(np.int64)[np.newaxis, :])[0, :, :]),
                pad_func(np.stack(
                    np.asarray([f.unpadded_segment_ids for f in batched_features]).astype(np.int64)[np.newaxis, :])[0, :, ])
            ]
            return fd
        def batched_list(lst, n):
            """Yield successive n-sized chunks from lst, with the last possibly short."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        # Extract features from queries and split into buckets
        input_features = [
            self.qsl.get_features(index) for index in indexes
        ]
        print(f"({time.time()}) Input generated")

        result = []
        for batched_features in batched_list(input_features, self.batch_size):
            unpadded_batch_size = len(batched_features)
            fd = process_batch(batched_features)

            scores = self.engine.run(fd, val_inp=False)

            output = np.stack(scores, axis=-1)  # (batch_size, seq_len, 2)
            assert output.shape[1] == MAX_SEQ_LEN

            result.extend(output[:unpadded_batch_size])
        result = np.stack(result, axis=0)
        
        print(f"({time.time()}) Result generated")

        return result
                
@ray.remote(num_cpus=2, num_gpus=1)
class BERT_PyTorch_Worker():
    def __init__(self, worker_id, bert_config, max_examples: int):
        self.worker_id = worker_id
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
            os.environ["http_proxy"] = proxy_addr
            os.environ["https_proxy"] = proxy_addr
            os.environ["all_proxy"] = proxy_addr
            self.model = MobileBertForQuestionAnswering.from_pretrained(model_name)
            if cast_to_fp16:
                self.model = self.model.to(torch.float16)
            self.model.to(self.dev)
        
        # Initialize QSL (Query Sample Library)
        self.qsl = get_squad_QSL(max_examples)

        for index in range(len(self.qsl.eval_features)):
            self.qsl.eval_features[index].input_ids = torch.tensor(self.qsl.eval_features[index].input_ids, device=self.dev, dtype=torch.int32)
            self.qsl.eval_features[index].segment_ids = torch.tensor(self.qsl.eval_features[index].segment_ids, device=self.dev, dtype=torch.int32)
    
    def init_ready(self) -> int:
        return 1
    
    def issue_queries(self, indexes: List[int]) -> torch.tensor:
        print(f"({time.time()}) Task received")
        with torch.no_grad():
            num_queries = len(indexes)
            hidden_size = 384

            input_ids = torch.empty((num_queries, hidden_size), device=self.dev, dtype=torch.int32)
            token_type_ids = torch.empty((num_queries, hidden_size), device=self.dev, dtype=torch.int32)
            attention_mask_prefix_lens = []

            for i, index in enumerate(indexes):
                eval_features = self.qsl.get_features(index)
                input_ids[i] = eval_features.input_ids
                token_type_ids[i] = eval_features.segment_ids
                attention_mask_prefix_lens.append(sum(eval_features.input_mask))

            attention_mask_prefix_lens = torch.tensor(attention_mask_prefix_lens, dtype=torch.int32, device=self.dev)
            print(f"({time.time()}) Input generated")

            num_batches = (num_queries + pytorch_worker_max_batchsize - 1) // pytorch_worker_max_batchsize
            start_scores = []
            end_scores = []

            for batch_index in range(num_batches):
                print(f"({time.time()}) Worker {self.worker_id} processing batch {batch_index}...")
                batch_start_index = batch_index * pytorch_worker_max_batchsize
                batch_end_index = min((batch_index + 1) * pytorch_worker_max_batchsize, num_queries)
                model_output = self.model.forward(
                    input_ids = input_ids[batch_start_index:batch_end_index],
                    attention_mask_prefix_lens = attention_mask_prefix_lens[batch_start_index:batch_end_index],
                    token_type_ids = token_type_ids[batch_start_index:batch_end_index]
                )
                batch_start_scores = model_output.start_logits
                batch_end_scores = model_output.end_logits
                start_scores.append(batch_start_scores)
                end_scores.append(batch_end_scores)
            
            start_scores = torch.cat(start_scores, dim=0)
            end_scores = torch.cat(end_scores, dim=0)

            output = torch.stack([start_scores, end_scores], axis=-1).cpu().numpy()
            print(f"({time.time()}) Result generated")
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
        print("Begin to create workers")
        if num_deepsparse_workers != 0:
            pg = placement_group(
                [{"CPU": num_cpus_per_deepsparse_worker}]*num_deepsparse_workers,
                strategy="STRICT_SPREAD"
            )
            ray.get(pg.ready())

            self.deepsparse_workers = []
            for i in range(num_deepsparse_workers):
                self.deepsparse_workers.append(
                    BERT_DeepSparse_Worker.options(
                        scheduling_strategy = PlacementGroupSchedulingStrategy(
                            placement_group = pg,
                            placement_group_bundle_index=i
                        )
                    ).remote(i, bert_config, args.max_examples)
                )

            handlers = []
            for worker in self.deepsparse_workers:
                handlers.append(worker.load_engine.remote())
            ray.get(handlers)
        else:
            self.deepsparse_workers = []

        self.pytorch_workers = []
        for i in range(num_pytorch_workers):
            self.pytorch_workers.append(BERT_PyTorch_Worker.remote(i, bert_config, args.max_examples))
        for worker in self.pytorch_workers:
            ray.get(worker.init_ready.remote())
        
        self.workers = self.pytorch_workers + self.deepsparse_workers

        # Initialize SUT (System Under Test)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        print(f"({time.time()}) Test starts")
        num_query_samples = len(query_samples)

        load_data_start_time = time.time()
        indexes = []
        for i in range(num_query_samples):
            indexes.append(query_samples[i].index)
        load_data_end_time = time.time()

        total_load_frac = deepsparse_pytorch_work_ratio*num_deepsparse_workers + num_pytorch_workers
        deepsparse_worker_load = int((deepsparse_pytorch_work_ratio / total_load_frac) * num_query_samples)
        pytorch_worker_load = int((1.0 / total_load_frac) * num_query_samples)
        worker_loads = [pytorch_worker_load] * num_pytorch_workers + [deepsparse_worker_load] * num_deepsparse_workers
        worker_loads[0] += num_query_samples - sum(worker_loads)
        print(f"Worker loads: {worker_loads}")
        
        forward_time_start = time.time()
        model_output_handlers = []
        work_start_index = 0
        for worker_id in range(len(self.workers)):
            cur_worker_load = worker_loads[worker_id]
            handler = self.workers[worker_id].issue_queries.remote(indexes[work_start_index:work_start_index+cur_worker_load])
            model_output_handlers.append(handler)
            work_start_index += cur_worker_load
        model_outputs = ray.get(model_output_handlers)
        forward_time_end = time.time()

        output = np.concatenate(model_outputs, axis=0).astype(np.float32)

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
        print(f"Estimated QPS: {str(num_query_samples / (submit_time_end - load_data_start_time))}")

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
