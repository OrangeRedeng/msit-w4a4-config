# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import random

import torch
from datasets import Dataset
from datasets import Features, Sequence, Value
from torch.utils.data import DataLoader
import sys
from functools import partial, lru_cache
from transformers import AutoTokenizer

import logging
import numpy as np
import os
import pickle
from datasets import load_dataset, concatenate_datasets

CALIB_DATASETS = {}

@lru_cache(None)
def warning_once(self, msg: str):
    self.warning(msg)
    
logging.Logger.warning_once = warning_once
logger = logging.getLogger("autoround")
logger.setLevel(logging.INFO)
logger.propagate = False

def register_dataset(name):
    """Class decorator to register a DATASET subclass to the registry.

    Decorator function used before a Pattern subclass.

    Args:
        name: A string. Define the dataset type.

    Returns:
        cls: The class of register.
    """

    def register(dataset):
        CALIB_DATASETS[name] = dataset
        return dataset

    return register


def get_tokenizer_function_ceval(tokenizer, seqlen):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
     of seqlen to the "code" field of examples.
    """

    def default_tokenizer_function(examples):

        promt = f"{examples['question']}\nA. {examples['A']}\nB. {examples['B']}\nC. {examples['C']}\nD. {examples['D']}\n答案：examples['answer']."
        example = tokenizer(promt, truncation=True, max_length=seqlen)

        return example

    return default_tokenizer_function


def get_combined_ceval(target_split: str = "val", shuffle=True, subset=None, seed=42):

    splits = ('accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine', 'business_administration', 'chinese_language_and_literature',
              'civil_servant', 'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics', 'college_programming',
              'computer_architecture', 'computer_network', 'discrete_mathematics', 'education_science', 'electrical_engineer', 'environmental_impact_assessment_engineer',
              'fire_engineer', 'high_school_biology', 'high_school_chemistry', 'high_school_chinese', 'high_school_geography', 'high_school_history', 'high_school_mathematics',
              'high_school_physics', 'high_school_politics', 'ideological_and_moral_cultivation', 'law', 'legal_professional', 'logic', 'mao_zedong_thought', 'marxism',
              'metrology_engineer', 'middle_school_biology', 'middle_school_chemistry', 'middle_school_geography', 'middle_school_history', 'middle_school_mathematics',
              'middle_school_physics', 'middle_school_politics', 'modern_chinese_history', 'operating_system', 'physician', 'plant_protection', 'probability_and_statistics',
              'professional_tour_guide', 'sports_science', 'tax_accountant', 'teacher_qualification', 'urban_and_rural_planner', 'veterinary_medicine')

    cache_file = f"ceval_combined.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            datasets = pickle.load(f)
    else:
        datasets = [load_dataset("ceval/ceval-exam", split)[target_split]
                    for split in splits]
        with open(cache_file, 'wb') as f:
            pickle.dump(datasets, f)

    min_length = min([len(dataset) for dataset in datasets])
    datasets_fixed = []
    for dataset in datasets:
        dataset = dataset.shuffle(seed=seed)
        dataset = select_dataset(dataset, range(min_length))
        datasets_fixed.append(dataset)
    datasets = datasets_fixed

    concatenated_dataset = concatenate_datasets(datasets)
    concatenated_dataset = concatenated_dataset.shuffle(seed=seed)
    return concatenated_dataset


@register_dataset("ceval/ceval-exam")
def get_ceval_dataset(tokenizer, seqlen, dataset_name="ceval/ceval-exam", split=None, seed=42,
                     apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    split = "test" if not split else split
    tokenizer_function = get_tokenizer_function_ceval(tokenizer, seqlen)
    
    try:
        calib_dataset = get_combined_ceval(target_split=split, seed=seed)
    except Exception as e:
        logger.error(f"Failed to load the dataset: {e}." \
                     "Consider using a backup dataset by `pip install modelscope`" \
                     " and set '--dataset ceval/ceval-exam' in AutoRound API.")
        sys.exit(1)

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset_tokenized = calib_dataset.map(tokenizer_function, batched=False, remove_columns=["question", "answer", "A", "B", "C", "D", "explanation"])

    return calib_dataset_tokenized


def get_tokenizer_function_gptqa(tokenizer, seqlen):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
     of seqlen to the "code" field of examples.
    """

    def default_tokenizer_function(examples):

        promt = f"{examples['question']}\nAnswer:"
        example = tokenizer(promt, truncation=True, max_length=seqlen)

        return example

    return default_tokenizer_function


@register_dataset("fingertap/GPQA-Diamond")
def get_gptqa_dataset(tokenizer, seqlen, dataset_name="fingertap/GPQA-Diamond", split=None, seed=42,
                     apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    split = "test" if not split else split
    tokenizer_function = get_tokenizer_function_gptqa(tokenizer, seqlen)
    
    calib_dataset = load_dataset(dataset_name, split=split)

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset_tokenized = calib_dataset.map(tokenizer_function, batched=False, remove_columns=["question", "answer"])

    return calib_dataset_tokenized


def get_tokenizer_function_arc(tokenizer, seqlen):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
     of seqlen to the "code" field of examples.
    """

    def default_tokenizer_function(examples):
        
        choises_str = '\n'.join([f'{label}. {text}' for label, text in zip(examples['choices']['label'], examples['choices']['text'])])
        promt = f"{examples['question']}\n{choises_str}\nAnswer:"
        example = tokenizer(promt, truncation=True, max_length=seqlen)

        return example

    return default_tokenizer_function


@register_dataset("allenai/ai2_arc")
def get_arc_dataset(tokenizer, seqlen, dataset_name="allenai/ai2_arc", split=None, seed=42,
                     apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    split = "validation" if not split else split
    tokenizer_function = get_tokenizer_function_arc(tokenizer, seqlen)
    
    calib_dataset = load_dataset(dataset_name, "ARC-Challenge", split=split)
    columns_to_remove = calib_dataset.column_names.copy()

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset_tokenized = calib_dataset.map(tokenizer_function, batched=False, remove_columns=columns_to_remove)

    return calib_dataset_tokenized


def apply_chat_template_to_samples(samples, tokenizer, seqlen, system_prompt=None):
    rendered_messages = []
    if system_prompt is None:
        system_prompt = "You are a helpful assistant."
    for text in samples:
        if system_prompt == "":
            message = [{"role": "user", "content": text}]
        else:
            message = [{"role": "system", "content": system_prompt},
                       {"role": "user", "content": text}]
        try:
            chat_templated = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,

            )
        except:
            logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            message_modified = [msg for msg in message if msg["role"] != "system"]
            chat_templated = tokenizer.apply_chat_template(
                message_modified,
                tokenize=False,
                add_generation_prompt=True,

            )

        rendered_messages.append(chat_templated)
    example = tokenizer(rendered_messages, truncation=True, max_length=seqlen)
    return example


def get_tokenizer_function(tokenizer, seqlen, apply_chat_template=False, system_prompt=None):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length of
    seqlen to the "text" field of examples.
    """

    def default_tokenizer_function(examples):
        if not apply_chat_template:
            example = tokenizer(examples["text"], truncation=True, max_length=seqlen)
        else:
            example = apply_chat_template_to_samples(examples["text"], tokenizer, seqlen, system_prompt)
        return example

    return default_tokenizer_function


def filter_func(example, seqlen=4096):
    if isinstance(example["input_ids"], list):
        example["input_ids"] = torch.tensor(example["input_ids"])
    if example["input_ids"].shape[-1] < seqlen:
        return False
    input_ids = example["input_ids"][:seqlen]
    input_ids_list = input_ids.tolist()
    if len(input_ids_list) > 1 and seqlen > 2 and input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
        return False
    return True


@register_dataset("NeelNanda/pile-10k")
def get_pile_dataset(tokenizer, seqlen, dataset_name="NeelNanda/pile-10k", split=None, seed=42,
                     apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    if not split:
        split = "train"

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_chat_template=apply_chat_template,
                                                system_prompt=system_prompt)
    try:
        calib_dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        import ssl
        error_message = str(e)
        # Check for proxy or SSL error
        if "proxy" in error_message.lower() or isinstance(e, ssl.SSLError) or "SSL" in error_message.upper():
            logger.error(f"Network error detected, please checking proxy settings." \
                         "Error: {error_message}. Or consider using a backup dataset by `pip install modelscope`" \
                         " and set '--dataset swift/pile-val-backup' in AutoRound API.")
        else:
            logger.error(f"Failed to load the dataset: {error_message}")
        sys.exit(1)
    
    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)

    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)
    calib_dataset = calib_dataset.filter(partial(filter_func, seqlen=seqlen))

    return calib_dataset


@register_dataset("togethercomputer/RedPajama-Data-1T-Sample")
def get_rp_dataset(tokenizer, seqlen, dataset_name="togethercomputer/RedPajama-Data-1T-Sample", split=None, seed=42,
                     apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    if not split:
        split = "train"

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_chat_template=apply_chat_template,
                                                system_prompt=system_prompt)
    try:
        calib_dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        import ssl
        error_message = str(e)
        # Check for proxy or SSL error
        if "proxy" in error_message.lower() or isinstance(e, ssl.SSLError) or "SSL" in error_message.upper():
            logger.error(f"Network error detected, please checking proxy settings." \
                         "Error: {error_message}. Or consider using a backup dataset by `pip install modelscope`" \
                         " and set '--dataset togethercomputer/RedPajama-Data-1T-Sample' in AutoRound API.")
        else:
            logger.error(f"Failed to load the dataset: {error_message}")
        sys.exit(1)

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


def get_tokenizer_function_bigcode(tokenizer, seqlen, apply_chat_template=False):
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        apply_chat_template: Whether to apply chat template in tokenization.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
         of seqlen to the "code" field of examples.
        """

        def default_tokenizer_function(examples, apply_chat_template=apply_chat_template):
            if not apply_chat_template:
                promt = f'{examples["complete_prompt"]}\n{examples["canonical_solution"]}'
                example = tokenizer(promt, truncation=True, max_length=seqlen)
            else:
                example = apply_chat_template_to_samples(examples["complete_prompt"], tokenizer, seqlen)
            return example

        return default_tokenizer_function


def get_tokenizer_function_humaneval(tokenizer, seqlen, apply_chat_template=False):
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        apply_chat_template: Whether to apply chat template in tokenization.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
         of seqlen to the "code" field of examples.
        """

        def default_tokenizer_function(examples, apply_chat_template=apply_chat_template):
            if not apply_chat_template:
                example = tokenizer(examples["prompt"], truncation=True, max_length=seqlen)
            else:
                example = apply_chat_template_to_samples(examples["prompt"], tokenizer, seqlen)
            return example

        return default_tokenizer_function


def get_tokenizer_function_gsm(tokenizer, seqlen):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
        of seqlen to the "code" field of examples.
    """

    def default_tokenizer_function(examples):
        promt = f'Question: {examples["question"]}\nAnswer:{examples["answer"]}'
        promt_cut = ' '.join(promt.strip().split(' ')[:-1])
        example = tokenizer(promt_cut, truncation=True, max_length=seqlen)
        
        return example

    return default_tokenizer_function



@register_dataset("bigcode/bigcodebench")
def get_bigcodebench_dataset(tokenizer, seqlen, dataset_name="bigcode/bigcodebench", split=None, seed=42,
                     apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    dataset_name = 'bigcode/bigcodebench'
    split = "v0.1.0_hf"
    tokenizer_function = get_tokenizer_function_bigcode(tokenizer, seqlen, apply_chat_template=apply_chat_template)

    try:
        calib_dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Failed to load the dataset: {e}." \
                     "Consider using a backup dataset by `pip install modelscope`" \
                     " and set 'bigcode/bigcodebench' in AutoRound API.")
        sys.exit(1)

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=False)

    return calib_dataset 



@register_dataset("openai/openai_humaneval")
def get_humaneval_dataset(tokenizer, seqlen, dataset_name="openai/openai_humaneval", split=None, seed=42,
                     apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    dataset_name = 'openai/openai_humaneval'
    split = "test"
    tokenizer_function = get_tokenizer_function_humaneval(tokenizer, seqlen, apply_chat_template=apply_chat_template)

    try:
        calib_dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Failed to load the dataset: {e}." \
                     "Consider using a backup dataset by `pip install modelscope`" \
                     " and set 'bigcode/bigcodebench' in AutoRound API.")
        sys.exit(1)

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset 


@register_dataset("openai/gsm8k")
def get_gsm_dataset(tokenizer, seqlen, dataset_name="bigcode/bigcodebench", split=None, seed=42,
                     apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    dataset_name = 'openai/gsm8k'

    if not split:
        split = "train"

    tokenizer_function = get_tokenizer_function_gsm(tokenizer, seqlen)

    try:
        calib_dataset = load_dataset(dataset_name, 'main', split=split)
    except Exception as e:
        logger.error(f"Failed to load the dataset: {e}." \
                     "Consider using a backup dataset by `pip install modelscope`" \
                     " and set '--dataset openai/gsm8k' in AutoRound API.")
        sys.exit(1)

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    
    calib_dataset = calib_dataset.map(tokenizer_function, batched=False, remove_columns=["question", "answer"])

    return calib_dataset 


def get_tokenizer_function_boolq(tokenizer, seqlen):
    """Returns a default tokenizer function.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
        of seqlen to the "code" field of examples.
    """

    def default_tokenizer_function(examples):

        promt = f'{examples["passage"]}\nQuestion: {examples["question"]}?\nAnswer:'
        example = tokenizer(promt, truncation=True, max_length=seqlen)

        return example

    return default_tokenizer_function


@register_dataset("google/boolq")
def get_boolq_dataset(tokenizer, seqlen, dataset_name="google/boolq", split=None, seed=42,
                     apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    if not split:
        split = "train"

    tokenizer_function = get_tokenizer_function_boolq(tokenizer, seqlen)

    try:
        calib_dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Failed to load the dataset: {e}." \
                     "Consider using a backup dataset by `pip install modelscope`" \
                     " and set '--dataset google/boolq' in AutoRound API.")
        sys.exit(1)
    
    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)

    calib_dataset_tokenized = calib_dataset.map(tokenizer_function, batched=False, remove_columns=["question", "passage"])

    return calib_dataset_tokenized


@register_dataset("swift/pile-val-backup")
def get_pile_val_dataset(tokenizer, seqlen, dataset_name="swift/pile-val-backup", split=None, seed=42,
                         apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test", "validation").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """

    split = "validation"

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_chat_template=apply_chat_template,
                                                system_prompt=system_prompt)
    from transformers.utils.versions import require_version
    require_version("modelscope", "Loading 'swift/pile-val-backup' dataset requires modelscope to be installed, " \
                                  "`pip install modelscope`")
    from modelscope import MsDataset  # pylint: disable=E0401
    calib_dataset = MsDataset.load('swift/pile-val-backup',
                                   'default', split=split).to_iterable_dataset()  # , use_streaming=True
    calib_dataset = calib_dataset.take(10000)

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset("BAAI/CCI3-HQ")
def get_cci3_hq_dataset(tokenizer, seqlen, dataset_name="BAAI/CCI3-HQ", split=None, seed=42, apply_chat_template=False,
                        system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_chat_template=apply_chat_template,
                                                system_prompt=system_prompt)

    calib_dataset = load_dataset(dataset_name, split='train', streaming=True)
    calib_dataset = calib_dataset.take(10000)

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset("codeparrot/github-code-clean")
def get_github_code_clean_dataset(tokenizer, seqlen, dataset_name="codeparrot/github-code-clean", split=None, seed=42,
                                  apply_chat_template=False, system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """

    def get_default_tokenizer_function():
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        apply_chat_template: Whether to apply chat template in tokenization.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
         of seqlen to the "code" field of examples.
        """

        def default_tokenizer_function(examples):
            if not apply_chat_template:
                example = tokenizer(examples["code"], truncation=True, max_length=seqlen)
            else:
                example = apply_chat_template_to_samples(examples["code"], tokenizer, seqlen,
                                                         system_prompt=system_prompt)
            return example

        return default_tokenizer_function

    from datasets import load_dataset

    tokenizer_function = get_default_tokenizer_function()

    calib_dataset = load_dataset(dataset_name, split='train', streaming=True)
    calib_dataset = calib_dataset.take(10000)

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset("madao33/new-title-chinese")
def get_new_chinese_title_dataset(
        tokenizer,
        seqlen,
        dataset_name="madao33/new-title-chinese",
        split=None,
        seed=42,
        apply_chat_template=False,
        system_prompt=None,
        shuffle=True,
        subset=None
):
    """
    Returns a tokenized dataset for the specified parameters.

    Args:
        tokenizer: The tokenizer to use.
        seqlen: Maximum sequence length.
        dataset_name: Name of the dataset to load.
        split: Which split of the dataset to use.
        seed: Random seed for shuffling.
        apply_template: Whether to apply a template to the data.

    Returns:
        A tokenized and shuffled dataset.
    """

    def get_tokenizer_function():
        """Returns a default tokenizer function.

        Args:
        tokenizer: The tokenizer to be used for tokenization.
        seqlen: The maximum sequence length.
        apply_chat_template: Whether to apply chat template in tokenization.

        Returns: A default tokenizer function that applies the provided tokenizer with truncation and a maximum length
        of seqlen to the "text" field of examples.
        """

        def default_tokenizer_function(examples):
            if not apply_chat_template:
                example = tokenizer(examples["content"], truncation=True, max_length=seqlen)
            else:
                example = apply_chat_template_to_samples(examples["content"], tokenizer, seqlen,
                                                         system_prompt=system_prompt)
            return example

        return default_tokenizer_function

    split = "train"
    from datasets import load_dataset

    tokenizer_function = get_tokenizer_function()

    calib_dataset = load_dataset(dataset_name, split=split)

    if shuffle:
        calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset("mbpp")
def get_mbpp_dataset(tokenizer, seqlen, dataset_name="mbpp", split=None, seed=42, apply_chat_template=False,
                     system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for the specified dataset and split.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name of the dataset.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for the specified dataset and split, using the provided tokenizer and sequence length.
    """
    from datasets import load_dataset

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_chat_template=apply_chat_template,
                                                system_prompt=system_prompt)

    samples = []
    splits = split
    if splits is None:
        splits = ["train", "validation", "test"]
    if isinstance(splits, str):
        splits = splits.split("+")

    for split in splits:
        dataset = load_dataset(dataset_name, split=split)
        for data in dataset:
            samples.append({"text": data["text"] + data["code"]})
    if shuffle:
        random.Random(seed).shuffle(samples)
    import datasets

    calib_dataset = datasets.Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)

    return calib_dataset


@register_dataset("local")
def get_local_dataset(tokenizer, seqlen, dataset_name="./tmp.json", split=None, seed=42, apply_chat_template=False,
                      system_prompt=None, shuffle=True, subset=None):
    """Returns a dataloader for a custom dataset and split.
    We allow the input of a json or text file containing a processed text sample each line.

    Args:
    tokenizer: The tokenizer to be used for tokenization.
    seqlen: The maximum sequence length.
    data_name: The name or path of the dataset, which is a json or jsonl file.
    split: The data split to be used (e.g., "train", "test").
    seed: The random seed for shuffling the dataset.
    apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
    A dataloader for a custom dataset and split, using the provided tokenizer and sequence length.
    """
    tokenizer_function = get_tokenizer_function(tokenizer, seqlen, apply_chat_template=apply_chat_template,
                                                system_prompt=system_prompt)

    def load_local_data(data_path):
        if data_path.endswith(".json"):
            with open(data_path, "r") as f:
                data = json.load(f)
            return data
        elif data_path.endswith(".jsonl"):
            data = []
            with open(data_path) as f:
                for line in f:
                    sample = json.loads(line)
                    data.append(sample)
            return data
        else:
            logger.error("invalid local file type, for now only support json/jsonl format data file.")

    samples = []
    dataset = load_local_data(dataset_name)
    if isinstance(dataset, dict):
        new_dataset = []
        for key in dataset.keys():
            new_dataset.append(dataset[key])
        dataset = new_dataset
    for data in dataset:
        text = data
        if isinstance(text, str):
            pass
        elif isinstance(data, dict) and len(data.keys()) == 1:
            for item in data.items():
                text = item[1]
        elif isinstance(data, dict) and "text" in data.keys():
            text = data["text"]
        elif isinstance(data, dict) and "input_ids" in data.keys():
            text = data["input_ids"]
        assert isinstance(text, str), "data must be string"
        text = text.rstrip()
        text = text.rstrip("\n")
        samples.append({"text": text})

    if shuffle:
        random.Random(seed).shuffle(samples)
    import datasets

    calib_dataset = datasets.Dataset.from_list(samples)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)
    return calib_dataset


def get_dataset_len(dataset):
    """Calculates the length of a dataset.

    Args:
        dataset: The dataset object, which can be any iterable or collection.

    Returns:
        int: The length of the dataset.

    Raises:
        If the dataset does not support `len()`, iterates through it to count the number of elements.
    """
    try:
        dataset_len = len(dataset)
        return dataset_len
    except:
        cnt = 0
        for _ in dataset:
            cnt += 1
        return cnt


def select(dataset, indices):
    """Selects specific elements from a dataset based on given indices.

    Args:
        dataset: The dataset object to iterate over.
        indices: An iterable of integers specifying the indices to select.

    Yields:
        Elements of the dataset corresponding to the specified indices.

    Notes:
        Stops iterating once the highest index in `indices` has been processed
        to optimize performance.
    """
    indices = set(indices)
    for idx, sample in enumerate(dataset):
        if idx in indices:
            yield sample
        if idx > max(indices):
            break


def select_dataset(dataset, indices):
    """Selects elements from a dataset using its native `select` method, if available.

    Args:
        dataset: The dataset object, which may have a `select` method.
        indices: An iterable of integers specifying the indices to select.

    Returns:
        A subset of the dataset, either using the dataset's `select` method or the
        `select` function defined above as a fallback.
    """
    try:
        return dataset.select(indices)
    except:
        list_data = list(select(dataset, indices))
        import pandas as pd
        df = pd.DataFrame(list_data)
        dataset = Dataset.from_pandas(df)
        return dataset


def get_dataloader(
        tokenizer,
        seqlen,
        dataset_name="NeelNanda/pile-10k",
        seed=42,
        bs=8,
        nsamples=512,
        target_idxs=None
):
    """Generate a DataLoader for calibration using specified parameters.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for tokenization.
        seqlen (int): The exact sequence length. samples < seqlen will be dropped,
                      samples longer than seqlen will be truncated
        dataset_name (str, optional): The name of the dataset or datasets separated by commas.
                                     Defaults to "NeelNanda/pile-10k".
        split (str, optional): The data split to use. Defaults to None.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        bs (int, optional): The batch size. Defaults to 4.
        nsamples (int, optional): The total number of samples to include. Defaults to 512.
        apply_chat_template: Whether to apply chat template in tokenization.

    Returns:
        DataLoader: The DataLoader for the calibrated dataset.
    """
    dataset_name = dataset_name.replace(" ", "")
    dataset_names = dataset_name.split(",")

    random.seed(seed)

    def filter_func(example, seqlen):
        if isinstance(example["input_ids"], list):
            example["input_ids"] = torch.tensor(example["input_ids"])
        if example["input_ids"].shape[-1] < seqlen:
            return False
        input_ids = example["input_ids"][:seqlen]
        input_ids_list = input_ids.tolist()
        if len(input_ids_list) > 1 and seqlen > 2 and input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
            return False
        return True


    def concat_dataset_element(dataset):
        input_ids, concat_input_ids = [eg['input_ids'] for eg in dataset], []
        attention_mask_list, attention_mask = [], torch.ones([1, seqlen]).to(torch.int64)
        buffer_input_id = torch.Tensor().to(torch.int64)
        bos_token_id, eos_token_id = tokenizer.bos_token_id, tokenizer.eos_token_id
        os_cnt, have_bos, have_eos = 0, False, False

        for input_id in input_ids:
            if input_id[0] == bos_token_id:
                input_id = input_id[1:]
                os_cnt, have_bos = os_cnt + 1, True
            if input_id[-1] == eos_token_id:
                input_id = input_id[:-1]
                os_cnt, have_eos = os_cnt + 1, True

            if buffer_input_id.shape[-1] + input_id.shape[-1] + os_cnt > seqlen:
                idx_keep = seqlen - buffer_input_id.shape[-1] - os_cnt
                input_id_to_append = [buffer_input_id, input_id[:idx_keep]]
                if have_bos:
                    input_id_to_append = [torch.tensor([bos_token_id])] + input_id_to_append
                if have_eos:
                    input_id_to_append.append(torch.tensor([eos_token_id]))

                concat_input_ids.append(torch.cat(input_id_to_append).to(torch.int64))
                attention_mask_list.append(attention_mask)
                buffer_input_id = input_id[idx_keep:]
            else:
                buffer_input_id = torch.cat([buffer_input_id, input_id])

            if buffer_input_id.shape[-1] + os_cnt == seqlen:
                input_id_to_append = [buffer_input_id]
                if have_bos:
                    input_id_to_append = [torch.tensor([bos_token_id])] + input_id_to_append
                if have_eos:
                    input_id_to_append.append(torch.tensor([eos_token_id]))
                concat_input_ids.append(torch.cat(input_id_to_append).to(torch.int64))
                attention_mask_list.append(attention_mask)
                buffer_input_id = torch.Tensor().to(torch.int64)
        
        data = [{'input_ids': a, 'attention_mask': b} for a, b in zip(concat_input_ids, attention_mask_list)]
        import datasets
        dataset_new = datasets.Dataset.from_list(data)
        return dataset_new

    datasets, data_lens = [], {}
    system_prompt = "You are a helpful assistant."
    for name in dataset_names:
        split = None
        do_concat = False
        apply_chat_template = False
        subset = None

        if ":" in name:
            name, split_list = name.split(":")[0], name.split(":")[1:]
            for ele in split_list:
                key, values = ele.split('=')[0], ele.split('=')[1:]
                if key == "split":
                    split = values[0].split('+')[0]
                if key == "num":
                    num_values = float(values[0])
                    if num_values < 1:
                        num_values = int(num_values * nsamples)
                    else:
                        num_values = int(num_values)
                    data_lens[name] = num_values
                if key == "concat":
                    do_concat = False if (len(values) > 0 and values[0].lower() == 'false') else True
                if key == "apply_chat_template":
                    apply_chat_template = False if (len(values) > 0 and values[0].lower() == 'false') else True
                if key == "system_prompt":
                    system_prompt = values[0]
                    apply_chat_template = True
                if key == "subset":
                    subset = values[0]

        calib_name = name
        if name not in CALIB_DATASETS.keys():
            calib_name = name.split('/')[-1]
            for key in CALIB_DATASETS.keys():
                if calib_name in key:
                    calib_name = key
                    break
        get_dataset = CALIB_DATASETS.get(calib_name)
        
        dataset = get_dataset(
            tokenizer,
            seqlen,
            seed=seed,
            split=split,
            dataset_name=name,
            apply_chat_template=apply_chat_template,
            system_prompt=system_prompt,
            shuffle=False if split == 'validation' else True,
            subset=subset
        )

        if do_concat:
            dataset = concat_dataset_element(dataset)
        # dataset = dataset.filter(filter_func)
        if target_idxs and name in target_idxs:
            dataset = select_dataset(dataset, target_idxs[name])
        elif name in data_lens:
            dataset = select_dataset(dataset, range(data_lens[name]))
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        new_features = {}
        for k, v in dataset.features.items():
            if k == "input_ids":
                new_features[k] = Sequence(Value('int64'))
            elif k == "attention_mask":
                new_features[k] = Sequence(Value('int8'))
            else:
                new_features[k] = v

        dataset = dataset.cast(Features(new_features))
        datasets.append(dataset)

    if len(datasets) == 1:
        set_num_samples = next(iter(data_lens.values()), nsamples)
        if set_num_samples != nsamples:
            logger.info(f"Take {nsamples} samples from {dataset_names[0]} dataset instead of {set_num_samples} samples.")
        dataset_final = datasets[0]
        # Dataset was shuffled previously
        if not target_idxs:
            dataset_final = select_dataset(dataset_final, range(nsamples))
    else:
        indices = range(len(datasets))
        lens = []
        for i in range(len(datasets)):
            cnt = get_dataset_len(datasets[i])
            lens.append(cnt)
        res = sorted(zip(indices, lens), key=lambda x: x[1])

        indices = [item[0] for item in res]
        datasets = [datasets[item[0]] for item in res]
        dataset_names = [dataset_names[index] for index in indices]
        cnt = 0 if not data_lens else sum(data_lens.values())
        dataset_cnt_info = {}
        if cnt > nsamples:
            cnt = 0

        for i in range(len(datasets)):
            name = dataset_names[i].split(':')[0]
            if name not in data_lens:
                target_cnt = (nsamples - cnt) // (len(datasets) - len(data_lens)) if data_lens \
                    else (nsamples - cnt) // (len(datasets) - i)
                target_cnt = min(target_cnt, lens[i])
                cnt += target_cnt
            else:
                target_cnt = data_lens[name]
            
            # Dataset was shuffled previously
            datasets[i] = select_dataset(datasets[i], range(target_cnt))
            dataset_cnt_info[name] = target_cnt

        if len(datasets) > 1:
            from datasets import concatenate_datasets

            dataset_final = concatenate_datasets(datasets)
            dataset_final = dataset_final.shuffle(seed=seed)

            logger.info(dataset_cnt_info)
        else:
            dataset_final = datasets[0]

    @torch.no_grad()
    def collate_batch(batch):
        input_ids_new = []
        attention_mask_new = []
        for text in batch:
            input_ids, attention_mask = text["input_ids"], text["attention_mask"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            if isinstance(attention_mask, list):
                attention_mask = torch.tensor(attention_mask)
            input_ids = input_ids[:seqlen]
            input_ids_list = input_ids.tolist()
            if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
                continue
            attention_mask = attention_mask[:seqlen]
            attention_mask_new.append(attention_mask)
            input_ids_new.append(input_ids)
        if len(input_ids_new) == 0:
            return None
        input_ids_new = torch.vstack(input_ids_new)
        attention_mask_new = torch.vstack(attention_mask_new)
        res = {"input_ids": input_ids_new, "attention_mask": attention_mask_new}
        return res
    
    return dataset_final

if __name__ == '__main__':
    common_seed = 42
    torch.manual_seed(common_seed)
    torch.cuda.manual_seed(common_seed)
    np.random.seed(common_seed)

    model_name = "/mnt/nvme0/ckpt/Qwen3-8B/"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    seq_length = 2048
    
    dataset = 'NeelNanda/pile-10k:num=506, openai/gsm8k:num=206'
    
    combined_dataset = get_dataloader(tokenizer,
                seqlen=seq_length,
                dataset_name=dataset,
                seed=common_seed)
    
    output_path = 'examples/val_data/new_val_test.jsonl'
    with open(output_path, 'a', encoding='utf-8') as fout:
        for i, line in enumerate(combined_dataset):
            json.dump({'id': i, 'inputs_pretokenized': tokenizer.decode(line['input_ids'])}, fout)
            fout.write('\n')
