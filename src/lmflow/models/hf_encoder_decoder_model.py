#!/usr/bin/env python
# coding=utf-8
"""This is a class called HFDecoderModel which is a wrapper around transformers model and
tokenizer classes. It has several methods such as __init__, tokenize, and train that are
used for training and fine-tuning the model. The __init__ method takes in several arguments
such as model_args, tune_strategy, and ds_config, which are used to load the pretrained
model and tokenizer, and initialize the training settings.

The tokenize method is used to tokenize the input text and return the input IDs and attention
masks that can be fed to the model for training or inference.

This class supports different tune_strategy options such as 'normal', 'none', 'lora', and
'adapter', which allow for different fine-tuning settings of the model. However, the 'lora'
and 'adapter' strategies are not yet implemented.

Overall, this class provides a convenient interface for loading and fine-tuning transformer
models and can be used for various NLP tasks such as language modeling, text classification,
and question answering.
"""

import copy
import hashlib
import logging
from typing import List, Union

import deepspeed

from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_config,
    get_peft_model,
)

import torch
import transformers
from transformers.deepspeed import HfDeepSpeedConfig

from transformers.testing_utils import CaptureLogger

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoModel,
    AutoProcessor,
)

from transformers import (Blip2VisionConfig,
                          Blip2QFormerConfig,
                          Blip2Config,
                          LlamaConfig)

from lmflow.datasets.dataset import Dataset
from lmflow.models.encoder_decoder_model import EncoderDecoderModel
from lmflow.models.interfaces.tunable import Tunable
from lmflow.models.vision2seq_model import CustomAutoVision2SeqModel

logger = logging.getLogger(__name__)


class HFEncoderDecoderModel(EncoderDecoderModel, Tunable):
    r"""
    Initializes a HFEncoderDecoderModel instance.

    Parameters
    ------------

    model_args :
        Model arguments such as model name, path, revision, etc.

    tune_strategy : str or none,  default="normal".
        A string representing the dataset backend. Defaults to "huggingface".

    ds_config :
        Deepspeed configuations.

    args : Optional.
        Positional arguments.

    kwargs : Optional.
        Keyword arguments.
    """

    def __init__(
        self,
        model_args,
        tune_strategy='normal',
        ds_config=None,
        device="gpu",
        use_accelerator=True,
        custom_model=False,
        *args,
        **kwargs
    ):
        """
        Initializes a HFDecoderModel instance.
        :param model_args: dictionary with model arguments such as model name, path, revision, etc.
        :param tune_strategy: tuning strategy: normal, none, lora or adapter
        :param ds_config: deepspeed configuration for distributed training
        """

        # See more about loading any type of standard or custom dataset (from
        # files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Load pretrained model and tokenizer
        #
        # Distributed training: The .from_pretrained methods guarantee that
        # only one local process can concurrently download model & vocab.

        self.device = device
        self.model_args = model_args
        self.arch_type = self.model_args.arch_type
        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        if model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **tokenizer_kwargs)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is"
                " not supported by this script. You can do it from another"
                " script, save it, and load it from here, using"
                " --tokenizer_name."
            )

        self.tokenizer = tokenizer  

        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **config_kwargs)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if model_args.config_overrides is not None:
                logger.info(f"Overriding config: {model_args.config_overrides}")
                config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {config}")


        if tune_strategy == 'normal':
            if model_args.model_name_or_path:
                import pdb; pdb.set_trace()
                model = AutoModel.from_pretrained(
                    model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    load_in_8bit=model_args.use_int8,
                    device_map="auto"
                )
            else:
                model = AutoModel.from_config(config)
                n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
                logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
            self.backend_model_full = model
            if model_args.use_lora:
                if model_args.lora_target_modules:
                    lora_target_modules = model_args.lora_target_modules
                else:
                    lora_target_modules = None
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=lora_target_modules,
                )
                model = get_peft_model(model, peft_config)
                # model.base_model.tie_weights()
                model.print_trainable_parameters()

            # We resize the embeddings only when necessary to avoid index errors.
            # If you are creating a model from scratch on a small vocab and want a
            # smaller embedding size, remove this test.
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if len(tokenizer) > embedding_size:
                model.resize_token_embeddings(len(tokenizer))

            self.config = config
            self.backend_model = model
            self.tune_strategy = tune_strategy
            # raise NotImplementedError(
            #     f"tune_strategy \"{tune_strategy}\" is not supported"
            # )
        elif tune_strategy == 'none':
            if use_accelerator:
                peft_model_id = model_args.lora_model_path
                self.backend_model = AutoModel.from_pretrained(
                        model_args.model_name_or_path,
                        config=config,
                        device_map="auto",
                        offload_folder="offload",
                        offload_state_dict=True,
                        torch_dtype=torch_dtype,
                        load_in_8bit = True,
                        trust_remote_code=True
                    )
                if peft_model_id is not None:
                    self.backend_model = PeftModel.from_pretrained(
                        self.backend_model, 
                        peft_model_id, 
                    )
                self.tokenizer.padding_side = "left"
            else:
                # dschf = HfDeepSpeedConfig(ds_config)
                peft_model_id = model_args.lora_model_path
                # NOTE: Currently offload is not supported by llama
                if "llama" in model_args.model_name_or_path and model_args.use_ram_optimized_load:
                    logger.warning(
                        "llama does not support RAM optimized load. Automatically"
                        " use original load instead."
                    )
                    model_args.use_ram_optimized_load = False

                # get model register
                self.arch_type = model_args.arch_type
                if self.arch_type == "encoder_decoder":
                    if model_args.model_name_or_path == 'THUDM/chatglm-6b':
                        model_register = AutoModel
                    else:
                        model_register = AutoModelForSeq2SeqLM
                elif self.arch_type == "vision_encoder_decoder":
                    if not custom_model:
                        model_register = AutoModelForVision2Seq
                    else:
                        model_register = CustomAutoVision2SeqModel
                else:
                    raise NotImplementedError
                if not custom_model:
                    if model_args.model_name_or_path == 'THUDM/chatglm-6b':
                        self.backend_model = model_register.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

                    elif model_args.use_ram_optimized_load and peft_model_id is None:
                        try:
                            # RAM-optimized load
                            self.backend_model = model_register.from_pretrained(
                                model_args.model_name_or_path,
                                device_map="auto",
                                offload_folder="offload",
                                offload_state_dict=True,
                            )
                        except:
                            logger.warning(
                                "Failed to use RAM optimized load. Automatically"
                                " use original load instead."
                            )
                            # Normal load
                            self.backend_model = model_register.from_pretrained(
                                model_args.model_name_or_path,
                            )
                    else:
                        if peft_model_id is not None:
                            logger.warning(
                                "LoRA does not support RAM optimized load currently."
                                " Automatically use original load instead."
                            )
                        self.backend_model = model_register.from_pretrained(
                            model_args.model_name_or_path,
                        )
                # else:
                #     self.backend_model = model_register.from_pretrained(
                #         model_args.model_name_or_path)
                else:
                    # model = CustomAutoVision2SeqModel.from_pretrained(
                    #     model_args.model_name_or_path,
                    # )
                    vision_config = Blip2VisionConfig.from_pretrained("Salesforce/blip2-flan-t5-xxl")
                    qformer_config = Blip2QFormerConfig.from_pretrained("Salesforce/blip2-flan-t5-xxl")
                    text_config = LlamaConfig.from_pretrained("/scratch/PI/tongzhang/qinglian/checkpoints/pretrained_weights/vicuna-7b/")
                    config = Blip2Config.from_vision_qformer_text_configs(vision_config, qformer_config, text_config)
                    model = CustomAutoVision2SeqModel(config)
                    model.vision_model_from_pretrained("Salesforce/blip2-flan-t5-xxl")
                    model.qformer_from_pretrained("Salesforce/blip2-flan-t5-xxl")
                    model.language_model_from_pretrained("/scratch/PI/tongzhang/qinglian/checkpoints/pretrained_weights/vicuna-7b/")
                    state_dict = torch.load(
                        "/scratch/PI/tongzhang/qinglian/checkpoints/pretrained_weights/minigpt4/prerained_minigpt4_7b_converted.pth",
                        map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                    self.backend_model = model

                if self.arch_type == "encoder_decoder":
                    tokenizer_register = AutoTokenizer
                elif self.arch_type == "vision_encoder_decoder":
                    tokenizer_register = AutoProcessor
                else:
                    raise NotImplementedError

                self.tokenizer = tokenizer_register.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
                self.backend_model_full = self.backend_model
                if peft_model_id is not None:
                    self.backend_model = PeftModel.from_pretrained(
                        self.backend_model, peft_model_id
                    )
                if device == "gpu":
                    deepspeed.init_distributed()
                    self.ds_engine = deepspeed.initialize(model=self.backend_model, config_params=ds_config)[0]
                    self.ds_engine.module.eval()

                self.tokenizer.padding_side = "left" #necessary for auto-gressive inference

        elif tune_strategy == 'adapter':
            raise NotImplementedError('adapter tune strategy not implemented')

        if self.arch_type == "encoder_decoder":
            if self.tokenizer.eos_token_id is None:
                self.tokenizer.eos_token_id = self.backend_model.config.eos_token_id
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def tokenize(self, dataset, add_special_tokens=True, *args, **kwargs):
        """
        Tokenize the full dataset.

        Parameters
        ------------
        dataset :
            Text dataset.

        args : Optional.
            Positional arguments.

        kwargs : Optional.
            Keyword arguments.

        Returns
        ------------
        tokenized_datasets :
            The tokenized dataset.
        """
        # raise NotImplementedError('tokenize not implemented')
        if dataset.get_backend() != "huggingface":
            raise NotImplementedError(
                "tokenization of datasets with non-huggingface backend are"
                "not supported yet"
            )

        dataset_type = dataset.get_type()

        # Requires three types of information for tokenizing different datasets
        #   1) Which fields require tokenization, e.g.
        #        "text2float": "text", but not "float"
        #        "text2text": both "input" and "output"
        #   2) How will there tokenized sequence concatenated together, e.g.
        #        "text_only": "text" -> "text"
        #        "text2text": "input", "output" -> "input" + "output"
        #   3) Which fields require loss in final computation, e.g.
        #        "text_only": "text"
        #        "text2text": "output" only
        tokenized_column_order = None       # Handles 1) and 2)
        label_columns = None                # Handles 3)
        if dataset_type == "text_only":
            tokenized_column_order = ["text"]
            label_columns = ["text"]
        elif dataset_type == "text2text":
            tokenized_column_order = ["input", "output"]
            label_columns = ["output"]
            add_special_tokens = False
        else:
            raise NotImplementedError(
                f"dataset type \"{dataset_type}\" is not supported, currently"
                " only support following data types:\n"
                f"    1) {TEXT_ONLY_DATASET_DESCRIPTION}\n"
                f"    2) {TEXT2TEXT_DATASET_DESCRIPTION}\n"
            )

        model_args = self.model_args
        raw_datasets = dataset
        hf_raw_datasets = dataset.get_backend_dataset()
        column_names = list(hf_raw_datasets.features)

        # since this will be pickled to avoid _LazyModule error in Hasher force
        # logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            num_example = len(examples[column_names[0]])
            token_dict = {
                "input_ids": [[] for _ in range(num_example)],
                "attention_mask": [[] for _ in range(num_example)],
                "labels": [[] for _ in range(num_example)],
            }
            with CaptureLogger(tok_logger) as cl:
                for column_name in tokenized_column_order:
                    encoding = self.tokenizer(
                        examples[column_name],
                        add_special_tokens=add_special_tokens,
                        truncation=True if model_args.use_lora else None,
                    )

                    if column_name in label_columns:
                        labels = encoding["input_ids"].copy()
                    else:
                        labels = [
                            [-100] * len(encoding["input_ids"][i])
                             for i in range(num_example)
                        ]

                    for i in range(num_example):
                        token_dict["input_ids"][i].extend(
                            encoding["input_ids"][i]
                        )
                        token_dict["attention_mask"][i].extend(
                            encoding["attention_mask"][i]
                        )
                        token_dict["labels"][i].extend(labels[i])

            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                    " before being passed to the model."
                )
            return token_dict

        data_args = raw_datasets.get_data_args()
        if not data_args.streaming:
            fingerprint = raw_datasets.get_fingerprint()
            new_fingerprint = hashlib.md5(
                (fingerprint + str(self.tokenizer)).encode("utf-8")
            ).hexdigest()

            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                new_fingerprint=new_fingerprint,
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
        return tokenized_datasets

    def encode(self, input: Union[str, List[str]], *args, **kwargs ) -> Union[List[int], List[List[int]]]:
        """
        Perform encoding process of the tokenizer.

        Parameters
        ------------
        inputs : str or list.
            The text sequence.

        args : Optional.
            Positional arguments.

        kwargs : Optional.
            Keyword arguments.

        Returns
        ------------
        outputs :
            The tokenized inputs.
        """
        if isinstance(input, dict):
            # TODO refactor the input type to make it elegant.
            kwargs.update(input)
            return self.tokenizer(*args, **kwargs)
        elif isinstance(input, list):
            return self.tokenizer(text=input, *args, **kwargs)#batch encode,will automatically do left padding
        elif isinstance(input, str):
            return self.tokenizer.encode(text=input, *args, **kwargs)
        else:
            raise NotImplementedError(f'type "{type(input)}" cannot be encoded')


    def decode(self, input, *args, **kwargs ) -> Union[str, List[str]]:
        """
        Perform decoding process of the tokenizer.

        Parameters
        ------------
        inputs : list.
            The token sequence.

        args : Optional.
            Positional arguments.

        kwargs : Optional.
            Keyword arguments.

        Returns
        ------------
        outputs :
            The text decoded from the token inputs.
        """
        if isinstance(input, List):
            input=torch.tensor(input)
        if input.dim()==2:
            return self.tokenizer.batch_decode(input, *args, **kwargs)#batch_decode
        else:
            # Can be list of ints or a Tensor
            return self.tokenizer.decode(input, *args, **kwargs)


    def inference(self, inputs, use_accelerator=True, *args, **kwargs):
        """
        Perform generation process of the model.

        Parameters
        ------------
        inputs :
            The sequence used as a prompt for the generation or as model inputs to the model.

        args : Optional.
            Positional arguments.

        kwargs : Optional.
            Keyword arguments.

        Returns
        ------------
        outputs :
            The generated sequence output
        """
        # TODO need to discuss how to handle pad_token_id
        if self.arch_type == "encoder_decoder":
            kwargs.update(pad_token_id=self.tokenizer.pad_token_id)
        elif self.arch_type == "vision_encoder_decoder":
            # TODO disucss how to modify the interface to remove this part.
            inputs = copy.deepcopy(inputs)
            input_ids = inputs.pop('input_ids')
            kwargs.update(**inputs)
            inputs = input_ids

        with torch.no_grad():
            if use_accelerator:
                outputs = self.backend_model.generate(
                    input_ids=inputs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    *args,
                    **kwargs
                )
            else:
                if self.device == "gpu":
                    outputs = self.ds_engine.module.generate(
                        input_ids=inputs,
                        synced_gpus=True,
                        *args,
                        **kwargs
                    )
                elif self.device == "cpu":
                    outputs = self.backend_model.generate(
                        input_ids=inputs,
                        synced_gpus=True,
                        *args,
                        **kwargs
                    )
                else:
                    raise NotImplementedError(
                        f"device \"{self.device}\" is not supported"
                    )
        return outputs


    def merge_lora_weights(self):
        if self.model_args.use_lora:
            self.get_backend_model().merge_and_unload()
        else:
            logger.warning("LoRA training is NOT enabled. Merging LoRA weights is not applicable.")


    def save(self, dir, save_full_model=False, *args, **kwargs):
        """
        Perform generation process of the model.

        Parameters
        ------------
        dir :
            The directory to save model and tokenizer

        save_full_model : Optional.
            Whether to save full model.

        kwargs : Optional.
            Keyword arguments.

        Returns
        ------------
        outputs :
            The generated sequence output
        """
        self.get_tokenizer().save_pretrained(dir)
        if save_full_model and self.model_args.use_lora:
            self.backend_model_full.save_pretrained(dir)
        else:
            self.get_backend_model().save_pretrained(dir)


    def get_max_length(self):
        """
        Return max acceptable input length in terms of tokens.
        """
        if "tokenizer" not in self.tokenizer.__dict__:
            return self.tokenizer.model_max_length
        else:
            # for the multi-modality processor,
            # the max length is stored in the inner text tokenizer
            return self.tokenizer.model_max_length
            # return self.tokenizer.tokenizer.model_max_length


    def get_tokenizer(self):
        """
        Return the tokenizer of the model.
        """
        return self.tokenizer


    def get_backend_model(self):
        """
        Return the backend model.
        """
        return self.backend_model
