import os
import csv
from os import path
import warnings
import urllib.request
from urllib.error import HTTPError
import json
from typing import Literal, Sequence, TypedDict, Tuple, List, Union, Any
import getpass
import logging
from peft import PeftModel
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pandas as pd
import torch
import tqdm
from sentence_transformers import SentenceTransformer
from languagechange.usages import TargetUsage

# Define types for chat dialog outside the class for clarity
Role = Literal["system", "user"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class DefinitionGenerator:
    def __init__(self, embedding_model: str = "all-mpnet-base-v2"):
        if embedding_model != None and embedding_model != "":
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = None

    def encode_definitions(self, definitions, encode = 'both'):
        if self.embedding_model != None and (encode == 'both' or encode == 'vectors'):
            vectors = self.embedding_model.encode(definitions)
        else:
            return definitions
        
        if encode == 'both':
            return definitions, vectors
        elif encode == 'vectors':
            return vectors
        else:
            return definitions
        

class LlamaDefinitionGenerator(DefinitionGenerator):
    """
    A tool to create short, clear definitions for words based on example sentences
    using fine-tuned Llama models.

    Attributes:
        model_name (str): Name of the base model (e.g., "meta-llama/Llama-2-7b-chat-hf").
        ft_model_name (str): Name of the fine-tuned model (e.g., "FrancescoPeriti/Llama2Dictionary").
        hf_token (str): Hugging Face token for authentication.
        max_length (int): Maximum token length for the prompt.
        batch_size (int): How many examples to process at once.
        max_time (float): Maximum time (in seconds) allowed per batch.
        temperature (float): Generation temperature.
        model: The loaded fine-tuned model ready for generation.
        tokenizer: The tokenizer that prepares text for the model.
        eos_tokens: Tokens that signal the end of a definition.
    """
    def __init__(self, model_name: str, ft_model_name: str, hf_token: str = None, is_adapter = False,
                 max_length: int = 512, batch_size: int = 32, max_time: float = 4.5, 
                 temperature: float = 1, sampling=False, language=None, embedding_model: str = "all-mpnet-base-v2", torch_dtype = torch.float16, low_cpu_mem_usage = True):
        """
        Sets up the LlamaDefinitionGenerator with model and data details.

        Args:
            model_name (str): The base model name from Hugging Face (e.g., "meta-llama/Llama-2-7b-chat-hf").
            ft_model_name (str): The fine-tuned model name (e.g., "FrancescoPeriti/Llama2Dictionary").
            hf_token (str): Hugging Face token to access models.
            max_length (int): Maximum token length for prompt and definitions (default is 512).
            batch_size (int): Number of examples to process in one go (default is 32).
            max_time (float): Maximum time in seconds per example (default is 4.5).
            temperature (float): Generation temperature if sampling (default is 1).
            sampling (bool): if True, do sampling when generating.
            language (str): a two letter language code to use for the prompts. If None, the English prompt will be used.
            embedding_model (str): the sentence embedding model to use, if encoding definitions.
            torch_dtype: the torch float type, applicable for the pre-trained models.
            low_cpu_mem_usage (bool): if True, set low_cpu_mem_usage=True for the pre-trained model.
        """
        super().__init__(embedding_model = embedding_model)
        self.model_name = model_name
        self.ft_model_name = ft_model_name
        self.name = "LlamaDefinitionGenerator_" + self.model_name + "_" + self.ft_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_time = max_time
        self.temperature = temperature
        self.sampling = sampling
        self.is_adapter = is_adapter
        self.language = language

        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token is None:
                hf_token = getpass.getpass("Enter HuggingFace token: ")

        # Log in to Hugging Face with token
        login(hf_token)

        # Set up the tokenizer with Llama-specific settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if "Llama-2" in self.model_name or ("Llama-3" in self.model_name and self.is_adapter):
            # Load the base model with explicit settings
            chat_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch_dtype,  # Use FP16 for memory efficiency
                low_cpu_mem_usage=low_cpu_mem_usage
            )
            chat_model.eval()
            # Load the fine-tuned model
            ft_model = PeftModel.from_pretrained(chat_model, self.ft_model_name)
            ft_model.eval()
            if "Llama-2" in self.model_name:
                # For the Llama2 definition generator, the model is just the finetuned model
                self.model = ft_model
            else:
                # If using an adapter, merge and unload the finetuned model
                ft_model = ft_model.merge_and_unload()

        elif "Llama-3" in self.model_name and not self.is_adapter:
            ft_model = self.ft_model_name
        
        if "Llama-3" in self.model_name:
            # For the Llama3 definition generators, use a pipeline containing the model and the tokenizer
            self.model = pipeline("text-generation", model=ft_model, tokenizer=self.tokenizer, device_map="auto")

        if "Llama-3" in self.model_name and self.is_adapter:
            self.eos_tokens = [self.model.tokenizer.eos_token_id] + [self.model.tokenizer.encode(token, add_special_tokens=False)[0] for token in ['.', ' .']] #TODO: look into this
        else:
            self.eos_tokens = [self.tokenizer.encode(token, add_special_tokens=False)[0] 
                        for token in [';', ' ;', '.', ' .']]
            self.eos_tokens.append(self.tokenizer.eos_token_id)
        if "Llama-3" in self.model_name and not self.is_adapter:
            self.model.tokenizer.padding_side='left'
            self.model.tokenizer.add_special_tokens = True
            self.model.tokenizer.add_eos_token = True
            self.model.tokenizer.add_bos_token = True

    # apply template for Llama 2
    def apply_chat_template_llama2(self, dataset, system_message, template):
        def apply_chat_template_func(record):
            dialog: Dialog = (Message(role='system', content=system_message),
                            Message(role='user', content=template.format(record['target'], record['example'])))
            prompt = self.tokenizer.decode(self.tokenizer.apply_chat_template(dialog, add_generation_prompt=True))
            return {'text': prompt}
        
        return dataset.map(apply_chat_template_func)
    
    # apply template for Llama 3
    def apply_chat_template_llama3(self, dataset, system_message, template):
        def apply_chat_template_func(record):
            dialog: Dialog = (Message(role='system', content=system_message),
                            Message(role='user', content=template.format(record['target'], record['example'])))
            return self.tokenizer.apply_chat_template(dialog, add_generation_prompt=True, tokenize=False)
        
        return [apply_chat_template_func(row) for row in dataset]
    
    def tokenize(self, dataset):
        def formatting_func(record):
            return record['text']
        
        def tokenization(dataset):
            result = self.tokenizer(formatting_func(dataset),
                            truncation=True,
                            max_length=self.max_length,
                            padding="max_length",
                            add_special_tokens=False)
            return result
        
        tokenized_dataset = dataset.map(tokenization)
        return tokenized_dataset

    def extract_definition(self, answer: str) -> str:
        """
        Extracts the actual definition from the model's response based on model type.

        Args:
            answer (str): The text generated by the model.

        Returns:
            str: A cleaned-up definition string with proper formatting.

        Notes:
            - For Llama-2: Extracts text after '[/INST]'.
            - For Llama-3: Takes the last line.
            - Warns if output appears abnormal.
        """
        if "Llama-2" in self.model_name:
            definition = answer.split('[/INST]')[-1].strip(" .,;:")
            if 'SYS>>' in definition:
                definition = ''
                warnings.warn("Abnormal output detected – the example may be too long. Try shortening it.")
        elif "Llama-3" in self.model_name:
            definition = answer.split('\n')[-1].strip(" .,;:")
            if not definition:
                warnings.warn("Empty definition generated – check if the example is too long.")
        else:
            definition = ''
            warnings.warn("Model type not supported – add handling logic in extract_definition if needed")
        return definition.replace('\n', ' ') + '\n'

    def generate_definitions(self, target_usages: List[TargetUsage],
                             language = None,
                             system_message: str = None,
                             user_message_template: str = None,
                             encode_definitions : str = None
                             ) -> List[str]:
        """
        Generates definitions for all examples in batches using the model.

        Args:
            target_usages (List[TargetUsage]): A list of TargetUsage objects.
            language (str): a two digit language code. If not None, it overrides self.language. Falls back to English.
            system_message (str): The system prompt message. If not None, it overrides the default message for the 
                chosen language.
            user_message_template (str): The template for the user prompt with placeholders {target} and {example}. 
                If not None, it overrides the default message template for the chosen language.
            encode_definitions (str): if not None, use self.embedding_model to encode the definitions. If 'vectors', 
                return only the sentence embeddings. If 'both', return both definitions and embeddings.

        Returns:
            Union[List[str], List[np.ndarray], Tuple[List[str],List[np.ndarray]]: Generated definitions corresponding to each TargetUsage, as text, sentence embeddings, or both.
        """
        # Choose the system and user messages for the specific language, fall back to English if the language is not available
        if language is None:
            if self.language is None:
                language = "EN"
            else:
                language = self.language

        if not system_message or not user_message_template:
            try:
                with urllib.request.urlopen(f'https://raw.githubusercontent.com/ChangeIsKey/languagechange/main/languagechange/locales/{language.lower()}.json') as url:
                    messages = json.load(url)
            except HTTPError:
                logging.info(f"Could not load system and user messages for {language}. Falling back to English.")
                with urllib.request.urlopen(f'https://raw.githubusercontent.com/ChangeIsKey/languagechange/main/languagechange/locales/en.json') as url:
                    messages = json.load(url)
                    
            if not system_message:
                system_message = messages["llama_definition_messages"]["system_message"]
            if not user_message_template:
                user_message_template = messages["llama_definition_messages"]["user_message"]

        examples = []
        for usage in target_usages:
            target = usage.text()[usage.offsets[0]:usage.offsets[1]]
            example = usage.text()
            examples.append({'target': target, 'example': example})

        dataset = Dataset.from_list(examples)

        if "Llama-2" in self.model_name:
            dataset = self.apply_chat_template_llama2(dataset, system_message, user_message_template)
            dataset = self.tokenize(dataset)
            device = next(self.model.parameters()).device

        elif "Llama-3" in self.model_name:
            dataset = self.apply_chat_template_llama3(dataset, system_message, user_message_template)
                    
        definitions = []
        total = len(dataset)
        for i in range(0, total, self.batch_size):
            logging.info(f"{i} out of {total} examples")
            batch = dataset[i:i + self.batch_size]

            if "Llama-2" in self.model_name:
                model_input = dict()
                for k in ['input_ids', 'attention_mask']:
                    model_input[k] = torch.tensor(batch[k]).to(device)
            try:
                if "Llama-2" in self.model_name:
                    output_ids = self.model.generate(
                        **model_input,
                        max_new_tokens=self.max_length,  # Generate up to self.max_length new tokens
                        forced_eos_token_id=self.eos_tokens,  # Note: depends on model API compatibility
                        max_time=self.max_time * self.batch_size,
                        eos_token_id=self.eos_tokens,
                        temperature=self.temperature,
                        do_sample = self.sampling,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    answers = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    definitions.extend([self.extract_definition(answer) for answer in answers])
                elif "Llama-3" in self.model_name:
                    if not self.is_adapter:
                        answers = self.model(
                            batch, 
                            max_new_tokens = self.max_length, 
                            forced_eos_token_id = self.eos_tokens,
                            max_time = self.max_time * self.batch_size, 
                            eos_token_id = self.eos_tokens, 
                            temperature = self.temperature,
                            do_sample = self.sampling,
                            pad_token_id = self.model.tokenizer.eos_token_id
                        )
                    else:                      
                        answers = self.model(
                            batch, 
                            max_new_tokens = self.max_length, 
                            forced_eos_token_id = self.eos_tokens,
                            max_time = self.max_time * self.batch_size, 
                            eos_token_id = self.eos_tokens, 
                            temperature = self.temperature,
                            do_sample = self.sampling,
                            pad_token_id = self.model.tokenizer.eos_token_id,
                            truncation = True,
                            batch_size = self.batch_size
                        )

                    definitions.extend([self.extract_definition(answer[0]['generated_text']) for answer in answers])
            except Exception as e:
                warnings.warn(f"Failed generation for batch starting at index {i}: {e}")
                logging.info(e)
                batch_size = len(batch)
                definitions.extend([''] * batch_size)
            
        if len(definitions) != total:
            raise ValueError(f"Generated definitions count ({len(definitions)}) doesn't match input count ({total}).")
        return self.encode_definitions(definitions, encode=encode_definitions)

    def print_results(self, target_usages: List[TargetUsage], definitions: List[str]) -> None:
        """
        Displays the target word, example sentence, and generated definition for each entry.

        Args:
            target_usages (List[TargetUsage]): List of TargetUsage objects.
            definitions (List[str]): List of generated definitions.
        """
        for usage, definition in zip(target_usages, definitions):
            target = usage.text()[usage.offsets[0]:usage.offsets[1]]
            print(f"Target: {target}\nExample: {usage.text()}\nDefinition: {definition}\n")

    def run(self, target_usages: List[TargetUsage],
            system_message: str = "You are a lexicographer familiar with providing concise definitions of word meanings.",
            template: str = 'Please provide a concise definition for the meaning of the word "{}" in the following sentence: {}'
            ) -> None:
        """
        Executes the complete workflow from generating definitions to printing the results.

        Args:
            target_usages (List[TargetUsage]): List of TargetUsage objects.
            system_message (str): The system prompt message.
            template (str): The template for the user prompt.
        """
        definitions = self.generate_definitions(target_usages, system_message, template)
        self.print_results(target_usages, definitions)


# Data model representing the definition of a target word within an example sentence.
class DefinitionOutput(BaseModel):
    """
    Represents the structured output for a word definition.

    Attributes:
        target (str): The target word.
        example (str): The example sentence.
        definition (str): The concise definition of the target word as used in the sentence.
    """
    target: str = Field(description="The target word")
    example: str = Field(description="The example sentence")
    definition: str = Field(description="The definition of the target word as used in the sentence")


class ChatModelDefinitionGenerator(DefinitionGenerator):
    """
    A model to generate concise definitions for target words using a chat model with structured output.

    The model leverages an underlying chat model (initialized with LangChain) that returns a 
    structured DefinitionOutput.
    """
    def __init__(self, model_name: str, model_provider: str, 
                 langsmith_key: str = None, provider_key_name: str = None, 
                 provider_key: str = None, language: str = None,
                 embedding_model: str = "all-mpnet-base-v2"):
        """
        Initializes the DefinitionModel.

        Args:
            model_name (str): The name of the model.
            model_provider (str): The model provider (e.g., "openai").
            langsmith_key (str, optional): API key for LangSmith. Defaults to None.
            provider_key_name (str, optional): Environment variable name for the provider API key. Defaults to None.
            provider_key (str, optional): The provider API key. Defaults to None.
            language (str, optional): Language code for potential lemmatization. Defaults to None.
        """
        super().__init__(embedding_model = embedding_model)
        self.model_name = model_name
        self.name = "ChatModelDefinitionGenerator_" + self.model_name
        self.language = language

        os.environ["LANGSMITH_TRACING"] = "true"

        # Set the LangSmith API key.
        if langsmith_key is not None:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
        elif not os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for LangSmith: ")

        # Determine the provider key name if not provided.
        if provider_key_name is None:
            provider_key_names = {
                "openai": "OPENAI_API_KEY",
                "groq": "GROQ_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "cohere": "COHERE_API_KEY",
                "nvidia": "NVIDIA_API_KEY",
                "fireworks": "FIREWORKS_API_KEY",
                "mistralai": "MISTRAL_API_KEY",
                "together": "TOGETHER_API_KEY",
                "xai": "XAI_API_KEY",
            }
            if model_provider in provider_key_names.keys():
                provider_key_name = provider_key_names[model_provider]
        # Set the provider API key.
        if provider_key is not None:
            os.environ[provider_key_name] = provider_key
        elif not os.environ.get(provider_key_name):
            os.environ[provider_key_name] = getpass.getpass(f"Enter API key for {model_provider}: ")

        try:
            llm = init_chat_model(model_name, model_provider=model_provider)
        except Exception as e:
            logging.error("Could not initialize chat model.")
            raise e

        # Configure the model to use structured output with the DefinitionOutput schema.
        self.model = llm.with_structured_output(DefinitionOutput)

    def generate_definitions(self, target_usages: List[TargetUsage],
                        user_prompt_template: str = ("Please provide a concise definition for the meaning of the word '{target}' as used in the following sentence:\nSentence: {example}"),
                        encode_definitions : str = None
                       ) -> List[str]:
        """
        Generates definitions for each TargetUsage using a chat model.
        
        Args:
            target_usages (List[TargetUsage]): List of target usages.
            user_prompt_template (str): Template for the user prompt.
        
        Returns:
            Union[List[str], List[np.ndarray], Tuple[List[str],List[np.ndarray]]: Generated definitions corresponding to each TargetUsage, as text, sentence embeddings, or both.
        """
        definitions = []
        system_message = "You are a lexicographer familiar with providing concise definitions of word meanings."
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_message), ("user", user_prompt_template)]
        )
        
        for usage in target_usages:
            # Extract target word and example sentence
            if hasattr(usage, 'target') and hasattr(usage, 'example'):
                target_word = usage.target
                example_sentence = usage.example
            else:
                target_word = usage.text()[usage.offsets[0]:usage.offsets[1]]
                example_sentence = usage.text()
            prompt = prompt_template.invoke({"target": target_word, "example": example_sentence})
            try:
                response = self.model.invoke(prompt)
                try:
                    definitions.append(response.definition)
                except Exception:
                    definitions.append(response)
            except Exception as e:
                logging.error(f"Could not run chat completion: {e}")
                definitions.append("")
        return self.encode_definitions(definitions, encode=encode_definitions)



class T5DefinitionGenerator(DefinitionGenerator):
    """Generates word definitions using a T5 model."""

    def __init__(self, model_path, bsize=4, max_length=256, filter_target=True,
                 sampling=False, temperature=1.0, repetition_penalty=1.0,
                 num_beams=1, num_beam_groups=1):
        """
        Initialize the T5 definition generator.

        Args:
            model_path (str): Path or name of the T5 model.
            bsize (int): Batch size for generation. Defaults to 4.
            max_length (int): Max input length for tokenization. Defaults to 256.
            filter_target (bool): Filter target word from definitions. Defaults to True.
            sampling (bool): Use sampling instead of greedy decoding. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 1.0.
            repetition_penalty (float): Penalty for repeated tokens. Defaults to 1.0.
            num_beams (int): Number of beams for beam search. Defaults to 1.
            num_beam_groups (int): Number of beam groups for diversity. Defaults to 1.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

        self.model_path = model_path
        self.name = "T5DefinitionGenerator_" + self.model_path
        self.bsize = bsize
        self.max_length = max_length
        self.filter_target = filter_target
        self.sampling = sampling
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = (AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto")
                      if torch.cuda.is_available() else
                      AutoModelForSeq2SeqLM.from_pretrained(model_path, low_cpu_mem_usage=True))
        self.model.to(self.device)
        self.logger.info("Model loaded.")

        self.prompts = [
            ["", "post"], ["Give the definition of <TRG>:", "pre"], ["Define <TRG>:", "pre"],
            ["Define the word <TRG>:", "pre"], ["What is the definition of <TRG>?", "pre"],
            ["Give the definition of <TRG>", "post"], ["Define <TRG>", "post"],
            ["Define the word <TRG>", "post"], ["What is the definition of <TRG>?", "post"],
            ["Quelle est la définition de <TRG>?", "post"], ["Что такое <TRG>?", "post"],
            ["Hva betyr <TRG>?", "post"], ["Was ist die Definition von <TRG>?", "post"],
            ["Mikä on <TRG>?", "post"]
        ]

    def load_data(self, path_to_data, split="test"):
        """
        Load and preprocess data from a file or directory.

        Args:
            path_to_data (str): Path to input file or directory.
            split (str): Data split (e.g., 'test', 'trial'). Defaults to 'test'.

        Returns:
            pd.DataFrame: Preprocessed data with 'Targets' and 'Real_Contexts' columns.

        Raises:
            FileNotFoundError: If the specified file or directory doesn’t exist.
        """
        if not path.exists(path_to_data):
            raise FileNotFoundError(f"Data path not found: {path_to_data}")
        
        if path.isfile(path_to_data):
            self.logger.info(f"Loading file: {path_to_data}")
            df = pd.read_csv(path_to_data, delimiter="\t", header="infer", quoting=csv.QUOTE_NONE,
                             encoding="utf-8", on_bad_lines="warn")
            if len(df.columns) == 2:
                df.columns = ["Targets", "Context"]
            df.dropna(subset=["Context"], inplace=True)
        else:
            self.logger.info(f"Loading directory: {path_to_data}")
            if "oxford" not in path_to_data and "wordnet" not in path_to_data:
                datafile = path.join(path_to_data, f"{split}.complete.tsv.gz" if split else "complete.tsv.gz")
                df = pd.read_csv(datafile, delimiter="\t", header=0, quoting=csv.QUOTE_NONE,
                                 encoding="utf-8", on_bad_lines="warn")
                df["Context"] = df.example
                df["Targets"] = [w.split("%")[0] for w in df.word]
                try:
                    df["Definition"] = df.gloss
                except AttributeError:
                    self.logger.info("No definitions found.")
            else:
                datafile = path.join(path_to_data, f"{split}.eg.gz")
                datafile_defs = path.join(path_to_data, f"{split}.txt.gz")
                df = pd.read_csv(datafile, delimiter="\t", quoting=csv.QUOTE_NONE,
                                 encoding="utf-8", on_bad_lines="warn")
                df_defs = pd.read_csv(datafile_defs, delimiter="\t", quoting=csv.QUOTE_NONE,
                                      encoding="utf-8", on_bad_lines="warn")
                df_defs.columns = ["Sense", "Ignore1", "Ignore2", "Definition", "Ignore3", "Ignore4"]
                df.columns = ["Sense", "Context"]
                df["Targets"] = [w.split("%")[0] for w in df.Sense]
                df["Definition"] = df_defs.Definition
        if "wordnet" in path_to_data:
            df["POS"] = [w.split("%")[1].split(".")[2] for w in df.Sense]
        df["Real_Contexts"] = [ctxt.replace("<TRG>", tgt).strip() for ctxt, tgt in zip(df.Context, df.Targets)]
        return df

    def _generate_definitions(self, prompts, targets):
        """Generate definitions for the given prompts (internal use)."""
        self.logger.info(f"Tokenizing with max length {self.max_length}...")
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.max_length)
        target_ids = torch.tensor([t[-1] for t in self.tokenizer(targets, add_special_tokens=False).input_ids])

        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            target_ids = target_ids.to(self.device)

        dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"], target_ids)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.bsize, shuffle=False)

        gen_args = {
            "do_sample": self.sampling, "num_beams": self.num_beams, "num_beam_groups": self.num_beam_groups,
            "temperature": self.temperature, "repetition_penalty": self.repetition_penalty
        }
        if self.num_beam_groups > 1:
            gen_args["diversity_penalty"] = 0.5

        definitions = []
        for inp, att, tgt_ids in tqdm.tqdm(dataloader):
            outputs = (self.model.generate(input_ids=inp, attention_mask=att, max_new_tokens=60,
                                          bad_words_ids=[[el] for el in tgt_ids.tolist()], **gen_args)
                       if self.filter_target else
                       self.model.generate(input_ids=inp, attention_mask=att, max_new_tokens=60, **gen_args))
            definitions.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return definitions

    def encode_definitions(self, df, encode = 'both', return_df = False):
        if encode == 'both' or encode == 'vectors':
            if self.embedding_model != None:
                vectors = self.embedding_model.encode(df["Generated_Definition"].tolist())
            else:
                logging.error("No embedding model specified, could not encode definitions.")
                raise AttributeError
        if not return_df:
            df = df["Generated_Definition"].tolist()
            
        if encode == 'both':
            return df, vectors
        elif encode == 'vectors':
            return vectors
        else:
            return df

    def generate_definitions(self, target_usage_list=None, df=None, prompt_index=8, encode_definitions : str = None, return_df = False):
        """
        Generate definitions for words in the DataFrame.

        Args:
            df (pd.DataFrame): Data with 'Targets' and 'Context' or 'Real_Contexts' columns.
            prompt_index (int): Index of the prompt template to use. Defaults to 8.

        Returns:
            pd.DataFrame: Updated DataFrame with 'Generated_Definition' column.

        Raises:
            ValueError: If prompt_index is invalid or required columns are missing.
        """
        if prompt_index < 0 or prompt_index >= len(self.prompts):
            raise ValueError(f"Prompt index {prompt_index} is out of range (0-{len(self.prompts)-1})")

        if target_usage_list is not None:
            targets = [u.text()[u.offsets[0]:u.offsets[1]] for u in target_usage_list]
            contexts = [u.text() for u in target_usage_list]
            df = pd.DataFrame({"Targets": targets, "Context": contexts})
        elif df == None:
            raise ValueError("Either target_usage_list or df must be provided")

        if "Targets" not in df.columns or ("Context" not in df.columns and "Real_Contexts" not in df.columns):
            raise ValueError("DataFrame must contain 'Targets' and either 'Context' or 'Real_Contexts'")

        task_prefix = self.prompts[prompt_index]
        self.logger.info(f"Using prompt: {task_prefix}")
        context_col = "Real_Contexts" if "Real_Contexts" in df.columns else "Context"
        prompts = [" ".join([task_prefix[0].replace("<TRG>", tgt), ctx]) if task_prefix[1] == "pre"
                   else " ".join([ctx, task_prefix[0].replace("<TRG>", tgt)])
                   for tgt, ctx in zip(df["Targets"], df[context_col])]
        df["Generated_Definition"] = self._generate_definitions(prompts, df["Targets"].tolist())
        df["Real_Contexts"] = prompts
        return self.encode_definitions(df, encode=encode_definitions, return_df=return_df)

    def save_definitions(self, df, output_file):
        """Save the DataFrame with definitions to a TSV file."""
        df.to_csv(output_file, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_NONE)
        self.logger.info(f"Saved definitions to {output_file}")