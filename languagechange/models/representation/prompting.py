from typing import List, Union
from languagechange.usages import TargetUsage
import getpass
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
import logging
import trankit


class SCFloat(BaseModel):
    change : float = Field(description='The semantic change on a scale from 0 to 1.',le=1, ge=0)


class CosSim(BaseModel):
    change : float = Field(description='The semantic similarity on a scale from 0 to 1.',le=1, ge=0)


class DURel(BaseModel):
    change : int = Field(description='The semantic similary from 1 to 4, where 1 is unrelated, 2 is distantly related, 3 is closely related and 4 is identical.',le=4, ge=1)


class PromptModel:
    def __init__(self, model_name : str, model_provider : str, langsmith_key : str = None, provider_key_name : str = None, provider_key : str = None, structure:Union[str,BaseModel]="float", language : str = None, **kwargs):
        self.model_name = model_name
        self.language = language

        os.environ["LANGSMITH_TRACING"] = "true"
        
        # The keys can either be passed as arguments, stored as an environment variable or put in manually
        if langsmith_key != None:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
        elif not os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for LangSmith: ")

        if provider_key_name is None:
            provider_key_names = {"openai":"OPENAI_API_KEY",
                                 "anthropic":"ANTHROPIC_API_KEY",
                                 "azure":"AZURE_OPENAI_API_KEY",
                                 "groq":"GROQ_API_KEY",
                                 "cohere":"COHERE_API_KEY",
                                 "nvidia":"NVIDIA_API_KEY",
                                 "fireworks":"FIREWORKS_API_KEY",
                                 "mistralai":"MISTRAL_API_KEY",
                                 "together":"TOGETHER_API_KEY",
                                 "ibm":"WATSONX_APIKEY",
                                 "databricks":"DATABRICKS_TOKEN",
                                 "xai":"XAI_API_KEY"}
            if model_provider in provider_key_names.keys():
                provider_key_name = provider_key_names[model_provider]
                
        if provider_key != None:
            os.environ[provider_key_name] = provider_key
        elif provider_key_name != None and not os.environ.get(provider_key_name):
            os.environ[provider_key_name] = getpass.getpass(f"Enter API key for {model_provider}: ")

        # special cases
        if model_provider == "azure":
            # pip install -qU "langchain[openai]"
            from langchain_openai import AzureChatOpenAI
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            )

        elif model_provider == "ibm":
            if 'url' in kwargs and 'project_id' in kwargs:
                # pip install -qU "langchain-ibm"
                from langchain_ibm import ChatWatsonx
                
                self.llm = ChatWatsonx(model_id = model_name,
                              url=kwargs.get('url'),
                              project_id=kwargs.get('project_id')
                              )
            else:
                raise Exception("Pass 'url' and 'project_id' to initialize a ChatWatsonx model.")
            
        elif model_provider == "databricks":
            if 'databricks_host_url' in kwargs:
                os.environ["DATABRICKS_HOST"] = kwargs.get('databricks_host_url')
            else:
                raise Exception("Pass 'databricks_host_url' to initialize a Databricks model.")
            # pip install -qU "databricks-langchain"
            from databricks_langchain import ChatDatabricks
            self.llm = ChatDatabricks(endpoint=model_name)
        else:
            try:
                self.llm = init_chat_model(model_name, model_provider=model_provider)
            except:
                logging.error("Could not initialize chat model.")
                raise Exception

        if not isinstance(structure,str) and issubclass(structure, BaseModel):
            if 'change' in structure.model_fields:
                self.structure = structure
            else:
                logging.error("A custom BaseModel needs to have a field named 'change'.")
                raise Exception
        
        self.set_structure(structure)

    def set_structure(self, structure):
        if isinstance(structure, str):
            structure = {"float": SCFloat, "DURel": DURel, "cosine": CosSim}.get(structure, None)
        if structure is not None:
            self.model = self.llm.with_structured_output(structure)
        else:
            self.model = self.llm

    def get_response(self, target_usages : List[TargetUsage], 
                     system_message = 'You are a lexicographer',
                     user_prompt_template = 'Please provide a number measuring how different the meaning of the word \'{target}\' is between the following example sentences: \n1. {usage_1}\n2. {usage_2}',
                     lemmatize = True):
        """
        Takes as input two target usages and returns the degree of semantic change between them, using a chat model with structured output.
        Args:
            target_usages (List[TargetUsage]): a list of target usages with the same target word.
            system_message (str): the system message to use in the prompt
            user_prompt_template (str): template to use for the user message in the prompt.
            lemmatize (bool): whether the target word should be lemmatized in the prompt or not. Uses trankit to lemmatize.
        Returns:
            int or float or str: the degree of semantic change between the two instances of the target word, alternatively the whole message content if the output is not structured.
        """
        
        assert len(target_usages) == 2

        words = []
        sentences = []
        for usage in target_usages:
            words.append(usage.text()[usage.offsets[0]:usage.offsets[1]])
            sentences.append(usage.text())

        def get_lemma(tokenized, usage):
            for token in tokenized['tokens']:
                if token['span'] == tuple(usage.offsets):
                    return(token['lemma'])
                
        if lemmatize:
            if self.language == None:
                logging.error("Could not lemmatize using trankit because no language is set. Please pass a value to 'language' when initializing the model.")
                raise Exception
            p = trankit.Pipeline(self.language)
            lemmatized = [p.lemmatize(sentence, is_sent = True) for sentence in sentences]
            lemmas = [get_lemma(lemmatized[i], target_usages[i]) for i in range(2)]
            
            if lemmas[0] != lemmas[1]:
                logging.info("Lemmas of the two target words differ, are you sure they are different forms of the same lexeme?")
            target = lemmas[0]
        else:
            target = words[0]

        prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", user_prompt_template)]
        )
            
        prompt = prompt_template.invoke({"target": target, "usage_1": sentences[0], "usage_2": sentences[1]})
        
        try:
            response = self.model.invoke(prompt)
        except:
            logging.error("Could not run chat completion.")
            raise Exception
        
        try:
            return response.change
        except:
            return response