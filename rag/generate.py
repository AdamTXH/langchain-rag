from rag import retrieve
from rag.utils import benchmark

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from ipex_llm.langchain.llms import TransformersLLM
import os

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_core.outputs import GenerationChunk
from typing import Iterator
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun, CallbackManager

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.load import dumpd
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.runnables import RunnableConfig, ensure_config, get_config_list

from langchain.llms.base import BaseLLM

from transformers import TextIteratorStreamer

from langfuse.utils import _get_timestamp

class MyTextIteratorStreamer(TextIteratorStreamer):
    '''
    Refer to https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
    Modified to return output token count alongside output
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokens_count = 0

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        
        # count output token
        self._tokens_count += 1

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put((text, self._tokens_count), timeout=self.timeout)
        if stream_end:
            #print("stop signal: ", self.stop_signal)
            self.text_queue.put((self.stop_signal, self._tokens_count), timeout=self.timeout)

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value[0] == self.stop_signal:
            raise StopIteration()
        else:
            return value

class MyTransformersLLM(TransformersLLM):

    _actual_input_tokens = 0
    _actual_output_tokens = 0
    _completion_start_timestamp = 0
  
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        '''
        Modified to count input token and get output tokens count from streamer
        '''
        
        from threading import Thread
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        type(self)._actual_input_tokens = input_ids.shape[1]
        input_ids = input_ids.to(self.model.device)
        streamer = MyTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        if stop is not None:
            from transformers.generation.stopping_criteria import StoppingCriteriaList
            from transformers.tools.agents import StopSequenceCriteria
            # stop generation when stop words are encountered
            # TODO: stop generation when the following one is stop word
            stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop,
                                                                            self.tokenizer)])
        else:
            stopping_criteria = None

        generation_kwargs = dict(inputs=input_ids, streamer=streamer,
                                        stopping_criteria=stopping_criteria)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        # output = self.model.generate(input_ids, streamer=streamer,
        #                                 stopping_criteria=stopping_criteria, **kwargs)
        thread.start()
        for item in streamer:
            #text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            #print(output)
            output, tokens_count = item
            type(self)._actual_output_tokens = tokens_count
            #print("Output: ", output)
            #print("Token count: ", tokens_count)
            yield GenerationChunk(text=output)
            
    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        '''
        Modified to send token counts and completion start time (for first token latency) to langfuse
        '''
        if type(self)._stream == BaseLLM._stream:
            # model doesn't implement streaming, so use default implementation
            yield self.invoke(input, config=config, stop=stop, **kwargs)
        else:
            prompt = self._convert_input(input).to_string()
            config = ensure_config(config)
            params = self.dict()
            params["stop"] = stop
            params = {**params, **kwargs}
            options = {"stop": stop}
            callback_manager = CallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                config.get("metadata"),
                self.metadata,
            )
            (run_manager,) = callback_manager.on_llm_start(
                dumpd(self),
                [prompt],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                batch_size=1,
            )

            # re-initialize metrics
            type(self)._actual_input_tokens = 0
            type(self)._actual_output_tokens = 0
            type(self)._completion_start_timestamp = 0

            generation: Optional[GenerationChunk] = None
            try:
                for chunk in self._stream(
                    prompt, stop=stop, run_manager=run_manager, **kwargs
                ):
                    if type(self)._completion_start_timestamp == 0:
                        type(self)._completion_start_timestamp = _get_timestamp()
                    yield chunk.text
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
            except BaseException as e:
                run_manager.on_llm_error(
                    e,
                    response=LLMResult(
                        generations=[[generation]] if generation else []
                    ),
                )
                raise e
            else:
                run_manager.on_llm_end(LLMResult(generations=[[generation]], llm_output=
                                                {
                                                    "usage":{
                                                        "input_tokens": type(self)._actual_input_tokens,
                                                        "output_tokens": type(self)._actual_output_tokens,
                                                    },
                                                    "completion": type(self)._completion_start_timestamp
                                                }
        ))

def get_llm_hf():
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    hf = HuggingFacePipeline(pipeline=pipe)
    return hf

# Create an llm chat model with Azure API
def get_llm_azure(deployment_name, max_tokens=2048, temperature=0.7):
    return AzureChatOpenAI(
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        azure_deployment=deployment_name,
        max_tokens=max_tokens,
        temperature=temperature,

    )

# Create a llamacpp model
def get_llm_llamacpp(model_path, n_gpu_layers=999, temperature=0, max_tokens=2048, top_p=1, verbose=False, n_ctx=2048):
    return LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        verbose=verbose,
        n_ctx=n_ctx
    )

def get_llm_ipex():
    # TODO: accept arguments
    return  MyTransformersLLM.from_model_id(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs={"temperature": 0, "max_length": 512, "trust_remote_code": True},
            device_map='xpu'
        )

# Create prompt template
def get_prompt_template(template=None):
    if template is None:
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.

        {context}

        Question: {question}

        Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    return custom_rag_prompt

# Format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the rag chain
def create_rag_chain(retriever, custom_rag_prompt, llm):
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
    )
    return rag_chain

# Streaming the output to improve responsiveness
def stream_output(rag_chain, query, trace=False, callback_handler=None):
    response = ""
    if trace:
        for text in rag_chain.stream(query, config={"callbacks": [callback_handler]}):
            print(text, end="", flush=True)
            response += text 
    else:
        for text in rag_chain.stream(query):
            print(text, end="", flush=True)
            response += text 
    return response