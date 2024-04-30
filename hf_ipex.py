from haystack import component
from haystack.dataclasses import ChatMessage

from typing import List, Optional, Dict, Any
from transformers import TextIteratorStreamer

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

@component
class IpexHuggingFaceLocalChatGenerator:
    def __init__(self, model, device, streaming_callback, model_kwargs) -> None:
        self.model = model
        self.device = device
        self.streaming_callback = streaming_callback
        self.generation_kwargs = model_kwargs
        self.tokenizer_id = None
        self.chat_template = None

    def warm_up(self):
        if self.llm is None:
            self.llm = self._warm_up()
    
    def _warm_up(self):
        model_id = self.model
        tokenizer_id = self.tokenizer_id
        device_map = self.device
        # model_kwargs = self.model_kwargs

        try:
            from ipex_llm.transformers import (
                AutoModel,
                AutoModelForCausalLM,
                # AutoModelForSeq2SeqLM,
            )
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            # invalidInputError(
            #     "Could not import transformers python package. "
            #     "Please install it with `pip install transformers`."
            # )
            pass

        #_model_kwargs = model_kwargs or {}
        # TODO: may refactore this code in the future
        if tokenizer_id is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
            except:
                tokenizer = LlamaTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            except:
                tokenizer = LlamaTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer = tokenizer

        # TODO: may refactore this code in the future
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True,
                                                         optimize_model=True,
                                                     trust_remote_code=True, use_cache=True, cpu_embedding=False).eval()
        except:
            pass
            #model = AutoModel.from_pretrained(model_id, load_in_4bit=True, **_model_kwargs)

        # TODO: may refactore this code in the future
        if 'xpu' in device_map:
            model = model.to(device_map)

        # if "trust_remote_code" in _model_kwargs:
        #     _model_kwargs = {
        #         k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
        #     }

        return model
    
    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :returns:
            A list containing the generated responses as ChatMessage instances.
        """
        if self.llm is None:
            raise RuntimeError("The generation model has not been loaded. Please call warm_up() before running.")

        tokenizer = self.tokenizer

        # Check and update generation parameters
        # generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # stop_words = generation_kwargs.pop("stop_words", []) + generation_kwargs.pop("stop_sequences", [])
        # # pipeline call doesn't support stop_sequences, so we need to pop it
        # stop_words = self._validate_stop_words(stop_words)

        # # Set up stop words criteria if stop words exist
        # stop_words_criteria = StopWordsCriteria(tokenizer, stop_words, self.pipeline.device) if stop_words else None
        # if stop_words_criteria:
        #     generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_words_criteria])

        # if self.streaming_callback:
        #     num_responses = generation_kwargs.get("num_return_sequences", 1)
        #     if num_responses > 1:
        #         logger.warning(
        #             "Streaming is enabled, but the number of responses is set to %d. "
        #             "Streaming is only supported for single response generation. "
        #             "Setting the number of responses to 1.",
        #             num_responses,
        #         )
        #         generation_kwargs["num_return_sequences"] = 1
        #     # streamer parameter hooks into HF streaming, HFTokenStreamingHandler is an adapter to our streaming
        #     generation_kwargs["streamer"] = HFTokenStreamingHandler(tokenizer, self.streaming_callback, stop_words)

        # Prepare the prompt for the model
        prepared_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, chat_template=self.chat_template, add_generation_prompt=True
        )

        # # Avoid some unnecessary warnings in the generation pipeline call
        # generation_kwargs["pad_token_id"] = (
        #     generation_kwargs.get("pad_token_id", tokenizer.pad_token_id) or tokenizer.eos_token_id
        # )

        # # Generate responses
        # output = self.pipeline(prepared_prompt, **generation_kwargs)
        # replies = [o.get("generated_text", "") for o in output]

        # # Remove stop words from replies if present
        # for stop_word in stop_words:
        #     replies = [reply.replace(stop_word, "").rstrip() for reply in replies]

        # # Create ChatMessage instances for each reply
        # chat_messages = [
        #     self.create_message(reply, r_index, tokenizer, prepared_prompt, generation_kwargs)
        #     for r_index, reply in enumerate(replies)
        # ]
        # return {"replies": chat_messages}

        print("Prepared prompt: ", prepared_prompt)

        from threading import Thread
        input_ids = self.tokenizer.encode(prepared_prompt, return_tensors="pt")
        type(self)._actual_input_tokens = input_ids.shape[1]
        input_ids = input_ids.to(self.model.device)
        streamer = MyTextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        stop = None
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
        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        # output = self.model.generate(input_ids, streamer=streamer,
        #                                 stopping_criteria=stopping_criteria, **kwargs)

        final_output = ''
        thread.start()
        for item in streamer:
            #text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            #print(output)
            output, tokens_count = item
            type(self)._actual_output_tokens = tokens_count
            #print("Output: ", output)
            #print("Token count: ", tokens_count)

            # streaming handler
            
            self.streaming_callback(output)

            final_output += output

        response = ChatMessage.from_assistant(final_output)

        return {"replies": response}

        



        
