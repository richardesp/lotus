import os
from typing import Any, Dict, List, Optional, Tuple, Union
import itertools

import ibm_watsonx_ai.wml_client_error
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

import lotus
from lotus.models.lm import LM

ERRORS = (
    ibm_watsonx_ai.wml_client_error.NoWMLCredentialsProvided,
    ibm_watsonx_ai.wml_client_error.WMLClientError,
)


class WatsonxAIModel(LM):

    def __init__(
        self,
        model_id: str = "ibm/granite-13b-instruct-v2",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        max_ctx_len: int = 8192,
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.model_id = model_id
        self.max_ctx_len = max_ctx_len
        self.max_tokens = kwargs.get("max_tokens", 2048)

        self.kwargs = {
            GenTextParamsMetaNames.DECODING_METHOD: "greedy",
            GenTextParamsMetaNames.MAX_NEW_TOKENS: self.max_tokens,
            GenTextParamsMetaNames.STOP_SEQUENCES: [],
            GenTextParamsMetaNames.REPETITION_PENALTY: 1,
            **kwargs,
        }

        credentials = {
            "url": url or os.environ.get("WATSONX_URL", None),
            "apikey": api_key or os.environ.get("WATSONX_APIKEY", None),
        }

        lotus.logger.debug(f"WatsonxAIModel.__init__ self.kwargs: {self.kwargs}")

        self.client = Model(
            model_id=self.model_id,
            params=self.kwargs,
            credentials=credentials,
            project_id=project_id or os.environ.get("WATSONX_PROJECTID", None),
            space_id=space_id,
        )

    def count_tokens(self, prompt: Union[str, list]) -> int:
        """
        Counts the number of tokens in a given prompt using the client's tokenizer.

        Args:
            prompt (Union[str, list]): The input prompt which can be a string or a list of strings.

        Returns:
            int: The token count from the model, or 0 if token count is not available.
        """

        lotus.logger.debug(f"WatsonxAIModel.count_tokens prompt: {prompt}")

        if isinstance(prompt, list):
            prompt_input = " ".join(
                [f"{batch['role']}: {batch['content']}\n" for batch in prompt]
            )
        else:
            prompt_input = f"{prompt}\n"

        lotus.logger.debug(f"WatsonxAIModel.count_tokens prompt_input: {prompt_input}")

        # Use the client's tokenize method to get the result
        response = self.client.tokenize(prompt=prompt_input, return_tokens=False)

        result = response.get("result", {})
        token_count = result.get("token_count", 0)

        # Return the token count (defaults to 0 if not found)
        return token_count

    def format_logprobs_for_cascade(
        self, logprobs: List[List[Tuple[str, float]]]
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Formats the logprobs for the cascade.

        Args:
            logprobs (List[List[Tuple[str, float]]]): A list of logprobs where each element is a list of tuples,
            each tuple containing a token (str) and its corresponding log probability (float).

        Returns:
            Tuple[List[List[str]], List[List[float]]]: A tuple containing two lists:
            - List of tokens (List[List[str]])
            - List of corresponding confidences (List[List[float]])
        """
        lotus.logger.debug(f"WatsonxAIModel.__call__ logprobs: {logprobs}")

    def __call__(
        self, messages_batch: List | List[List], **kwargs: Dict[str, Any]
    ) -> List | Tuple[List]:
        """Invoke the LLM.

        Args:
            messages_batch (Union[List, List[List]]): Either one prompt or a list of prompts in message format.
            kwargs (Dict[str, Any]): Additional keyword arguments. They can be used to specify inference parameters.

        Returns:
            Union[List, Tuple[List, List]]: A list of outputs for each prompt in the batch. If logprobs is specified in the keyword arguments,
            then a list of logprobs is also returned.
        """

        new_messages_batch = []

        for batch in messages_batch:
            combined_message = " ".join(
                [f"{message['role']}: {message['content']}\n" for message in batch]
            )
            new_messages_batch.append(combined_message)

        lotus.logger.debug(
            f"WatsonxAIModel.__call__ messages_batch: {new_messages_batch}"
        )
        lotus.logger.debug(f"WatsonxAIModel.__call__ kwargs: {kwargs}")

        return_options = {
            "input_text": kwargs.get(
                "input_text", False
            ),  # Add the input text at the text returned
            "generated_tokens": True,
            "input_tokens": True,
            "token_logprobs": kwargs.get("logprobs", False),  # Enable token logprobs
            "token_ranks": False,
        }

        params = {
            GenTextParamsMetaNames.RETURN_OPTIONS: return_options,
            GenTextParamsMetaNames.MAX_NEW_TOKENS: self.max_tokens,
        }

        lotus.logger.debug(f"WatsonxAIModel.__call__ return_options: {return_options}")

        generated_outputs = self.client.generate(
            prompt=new_messages_batch, params=params
        )

        lotus.logger.debug(
            f"WatsonxAIModel.__call__ generated_outputs: {generated_outputs}"
        )

        generated_texts = [
            item["results"][0]["generated_text"] for item in generated_outputs
        ]

        lotus.logger.debug(
            f"WatsonxAIModel.__call__ generated_texts (True/False): {generated_texts}"
        )

        if kwargs.get("logprobs", False):

            print(generated_outputs)
            generated_tokens = [result for result in generated_outputs["results"]]

            generated_logprobs = [
                [text.get("logprob", float("-inf")) for text in generated_token]
                for generated_token in generated_tokens
            ]

            return generated_texts, generated_logprobs

        # Return only the generated text if logprobs are not requested
        return generated_texts
