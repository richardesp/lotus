import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import ibm_watsonx_ai.wml_client_error
from ibm_watsonx_ai.foundation_models import Model

import lotus
from lotus.models.lm import LM

ERRORS = (
    ibm_watsonx_ai.wml_client_error.NoWMLCredentialsProvided,
    ibm_watsonx_ai.wml_client_error.WMLClientError,
)


class WatsonxAIModel(LM):

    def __init__(
        self,
        model_id: str = "ibm/granite-13b-chat-v2",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.model_id = model_id

        self.kwargs = {
            "decoding_method": "greedy",
            "max_new_tokens": 512,
            "stop_sequences": ["\n"],
            "repetition_penalty": 1,
            **kwargs,
        }

        credentials = {
            "url": url or os.environ.get("WATSONX_URL", None),
            "apikey": api_key or os.environ.get("WATSONX_APIKEY", None),
        }

        self.client = Model(
            model_id=self.model_id,
            params=self.kwargs,
            credentials=credentials,
            project_id=project_id,
            space=space_id,
        )
        
        
