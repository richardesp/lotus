import pandas as pd

import lotus
from lotus.models import OpenAIModel, WatsonxAIModel

lotus.logger.setLevel("DEBUG")

lm = WatsonxAIModel(
    model_id="ibm/granite-7b-lab",
    max_tokens=128,
    logprobs=False
)

lotus.settings.configure(lm=lm)
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
        "Operating Systems and Systems Programming",
        "Compilers",
        "Computer Networks",
        "Deep Learning",
        "Graphics",
        "Databases",
        "Art History",
    ]
}
df = pd.DataFrame(data)
user_instruction = "{Course Name} is related with compilers?"
new_df = df.sem_filter(user_instruction)
new_df
