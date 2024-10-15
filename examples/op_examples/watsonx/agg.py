import pandas as pd

import lotus
from lotus.models import WatsonxAIModel

lotus.logger.setLevel("DEBUG")

lm = WatsonxAIModel(
    model_id="mistralai/mistral-large",
    max_tokens=4095,
    logprobs=False
)

lotus.settings.configure(lm=lm)
data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
        "Cooking",
        "Food Sciences",
    ]
}
df = pd.DataFrame(data)
df = df.sem_agg("translate all {Course Name}")
print(df._output[0])
