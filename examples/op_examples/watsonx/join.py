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
        "History of the Atlantic World",
        "Riemannian Geometry",
        "Operating Systems",
        "Food Science",
        "Compilers",
        "Intro to computer science",
    ]
}

data2 = {"Skill": ["Math", "Computer Science"]}

df1 = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
join_instruction = "Taking {Course Name:left} will help me learn {Skill:right}"
res = df1.sem_join(df2, join_instruction)
print(res)
