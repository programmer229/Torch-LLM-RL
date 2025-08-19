
from openai import OpenAI

def inference(model, system_prompt, problem, max_tokens=1024, temperature=0.7):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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"  
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content

