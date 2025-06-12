from mistralai import Mistral, UserMessage

def get_insights(stats_dict, api_key, model="mistral-large-latest"):
    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=f"""
You are an audio signal processing expert.
Analyze these stats:

{stats_dict}

Give insight about speech vs silence, energy and anomalies.
""")]
    response = client.chat.complete(model=model, messages=messages)
    return response.choices[0].message.content
