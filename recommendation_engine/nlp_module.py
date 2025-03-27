from datetime import datetime

negative_cancellation_texts = [
    "I cannot come in tomorrow morning due to a family emergency.",
    "I'm unable to work my shift this evening because I'm feeling unwell.",
    "I regret to inform you that I can't make it to my shift on Friday.",
    "Unfortunately, I have to cancel my shift scheduled for Saturday evening.",
    "I won't be able to come in for my morning shift on Monday due to personal reasons.",
    "I have to withdraw from my evening shift on Tuesday because of a prior commitment.",
    "I'm really sorry, but I cannot fulfill my shift on Wednesday.",
    "Due to unforeseen circumstances, I must cancel my shift on Thursday.",
    "I cannot work my scheduled shift this weekend because of a family obligation.",
    "I regret to say that I can't make it to my shift on Friday afternoon.",
    "I won't be able to cover my shift on Sunday due to a last-minute issue.",
    "I'm unable to come in for my shift tomorrow morning because of transportation problems.",
    "I have to cancel my evening shift on Tuesday due to a personal emergency.",
    "Unfortunately, I cannot work my shift on Saturday because of a scheduling conflict.",
    "I regret to inform you that I can't make it to my shift on Thursday night."
]

from google import genai

client = genai.Client(api_key="AIzaSyD3znXcc5cVneQuQZ42LL_g5v5YQuA3mp0")

prompt = f'''
            Today's date is {datetime.now().strftime("%Y-%m-%d")}
            "Extract the cancellation intent and the date from the following text:"
            {negative_cancellation_texts}

            #Important Instruction:
            The date is in the format of "MM/DD/YYYY".
            Calculate the date of the cancellation based on the text. Ignore the time of the day.

            Return your response as JSON with:
            - 'text': text,
            - 'cancellation_intent': cancellation intent,
            - 'date': date.
            '''


response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
)

print(response.text)
