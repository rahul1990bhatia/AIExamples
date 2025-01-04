import re
from openai import OpenAI
from dotenv import load_dotenv

_ = load_dotenv()

class RestaurantAgent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        self.client = OpenAI()
        if self.system:
            self.messages.append({'role': 'system', 'content': system})

    def __call__(self, message):
        self.messages.append({'role': 'user', 'content': message})
        result = self.execute()
        self.messages.append({'role': 'assistant', 'content': result})
        return result
    
    def execute(self):
        completion = self.client.chat.completions.create(
            model='gpt-4',
            temperature=0,
            messages=self.messages
        )
        return completion.choices[0].message.content

def get_restaurant_rating(name):
    ratings = {
        "Pizza Palace": {"rating": 4.5, "reviews": 230},
        "Burger Barn": {"rating": 4.2, "reviews": 185},
        "Sushi Supreme": {"rating": 4.8, "reviews": 320}
    }
    return ratings.get(name, {"rating": 0, "reviews": 0})

known_actions = {
    "get_rating": get_restaurant_rating
}

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your reasoning about the restaurant comparison.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:
get_rating:
e.g. get_rating: Pizza Palace
Returns rating and review count for the specified restaurant

Example session:
Question: Which restaurant has better ratings, Pizza Palace or Burger Barn?
Thought: I should check the ratings for both restaurants
Action: get_rating: Pizza Palace
PAUSE
"""

def query(question, max_turns=5):
    action_re = re.compile('^Action: (\w+): (.*)$')
    bot = RestaurantAgent(prompt)
    next_prompt = question
    
    for i in range(max_turns):
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a) 
            for a in result.split('\n') 
            if action_re.match(a)
        ]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            observation = known_actions[action](action_input)
            next_prompt = f"Observation: {observation}"
        else:
            return
        
question = """which resturant have better rating, Pizza Palace or Burger Barn?"""
query(question)