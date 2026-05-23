from langgraph_groq import ChatGroq

def planner_node(state):
    planner = ChatGroq(model="gpt-4o", temperature=0.7)
    plan = planner(
        f"""
        You are a helpful assistant that creates a plan to achieve the following goal: {state['goal']}
        The plan should be broken down into clear, actionable steps.
        """
    )
    return plan

