from crewai.flow.flow import flow, listen, start, router  # type: ignore
from litellm import completion
from textwrap import dedent
from trip_agents import TripAgents
from trip_tasks import TripTasks


@flow
def trip_flow(origin: str, cities: str, date_range: str, interests: str) -> str:
    """
    A CrewAI flow to plan a trip.
    """
    agents = TripAgents()
    tasks = TripTasks()

    # Get the agents
    city_selector_agent = agents.city_selection_agent()
    local_expert_agent = agents.local_expert()
    travel_concierge_agent = agents.travel_concierge()

    # Create tasks using the agents and inputs
    identify_task = tasks.identify_task(
        city_selector_agent,
        origin,
        cities,
        interests,
        date_range
    )
    gather_task = tasks.gather_task(
        local_expert_agent,
        origin,
        interests,
        date_range
    )
    plan_task = tasks.plan_task(
        travel_concierge_agent, 
        origin,
        interests,
        date_range
    )

    # Create and run the Crew flow
    crew = Crew(
        agents=[
            city_selector_agent, local_expert_agent, travel_concierge_agent
        ],
        tasks=[identify_task, gather_task, plan_task],
        verbose=True
    )
    
    result = crew.kickoff()
    return result


if __name__ == "__main__":
    print("## Welcome to Trip Planner Crew")
    print('-------------------------------')
    location = input(dedent("""\
        From where will you be traveling from?
    """))
    cities = input(dedent("""\
        What are the cities options you are interested in visiting?
    """))
    date_range = input(dedent("""\
        What is the date range you are interested in traveling?
    """))
    interests = input(dedent("""\
        What are some of your high level interests and hobbies?
    """))

    # Run the decorated flow function
    result = trip_flow(origin=location, cities=cities, date_range=date_range, interests=interests)
    
    print("\n\n########################")
    print("## Here is your Trip Plan")
    print("########################\n")
    print(result)
