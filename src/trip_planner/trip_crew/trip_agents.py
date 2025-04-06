from crewai import Agent, Crew, Process  # type: ignore
from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools

class TripCrew:
    def city_selection_agent(self) -> Agent:
        return Agent(
            role="City Selection Expert",
            goal="Select the best city based on weather, season, and prices",
            backstory="An expert in analyzing travel data to pick ideal destinations",
            tools=[
                SearchTools.search_internet,
                BrowserTools.scrape_and_summarize_website,
            ],
            verbose=True
        )
        
    def local_expert_agent(self) -> Agent:
        return Agent(
            role="Local Expert at this city",
            goal="Provide the BEST insights about the selected city",
            backstory=(
                "A knowledgeable local guide with extensive information "
                "about the city, its attractions and customs"
            ),
            tools=[
                SearchTools.search_internet,
                BrowserTools.scrape_and_summarize_website,
            ],
            verbose=True
        )
        
    def travel_concierge_agent(self) -> Agent:
        return Agent(
            role="Amazing Travel Concierge",
            goal=(
                "Create the most amazing travel itineraries with budget and "
                "packing suggestions for the city"
            ),
            backstory=(
                "Specialist in travel planning and logistics with decades of experience"
            ),
            tools=[
                SearchTools.search_internet,
                BrowserTools.scrape_and_summarize_website,
                CalculatorTools.calculate,
            ],
            verbose=True
        )
        
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.city_selection_agent(),
                self.local_expert_agent(),
                self.travel_concierge_agent()
            ],
            tasks=[],  # You can add Task instances here if needed.
            process=Process.sequential,
            verbose=True
        )
