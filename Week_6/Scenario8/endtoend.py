import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# Prefer Serper; it’s the documented tool
try:
    from crewai_tools import SerperApiTool as SerperTool  # newer naming
except ImportError:
    from crewai_tools import SerperDevTool as SerperTool  # legacy naming

load_dotenv()

# Gemini (text model; free-tier friendly)
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
    max_tokens=900,
    timeout=120,
)

# Serper search tool (uses SERPER_API_KEY)
search_tool = SerperTool()

researcher = Agent(
    role="AI News Researcher",
    goal="Find 5 important AI news stories from the last 7 days with sources.",
    backstory="You use reputable web sources and return concise, sourced bullets.",
    tools=[search_tool],
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Content Creator",
    goal="Turn the research into an engaging LinkedIn post for engineering leaders.",
    backstory="You write crisp, credible, friendly posts.",
    llm=llm,
    verbose=True,
)

research_task = Task(
    description=(
        "Use the search tool to find exactly 5 relevant AI news items from the last 7 days. "
        "For each, return a headline, one‑line summary, and a clickable URL."
    ),
    expected_output=(
        "- Headline 1 — one line\n  URL\n"
        "- Headline 2 — one line\n  URL\n"
        "- Headline 3 — one line\n  URL\n"
        "- Headline 4 — one line\n  URL\n"
        "- Headline 5 — one line\n  URL"
    ),
    agent=researcher,
    verbose=True,
)

write_post_task = Task(
    description=(
        "Using the 5 items, write a ~200‑word LinkedIn post with a punchy hook, "
        "scannable bullets, and a closing question."
    ),
    expected_output="~200‑word LinkedIn post with hook, bullets, and a closing question.",
    agent=writer,
    context=[research_task],
    verbose=True,
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_post_task], verbose=True)
print(crew.kickoff())