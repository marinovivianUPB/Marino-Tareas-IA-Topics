import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, CodeDocsSearchTool

load_dotenv()

llm = LLM(model="gpt-4o-mini")
current_dir = os.path.dirname(os.path.abspath(__file__))

analyzer_agent = Agent(
    role="Python Code Analyzer",
    goal="Analyze documents and provide relevant code insights.",
    backstory="""You are an expert Python developer tasked with reading and analyzing a scientific paper. This scientific paper is in a PDF format. You are responsible for analyzing the code in the paper or to translate what it says into code. 
    You must also search and read the FastAPI and SciPy documentation so that this code can be implemented on a python script.""",
    llm=llm,
)

coder_agent = Agent(
    role="Senior Python Developer",
    goal="Generate code based on what is described in the analysis and put it in a Python file.",
    backstory="""You are a senior Python developer with extensive experience in translating documentation or 
    analysis into clean, executable code. You are responsible for replicating the code from the analysis 
    provided by the analyzer agent. You must only replicate the code detailed in the paper. You must use 
    the FastAPI and SciPy documentation to replicate the code. You must write a Python script with clean 
    and well-documented code. The Python script must be executable. You will execute it once you finish writing it.""",
    allow_code_execution=True,
    llm=llm,
)

read_pdf_task = Task(
    description="Read the provided PDF document and analyze the code described in it or translate what it say into code.",
    expected_output="A detailed analysis of the code in the PDF document.",
    agent=analyzer_agent,
    tools=[FileReadTool(file_path=os.path.join(current_dir, "Lect-7-DM.pdf"))],
)

fetch_fastapi_docs_task = Task(
    description="Fetch and analyze the FastAPI documentation.",
    expected_output="A summary of key FastAPI documentation points relevant to the code.",
    agent=analyzer_agent,
    tools=[CodeDocsSearchTool(query="https://fastapi.tiangolo.com/")],
)

fetch_scipy_docs_task = Task(
    description="Fetch and analyze the SciPy documentation.",
    expected_output="A summary of key SciPy documentation points relevant to the code.",
    agent=analyzer_agent,
    tools=[CodeDocsSearchTool(query="https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide")],
)

generate_code_task = Task(
    description="Generate an executable Python script based on the PDF document, FastAPI docs, and SciPy docs analysis. The code should implement the core functionality described and be well-documented. The output should be valid Python code, not a description.",
    expected_output="A valid Python `.py` file implementing the functionality described in the PDF document, with clear comments and code structure. No descriptions, just the actual code.",
    agent=coder_agent,
    output_file="replicated_gen.py"
)

dev_crew = Crew(
    agents=[analyzer_agent, coder_agent],
    tasks=[read_pdf_task, fetch_fastapi_docs_task, fetch_scipy_docs_task, generate_code_task],
    verbose=True
)

result = dev_crew.kickoff()

print(result)
